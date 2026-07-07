from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from data_utils import (
    ASSET_COLUMN,
    DATE_COLUMN,
    FINAL_REQUIRED_COLUMNS,
    add_causal_features,
    attach_market_state,
    compute_market_state,
    feature_metadata,
    load_config,
    read_table,
    sort_panel,
    universe_metadata,
    write_json,
    write_table,
)


def build_feature_panel(config: dict) -> pd.DataFrame:
    paths = config.get("paths", {})
    intermediate_dir = Path(paths.get("intermediate_dir", "data/intermediate"))
    prices_path = Path(paths.get("prices_daily", intermediate_dir / "prices_daily.parquet"))
    universe_path = Path(paths.get("universe_daily", intermediate_dir / "universe_daily.parquet"))

    if not prices_path.exists():
        raise FileNotFoundError(f"missing prices table: {prices_path}")
    if not universe_path.exists():
        raise FileNotFoundError(f"missing universe table: {universe_path}")

    prices = read_table(prices_path)
    universe = read_table(universe_path)
    panel = merge_price_universe(prices, universe)

    optional_tables = [
        ("fundamentals_asof", "fundamentals_asof.parquet"),
        ("macro_asof", "macro_asof.parquet"),
        ("factor_exposures_daily", "factor_exposures_daily.parquet"),
        ("liquidity_daily", "liquidity_daily.parquet"),
        ("events_asof", "events_asof.parquet"),
        ("short_interest_daily", "short_interest_daily.parquet"),
    ]
    for key, default_name in optional_tables:
        table_path = Path(paths.get(key, intermediate_dir / default_name))
        if table_path.exists():
            panel = merge_optional_table(panel, read_table(table_path))

    panel = add_required_microstructure_defaults(panel)
    panel = add_causal_features(panel)
    state_path = Path(paths.get("market_state_daily", intermediate_dir / "market_state_daily.parquet"))
    state = read_table(state_path) if state_path.exists() else compute_market_state(panel)
    panel = attach_market_state(panel, state)
    ordered = [c for c in FINAL_REQUIRED_COLUMNS if c in panel.columns]
    panel = panel[ordered + [c for c in panel.columns if c not in ordered]]
    return sort_panel(panel)


def merge_price_universe(prices: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    prices = sort_panel(prices)
    universe = sort_panel(universe)
    common = [DATE_COLUMN, ASSET_COLUMN]
    duplicate_price_cols = [c for c in universe.columns if c in prices.columns and c not in common]
    universe = universe.drop(columns=duplicate_price_cols)
    return prices.merge(universe, on=common, how="left")


def merge_optional_table(panel: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table[DATE_COLUMN] = pd.to_datetime(table[DATE_COLUMN])
    keys = [DATE_COLUMN]
    if ASSET_COLUMN in table.columns:
        keys.append(ASSET_COLUMN)
    duplicate_cols = [c for c in table.columns if c in panel.columns and c not in keys]
    table = table.drop(columns=duplicate_cols)
    return panel.merge(table, on=keys, how="left")


def add_required_microstructure_defaults(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    if "adjusted_close" not in df and "close" in df:
        df["adjusted_close"] = df["close"]
    if "price" not in df:
        df["price"] = df.get("adjusted_close", df.get("close"))
    if "dollar_volume" not in df:
        df["dollar_volume"] = df["price"].abs() * df["volume"]
    if "market_cap" not in df and "shares_outstanding" in df:
        df["market_cap"] = df["price"].abs() * df["shares_outstanding"]
    if "turnover" not in df:
        shares = df.get("shares_outstanding", pd.Series(1.0, index=df.index)).replace(0, pd.NA)
        df["turnover"] = df["volume"] / shares
    if "amihud_illiquidity" not in df:
        df["amihud_illiquidity"] = df["ret_1d"].abs() / (df["dollar_volume"] / 1_000_000.0).clip(lower=1.0)
    if "spread_proxy" not in df:
        high = df.get("high", df["price"])
        low = df.get("low", df["price"])
        df["spread_proxy"] = ((high - low).abs() / df["price"].abs().replace(0, pd.NA)).fillna(0.0)
    if "liquidity_score" not in df:
        df["liquidity_score"] = (1.0 - 20.0 * df["spread_proxy"] - 8.0 * df["amihud_illiquidity"]).clip(0.0, 1.0)
    for col, value in {
        "country": "US",
        "currency": "USD",
        "sector": "unknown",
        "industry": "unknown",
        "exchange": "unknown",
        "is_active": True,
        "is_tradable": True,
        "delisting_return": 0.0,
        "ret_1d_ex_delist": df["ret_1d"] if "ret_1d" in df else 0.0,
    }.items():
        if col not in df:
            df[col] = value
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final causal tabular feature panel.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    panel = build_feature_panel(config)
    output = args.output or config.get("paths", {}).get("panel_path", "data/processed/panel_daily_2015_2024.parquet")
    write_table(panel, output)
    processed_dir = Path(config.get("paths", {}).get("processed_dir", "data/processed"))
    write_json(processed_dir / "feature_metadata.json", feature_metadata(panel))
    write_json(processed_dir / "universe_metadata.json", universe_metadata(panel, config.get("dataset", {}).get("grade", "unknown")))
    print(f"wrote feature panel: {output} ({len(panel):,} rows)")


if __name__ == "__main__":
    main()
