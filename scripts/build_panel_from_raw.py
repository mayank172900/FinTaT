from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from data_utils import (
    ASSET_COLUMN,
    DATE_COLUMN,
    build_sample_panel,
    feature_metadata,
    label_metadata,
    load_config,
    read_table,
    sample_metadata,
    sort_panel,
    universe_metadata,
    write_json,
    write_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FinTTA data panel from normalized raw vendor exports.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-sample", action="store_true", help="If licensed raw files are missing, build only data/sample.")
    args = parser.parse_args()

    config = load_config(args.config)
    mode = config.get("dataset", {}).get("mode", "licensed")
    if mode == "sample":
        build_sample_outputs(config)
        return

    missing = missing_required_raw(config)
    if missing:
        message = (
            "Licensed/raw empirical inputs are unavailable, so no CRSP/TAQ/Compustat "
            "2015-2024 panel was generated. Missing files:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )
        if args.allow_sample:
            print(message)
            print("Building schema-compatible sample dataset instead.")
            sample_config = {"paths": {"sample_dir": "data/sample"}}
            build_sample_outputs(sample_config)
            return
        raise SystemExit(message + "\nUse configs/data_sample.yaml for the committed sample fixture.")

    normalize_intermediate_tables(config)
    run_pipeline(config, args.config)


def missing_required_raw(config: dict) -> list[str]:
    raw_inputs = config.get("paths", {}).get("raw_inputs", {})
    required = config.get("requirements", {}).get("required_raw_inputs", ["prices", "universe"])
    return [str(raw_inputs.get(key, f"<paths.raw_inputs.{key}>")) for key in required if not Path(raw_inputs.get(key, "")).exists()]


def normalize_intermediate_tables(config: dict) -> None:
    paths = config.get("paths", {})
    raw_inputs = paths.get("raw_inputs", {})
    intermediate_dir = Path(paths.get("intermediate_dir", "data/intermediate"))
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    prices = normalize_prices(read_table(raw_inputs["prices"]))
    universe = normalize_universe(read_table(raw_inputs["universe"]))
    write_table(prices, intermediate_dir / "prices_daily.parquet")
    write_table(universe, intermediate_dir / "universe_daily.parquet")

    passthrough = {
        "fundamentals": "fundamentals_asof.parquet",
        "macro": "macro_asof.parquet",
        "factor_returns": "factor_returns.parquet",
        "factor_exposures": "factor_exposures_daily.parquet",
        "sector_industry": "sector_industry_daily.parquet",
        "etf_holdings": "etf_holdings_daily.parquet",
        "liquidity": "liquidity_daily.parquet",
        "events": "events_asof.parquet",
        "short_interest": "short_interest_daily.parquet",
    }
    for key, filename in passthrough.items():
        source = raw_inputs.get(key)
        if source and Path(source).exists():
            table = read_table(source)
            if key == "macro":
                table = normalize_open_macro(table, raw_inputs.get("factor_returns"))
            elif key == "short_interest":
                table = normalize_short_interest(table)
            write_table(sort_panel(table) if key != "macro" else table, intermediate_dir / filename)


def normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    if "asset_id" not in df and "permno" in df:
        df["asset_id"] = df["permno"].astype(str)
    if "ret_1d_ex_delist" not in df:
        df["ret_1d_ex_delist"] = df.get("raw_return", df.get("adjusted_return", df.get("ret_1d", 0.0)))
    if "delisting_return" not in df:
        df["delisting_return"] = 0.0
    if "ret_1d" not in df:
        df["ret_1d"] = (1.0 + df["ret_1d_ex_delist"].fillna(0.0)) * (1.0 + df["delisting_return"].fillna(0.0)) - 1.0
    if "adjusted_close" not in df and "close" in df:
        df["adjusted_close"] = df["close"]
    if "price" not in df:
        df["price"] = df.get("adjusted_close", df.get("close"))
    if "dollar_volume" not in df:
        df["dollar_volume"] = df["price"].abs() * df["volume"]
    if "market_cap" not in df and "shares_outstanding" in df:
        df["market_cap"] = df["price"].abs() * df["shares_outstanding"]
    if "spread_proxy" not in df:
        high = df.get("high", df["price"])
        low = df.get("low", df["price"])
        df["spread_proxy"] = ((high - low).abs() / df["price"].abs().replace(0, pd.NA)).fillna(0.0)
    if "liquidity_score" not in df:
        df["liquidity_score"] = (1.0 - 20.0 * df["spread_proxy"]).clip(0.0, 1.0)
    return sort_panel(df)


def normalize_universe(universe: pd.DataFrame) -> pd.DataFrame:
    df = universe.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    if "asset_id" not in df and "permno" in df:
        df["asset_id"] = df["permno"].astype(str)
    for col, value in {
        "country": "US",
        "currency": "USD",
        "is_active": True,
        "is_tradable": True,
        "universe_inclusion_reason": "vendor_universe_row",
    }.items():
        if col not in df:
            df[col] = value
    return sort_panel(df)


def run_pipeline(config: dict, config_path: str) -> None:
    paths = config.get("paths", {})
    processed_dir = Path(paths.get("processed_dir", "data/processed"))
    panel_path = Path(paths.get("panel_path", processed_dir / "panel_daily_2015_2024.parquet"))
    script_dir = Path(__file__).parent
    subprocess.run([sys.executable, str(script_dir / "make_features.py"), "--config", config_path, "--output", str(panel_path)], check=True)
    subprocess.run([sys.executable, str(script_dir / "make_labels.py"), "--input", str(panel_path)], check=True)
    labels = read_table("data/intermediate/labels_daily.parquet")
    panel = read_table(panel_path).merge(labels, on=[DATE_COLUMN, ASSET_COLUMN], how="left")
    write_table(panel, panel_path)
    if config.get("graph", {}).get("precompute_edges", False):
        subprocess.run([sys.executable, str(script_dir / "make_graph_inputs.py"), "--panel", str(panel_path)], check=True)
    subprocess.run([sys.executable, str(script_dir / "make_splits.py"), "--config", "configs/experiment_2015_2024.yaml"], check=True)
    subprocess.run([sys.executable, str(script_dir / "validate_panel.py"), "--panel", str(panel_path), "--config", "configs/experiment_2015_2024.yaml"], check=True)
    write_json(processed_dir / "label_metadata.json", label_metadata())
    write_json(processed_dir / "universe_metadata.json", universe_metadata(panel, config.get("dataset", {}).get("grade", "conference-grade-candidate")))


def normalize_open_macro(macro: pd.DataFrame, factor_returns_path: str | None = None) -> pd.DataFrame:
    df = macro.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    rename = {
        "vixcls": "vix",
        "t10y2y": "yield_curve_10y_2y",
        "baa10y": "credit_spread",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if factor_returns_path and Path(factor_returns_path).exists():
        factors = read_table(factor_returns_path)
        factors[DATE_COLUMN] = pd.to_datetime(factors[DATE_COLUMN])
        keep = [DATE_COLUMN] + [c for c in ["mom", "hml", "mkt_rf"] if c in factors.columns]
        df = df.merge(factors[keep], on=DATE_COLUMN, how="left")
    for col in ["vix", "yield_curve_10y_2y", "credit_spread"]:
        if col not in df:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").ffill().fillna(0.0)
    for col in ["mom", "hml", "mkt_rf"]:
        if col not in df:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["rates_factor"] = df["yield_curve_10y_2y"].diff().fillna(0.0) / 100.0
    if "dcoilwtico" in df:
        oil = pd.to_numeric(df["dcoilwtico"], errors="coerce").ffill()
        df["oil_factor"] = oil.pct_change().replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    elif "oil_factor" not in df:
        df["oil_factor"] = 0.0
    if "vintage_date" not in df:
        df["vintage_date"] = df[DATE_COLUMN]
    return df


def normalize_short_interest(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    if "short_interest_ratio" not in df and "short_sale_volume_ratio" in df:
        df["short_interest_ratio"] = df["short_sale_volume_ratio"]
    return df


def build_sample_outputs(config: dict) -> None:
    sample_dir = Path(config.get("paths", {}).get("sample_dir", "data/sample"))
    sample_dir.mkdir(parents=True, exist_ok=True)
    panel, metadata = build_sample_panel()
    panel_path = sample_dir / "sample_panel.parquet"
    metadata_path = sample_dir / "sample_metadata.json"
    write_table(panel, panel_path)
    write_json(metadata_path, metadata)
    write_json(sample_dir / "sample_feature_metadata.json", feature_metadata(panel))
    write_json(sample_dir / "sample_label_metadata.json", label_metadata())
    write_json(sample_dir / "sample_universe_metadata.json", universe_metadata(panel, "sample-only"))
    print(f"wrote sample panel: {panel_path} ({len(panel):,} rows)")
    print(f"wrote sample metadata: {metadata_path}")


if __name__ == "__main__":
    main()
