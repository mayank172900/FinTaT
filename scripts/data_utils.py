from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


DATE_COLUMN = "date"
ASSET_COLUMN = "asset_id"

MODEL_FEATURE_NAMES = [
    "ret_lag_1",
    "ret_lag_2",
    "ret_lag_3",
    "ret_lag_5",
    "ret_lag_10",
    "ret_lag_20",
    "ret_lag_60",
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "reversal_1",
    "realized_vol_5",
    "realized_vol_20",
    "realized_vol_60",
    "realized_vol_252",
    "downside_vol_20",
    "skew_60",
    "kurtosis_60",
    "drawdown_20",
    "drawdown_60",
    "drawdown_252",
    "volume_z_20",
    "volume_z_60",
    "dollar_volume_z_20",
    "turnover_20",
    "amihud_20",
    "spread_proxy",
    "liquidity_score",
    "market_beta_60",
    "market_beta_252",
    "sector_beta_60",
    "sector_beta_252",
    "mom_beta_252",
    "value_beta_252",
    "rates_beta_252",
    "oil_beta_252",
    "residual_vol_60",
    "sector_relative_return_5",
    "sector_relative_return_20",
    "sector_relative_return_60",
    "market_relative_return_20",
    "book_to_market",
    "earnings_yield",
    "profitability",
    "leverage",
    "asset_growth",
    "sales_growth",
    "earnings_event_flag",
    "days_to_earnings",
    "days_since_earnings",
    "analyst_revision_30d",
    "short_interest_ratio",
    "borrow_fee",
    "vix",
    "yield_curve_10y_2y",
    "credit_spread",
    "market_state_vol",
    "market_state_corr",
    "market_state_liquidity",
]

FEATURE_COLUMNS = [f"feat_{name}" for name in MODEL_FEATURE_NAMES]

FACTOR_COLUMNS = [
    "factor_market_beta",
    "factor_sector_beta",
    "factor_rates_beta",
    "factor_oil_beta",
    "factor_mom_beta",
]

MARKET_STATE_COLUMNS = [
    "mkt_median_realized_vol_20",
    "mkt_cross_sectional_mad_return",
    "mkt_average_abs_corr_60",
    "mkt_market_mode_eigen_share_60",
    "mkt_liquidity_stress",
    "mkt_vix",
    "mkt_credit_spread",
    "mkt_yield_curve_10y_2y",
]

LABEL_COLUMNS = [
    "label_5class_h1_quantile",
    "label_5class_h5_quantile",
    "label_5class_h20_quantile",
    "label_5class_h1_abs",
    "label_5class_h5_abs",
    "label_5class_h20_abs",
]

FORWARD_RETURN_COLUMNS = [
    "forward_return_1d",
    "forward_return_5d",
    "forward_return_20d",
]

FUTURE_NORM_RETURN_COLUMNS = [
    "future_norm_return_1d",
    "future_norm_return_5d",
    "future_norm_return_20d",
]

EVALUATION_ONLY_COLUMNS = LABEL_COLUMNS + FORWARD_RETURN_COLUMNS + FUTURE_NORM_RETURN_COLUMNS

FINAL_BASE_COLUMNS = [
    "date",
    "asset_id",
    "ticker",
    "permno",
    "permco",
    "cusip",
    "figi",
    "ret_1d",
    "ret_1d_ex_delist",
    "delisting_return",
    "volume",
    "dollar_volume",
    "market_cap",
    "price",
    "sector",
    "industry",
    "exchange",
    "country",
    "currency",
    "is_active",
    "is_tradable",
    "liquidity_score",
]

FINAL_REQUIRED_COLUMNS = (
    FINAL_BASE_COLUMNS
    + FEATURE_COLUMNS
    + FACTOR_COLUMNS
    + MARKET_STATE_COLUMNS
    + EVALUATION_ONLY_COLUMNS
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return loaded


def write_json(path: str | Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pq.read_table(path, memory_map=False, use_threads=False).to_pandas()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError(f"unsupported table extension for {path}")


def write_table(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        raise ValueError(f"unsupported output extension for {path}")


def ensure_datetime(frame: pd.DataFrame, column: str = DATE_COLUMN) -> pd.DataFrame:
    out = frame.copy()
    out[column] = pd.to_datetime(out[column])
    return out


def sort_panel(frame: pd.DataFrame) -> pd.DataFrame:
    return ensure_datetime(frame).sort_values([DATE_COLUMN, ASSET_COLUMN]).reset_index(drop=True)


def safe_divide(numer: pd.Series | np.ndarray, denom: pd.Series | np.ndarray) -> pd.Series:
    result = pd.Series(numer).astype(float) / pd.Series(denom).replace(0, np.nan).astype(float)
    return result.replace([np.inf, -np.inf], np.nan)


def rolling_beta(
    frame: pd.DataFrame,
    y_col: str,
    x_col: str,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.Series:
    min_periods = min_periods or max(3, min(20, window // 3))
    out = pd.Series(index=frame.index, dtype=float)
    for _, idx in frame.groupby(ASSET_COLUMN, sort=False).groups.items():
        sub = frame.loc[idx, [y_col, x_col]].astype(float)
        cov = sub[y_col].rolling(window, min_periods=min_periods).cov(sub[x_col])
        var = sub[x_col].rolling(window, min_periods=min_periods).var()
        out.loc[idx] = (cov / var.replace(0, np.nan)).to_numpy()
    return out


def rolling_sum_by_asset(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    return (
        frame.groupby(ASSET_COLUMN, sort=False)[column]
        .rolling(window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )


def rolling_std_by_asset(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    return (
        frame.groupby(ASSET_COLUMN, sort=False)[column]
        .rolling(window, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
    )


def rolling_mean_by_asset(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    return (
        frame.groupby(ASSET_COLUMN, sort=False)[column]
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )


def rolling_z_by_asset(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    mean = rolling_mean_by_asset(frame, column, window)
    std = rolling_std_by_asset(frame, column, window)
    return safe_divide(frame[column] - mean, std)


def add_causal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add FinTTA feature columns using information available through `date`."""

    df = sort_panel(frame)
    grouped = df.groupby(ASSET_COLUMN, sort=False)

    for lag in [1, 2, 3, 5, 10, 20, 60]:
        df[f"feat_ret_lag_{lag}"] = grouped["ret_1d"].shift(lag)

    for window in [5, 20, 60]:
        df[f"feat_momentum_{window}"] = rolling_sum_by_asset(df, "ret_1d", window)

    df["feat_reversal_1"] = -(df["feat_ret_lag_1"] - df["feat_momentum_5"] / 5.0)

    for window in [5, 20, 60, 252]:
        vol = np.sqrt(rolling_mean_by_asset(df.assign(_ret_sq=df["ret_1d"] ** 2), "_ret_sq", window))
        df[f"feat_realized_vol_{window}"] = vol

    downside = df["ret_1d"].clip(upper=0.0) ** 2
    df["feat_downside_vol_20"] = np.sqrt(rolling_mean_by_asset(df.assign(_downside=downside), "_downside", 20))
    df["feat_skew_60"] = (
        grouped["ret_1d"].rolling(60, min_periods=5).skew().reset_index(level=0, drop=True)
    )
    df["feat_kurtosis_60"] = (
        grouped["ret_1d"].rolling(60, min_periods=5).kurt().reset_index(level=0, drop=True)
    )

    for window in [20, 60, 252]:
        drawdowns = []
        for _, sub in df.groupby(ASSET_COLUMN, sort=False):
            roll_max = sub["adjusted_close"].rolling(window, min_periods=1).max()
            drawdowns.append((sub["adjusted_close"] / roll_max.replace(0, np.nan) - 1.0))
        df[f"feat_drawdown_{window}"] = pd.concat(drawdowns).sort_index()

    df["feat_volume_z_20"] = rolling_z_by_asset(df, "volume", 20)
    df["feat_volume_z_60"] = rolling_z_by_asset(df, "volume", 60)
    df["feat_dollar_volume_z_20"] = rolling_z_by_asset(df, "dollar_volume", 20)
    df["feat_turnover_20"] = rolling_mean_by_asset(df, "turnover", 20)
    df["feat_amihud_20"] = rolling_mean_by_asset(df, "amihud_illiquidity", 20)
    df["feat_spread_proxy"] = df["spread_proxy"]
    df["feat_liquidity_score"] = df["liquidity_score"]

    market_return = df.groupby(DATE_COLUMN, sort=False)["ret_1d"].mean().rename("_market_return")
    df = df.merge(market_return, on=DATE_COLUMN, how="left")
    df["_market_momentum_20"] = (
        market_return.rolling(20, min_periods=1).sum().rename("_market_momentum_20").reindex(df[DATE_COLUMN]).to_numpy()
    )

    sector_return = (
        df.groupby([DATE_COLUMN, "sector"], sort=False)["ret_1d"].mean().rename("_sector_return").reset_index()
    )
    df = df.merge(sector_return, on=[DATE_COLUMN, "sector"], how="left")

    df["feat_market_beta_60"] = rolling_beta(df, "ret_1d", "_market_return", 60)
    df["feat_market_beta_252"] = rolling_beta(df, "ret_1d", "_market_return", 252)
    df["feat_sector_beta_60"] = rolling_beta(df, "ret_1d", "_sector_return", 60)
    df["feat_sector_beta_252"] = rolling_beta(df, "ret_1d", "_sector_return", 252)

    for col, feature in [
        ("mom", "feat_mom_beta_252"),
        ("hml", "feat_value_beta_252"),
        ("rates_factor", "feat_rates_beta_252"),
        ("oil_factor", "feat_oil_beta_252"),
    ]:
        if col in df:
            df[feature] = rolling_beta(df, "ret_1d", col, 252)
        else:
            df[feature] = 0.0

    df["feat_residual_vol_60"] = np.sqrt(
        rolling_mean_by_asset(
            df.assign(_resid=(df["ret_1d"] - df["feat_market_beta_60"].fillna(1.0) * df["_market_return"]) ** 2),
            "_resid",
            60,
        )
    )

    for window in [5, 20, 60]:
        mom_col = f"feat_momentum_{window}"
        sector_avg = df.groupby([DATE_COLUMN, "sector"], sort=False)[mom_col].transform("mean")
        df[f"feat_sector_relative_return_{window}"] = df[mom_col] - sector_avg
    df["feat_market_relative_return_20"] = df["feat_momentum_20"] - df["_market_momentum_20"]

    optional_defaults = {
        "feat_book_to_market": "book_to_market",
        "feat_earnings_yield": "earnings_yield",
        "feat_profitability": "profitability",
        "feat_leverage": "leverage",
        "feat_asset_growth": "asset_growth",
        "feat_sales_growth": "sales_growth",
        "feat_earnings_event_flag": "earnings_announcement_flag",
        "feat_days_to_earnings": "days_to_earnings",
        "feat_days_since_earnings": "days_since_earnings",
        "feat_analyst_revision_30d": "analyst_revision_30d",
        "feat_short_interest_ratio": "short_interest_ratio",
        "feat_borrow_fee": "borrow_fee",
        "feat_vix": "vix",
        "feat_yield_curve_10y_2y": "yield_curve_10y_2y",
        "feat_credit_spread": "credit_spread",
    }
    for feature, source in optional_defaults.items():
        df[feature] = df[source] if source in df else 0.0

    df["factor_market_beta"] = df["feat_market_beta_252"]
    df["factor_sector_beta"] = df["feat_sector_beta_252"]
    df["factor_rates_beta"] = df["feat_rates_beta_252"]
    df["factor_oil_beta"] = df["feat_oil_beta_252"]
    df["factor_mom_beta"] = df["feat_mom_beta_252"]

    df = df.drop(columns=[c for c in ["_market_return", "_market_momentum_20", "_sector_return"] if c in df])
    cleanup_columns = [c for c in FEATURE_COLUMNS + FACTOR_COLUMNS if c in df.columns]
    df[cleanup_columns] = df[cleanup_columns].replace([np.inf, -np.inf], np.nan)
    df[cleanup_columns] = df[cleanup_columns].fillna(0.0)
    return df


def make_labels(frame: pd.DataFrame, horizons: list[int] | None = None) -> pd.DataFrame:
    df = sort_panel(frame)
    horizons = horizons or [1, 5, 20]
    labels = df[[DATE_COLUMN, ASSET_COLUMN]].copy()
    if "feat_realized_vol_20" in df:
        realized_vol = df["feat_realized_vol_20"].replace(0, np.nan)
    else:
        realized_vol = rolling_std_by_asset(df, "ret_1d", 20).replace(0, np.nan)

    for horizon in horizons:
        fwd = pd.Series(index=df.index, dtype=float)
        for _, idx in df.groupby(ASSET_COLUMN, sort=False).groups.items():
            sub = df.loc[idx]
            if "adjusted_close" in sub:
                value = sub["adjusted_close"].shift(-horizon) / sub["adjusted_close"] - 1.0
            else:
                shifted = [sub["ret_1d"].shift(-i) for i in range(1, horizon + 1)]
                value = pd.concat(shifted, axis=1).add(1.0).prod(axis=1) - 1.0
            fwd.loc[idx] = value.to_numpy()
        norm = fwd / realized_vol
        labels[f"forward_return_{horizon}d"] = fwd
        labels[f"future_norm_return_{horizon}d"] = norm
        labels[f"label_5class_h{horizon}_quantile"] = _quantile_labels_by_date(df[DATE_COLUMN], norm)
        labels[f"label_5class_h{horizon}_abs"] = _absolute_labels(norm)
    return labels


def compute_market_state(frame: pd.DataFrame) -> pd.DataFrame:
    df = sort_panel(frame)
    dates = list(df[DATE_COLUMN].drop_duplicates())
    ret_pivot = df.pivot(index=DATE_COLUMN, columns=ASSET_COLUMN, values="ret_1d").sort_index()
    market_return = ret_pivot.mean(axis=1)
    cumulative = (1.0 + market_return.fillna(0.0)).cumprod()
    rows: list[dict[str, Any]] = []

    for date in dates:
        day = df[df[DATE_COLUMN] == date]
        returns_until_date = ret_pivot.loc[:date]
        row: dict[str, Any] = {DATE_COLUMN: date}
        row["median_realized_vol_20"] = float(day.get("feat_realized_vol_20", pd.Series([np.nan])).median())
        row["median_realized_vol_60"] = float(day.get("feat_realized_vol_60", pd.Series([np.nan])).median())
        row["cross_sectional_mad_return"] = float((day["ret_1d"] - day["ret_1d"].median()).abs().median())
        for window in [20, 60]:
            corr_window = returns_until_date.tail(window)
            corr = corr_window.corr(min_periods=min(5, len(corr_window)))
            values = corr.to_numpy(dtype=float)
            if values.size and values.shape[0] > 1:
                upper = values[np.triu_indices_from(values, k=1)]
                eig = np.linalg.eigvalsh(np.nan_to_num(values, nan=0.0))
                upper_clean = upper[~np.isnan(upper)]
                row[f"average_abs_correlation_{window}"] = float(np.mean(np.abs(upper_clean))) if len(upper_clean) else 0.0
                row[f"market_mode_eigen_share_{window}"] = float(eig[-1] / max(eig.sum(), 1e-12))
            else:
                row[f"average_abs_correlation_{window}"] = 0.0
                row[f"market_mode_eigen_share_{window}"] = 0.0
        row["market_return_1d"] = float(market_return.loc[date])
        row["market_return_5d"] = float(market_return.loc[:date].tail(5).sum())
        for window in [20, 60]:
            cum_window = cumulative.loc[:date].tail(window)
            row[f"market_drawdown_{window}"] = float(cum_window.iloc[-1] / cum_window.max() - 1.0)
        row["vix"] = float(day["vix"].median()) if "vix" in day else 0.0
        prior_vix = (
            df[df[DATE_COLUMN] <= date].drop_duplicates(DATE_COLUMN).tail(6)["vix"]
            if "vix" in df
            else pd.Series(dtype=float)
        )
        row["vix_change_5d"] = float(prior_vix.iloc[-1] - prior_vix.iloc[0]) if len(prior_vix) >= 2 else 0.0
        row["liquidity_stress"] = float(1.0 - day["liquidity_score"].median())
        row["spread_zscore"] = float(day.get("feat_spread_proxy", pd.Series([0.0])).median())
        row["volume_zscore"] = float(day.get("feat_volume_z_20", pd.Series([0.0])).median())
        row["order_imbalance_market"] = float(day.get("order_imbalance_proxy", pd.Series([0.0])).median())
        row["yield_curve_10y_2y"] = float(day.get("yield_curve_10y_2y", pd.Series([0.0])).median())
        row["credit_spread"] = float(day.get("credit_spread", pd.Series([0.0])).median())
        for sector in ["technology", "financials", "energy", "healthcare"]:
            sector_day = day[day["sector"] == sector]
            row[f"sector_return_{sector.replace('technology', 'tech')}"] = float(sector_day["ret_1d"].mean()) if len(sector_day) else 0.0
        row["sector_dispersion"] = float(day.groupby("sector")["ret_1d"].mean().std() or 0.0)
        rows.append(row)

    state = pd.DataFrame(rows)
    state["mkt_median_realized_vol_20"] = state["median_realized_vol_20"]
    state["mkt_cross_sectional_mad_return"] = state["cross_sectional_mad_return"]
    state["mkt_average_abs_corr_60"] = state["average_abs_correlation_60"]
    state["mkt_market_mode_eigen_share_60"] = state["market_mode_eigen_share_60"]
    state["mkt_liquidity_stress"] = state["liquidity_stress"]
    state["mkt_vix"] = state["vix"]
    state["mkt_credit_spread"] = state["credit_spread"]
    state["mkt_yield_curve_10y_2y"] = state["yield_curve_10y_2y"]
    return state.fillna(0.0)


def attach_market_state(frame: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    df = ensure_datetime(frame)
    state = ensure_datetime(state)
    merged = df.merge(state[[DATE_COLUMN] + MARKET_STATE_COLUMNS], on=DATE_COLUMN, how="left")
    merged["feat_market_state_vol"] = merged["mkt_median_realized_vol_20"]
    merged["feat_market_state_corr"] = merged["mkt_average_abs_corr_60"]
    merged["feat_market_state_liquidity"] = merged["mkt_liquidity_stress"]
    merged[MARKET_STATE_COLUMNS + ["feat_market_state_vol", "feat_market_state_corr", "feat_market_state_liquidity"]] = (
        merged[MARKET_STATE_COLUMNS + ["feat_market_state_vol", "feat_market_state_corr", "feat_market_state_liquidity"]]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return merged


def build_sample_panel(seed: int = 17) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-10-01", periods=120)
    assets = [
        ("SAMP001", "ALFA", 10001, 5001, "technology", "software", 1.15, 0.30),
        ("SAMP002", "BETA", 10002, 5002, "technology", "semiconductors", 1.25, -0.15),
        ("SAMP003", "CASH", 10003, 5003, "financials", "banks", 0.80, 0.10),
        ("SAMP004", "DURA", 10004, 5004, "financials", "insurance", 0.70, -0.20),
        ("SAMP005", "ENRG", 10005, 5005, "energy", "oil_gas", 1.05, 0.05),
        ("SAMP006", "HLTH", 10006, 5006, "healthcare", "biotech", 0.90, 0.25),
        ("SAMP007", "INDU", 10007, 5007, "industrials", "machinery", 1.00, -0.05),
        ("SAMP008", "SHOP", 10008, 5008, "consumer", "retail", 1.10, -0.30),
    ]
    sector_offsets = {
        "technology": 0.0005,
        "financials": 0.0002,
        "energy": -0.0001,
        "healthcare": 0.0003,
        "industrials": 0.0001,
        "consumer": -0.0002,
    }
    prices = {asset_id: 30.0 + i * 7.5 for i, (asset_id, *_rest) in enumerate(assets)}
    shares = {asset_id: 20_000_000 + i * 2_500_000 for i, (asset_id, *_rest) in enumerate(assets)}
    delist_date = dates[95]
    rows: list[dict[str, Any]] = []

    for t, date in enumerate(dates):
        stress = -0.035 if pd.Timestamp("2020-03-02") <= date <= pd.Timestamp("2020-03-20") else 0.0
        vol_boost = 2.8 if stress else 1.0
        common = rng.normal(0.00035 + stress, 0.010 * vol_boost)
        mom = rng.normal(0.0, 0.006)
        hml = rng.normal(0.0, 0.004)
        rates_factor = rng.normal(0.0, 0.003) - 0.002 * bool(stress)
        oil_factor = rng.normal(0.0, 0.012) - 0.015 * bool(stress)
        vix = 16.0 + 45.0 * bool(stress) + 3.0 * np.sin(t / 9.0)
        yield_curve = 0.45 - 0.15 * bool(stress) + 0.05 * np.sin(t / 21.0)
        credit_spread = 0.75 + 1.10 * bool(stress) + 0.05 * np.cos(t / 13.0)

        for i, (asset_id, ticker, permno, permco, sector, industry, beta, size_loading) in enumerate(assets):
            if asset_id == "SAMP008" and date > delist_date:
                continue
            sector_ret = sector_offsets[sector] + rng.normal(0.0, 0.004 * vol_boost)
            idio = rng.normal(0.0, 0.008 * vol_boost)
            raw_ret = beta * common + sector_ret + 0.15 * mom + 0.05 * hml + idio
            delisting_return = -0.35 if asset_id == "SAMP008" and date == delist_date else 0.0
            ret_1d = (1.0 + raw_ret) * (1.0 + delisting_return) - 1.0
            price_prev = prices[asset_id]
            price = max(1.0, price_prev * (1.0 + ret_1d))
            prices[asset_id] = price
            volume = int((450_000 + i * 45_000) * (1.0 + 3.0 * abs(ret_1d)) * (1.8 if stress else 1.0))
            dollar_volume = float(volume * price)
            spread_proxy = float(min(0.15, 0.0025 + 0.35 / np.sqrt(max(dollar_volume, 1.0))))
            turnover = volume / shares[asset_id]
            amihud = abs(ret_1d) / max(dollar_volume / 1_000_000.0, 1.0)
            liquidity_score = float(np.clip(1.0 - 20.0 * spread_proxy - 8.0 * amihud, 0.05, 1.0))
            rows.append(
                {
                    "date": date,
                    "asset_id": asset_id,
                    "ticker": ticker,
                    "permno": permno,
                    "permco": permco,
                    "cusip": f"{permno:08d}",
                    "figi": f"BBG00SAMP{i:03d}",
                    "exchange": "NYSE" if i % 2 == 0 else "NASDAQ",
                    "share_code": 10,
                    "country": "US",
                    "currency": "USD",
                    "sector": sector,
                    "industry": industry,
                    "is_active": not (asset_id == "SAMP008" and date == delist_date),
                    "is_tradable": not (asset_id == "SAMP008" and date == delist_date),
                    "index_membership_sp500": i < 4,
                    "index_membership_russell1000": i < 6,
                    "universe_inclusion_reason": "sample_static_membership" if asset_id != "SAMP008" else "sample_delisted_asset",
                    "open": price_prev,
                    "high": max(price_prev, price) * (1.0 + abs(rng.normal(0, 0.002))),
                    "low": min(price_prev, price) * (1.0 - abs(rng.normal(0, 0.002))),
                    "close": price,
                    "adjusted_close": price,
                    "price": price,
                    "raw_return": raw_ret,
                    "adjusted_return": raw_ret,
                    "ret_1d": ret_1d,
                    "ret_1d_ex_delist": raw_ret,
                    "delisting_return": delisting_return,
                    "volume": volume,
                    "shares_outstanding": shares[asset_id],
                    "dollar_volume": dollar_volume,
                    "market_cap": price * shares[asset_id],
                    "bid": price * (1.0 - spread_proxy / 2.0),
                    "ask": price * (1.0 + spread_proxy / 2.0),
                    "spread_proxy": spread_proxy,
                    "liquidity_score": liquidity_score,
                    "halt_flag": False,
                    "split_factor": 1.0,
                    "dividend_amount": 0.0,
                    "turnover": turnover,
                    "amihud_illiquidity": amihud,
                    "book_to_market": 0.35 + 0.05 * size_loading + rng.normal(0.0, 0.02),
                    "earnings_yield": 0.04 + 0.01 * size_loading + rng.normal(0.0, 0.004),
                    "profitability": 0.12 + 0.03 * beta + rng.normal(0.0, 0.01),
                    "leverage": 0.25 - 0.03 * size_loading + rng.normal(0.0, 0.015),
                    "asset_growth": 0.02 + rng.normal(0.0, 0.01),
                    "sales_growth": 0.03 + rng.normal(0.0, 0.012),
                    "earnings_announcement_flag": int(t % 63 == i % 17),
                    "days_to_earnings": int((63 - (t - i) % 63) % 63),
                    "days_since_earnings": int((t - i) % 63),
                    "analyst_revision_30d": rng.normal(0.0, 0.02),
                    "short_interest_ratio": max(0.0, 0.04 + rng.normal(0.0, 0.01)),
                    "borrow_fee": max(0.0, 0.002 + 0.004 * (asset_id == "SAMP008") + rng.normal(0.0, 0.001)),
                    "vix": vix,
                    "yield_curve_10y_2y": yield_curve,
                    "credit_spread": credit_spread,
                    "mkt_rf": common,
                    "smb": size_loading * 0.001 + rng.normal(0.0, 0.002),
                    "hml": hml,
                    "rmw": rng.normal(0.0, 0.002),
                    "cma": rng.normal(0.0, 0.002),
                    "mom": mom,
                    "risk_free_rate": 0.00008,
                    "market_return": common,
                    "rates_factor": rates_factor,
                    "oil_factor": oil_factor,
                    "volatility_factor": vix / 100.0,
                }
            )

    panel = pd.DataFrame(rows)
    panel = add_causal_features(panel)
    state = compute_market_state(panel)
    panel = attach_market_state(panel, state)
    label_frame = make_labels(panel)
    panel = panel.merge(label_frame, on=[DATE_COLUMN, ASSET_COLUMN], how="left")
    panel[EVALUATION_ONLY_COLUMNS] = panel[EVALUATION_ONLY_COLUMNS].fillna(
        {
            **{col: 2 for col in LABEL_COLUMNS},
            **{col: 0.0 for col in FORWARD_RETURN_COLUMNS + FUTURE_NORM_RETURN_COLUMNS},
        }
    )
    for col in LABEL_COLUMNS:
        panel[col] = panel[col].astype("int64")

    ordered = [col for col in FINAL_REQUIRED_COLUMNS if col in panel.columns]
    remaining = [col for col in panel.columns if col not in ordered]
    panel = panel[ordered + remaining].sort_values([DATE_COLUMN, ASSET_COLUMN]).reset_index(drop=True)
    metadata = sample_metadata(panel)
    return panel, metadata


def sample_metadata(panel: pd.DataFrame) -> dict[str, Any]:
    by_year = panel.assign(year=pd.to_datetime(panel[DATE_COLUMN]).dt.year)
    return {
        "dataset_grade": "sample-only",
        "purpose": "Tiny schema-compatible fixture for tests and demos; not empirical evidence.",
        "date_min": str(pd.to_datetime(panel[DATE_COLUMN]).min().date()),
        "date_max": str(pd.to_datetime(panel[DATE_COLUMN]).max().date()),
        "rows": int(len(panel)),
        "assets": int(panel[ASSET_COLUMN].nunique()),
        "rows_by_year": {str(k): int(v) for k, v in by_year.groupby("year").size().items()},
        "feature_columns": FEATURE_COLUMNS,
        "adaptation_feature_columns": FEATURE_COLUMNS,
        "evaluation_only_columns": EVALUATION_ONLY_COLUMNS,
        "factor_columns": FACTOR_COLUMNS,
        "market_state_columns": MARKET_STATE_COLUMNS,
        "leakage_note": "All feat_ columns are built from contemporaneous or lagged sample values; labels and forward returns are evaluation-only.",
    }


def feature_metadata(panel: pd.DataFrame) -> dict[str, Any]:
    return {
        "feature_prefix": "feat_",
        "feature_columns": [c for c in panel.columns if c.startswith("feat_")],
        "excluded_from_adaptation": EVALUATION_ONLY_COLUMNS,
        "construction_rule": "Features must be point-in-time and use information available on or before date.",
        "generated_by": "scripts/make_features.py",
    }


def label_metadata() -> dict[str, Any]:
    return {
        "labels": LABEL_COLUMNS,
        "forward_returns": FORWARD_RETURN_COLUMNS,
        "future_norm_returns": FUTURE_NORM_RETURN_COLUMNS,
        "classes": {
            "0": "strong sell",
            "1": "sell",
            "2": "hold",
            "3": "buy",
            "4": "strong buy",
        },
        "quantile_buckets": "<q10, q10-q35, q35-q65, q65-q90, >=q90 by date",
        "absolute_thresholds_on_future_norm_return": [-1.0, -0.25, 0.25, 1.0],
        "offline_only": True,
    }


def universe_metadata(panel: pd.DataFrame, dataset_grade: str) -> dict[str, Any]:
    return {
        "dataset_grade": dataset_grade,
        "asset_id_rule": "Use stable PERMNO/FIGI/vendor security id; tickers are descriptive and may change.",
        "rows": int(len(panel)),
        "assets": int(panel[ASSET_COLUMN].nunique()),
        "date_min": str(pd.to_datetime(panel[DATE_COLUMN]).min().date()),
        "date_max": str(pd.to_datetime(panel[DATE_COLUMN]).max().date()),
        "active_rows": int(panel.get("is_active", pd.Series(dtype=bool)).fillna(False).sum()),
        "inactive_rows": int((~panel.get("is_active", pd.Series(True, index=panel.index)).fillna(True)).sum()),
        "delisting_return_rows": int((panel.get("delisting_return", pd.Series(0.0, index=panel.index)).fillna(0.0) != 0).sum()),
        "survivorship_rule": "Inactive/delisted securities must remain in historical rows when supported by the source.",
    }


def _quantile_labels_by_date(dates: pd.Series, values: pd.Series) -> pd.Series:
    out = pd.Series(2, index=values.index, dtype="int64")
    temp = pd.DataFrame({"date": dates, "value": values})
    for _, idx in temp.groupby("date", sort=False).groups.items():
        sub = temp.loc[idx, "value"].dropna()
        if len(sub) < 5:
            continue
        q10, q35, q65, q90 = sub.quantile([0.10, 0.35, 0.65, 0.90]).to_numpy()
        out.loc[idx] = pd.cut(
            temp.loc[idx, "value"],
            bins=[-np.inf, q10, q35, q65, q90, np.inf],
            labels=[0, 1, 2, 3, 4],
        ).astype("float").fillna(2).astype("int64")
    return out


def _absolute_labels(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[-np.inf, -1.0, -0.25, 0.25, 1.0, np.inf],
        labels=[0, 1, 2, 3, 4],
    ).astype("float").fillna(2).astype("int64")


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return str(value)
