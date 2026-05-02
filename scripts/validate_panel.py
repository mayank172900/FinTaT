from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_utils import (
    ASSET_COLUMN,
    DATE_COLUMN,
    EVALUATION_ONLY_COLUMNS,
    FINAL_REQUIRED_COLUMNS,
    LABEL_COLUMNS,
    MARKET_STATE_COLUMNS,
    FEATURE_COLUMNS,
    load_config,
    read_table,
    write_json,
)


def validate_panel(panel_path: str | Path, config_path: str | Path | None = None, graph_path: str | Path | None = None) -> dict[str, Any]:
    panel_path = Path(panel_path)
    panel = read_table(panel_path)
    panel[DATE_COLUMN] = pd.to_datetime(panel[DATE_COLUMN])
    config = load_config(config_path) if config_path else {}
    feature_columns = _adaptation_features(panel, config)

    report: dict[str, Any] = {
        "panel_path": str(panel_path),
        "dataset_grade": "unknown",
        "schema": {},
        "leakage": {},
        "survivorship": {},
        "missingness": {},
        "distribution": {},
        "graph": {},
        "market_state": {},
        "status": "pass",
        "errors": [],
        "warnings": [],
    }

    missing_required = [c for c in ["date", "asset_id", "ret_1d", "volume", "sector", "industry", "liquidity_score"] if c not in panel.columns]
    missing_final = [c for c in FINAL_REQUIRED_COLUMNS if c not in panel.columns]
    duplicates = int(panel.duplicated([DATE_COLUMN, ASSET_COLUMN]).sum())
    sorted_ok = panel.sort_values([DATE_COLUMN, ASSET_COLUMN]).index.equals(panel.index)
    report["schema"] = {
        "rows": int(len(panel)),
        "columns": int(len(panel.columns)),
        "missing_minimum_columns": missing_required,
        "missing_final_schema_columns": missing_final,
        "duplicate_date_asset_rows": duplicates,
        "dates_sorted": bool(sorted_ok),
        "date_min": str(panel[DATE_COLUMN].min().date()) if len(panel) else None,
        "date_max": str(panel[DATE_COLUMN].max().date()) if len(panel) else None,
        "asset_count": int(panel[ASSET_COLUMN].nunique()) if ASSET_COLUMN in panel else 0,
    }
    if missing_required or duplicates:
        report["errors"].append("schema failed: missing required columns or duplicate date/asset rows")

    eval_in_features = sorted(set(feature_columns).intersection(EVALUATION_ONLY_COLUMNS))
    feature_timestamp_violations = _timestamp_violations(panel)
    report["leakage"] = {
        "adaptation_feature_count": len(feature_columns),
        "evaluation_columns_in_adaptation_features": eval_in_features,
        "timestamp_columns_after_date": feature_timestamp_violations,
        "macro_vintage_after_date_rows": _date_order_violations(panel, "vintage_date"),
        "filing_date_after_date_rows": _date_order_violations(panel, "filing_date"),
        "feature_columns_all_prefixed": all(c.startswith("feat_") for c in feature_columns),
        "result": "pass" if not eval_in_features and not feature_timestamp_violations else "fail",
    }
    if eval_in_features or feature_timestamp_violations:
        report["errors"].append("leakage failed: evaluation columns or future timestamps found in features")

    active = panel.get("is_active", pd.Series(True, index=panel.index)).fillna(True).astype(bool)
    delisting = panel.get("delisting_return", pd.Series(0.0, index=panel.index)).fillna(0.0)
    report["survivorship"] = {
        "active_rows": int(active.sum()),
        "inactive_rows": int((~active).sum()),
        "delisting_return_rows": int((delisting != 0).sum()),
        "assets_with_inactive_rows": int(panel.loc[~active, ASSET_COLUMN].nunique()) if ASSET_COLUMN in panel else 0,
        "ticker_changes_by_asset": _ticker_change_counts(panel),
        "note": "For conference-grade CRSP/Norgate builds, inactive/delisted rows must be nonzero when applicable.",
    }

    feature_existing = [c for c in FEATURE_COLUMNS if c in panel.columns]
    missing_by_feature = panel[feature_existing].isna().mean().sort_values(ascending=False).head(50).to_dict() if feature_existing else {}
    missing_by_date = panel.assign(_missing=panel[feature_existing].isna().mean(axis=1) if feature_existing else 0.0).groupby(DATE_COLUMN)["_missing"].mean()
    missing_by_sector = panel.assign(_missing=panel[feature_existing].isna().mean(axis=1) if feature_existing else 0.0).groupby("sector")["_missing"].mean() if "sector" in panel else pd.Series(dtype=float)
    report["missingness"] = {
        "feature_missing_rate_top50": {k: float(v) for k, v in missing_by_feature.items()},
        "mean_missing_rate_by_date_top10": {str(k.date()): float(v) for k, v in missing_by_date.sort_values(ascending=False).head(10).items()},
        "mean_missing_rate_by_sector": {str(k): float(v) for k, v in missing_by_sector.items()},
        "rows_with_any_feature_missing": int(panel[feature_existing].isna().any(axis=1).sum()) if feature_existing else 0,
    }

    panel["_year"] = panel[DATE_COLUMN].dt.year
    report["distribution"] = {
        "rows_by_year": {str(k): int(v) for k, v in panel.groupby("_year").size().items()},
        "asset_count_by_year": {str(k): int(v) for k, v in panel.groupby("_year")[ASSET_COLUMN].nunique().items()},
        "average_assets_per_day": float(panel.groupby(DATE_COLUMN)[ASSET_COLUMN].nunique().mean()) if len(panel) else 0.0,
        "label_distribution_by_year": _label_distribution_by_year(panel),
        "sector_distribution_by_year": _sector_distribution_by_year(panel),
        "market_cap_summary": _summary(panel.get("market_cap")),
        "liquidity_summary": _summary(panel.get("liquidity_score")),
        "feature_mean_std_by_split": _feature_mean_std_by_split(panel, feature_columns),
    }

    graph_default = panel_path.parent.parent / "intermediate" / "graph_edges_daily.parquet"
    graph_path = Path(graph_path) if graph_path else graph_default
    if graph_path.exists():
        graph = read_table(graph_path)
        report["graph"] = _graph_report(graph, panel)
    else:
        report["graph"] = {"available": False, "reason": f"graph edge file not found: {graph_path}"}

    report["market_state"] = _market_state_report(panel)

    if report["errors"]:
        report["status"] = "fail"
    elif report["warnings"]:
        report["status"] = "pass_with_warnings"
    return report


def write_markdown_report(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FinTTA Data Validation Report",
        "",
        f"Status: `{report['status']}`",
        f"Panel: `{report['panel_path']}`",
        "",
        "## Schema",
        f"- Rows: {report['schema'].get('rows', 0):,}",
        f"- Assets: {report['schema'].get('asset_count', 0):,}",
        f"- Duplicate `(date, asset_id)` rows: {report['schema'].get('duplicate_date_asset_rows', 0):,}",
        f"- Missing minimum columns: {report['schema'].get('missing_minimum_columns', [])}",
        "",
        "## Leakage",
        f"- Result: {report['leakage'].get('result')}",
        f"- Evaluation columns in adaptation feature list: {report['leakage'].get('evaluation_columns_in_adaptation_features')}",
        f"- Timestamp violations: {report['leakage'].get('timestamp_columns_after_date')}",
        "",
        "## Survivorship",
        f"- Inactive rows: {report['survivorship'].get('inactive_rows', 0):,}",
        f"- Delisting-return rows: {report['survivorship'].get('delisting_return_rows', 0):,}",
        "",
        "## Graph",
        f"- {report['graph']}",
        "",
        "## Market State",
        f"- {report['market_state']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _adaptation_features(panel: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    prefixes = config.get("features", {}).get("include_prefixes", ["feat_"])
    excludes = set(config.get("features", {}).get("exclude_columns", []))
    return [c for c in panel.columns if any(c.startswith(prefix) for prefix in prefixes) and c not in excludes]


def _timestamp_violations(panel: pd.DataFrame) -> dict[str, int]:
    out = {}
    for column in panel.columns:
        if column.endswith("_timestamp") or column.endswith("_asof_date") or column.endswith("_available_date"):
            out[column] = _date_order_violations(panel, column)
    return {k: v for k, v in out.items() if v}


def _date_order_violations(panel: pd.DataFrame, column: str) -> int:
    if column not in panel:
        return 0
    values = pd.to_datetime(panel[column], errors="coerce")
    return int((values.notna() & (values > panel[DATE_COLUMN])).sum())


def _ticker_change_counts(panel: pd.DataFrame) -> dict[str, int]:
    if "ticker" not in panel:
        return {}
    counts = panel.groupby(ASSET_COLUMN)["ticker"].nunique()
    return {str(k): int(v) for k, v in counts[counts > 1].items()}


def _label_distribution_by_year(panel: pd.DataFrame) -> dict[str, dict[str, dict[str, int]]]:
    out = {}
    for label in [c for c in LABEL_COLUMNS if c in panel]:
        dist = panel.groupby(["_year", label]).size()
        out[label] = {}
        for (year, value), count in dist.items():
            out[label].setdefault(str(year), {})[str(int(value))] = int(count)
    return out


def _sector_distribution_by_year(panel: pd.DataFrame) -> dict[str, dict[str, int]]:
    if "sector" not in panel:
        return {}
    dist = panel.groupby(["_year", "sector"]).size()
    out: dict[str, dict[str, int]] = {}
    for (year, sector), count in dist.items():
        out.setdefault(str(year), {})[str(sector)] = int(count)
    return out


def _feature_mean_std_by_split(panel: pd.DataFrame, features: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    if not features:
        return {}
    split = np.where(panel[DATE_COLUMN] <= pd.Timestamp("2019-12-31"), "source_2015_2019_or_sample_pre2020", "tta_2020_2024_or_sample_post2019")
    panel = panel.assign(_split=split)
    out = {}
    for name, sub in panel.groupby("_split"):
        stats = sub[features].agg(["mean", "std"]).T.head(100)
        out[str(name)] = {
            col: {"mean": float(row["mean"]), "std": float(row["std"] or 0.0)}
            for col, row in stats.iterrows()
        }
    return out


def _summary(series: pd.Series | None) -> dict[str, float]:
    if series is None:
        return {}
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {}
    return {
        "min": float(clean.min()),
        "p25": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "p75": float(clean.quantile(0.75)),
        "max": float(clean.max()),
    }


def _graph_report(graph: pd.DataFrame, panel: pd.DataFrame) -> dict[str, Any]:
    graph[DATE_COLUMN] = pd.to_datetime(graph[DATE_COLUMN])
    counts = graph.groupby(DATE_COLUMN).size()
    positive = int((graph["edge_sign"] > 0).sum()) if "edge_sign" in graph else 0
    negative = int((graph["edge_sign"] < 0).sum()) if "edge_sign" in graph else 0
    isolated = {}
    for date, day_assets in panel.groupby(DATE_COLUMN)[ASSET_COLUMN]:
        edges = graph[graph[DATE_COLUMN] == date]
        connected = set(edges.get("asset_id_i", pd.Series(dtype=str))).union(set(edges.get("asset_id_j", pd.Series(dtype=str))))
        isolated[str(pd.Timestamp(date).date())] = int(len(set(day_assets.astype(str)) - connected))
    endpoint_counts = pd.concat([graph.get("asset_id_i", pd.Series(dtype=str)), graph.get("asset_id_j", pd.Series(dtype=str))]).value_counts()
    return {
        "available": True,
        "edge_count": int(len(graph)),
        "edge_count_per_date_summary": _summary(counts),
        "positive_edges": positive,
        "negative_edges": negative,
        "positive_negative_ratio": float(positive / max(negative, 1)),
        "isolated_asset_count_top10_dates": dict(list(isolated.items())[:10]),
        "top_connected_assets": {str(k): int(v) for k, v in endpoint_counts.head(20).items()},
        "average_correlation_by_date_available": "rolling_corr flags are encoded in edge_source_flags",
    }


def _market_state_report(panel: pd.DataFrame) -> dict[str, Any]:
    existing = [c for c in MARKET_STATE_COLUMNS if c in panel]
    out = {
        "required_columns_present": existing,
        "missing_columns": [c for c in MARKET_STATE_COLUMNS if c not in panel],
    }
    if existing:
        state = panel.drop_duplicates(DATE_COLUMN).sort_values(DATE_COLUMN)
        out["summary"] = {c: _summary(state[c]) for c in existing}
        crisis = state[(state[DATE_COLUMN] >= "2020-03-01") & (state[DATE_COLUMN] <= "2020-04-30")]
        baseline = state[state[DATE_COLUMN] < "2020-03-01"]
        if len(crisis) and len(baseline) and "mkt_median_realized_vol_20" in state:
            out["crisis_volatility_elevated"] = bool(
                crisis["mkt_median_realized_vol_20"].median() > baseline["mkt_median_realized_vol_20"].median()
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a FinTTA panel for schema, leakage, and diagnostics.")
    parser.add_argument("--panel", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--graph", default=None)
    parser.add_argument("--output", default="data/processed/data_validation_report.json")
    parser.add_argument("--markdown-output", default="data/processed/data_validation_report.md")
    args = parser.parse_args()

    report = validate_panel(args.panel, args.config, args.graph)
    write_json(args.output, report)
    write_markdown_report(report, args.markdown_output)
    print(f"validation {report['status']}: {args.output}")


if __name__ == "__main__":
    main()
