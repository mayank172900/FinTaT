from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from data_utils import ASSET_COLUMN, DATE_COLUMN, FACTOR_COLUMNS, read_table, sort_panel, write_table


def build_graph_edges(
    panel: pd.DataFrame,
    *,
    topk_per_asset: int = 20,
    min_abs_weight: float = 0.05,
) -> pd.DataFrame:
    panel = sort_panel(panel)
    returns = panel.pivot(index=DATE_COLUMN, columns=ASSET_COLUMN, values="ret_1d").sort_index()
    dates = list(panel[DATE_COLUMN].drop_duplicates())
    rows: list[dict] = []

    for date in dates:
        day = panel[panel[DATE_COLUMN] == date].sort_values(ASSET_COLUMN)
        assets = day[ASSET_COLUMN].astype(str).tolist()
        if len(assets) < 2:
            continue
        corr_by_window = {}
        for window in [20, 60, 252]:
            hist = returns.loc[:date, assets].tail(window)
            corr_by_window[window] = hist.corr(min_periods=min(5, len(hist)))

        date_edges = []
        records = day.set_index(ASSET_COLUMN).to_dict("index")
        for asset_i, asset_j in itertools.combinations(assets, 2):
            rec_i = records[asset_i]
            rec_j = records[asset_j]
            flags: list[str] = []
            positive = 0.0
            negative = 0.0

            if rec_i.get("sector") == rec_j.get("sector"):
                positive += 0.20
                flags.append("same_sector")
            if rec_i.get("industry") == rec_j.get("industry"):
                positive += 0.20
                flags.append("same_industry")
            if rec_i.get("country") == rec_j.get("country"):
                positive += 0.05
                flags.append("same_country")
            if rec_i.get("exchange") == rec_j.get("exchange"):
                positive += 0.03
                flags.append("same_exchange")

            for window, weight in [(20, 0.20), (60, 0.22), (252, 0.15)]:
                corr = corr_by_window[window].loc[asset_i, asset_j] if asset_i in corr_by_window[window].index and asset_j in corr_by_window[window].columns else np.nan
                if pd.notna(corr):
                    if corr >= 0:
                        positive += weight * float(corr)
                        if corr > 0.1:
                            flags.append(f"rolling_corr_{window}_positive")
                    else:
                        negative += weight * abs(float(corr))
                        if corr < -0.1:
                            flags.append(f"rolling_corr_{window}_negative")

            beta_i = np.array([float(rec_i.get(c, 0.0) or 0.0) for c in FACTOR_COLUMNS], dtype=float)
            beta_j = np.array([float(rec_j.get(c, 0.0) or 0.0) for c in FACTOR_COLUMNS], dtype=float)
            denom = np.linalg.norm(beta_i) * np.linalg.norm(beta_j)
            cosine = float(beta_i.dot(beta_j) / denom) if denom > 0 else 0.0
            if cosine >= 0:
                positive += 0.15 * cosine
                flags.append("factor_beta_cosine_positive")
            else:
                negative += 0.15 * abs(cosine)
                flags.append("factor_beta_cosine_negative")

            if np.sign(rec_i.get("factor_rates_beta", 0.0)) != np.sign(rec_j.get("factor_rates_beta", 0.0)):
                negative += 0.03
                flags.append("rates_exposure_opposite")
            if np.sign(rec_i.get("factor_oil_beta", 0.0)) != np.sign(rec_j.get("factor_oil_beta", 0.0)):
                negative += 0.03
                flags.append("commodity_exposure_opposite")

            edge_sign = 1 if positive >= negative else -1
            edge_weight = max(positive, negative)
            if edge_weight >= min_abs_weight:
                date_edges.append(
                    {
                        "date": date,
                        "asset_id_i": asset_i,
                        "asset_id_j": asset_j,
                        "same_sector_flag": int(rec_i.get("sector") == rec_j.get("sector")),
                        "same_industry_flag": int(rec_i.get("industry") == rec_j.get("industry")),
                        "same_country_flag": int(rec_i.get("country") == rec_j.get("country")),
                        "same_exchange_flag": int(rec_i.get("exchange") == rec_j.get("exchange")),
                        "positive_score": positive,
                        "negative_score": negative,
                        "edge_sign": edge_sign,
                        "edge_weight_raw": edge_weight,
                        "edge_source_flags": ",".join(sorted(set(flags))),
                    }
                )

        if topk_per_asset > 0 and date_edges:
            edge_df = pd.DataFrame(date_edges)
            keep_idx = set()
            for side in ["asset_id_i", "asset_id_j"]:
                for _, sub in edge_df.groupby(side):
                    keep_idx.update(sub.nlargest(topk_per_asset, "edge_weight_raw").index.tolist())
            edge_df = edge_df.loc[sorted(keep_idx)]
            rows.extend(edge_df.to_dict("records"))
        else:
            rows.extend(date_edges)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create signed cross-asset graph inputs.")
    parser.add_argument("--panel", required=True)
    parser.add_argument("--output", default="data/intermediate/graph_edges_daily.parquet")
    parser.add_argument("--topk-per-asset", type=int, default=20)
    parser.add_argument("--min-abs-weight", type=float, default=0.05)
    args = parser.parse_args()

    panel = read_table(args.panel)
    edges = build_graph_edges(panel, topk_per_asset=args.topk_per_asset, min_abs_weight=args.min_abs_weight)
    write_table(edges, args.output)
    print(f"wrote graph edges: {args.output} ({len(edges):,} rows)")


if __name__ == "__main__":
    main()
