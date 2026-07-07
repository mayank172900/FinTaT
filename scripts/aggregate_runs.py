from __future__ import annotations

import argparse
import glob
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multiple FinTTA run directories.")
    parser.add_argument("run_globs", nargs="+", help="Glob(s) pointing at run output directories.")
    parser.add_argument("--output-dir", default="outputs/aggregate_runs")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    aggregate_runs(args.run_globs, args.output_dir, bootstrap_samples=args.bootstrap_samples, seed=args.seed)


def aggregate_runs(
    run_globs: Sequence[str],
    output_dir: str | Path,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dirs = _expand_run_dirs(run_globs)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = _load_metrics(run_dirs)
    aggregate_metrics = _aggregate_metrics(metrics)
    aggregate_metrics.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    (output_dir / "aggregate_metrics.md").write_text(_format_markdown(aggregate_metrics), encoding="utf-8")

    paired = _paired_tests(run_dirs, bootstrap_samples=bootstrap_samples, seed=seed)
    paired.to_csv(output_dir / "paired_tests.csv", index=False)
    return aggregate_metrics, paired


def _expand_run_dirs(run_globs: Sequence[str]) -> list[Path]:
    run_dirs: list[Path] = []
    for pattern in run_globs:
        run_dirs.extend(Path(path) for path in sorted(glob.glob(pattern)))
    unique = []
    seen = set()
    for run_dir in sorted(run_dirs, key=lambda path: str(path)):
        if run_dir.is_dir() and str(run_dir) not in seen:
            seen.add(str(run_dir))
            unique.append(run_dir)
    if not unique:
        raise FileNotFoundError(f"no run directories matched: {list(run_globs)}")
    return unique


def _load_metrics(run_dirs: Sequence[Path]) -> pd.DataFrame:
    frames = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"missing metrics.csv in {run_dir}")
        frame = pd.read_csv(metrics_path)
        frame["run_dir"] = str(run_dir)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [col for col in metrics.columns if col not in {"variant", "run_dir"} and pd.api.types.is_numeric_dtype(metrics[col])]
    rows = []
    for variant, group in metrics.sort_values("variant").groupby("variant", sort=True):
        row: dict[str, float | str] = {"variant": variant, "seed_count": int(group["run_dir"].nunique())}
        for col in numeric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def _format_markdown(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "# Aggregate Metrics\n"
    metric_bases = sorted({col[:-5] for col in summary.columns if col.endswith("_mean")})
    headers = ["variant"] + metric_bases + ["seed_count"]
    lines = ["# Aggregate Metrics", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in summary.sort_values("variant").iterrows():
        cells = [str(row["variant"])]
        for metric in metric_bases:
            cells.append(_format_mean_std(row.get(f"{metric}_mean", float("nan")), row.get(f"{metric}_std", float("nan"))))
        cells.append(str(int(row["seed_count"])))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _format_mean_std(mean: float, std: float) -> str:
    if not np.isfinite(mean) or not np.isfinite(std):
        return "nan ± nan"
    return f"{mean:.6f} ± {std:.6f}"


def _paired_tests(run_dirs: Sequence[Path], *, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    variants = sorted({variant for run_dir in run_dirs for variant in pd.read_csv(run_dir / "metrics.csv")["variant"].astype(str)})
    variants = [variant for variant in variants if variant != "no_adaptation"]
    rng = np.random.default_rng(seed)
    rows = []
    for variant in variants:
        for metric in ("nll", "brier"):
            pooled_diffs: list[float] = []
            seed_wins: list[bool] = []
            for run_dir in run_dirs:
                baseline = _read_daily_metric(run_dir, "no_adaptation")
                challenger = _read_daily_metric(run_dir, variant)
                if baseline is None or challenger is None:
                    continue
                merged = baseline[["date", metric]].rename(columns={metric: "baseline"}).merge(
                    challenger[["date", metric]].rename(columns={metric: "challenger"}),
                    on="date",
                    how="inner",
                )
                merged = merged.dropna(subset=["baseline", "challenger"])
                if merged.empty:
                    continue
                diffs = merged["challenger"].to_numpy(dtype=np.float64) - merged["baseline"].to_numpy(dtype=np.float64)
                pooled_diffs.extend(diffs.tolist())
                base_mean = float(baseline[metric].mean())
                chal_mean = float(challenger[metric].mean())
                if np.isfinite(base_mean) and np.isfinite(chal_mean):
                    seed_wins.append(chal_mean < base_mean)
            if not pooled_diffs:
                rows.append(
                    {
                        "variant": variant,
                        "baseline": "no_adaptation",
                        "metric": metric,
                        "mean_diff": float("nan"),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                        "win_fraction": float("nan"),
                        "n_days": 0,
                        "n_seeds": 0,
                    }
                )
                continue
            diffs = np.asarray(pooled_diffs, dtype=np.float64)
            boot = np.empty(bootstrap_samples, dtype=np.float64)
            for idx in range(bootstrap_samples):
                sample = rng.choice(diffs, size=diffs.size, replace=True)
                boot[idx] = float(sample.mean())
            ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
            rows.append(
                {
                    "variant": variant,
                    "baseline": "no_adaptation",
                    "metric": metric,
                    "mean_diff": float(diffs.mean()),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "win_fraction": float(np.mean(seed_wins)) if seed_wins else float("nan"),
                    "n_days": int(diffs.size),
                    "n_seeds": int(len(seed_wins)),
                }
            )
    return pd.DataFrame(rows)


def _read_daily_metric(run_dir: Path, variant: str) -> pd.DataFrame | None:
    path = run_dir / "daily" / f"{variant}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


if __name__ == "__main__":
    main()
