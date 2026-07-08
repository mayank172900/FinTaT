from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REGIME_BANDS = [
    ("2020-02-19", "2020-04-30", "COVID crash"),
    ("2022-01-03", "2022-10-14", "2022 rate shock"),
]

VARIANT_STYLE = {
    "no_adaptation": ("Frozen source", "#444444", "-"),
    "calibration_bias_prequential": ("Calibration-restricted (label-free)", "#1a7f37", "-"),
    "online_temp": ("Online temp. scaling (delayed labels)", "#0969da", "--"),
    "fintta_prequential": ("Entropy TTA + safeguards", "#bc4c00", "-"),
    "tent_full": ("Tent", "#cf222e", ":"),
}


def load_daily(run_glob: str, variant: str) -> pd.DataFrame | None:
    frames = []
    for run_dir in sorted(glob.glob(run_glob)):
        path = Path(run_dir) / "daily" / f"{variant}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, parse_dates=["date"])
        frame["run"] = run_dir
        frames.append(frame)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def calibration_gap_figure(run_glob: str, out_path: Path, window: int = 21) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    for variant, (label, color, linestyle) in VARIANT_STYLE.items():
        daily = load_daily(run_glob, variant)
        if daily is None:
            continue
        gap = (
            daily.assign(gap=daily["mean_confidence"] - daily["top1_accuracy"])
            .groupby("date")["gap"]
            .mean()
            .sort_index()
            .rolling(window, min_periods=window // 2)
            .mean()
        )
        ax.plot(gap.index, gap.values, label=label, color=color, linestyle=linestyle, linewidth=1.4)
    for start, end, band_label in REGIME_BANDS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="#d0d7de", alpha=0.45, lw=0)
        ax.text(
            pd.Timestamp(start), ax.get_ylim()[1], f" {band_label}", fontsize=7, va="top", color="#57606a"
        )
    ax.axhline(0.0, color="#888888", linewidth=0.8, linestyle="-")
    ax.set_ylabel(f"{window}-day mean confidence $-$ accuracy")
    ax.set_xlabel("")
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures from sweep run directories.")
    parser.add_argument("--runs", default="outputs/full_seed*", help="Glob of run output directories.")
    parser.add_argument("--out-dir", default="paper/figures")
    parser.add_argument("--window", type=int, default=21)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    calibration_gap_figure(args.runs, out_dir / "fig_calibration_gap.pdf", window=args.window)


if __name__ == "__main__":
    main()
