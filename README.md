# FinTaT

FinTaT is an end-to-end research prototype for **source-free test-time adaptation on financial tabular prediction**. It implements FinTTA, a regime-aware, graph-structured, risk-sensitive TTA framework, plus a reproducible open-data benchmark pipeline.

The repository is intentionally honest about the current empirical result: on the bundled open-data S&P 500 prototype, the tested FinTTA variants did **not** beat no-adaptation on the main classification/calibration metrics. That negative result is preserved because it is useful research evidence and a strong baseline for future work.

## What Is Implemented

- Regime-aware distribution tracker with jump-hazard prior resets.
- Signed cross-asset graph reliability and consistency loss.
- Risk-aware entropy weighting and risk-adjusted label priors.
- Regime-specific adapter and EMA teacher memory.
- Same-batch and prequential TTA evaluation modes.
- Open-data pipeline for a survivor-bias-reduced S&P 500 prototype.
- Validation scripts for leakage, schema, splits, and feature metadata.
- Experiment runner with no-adaptation, FinTTA, ablations, Tent-lite, and conservative rescue variants.

## Repository Layout

```text
src/fintta/              Core FinTTA package
scripts/                 Data builders, validators, and experiment runners
configs/                 Data and experiment configs
data/sample/             Tiny committed schema-compatible fixture
data/README.md           Data contract and open-data caveats
docs/                    Protocol, schema, and result notes
outputs/*/metrics.csv    Committed summary result tables
```

Large generated parquet files are intentionally ignored because they exceed normal GitHub limits. Rebuild them locally with the commands below.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

If your system Python already has the dependencies, you can run with:

```bash
PYTHONPATH=src pytest -q
```

## Rebuild Open Data

This downloads no-key/free public data where possible and creates the open-data prototype panel.

```bash
python3 scripts/download_open_data.py
python3 scripts/build_panel_from_raw.py --config configs/data_open.yaml
python3 scripts/validate_panel.py \
  --panel data/processed/panel_daily_2015_2024.parquet \
  --config configs/experiment_2015_2024.yaml
```

Notes:

- This is not CRSP-grade.
- Yahoo/yfinance misses many delisted/renamed tickers.
- FRED values are latest/revised, not ALFRED vintage.
- Current public sector metadata is not true point-in-time GICS.
- FINRA CNMS short-sale volume is a proxy, not borrow cost.

## Run Experiments

Quick smoke run:

```bash
python3 scripts/run_open_panel_experiment.py \
  --quick \
  --variants all \
  --output-dir outputs/open_panel_quick_all \
  --epochs 1 \
  --max-assets-per-day 40
```

Main open-data run:

```bash
python3 scripts/run_open_panel_experiment.py \
  --variants all \
  --output-dir outputs/open_panel_full \
  --epochs 8 \
  --max-assets-per-day 180
```

Final conservative rescue run:

```bash
python3 scripts/run_open_panel_experiment.py \
  --variants no_adaptation,conservative_bias_prequential,conservative_bias_same_batch,calibration_bias_prequential \
  --output-dir outputs/open_panel_rescue \
  --epochs 8 \
  --max-assets-per-day 180
```

## Current Result

The main full run is in [outputs/open_panel_full/metrics.csv](outputs/open_panel_full/metrics.csv).

Summary:

- No-adaptation had the best balanced accuracy, macro-F1, NLL, Brier, ECE, and false-positive-buy loss among the serious variants.
- Full FinTTA improved some trading drawdown/Sharpe values relative to no-adaptation, but the classification and calibration degradation was too large.
- Conservative bias/temperature-only rescue variants did not recover the result.

Conclusion: **freeze this as a research-engineering artifact and negative empirical finding, not a current conference-paper bet.**

## Tests

```bash
pytest -q
```

Current local verification: `6 passed`.

## Citation / Project Statement

This project is best described as:

> A source-free financial test-time adaptation framework and open-data benchmark showing that entropy-style online adaptation can be fragile under realistic financial tabular drift, even with regime tracking, graph reliability, risk weighting, and conservative calibration-only rescue variants.

