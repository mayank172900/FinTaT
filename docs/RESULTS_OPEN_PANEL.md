# Open-Panel Results

This document records the empirical outcome of the open-data prototype run.

Dataset:

- Period: 2014-01-01 to 2024-12-31.
- Source training split: 2015-01-01 to 2019-12-31.
- TTA test split: 2020-01-01 to 2024-12-31.
- Universe: survivor-bias-reduced S&P 500 point-in-time constituent proxy from public data.
- Prices: Yahoo/yfinance daily OHLCV for available tickers.
- Macro/factors: FRED latest/revised values, Cboe VIX, Kenneth French factors.
- Short pressure: FINRA CNMS short-sale volume from 2018 onward.

Important limitation:

- This is not a CRSP/Compustat/TAQ institutional dataset.

## Main Full Run

Source file:

- `outputs/open_panel_full/metrics.csv`

Key baseline:

```text
no_adaptation
balanced_accuracy = 0.22699
macro_f1          = 0.22128
nll               = 1.57304
brier             = 0.78875
ece               = 0.06431
trade_sharpe      = -1.39784
fp_buy_loss       = 0.03157
```

Best full FinTTA-style variants did not beat no-adaptation on classification or calibration. For example:

```text
fintta_prequential
balanced_accuracy = 0.20037
macro_f1          = 0.09250
nll               = 3.36676
brier             = 1.25781
ece               = 0.59398
fp_buy_loss       = 0.03598
```

## Conservative Rescue

Source files:

- `outputs/open_panel_rescue_180/metrics.csv`
- `outputs/open_panel_rescue_calibration_180/metrics.csv`

The strict conservative bias/temperature variants made zero online updates and improved only ECE while hurting balanced accuracy, macro-F1, NLL, and false-positive-buy loss.

The lower-gate calibration-only variant updated on most batches but still lost to no-adaptation:

```text
calibration_bias_prequential
balanced_accuracy = 0.21560
macro_f1          = 0.10796
nll               = 1.64542
brier             = 0.81721
ece               = 0.12953
fp_buy_loss       = 0.03680
```

## Decision

Freeze the project as:

- a strong implementation/research-engineering artifact;
- a negative empirical finding about financial tabular TTA fragility;
- a possible future project only if better licensed data or a simpler adaptation hypothesis becomes available.

Do not make this the primary research bet for graduate applications without a new empirical breakthrough.

## Post-Fix Rerun (2026-07-08)

After the audit fixes (issues #1-#9: adapter blend initialization, LayerNorm
adaptability, strict prequential ordering, universe-aligned trading metrics,
validator leakage gate, and downloader robustness), the open-data stack was
re-downloaded from scratch and the full experiment was rerun:

- `outputs/open_panel_full_postfix/metrics.csv`
- `outputs/open_panel_full_postfix/run_summary.json` (includes provenance)

```text
variant                        bal_acc  macro_f1   nll     brier   ece     adapt_rate
no_adaptation                  0.2236   0.2125     1.5750  0.7888  0.0672  -
fintta_prequential             0.2001   0.0912     2.6376  1.1686  0.5330  0.9928
fintta_same_batch              0.2001   0.0912     2.6398  1.1692  0.5335  0.9928
conservative_bias_prequential  0.2218   0.2062     1.5598  0.7821  0.0544  0.0000
conservative_bias_same_batch   0.2219   0.2063     1.5598  0.7821  0.0544  0.0000
calibration_bias_prequential   0.2232   0.2088     1.5551  0.7805  0.0076  0.9889
fintta_no_risk                 0.2047   0.1190     2.6366  1.1369  0.4639  0.9928
fintta_no_graph                0.2001   0.0912     2.6322  1.1677  0.5324  0.9928
fintta_no_prior                0.2047   0.0681     4.9127  1.5692  0.7383  0.9928
fintta_no_teacher              0.2001   0.0911     2.6237  1.1659  0.5311  0.9928
tent_lite                      0.2014   0.1035     3.3697  1.2032  0.5232  0.9928
```

Reading:

- The core negative result stands: full FinTTA still loses to no-adaptation
  on balanced accuracy and macro-F1, though far less pathologically than in
  the pre-fix run (NLL 2.64 vs 3.37).
- `calibration_bias_prequential` now beats no-adaptation on NLL, Brier, and
  ECE (0.0076 vs 0.0672) at essentially equal accuracy. This flips the
  calibration-only rescue from a loss to a clear win: source-free test-time
  calibration works here even though source-free accuracy adaptation does not.
- The prior-tracking ablation (`fintta_no_prior`) remains the worst variant,
  confirming the label-prior correction is the load-bearing safeguard.
- Trading metrics are not comparable to the pre-fix table: the turnover
  computation was fixed and the data was rebuilt. Compare within a run only.

## Revised Decision

The project is now:

- a research-engineering artifact with a reproducible open-data pipeline;
- a preserved negative result on entropy-style source-free accuracy TTA;
- a small positive result on calibration-only source-free TTA
  (adapt the confidence, not the decision function).
