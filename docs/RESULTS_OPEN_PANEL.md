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
