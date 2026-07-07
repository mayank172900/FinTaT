# FinTaT

FinTaT is my research prototype for **fully source-free test-time adaptation on financial tabular data**.

The original question was simple:

> If a financial model is trained on one market regime, can it adapt online during deployment using only the unlabeled market stream?

I wanted to test this seriously, not just run entropy minimization on stock rows and call it finance. Financial data has abrupt regime changes, asymmetric losses, cross-asset structure, shifting class priors, and a lot of ways to accidentally leak the future. So this repo contains both the method implementation and the data/evaluation pipeline I built to stress-test the idea.

The current result is also included honestly: on the open-data S&P 500 prototype, the FinTTA variants did **not** beat no-adaptation on the main classification and calibration metrics. I am keeping that result because it is useful. It shows that source-free entropy-style adaptation in finance is fragile, even after adding regime tracking, graph reliability, risk weighting, and conservative calibration-only variants.

## The Idea

Existing test-time adaptation methods give useful ingredients, but finance breaks a lot of their assumptions. Vanilla entropy minimization can push the model toward overconfident predictions. Smooth label-prior tracking can lag during crashes. Tabular augmentation is weak. And a bad buy signal in a crash is not symmetric with a bad hold signal in a calm regime.

The FinTTA idea I implemented is:

```text
FinTTA =
  regime jump detection
+ cross-asset graph reliability
+ risk-weighted entropy
+ regime-specific adapter memory
```

At each timestamp `t`, the model receives a cross-sectional unlabeled batch:

```math
B_t=\{(x_{i,t}, a_i, m_i)\}_{i=1}^{n_t}
```

where `x` is the asset feature vector, `a_i` is the asset id, and `m_i` contains metadata such as sector, industry, exchange, and factor information.

For the default five-class setup:

```text
0 = strong sell
1 = sell
2 = hold
3 = buy
4 = strong buy
```

During adaptation, the engine does **not** use source training rows, future labels, or future returns.

## Objective

The online FinTTA objective is:

```math
\mathcal{L}_{\text{FinTTA}}
=
\mathcal{L}_{\text{risk-ent}}
+ \alpha_G\mathcal{L}_{\text{graph}}
+ \alpha_\pi\mathcal{L}_{\text{prior}}
+ \alpha_T\mathcal{L}_{\text{teacher}}
+ \alpha_F\mathcal{L}_{\text{anchor}}.
```

The main adaptation term is a risk-weighted entropy loss:

```math
\mathcal{L}_{\text{risk-ent}}
=
\frac{1}{\sum_i \omega_i+\varepsilon}
\sum_i \omega_i
\left[
-\sum_{k=1}^{K}
\lambda_{t,k}p_{i,t,k}\log(p_{i,t,k}+\varepsilon)
\right].
```

The important detail is that financial cost is converted into a **trust weight**, not a naive larger-cost-larger-gradient coefficient. Entropy minimization sharpens predictions. If a class is dangerous to sharpen in the current regime, its entropy weight should go down:

```math
\lambda_{t,k}
\propto
(D_{t,k}+\varepsilon)^{-\rho_\lambda}.
```

Here `D_{t,k}` is the expected regime-conditioned cost of sharpening class `k`.

## Modules

### 1. Regime-Aware Distribution Tracker

The tracker builds a market-state vector `psi_t` from causal market information: volatility, dispersion, cross-sectional correlation, market-mode eigenvalue share, liquidity stress, VIX, rates, and credit features.

It maintains a latent regime posterior:

```math
\gamma_t(r)=P(z_t=r\mid \psi_{1:t}).
```

Each regime stores a Dirichlet label-prior state:

```math
\alpha_{r,t}\in\mathbb{R}_+^K,
\qquad
\pi_{r,t,k}
=
\frac{\alpha_{r,t,k}}{\sum_j\alpha_{r,t,j}}.
```

During stable periods, the label prior updates slowly. During shocks, the concentration is reset so the prior can move quickly:

```math
\alpha_{r^\star,t+1}
=
c_{\text{reset}}\frac{\mathbf{1}}{K}
+ m_{\text{jump}}\hat q_t.
```

This was meant to avoid dragging a bull-market prior into a crash.

### 2. Signed Cross-Asset Graph

Instead of tabular augmentation, FinTTA uses financial structure. Assets are connected by a signed graph:

```math
G_t=(V_t,E_t).
```

Positive edges connect assets that should often move or be classified similarly: same sector, same industry, positive rolling correlation, similar factor exposures.

Negative edges connect inverse or hedge-like relationships: negative correlation, opposite factor exposure, or known inverse relations.

For ordinal labels, negative edges use a reversal operator:

```text
strong sell <-> strong buy
sell        <-> buy
hold        <-> hold
```

The graph loss combines distributional and directional consistency:

```math
\mathcal{L}_{\text{graph}}
=
\mathcal{L}_{\text{JS}}
+ \eta_{\text{dir}}\mathcal{L}_{\text{dir}}.
```

Graph disagreement is also used as a reliability score:

```math
g_{i,t}
=
\exp(-\Delta_{i,t}/\tau_G).
```

So if an asset prediction is isolated or economically inconsistent, its entropy-gradient weight is reduced.

### 3. Risk-Aware Entropy

Classes map to trading exposure:

```math
u=[-1,-0.5,0,0.5,1].
```

The expected cost of sharpening class `k` is:

```math
D_{t,k}
=
\sum_r \gamma_t(r)
\sum_y \pi_{r,t,y}c_r(k,y).
```

This feeds a risk-adjusted prior:

```math
\pi^R_{t,k}
=
\frac{
(\pi_{t,k}+\epsilon_\pi)
\exp(-D_{t,k}/T_R)
}{
\sum_j
(\pi_{t,j}+\epsilon_\pi)
\exp(-D_{t,j}/T_R)
}.
```

The goal was to prevent the model from aggressively sharpening high-risk buy predictions during high-volatility regimes.

### 4. Regime Adapter Memory

Instead of letting one model drift forever, FinTTA stores small adaptation states per regime:

```math
\theta_t = \theta_0 + \phi_{r_t^\star}.
```

The base model stays frozen. Only small adaptation parameters are updated: calibration terms, normalization affine parameters, and small adapters. When a regime recurs, its adapter can be reused instead of overwritten.

## What Is In This Repo

- Core FinTTA implementation in `src/fintta/`
- Data download/build/validation scripts in `scripts/`
- Open-data S&P 500 prototype pipeline
- Tiny committed sample fixture for tests
- Experiment runner with:
  - no adaptation
  - FinTTA prequential
  - FinTTA same-batch
  - risk/graph/prior/teacher ablations
  - Tent-lite
  - conservative bias/temperature rescue variants
- Result summaries under `outputs/*/metrics.csv`
- Notes on the negative result in `docs/RESULTS_OPEN_PANEL.md`

## Repository Layout

```text
src/fintta/              Core FinTTA package
scripts/                 Data builders, validators, and experiment runners
configs/                 Data and experiment configs
data/sample/             Small committed schema-compatible fixture
data/README.md           Data contract and caveats
docs/                    Protocol, schema, and result notes
outputs/*/metrics.csv    Committed summary result tables
```

Large generated parquet files are ignored because they exceed normal GitHub limits. The scripts rebuild them locally.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For the optional open-data prototype downloader, install the open-data extras too:

```bash
pip install -e ".[dev,open-data]"
```

Run tests:

```bash
pytest -q
```

Current local verification:

```text
6 passed
```

## Rebuild The Open-Data Prototype

```bash
python3 scripts/download_open_data.py
python3 scripts/build_panel_from_raw.py --config configs/data_open.yaml
python3 scripts/validate_panel.py \
  --panel data/processed/panel_daily_2015_2024.parquet \
  --config configs/experiment_2015_2024.yaml
```

The open-data stack uses:

- PIT S&P 500 constituent proxy from `fja05680/sp500`
- Yahoo/yfinance OHLCV for available tickers
- FRED macro/rates/credit series
- Cboe/FRED VIX
- Kenneth French factors
- FINRA CNMS short-sale volume proxy
- SEC ticker-to-CIK map

Important caveats:

- This is not CRSP-grade.
- Yahoo/yfinance misses many delisted or renamed tickers.
- FRED values are latest/revised, not ALFRED vintage.
- Current public sector metadata is not true point-in-time GICS.
- FINRA short-sale volume is not borrow cost.

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

Main run:

```bash
python3 scripts/run_open_panel_experiment.py \
  --variants all \
  --output-dir outputs/open_panel_full \
  --epochs 8 \
  --max-assets-per-day 180
```

Conservative rescue run:

```bash
python3 scripts/run_open_panel_experiment.py \
  --variants no_adaptation,conservative_bias_prequential,conservative_bias_same_batch,calibration_bias_prequential \
  --output-dir outputs/open_panel_rescue \
  --epochs 8 \
  --max-assets-per-day 180
```

## Empirical Outcome

The main full run is in:

```text
outputs/open_panel_full/metrics.csv
```

The best baseline was no-adaptation:

```text
no_adaptation
balanced_accuracy = 0.22699
macro_f1          = 0.22128
nll               = 1.57304
brier             = 0.78875
ece               = 0.06431
fp_buy_loss       = 0.03157
```

FinTTA prequential did not beat it:

```text
fintta_prequential
balanced_accuracy = 0.20037
macro_f1          = 0.09250
nll               = 3.36676
brier             = 1.25781
ece               = 0.59398
fp_buy_loss       = 0.03598
```

I also tried conservative bias/temperature-only variants. The strict version mostly refused to update, and the looser calibration-only version updated but still lost to no-adaptation:

```text
calibration_bias_prequential
balanced_accuracy = 0.21560
macro_f1          = 0.10796
nll               = 1.64542
brier             = 0.81721
ece               = 0.12953
fp_buy_loss       = 0.03680
```

So the final research conclusion is:

> On this open-data financial tabular benchmark, entropy-style source-free TTA was fragile. The added finance-aware safeguards helped relative to naive TTA in some cases, but did not beat a frozen source model on the main metrics.

That makes the project a negative empirical study and a research-engineering artifact rather than a current conference-paper result.

## Why I Still Think This Was Worth Doing

The negative result is useful because it narrows the space. It suggests that for financial tabular deployment, simply making TTA more elaborate is not enough. Future work probably needs either:

- better point-in-time institutional data;
- stronger uncertainty/rejection mechanisms;
- objectives that are not entropy-minimization-first;
- explicit causal or portfolio-level constraints;
- or supervised periodic recalibration rather than fully source-free adaptation.

For now, this repo preserves the implementation, the benchmark pipeline, and the empirical failure mode clearly.

