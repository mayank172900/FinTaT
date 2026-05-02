# Experiment Protocol

This repository separates implementation verification from conference experiments.

## Verification Mode

Use the deterministic synthetic stream only to verify:

- regime jump control flow;
- signed graph construction;
- risk-aware entropy and prior-volume losses;
- adapter-bank switching and teacher updates;
- prequential and same-batch online execution.

Synthetic results should not be reported as empirical evidence.

## Conference Mode

Recommended source window:

- train source models on 2015-2019;
- deploy source-free TTA on 2020-2024 chronological batches.
- serialize source-period market-state summaries/regime emissions before deployment; do not carry source rows into test-time adaptation.

Recommended universes:

- CRSP active/inactive U.S. equities, or a point-in-time constituent product;
- optionally Russell 1000/top-N liquid universe with inactive securities retained.

Core variants:

- `no_adaptation`;
- `fintta`;
- `fintta --ablation no-risk`;
- `fintta --ablation no-graph`;
- `fintta --ablation no-prior`;
- `fintta --ablation no-teacher`.

Additional baselines such as Tent, EATA, CoTTA, SAR, and FTaT should share the same `PanelDataset` batches and the same source model checkpoint. Implement them as separate engines with the same `step(batch)` contract so the evaluation script remains identical.

## Metrics

Report by full period and market segment:

- accuracy, balanced accuracy, macro F1, NLL, Brier, ECE;
- detection delay, false alarms, posterior entropy, prior adaptation half-life;
- annualized return, Sharpe, max drawdown, Calmar, turnover, tail loss, false-positive-buy crash loss.

## Evaluation Modes

- Prequential: predict at timestamp `t`, adapt on unlabeled `B_t`, and apply the update from `t+1`.
- Same-batch TTA: adapt on unlabeled `B_t`, then emit `t` probabilities. This is valid only when the deployment permits using the full contemporaneous cross-section before trading and no future returns or labels enter the loss.
