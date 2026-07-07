# Data Schema

FinTTA expects a chronological point-in-time panel with one row per `(date, asset_id)`.
The engine never reads labels during adaptation; label columns are only used by evaluation code outside `FinTTAEngine`.

## Required Columns

- `date`: timestamp available at decision time.
- `asset_id`: stable identifier such as PERMNO, FIGI, or vendor security id.
- `ret_1d`: realized return ending at `date`, used only for causal rolling market state and graph statistics.
- `volume`: causal volume measure.
- `sector`: point-in-time sector or equivalent grouping.
- `industry`: point-in-time industry or equivalent grouping.

## Typical Feature Columns

Feature columns should be precomputed with as-of joins:

- return lags and momentum;
- short-term reversal residuals such as `feat_reversal_1`, which should not be an exact duplicate transform of a return lag;
- realized volatility and drawdown ending at `date`;
- spread/liquidity estimates available at `date`;
- rolling factor betas estimated only on data through `date`;
- sector-relative strength;
- event flags whose announcement timestamp is no later than `date`;
- macro/rates/volatility values from vintage data where revisions matter.

## Optional Evaluation Columns

- `label`: future-return bucket for offline metrics only.
- `forward_return`: next-period realized return for trading metrics only.
- factor exposure columns, passed through `PanelSpec.factor_columns`.
- liquidity gate column in `[0, 1]`, passed through `PanelSpec.liquidity_column`.

For the empirical data layer, model features are prefixed with `feat_`.
Evaluation-only columns use `label_`, `forward_return_`, or
`future_norm_return_` prefixes and must be excluded from adaptation feature
lists.

The committed `data/sample/sample_panel.parquet` is a small historical schema
fixture. Its feature list is kept stable unless the schema changes; future
panel rebuilds use the current feature formulas in `scripts/data_utils.py`.

## Leakage Rules

- Do not backfill current index constituents over historical periods.
- Include delisted names and delisting returns where the vendor supports them.
- For macro data, use vintage/as-of releases for historical experiments.
- Fit feature scalers only on the source training window, then apply them chronologically.
- Report both prequential prediction-first and same-batch unlabeled adaptation variants.
