# FinTTA Data Validation Report

Status: `pass`
Panel: `data/processed/panel_daily_2015_2024.parquet`

## Schema
- Rows: 1,563,085
- Assets: 598
- Duplicate `(date, asset_id)` rows: 0
- Missing minimum columns: []

## Leakage
- Result: pass
- Evaluation columns in adaptation feature list: []
- Timestamp violations: {}

## Survivorship
- Inactive rows: 0
- Delisting-return rows: 0

## Graph
- {'available': False, 'reason': 'graph edge file not found: data/intermediate/graph_edges_daily.parquet'}

## Market State
- {'required_columns_present': ['mkt_median_realized_vol_20', 'mkt_cross_sectional_mad_return', 'mkt_average_abs_corr_60', 'mkt_market_mode_eigen_share_60', 'mkt_liquidity_stress', 'mkt_vix', 'mkt_credit_spread', 'mkt_yield_curve_10y_2y'], 'missing_columns': [], 'summary': {'mkt_median_realized_vol_20': {'min': 0.0, 'p25': 0.011897597899146463, 'median': 0.014335709557260004, 'p75': 0.017341502771793957, 'max': 0.07494883463893053}, 'mkt_cross_sectional_mad_return': {'min': 0.0, 'p25': 0.005966889219848309, 'median': 0.007204393927066233, 'p75': 0.008885423100824921, 'max': 0.04733736294051316}, 'mkt_average_abs_corr_60': {'min': 0.0, 'p25': 0.24389572283241445, 'median': 0.3002406645148047, 'p75': 0.3684092516437616, 'max': 0.7143086381979058}, 'mkt_market_mode_eigen_share_60': {'min': 0.0, 'p25': 0.26010483003258367, 'median': 0.32771090814042214, 'p75': 0.39770986170890515, 'max': 0.7313538651558844}, 'mkt_liquidity_stress': {'min': 0.2083654193777772, 'p25': 0.3667958572552947, 'median': 0.42450212354178146, 'p75': 0.5242408162718616, 'max': 1.0}, 'mkt_vix': {'min': 9.14, 'p25': 13.1975, 'median': 15.945, 'p75': 20.7925, 'max': 82.69}, 'mkt_credit_spread': {'min': 1.36, 'p25': 1.9, 'median': 2.19, 'p75': 2.45, 'max': 4.31}, 'mkt_yield_curve_10y_2y': {'min': -1.08, 'p25': 0.14, 'median': 0.57, 'p75': 1.19, 'max': 2.61}}, 'crisis_volatility_elevated': True}
