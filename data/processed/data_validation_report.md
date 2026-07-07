# FinTTA Data Validation Report

Status: `pass`
Panel: `data/processed/panel_daily_2015_2024.parquet`

## Schema
- Rows: 1,251,504
- Assets: 460
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
- {'available': True, 'edge_count': 1251504, 'edge_count_per_date_summary': {'min': 438.0, 'p25': 450.0, 'median': 454.0, 'p75': 455.0, 'max': 459.0}, 'positive_edges': 0, 'negative_edges': 0, 'positive_negative_ratio': 0.0, 'isolated_asset_count_top10_dates': {'2014-01-02': 438, '2014-01-03': 438, '2014-01-06': 438, '2014-01-07': 438, '2014-01-08': 438, '2014-01-09': 438, '2014-01-10': 438, '2014-01-13': 438, '2014-01-14': 438, '2014-01-15': 438}, 'top_connected_assets': {}, 'average_correlation_by_date_available': 'rolling_corr flags are encoded in edge_source_flags'}

## Market State
- {'required_columns_present': ['mkt_median_realized_vol_20', 'mkt_cross_sectional_mad_return', 'mkt_average_abs_corr_60', 'mkt_market_mode_eigen_share_60', 'mkt_liquidity_stress', 'mkt_vix', 'mkt_credit_spread', 'mkt_yield_curve_10y_2y'], 'missing_columns': [], 'summary': {'mkt_median_realized_vol_20': {'min': 0.0, 'p25': 0.01207356985580631, 'median': 0.01426773940323349, 'p75': 0.017221828672965858, 'max': 0.07617294210445574}, 'mkt_cross_sectional_mad_return': {'min': 0.0, 'p25': 0.005793688943949771, 'median': 0.007014244324308638, 'p75': 0.008706402066928715, 'max': 0.049120314787510666}, 'mkt_average_abs_corr_60': {'min': 0.0, 'p25': 0.2404152781418673, 'median': 0.2997356607354845, 'p75': 0.3644955458056594, 'max': 1.0}, 'mkt_market_mode_eigen_share_60': {'min': 0.0, 'p25': 0.25492834572286815, 'median': 0.32090688192356254, 'p75': 0.3900651239639796, 'max': 1.0000000000000002}, 'mkt_liquidity_stress': {'min': 0.20535801051231273, 'p25': 0.3620756638464968, 'median': 0.4200573719285137, 'p75': 0.5182625601210239, 'max': 1.0}, 'mkt_vix': {'min': 9.14, 'p25': 13.1975, 'median': 15.945, 'p75': 20.7925, 'max': 82.69}, 'mkt_credit_spread': {'min': 1.36, 'p25': 1.9, 'median': 2.19, 'p75': 2.45, 'max': 4.31}, 'mkt_yield_curve_10y_2y': {'min': -1.08, 'p25': 0.14, 'median': 0.57, 'p75': 1.19, 'max': 2.61}}, 'crisis_volatility_elevated': True}
