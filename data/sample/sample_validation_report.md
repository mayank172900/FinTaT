# FinTTA Data Validation Report

Status: `pass`
Panel: `data/sample/sample_panel.parquet`

## Schema
- Rows: 936
- Assets: 8
- Duplicate `(date, asset_id)` rows: 0
- Missing minimum columns: []

## Leakage
- Result: pass
- Evaluation columns in adaptation feature list: []
- Timestamp violations: {}

## Survivorship
- Inactive rows: 1
- Delisting-return rows: 1

## Graph
- {'available': True, 'edge_count': 3192, 'edge_count_per_date_summary': {'min': 21.0, 'p25': 28.0, 'median': 28.0, 'p75': 28.0, 'max': 28.0}, 'positive_edges': 3188, 'negative_edges': 4, 'positive_negative_ratio': 797.0, 'isolated_asset_count_top10_dates': {'2019-10-01': 0, '2019-10-02': 0, '2019-10-03': 0, '2019-10-04': 0, '2019-10-07': 0, '2019-10-08': 0, '2019-10-09': 0, '2019-10-10': 0, '2019-10-11': 0, '2019-10-14': 0}, 'top_connected_assets': {'SAMP001': 816, 'SAMP002': 816, 'SAMP003': 816, 'SAMP004': 816, 'SAMP005': 816, 'SAMP006': 816, 'SAMP007': 816, 'SAMP008': 672}, 'average_correlation_by_date_available': 'rolling_corr flags are encoded in edge_source_flags'}

## Market State
- {'required_columns_present': ['mkt_median_realized_vol_20', 'mkt_cross_sectional_mad_return', 'mkt_average_abs_corr_60', 'mkt_market_mode_eigen_share_60', 'mkt_liquidity_stress', 'mkt_vix', 'mkt_credit_spread', 'mkt_yield_curve_10y_2y'], 'missing_columns': [], 'summary': {'mkt_median_realized_vol_20': {'min': 0.007534174627417567, 'p25': 0.011620960242021024, 'median': 0.012340346069442034, 'p75': 0.013566112651921446, 'max': 0.04062977694791543}, 'mkt_cross_sectional_mad_return': {'min': 0.00042801720757590367, 'p25': 0.0035620139445268484, 'median': 0.005201023777700736, 'p75': 0.006504273953366585, 'max': 0.026401966059514836}, 'mkt_average_abs_corr_60': {'min': 0.0, 'p25': 0.4970134231696661, 'median': 0.5780986842129003, 'p75': 0.6046644294329849, 'max': 1.0}, 'mkt_market_mode_eigen_share_60': {'min': 0.0, 'p25': 0.5650396489827372, 'median': 0.6376427945473049, 'p75': 0.6622639385798526, 'max': 1.0}, 'mkt_liquidity_stress': {'min': 0.05168287773544278, 'p25': 0.052552614042878865, 'median': 0.05317628913617417, 'p75': 0.054137897428636506, 'max': 0.05954273829321821}, 'mkt_vix': {'min': 13.00002938034789, 'p25': 13.99377812400309, 'median': 16.63235682493358, 'p75': 18.58010294529559, 'max': 62.829503164621144}, 'mkt_credit_spread': {'min': 0.7000037536597459, 'p25': 0.7289156961632057, 'median': 0.7660955892789034, 'p75': 0.7943491203503338, 'max': 1.8246959977909507}, 'mkt_yield_curve_10y_2y': {'min': 0.25560617069325037, 'p25': 0.40527978771701445, 'median': 0.4570872934463033, 'p75': 0.48776770554802135, 'max': 0.4999999900066684}}, 'crisis_volatility_elevated': True}
