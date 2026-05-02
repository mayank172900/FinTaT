import json
from pathlib import Path

import pandas as pd


def test_sample_panel_is_schema_compatible_and_sample_only():
    root = Path(__file__).resolve().parents[1]
    panel = pd.read_parquet(root / "data" / "sample" / "sample_panel.parquet")
    metadata = json.loads((root / "data" / "sample" / "sample_metadata.json").read_text())

    minimum = {"date", "asset_id", "ret_1d", "volume", "sector", "industry", "liquidity_score"}
    label_cols = {c for c in panel.columns if c.startswith("label_")}
    forward_cols = {c for c in panel.columns if c.startswith("forward_return_")}
    feature_cols = {c for c in panel.columns if c.startswith("feat_")}

    assert metadata["dataset_grade"] == "sample-only"
    assert minimum.issubset(panel.columns)
    assert len(panel) > 0
    assert panel.duplicated(["date", "asset_id"]).sum() == 0
    assert len(feature_cols) >= 50
    assert label_cols
    assert forward_cols
    assert set(metadata["adaptation_feature_columns"]) == feature_cols
    assert not (feature_cols & label_cols)
    assert not (feature_cols & forward_cols)


def test_sample_validation_report_passes_leakage_checks():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "data" / "sample" / "sample_validation_report.json").read_text())

    assert report["status"] == "pass"
    assert report["leakage"]["result"] == "pass"
    assert report["leakage"]["evaluation_columns_in_adaptation_features"] == []
