import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from data_utils import add_causal_features
from validate_panel import validate_panel

from fintta.config import FinTTAConfig
from fintta.data import PanelDataset, PanelSpec, _future_bucket_labels, make_synthetic_market
from fintta.engine import FinTTAEngine
from fintta.graph import build_signed_graph
from fintta.losses import risk_weighted_entropy
from fintta.metrics import expected_calibration_error, trading_metrics
from fintta.model import AdaptableMLP, FTTransformerLite, RegimeAdapterBank
from fintta.types import AssetBatch


def _minimal_batch(n_assets: int = 8, input_dim: int = 2) -> AssetBatch:
    return AssetBatch(
        x=torch.zeros(n_assets, input_dim),
        asset_ids=[f"A{i}" for i in range(n_assets)],
        metadata={
            "sector": [f"S{i}" for i in range(n_assets)],
            "industry": [f"I{i}" for i in range(n_assets)],
        },
        returns_window=np.zeros((n_assets, 5), dtype=np.float64),
        liquidity=torch.ones(n_assets),
        factor_exposures=np.zeros((n_assets, 2), dtype=np.float64),
    )


def test_adapter_blend_renormalizes_over_existing_regimes_and_falls_back_to_source():
    model = AdaptableMLP(input_dim=3, num_classes=5, hidden_dim=4, depth=1)
    bank = RegimeAdapterBank(model, max_regimes=4)
    bank.adapters[0] = {name: torch.full_like(value, 2.0) for name, value in bank.source_state.items()}

    bank.activate(1, blend_weights={0: 0.1, 1: 0.9})
    for name, value in bank.adapters[1].items():
        assert torch.allclose(value, bank.adapters[0][name]), name

    bank.activate(2, blend_weights={99: 1.0})
    for name, value in bank.adapters[2].items():
        assert torch.allclose(value, bank.source_state[name]), name


def test_all_layernorm_affine_parameters_are_adaptable_at_depth_two_and_three():
    for depth in (2, 3):
        model = AdaptableMLP(input_dim=4, num_classes=5, hidden_dim=6, depth=depth)
        layernorm_names = []
        for module_name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                layernorm_names.extend([f"{module_name}.weight", f"{module_name}.bias"])

        assert len(layernorm_names) == 2 * depth
        assert all(model.is_adaptation_parameter(name) for name in layernorm_names)


def test_prequential_predictions_use_state_from_before_current_batch_update():
    model = AdaptableMLP(input_dim=2, num_classes=5, hidden_dim=4, depth=1)
    with torch.no_grad():
        model.logit_bias.copy_(torch.tensor([4.0, 0.0, 0.0, 0.0, 0.0]))

    cfg = FinTTAConfig(
        num_classes=5,
        same_batch_adaptation=False,
        beta_pi=1.0,
        confidence_floor=0.0,
        min_effective_assets=1.0,
        risk_temperature=1e6,
    )
    engine = FinTTAEngine(model, cfg)
    batch = _minimal_batch()
    batch_dev = batch.to(engine.device)

    engine.bank.load_teacher()
    with torch.no_grad():
        teacher_logits, teacher_features = engine.model(batch_dev.x, return_features=True)
        teacher_p = torch.softmax(teacher_logits, dim=-1)
    engine.bank.load_student()
    graph = build_signed_graph(batch_dev, cfg, engine.device)
    typicality = engine._typicality(teacher_features.detach())
    from fintta.graph import graph_reliability

    omega, _ = graph_reliability(teacher_p, graph, batch_dev.liquidity, cfg, typicality=typicality)
    prior = torch.full((cfg.num_classes,), 1.0 / cfg.num_classes, device=engine.device)
    costs = engine.risk.class_costs(np.array([1.0]), [np.full(cfg.num_classes, 1.0 / cfg.num_classes)], regime_ids=[0])
    pi_risk = engine.risk.risk_adjusted_prior(prior, costs, engine.device)
    expected = engine._predict_with_prior(batch_dev, omega, pi_risk).cpu()

    out = engine.step(batch, adapt=False)

    assert torch.allclose(out.probabilities, expected, atol=1e-6)
    assert engine.regimes.regime_ids()


def test_trading_metrics_aligns_changing_universes_and_handles_single_asset_days():
    result = trading_metrics(
        scores=[torch.tensor([1.0, -1.0]), torch.tensor([0.5])],
        forward_returns=[np.array([0.01, -0.02]), np.array([0.03])],
        labels=[torch.tensor([4, 0]), torch.tensor([4])],
        asset_ids=[["A", "B"], ["B"]],
        q=0.5,
        cost_bps=10.0,
    )

    assert result["mean_daily_return"] == pytest.approx(0.0065)


def test_expected_calibration_error_matches_two_bin_hand_calculation():
    p = np.array([[0.8, 0.2], [0.6, 0.4]], dtype=np.float64)
    y = np.array([0, 1], dtype=np.int64)

    assert expected_calibration_error(p, y, bins=2) == pytest.approx(0.2)


def test_validate_panel_fails_when_vintage_or_filing_dates_are_after_panel_date(tmp_path):
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"]),
            "asset_id": ["A"],
            "ret_1d": [0.0],
            "volume": [100.0],
            "sector": ["tech"],
            "industry": ["software"],
            "liquidity_score": [1.0],
            "feat_x": [0.0],
            "vintage_date": pd.to_datetime(["2020-01-03"]),
            "filing_date": pd.to_datetime(["2020-01-04"]),
        }
    )
    path = tmp_path / "panel.parquet"
    panel.to_parquet(path, index=False)

    report = validate_panel(path)

    assert report["status"] == "fail"
    assert report["leakage"]["result"] == "fail"
    assert report["leakage"]["macro_vintage_after_date_rows"] == 1
    assert report["leakage"]["filing_date_after_date_rows"] == 1


def test_make_synthetic_market_rejects_feature_dimensions_below_required_core_features():
    with pytest.raises(ValueError, match="input_dim"):
        make_synthetic_market(input_dim=9)


def test_synthetic_future_bucket_cuts_can_be_limited_to_source_period():
    returns = np.array([[0.0, 0.01, 0.02, 10.0, 11.0], [0.0, 0.02, 0.03, 12.0, 13.0]])

    labels = _future_bucket_labels(returns, horizon=1, num_classes=2, calibration_end=2)

    assert labels[:, 2].tolist() == [1, 1]


def test_validate_panel_finds_sibling_sample_graph_by_default():
    report = validate_panel(ROOT / "data" / "sample" / "sample_panel.parquet", ROOT / "configs" / "data_sample.yaml")

    assert report["graph"]["available"] is True


def test_build_signed_graph_skips_positive_edges_when_all_positive_scores_are_zero():
    cfg = FinTTAConfig(topk_edges=2)
    graph = build_signed_graph(_minimal_batch(n_assets=3), cfg, "cpu")

    assert graph.n_edges == 0


def test_panel_dataset_normalizes_asset_id_dtype_before_return_window_lookup():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
            "asset_id": [1, 2, 1, 2],
            "ret_1d": [0.01, 0.02, 0.03, 0.04],
            "volume": [100, 100, 100, 100],
            "sector": ["a", "b", "a", "b"],
            "industry": ["a1", "b1", "a1", "b1"],
            "feat_x": [1.0, 2.0, 3.0, 4.0],
        }
    )
    dataset = PanelDataset(frame, PanelSpec(feature_columns=["feat_x"], lookback=2))

    batches = list(dataset.iter_batches())

    assert batches[-1].asset_ids == ["1", "2"]
    assert batches[-1].returns_window.shape == (2, 2)


def test_engine_state_file_loads_with_torch_weights_only(tmp_path):
    market = make_synthetic_market(n_assets=8, source_days=3, test_days=2, lookback=4, input_dim=10, seed=5)
    model = AdaptableMLP(market.input_dim, market.num_classes, hidden_dim=8, depth=1)
    cfg = FinTTAConfig(num_classes=market.num_classes, min_effective_assets=1.0)
    engine = FinTTAEngine(model, cfg)
    engine.step(market.test_batches[0], adapt=True)
    path = tmp_path / "engine.pt"

    engine.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert isinstance(payload["config"], dict)
    assert isinstance(payload["regimes"], dict)
    restored = FinTTAEngine(model.clone(), cfg)
    restored.load(path)
    out = restored.step(market.test_batches[1], adapt=False)
    assert out.probabilities.shape == (8, 5)


def test_risk_weighted_entropy_is_lower_when_dominant_costly_class_is_downweighted():
    cfg = FinTTAConfig(num_classes=2)
    probs = torch.tensor([[0.9, 0.1]])
    omega = torch.ones(1)

    low_costly_weight = risk_weighted_entropy(probs, omega, torch.tensor([0.2, 1.0]), cfg)
    high_costly_weight = risk_weighted_entropy(probs, omega, torch.tensor([2.0, 1.0]), cfg)

    assert low_costly_weight < high_costly_weight


def test_engine_health_gate_restores_adapter_state_when_update_is_rejected():
    market = make_synthetic_market(n_assets=8, source_days=3, test_days=2, lookback=4, input_dim=10, seed=13)
    model = AdaptableMLP(market.input_dim, market.num_classes, hidden_dim=8, depth=1)
    cfg = FinTTAConfig(num_classes=market.num_classes, min_effective_assets=1.0, health_max=-1_000_000.0)
    engine = FinTTAEngine(model, cfg)
    before = engine.bank.snapshot()

    out = engine.step(market.test_batches[0], adapt=True)

    assert out.adapted is False
    after = engine.bank.snapshot()
    for name, value in before.items():
        assert torch.allclose(after[name], value), name


def test_reversal_feature_is_not_a_duplicate_negative_return_lag():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=8),
            "asset_id": ["A"] * 8,
            "ret_1d": [0.01, -0.03, 0.02, 0.04, -0.02, 0.03, -0.01, 0.02],
            "adjusted_close": [100, 97, 99, 103, 101, 104, 103, 105],
            "volume": [1000, 1100, 1050, 1200, 1150, 1300, 1250, 1400],
            "dollar_volume": [100_000, 106_700, 103_950, 123_600, 116_150, 135_200, 128_750, 147_000],
            "turnover": [0.1] * 8,
            "amihud_illiquidity": [0.01] * 8,
            "spread_proxy": [0.001] * 8,
            "liquidity_score": [0.9] * 8,
            "sector": ["tech"] * 8,
            "mom": [0.0] * 8,
            "hml": [0.0] * 8,
            "rates_factor": [0.0] * 8,
            "oil_factor": [0.0] * 8,
        }
    )

    featured = add_causal_features(frame)
    valid = featured[["feat_reversal_1", "feat_ret_lag_1"]].dropna()

    assert not np.allclose(valid["feat_reversal_1"], -valid["feat_ret_lag_1"])


def test_open_panel_provenance_builder_records_git_command_versions_and_panel_hash(tmp_path):
    from run_open_panel_experiment import build_provenance

    panel = tmp_path / "panel.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(panel, index=False)
    args = type("Args", (), {"seed": 123, "model_arch": "mlp"})()

    provenance = build_provenance(args, panel)

    assert provenance["command_line"]
    assert provenance["python"]["version"]
    assert "numpy" in provenance["packages"]
    assert provenance["input_panel"]["sha256"]
    assert provenance["seeds"]["numpy"] == 123
    assert provenance["model_arch"] == "mlp"


def test_open_panel_runner_selects_source_model_architecture():
    from run_open_panel_experiment import build_source_model, source_checkpoint_payload

    mlp_args = argparse.Namespace(model_arch="mlp", hidden_dim=12, depth=1)
    ft_args = argparse.Namespace(model_arch="ft_lite", hidden_dim=12, depth=1)

    mlp = build_source_model(mlp_args, input_dim=10, num_classes=5)
    ft_lite = build_source_model(ft_args, input_dim=10, num_classes=5)
    payload = source_checkpoint_payload(ft_lite, ["feat_a"], ft_args)

    assert isinstance(mlp, AdaptableMLP)
    assert isinstance(ft_lite, FTTransformerLite)
    assert payload["model_arch"] == "ft_lite"
    assert payload["args"]["model_arch"] == "ft_lite"


def test_tent_lite_variant_uses_same_batch_adaptation():
    from run_open_panel_experiment import variant_config

    assert variant_config("tent_lite", seed=7).same_batch_adaptation is True


def test_sp500_history_download_tries_configured_fallback_urls(tmp_path):
    from download_open_data import fetch_first_csv

    class Response:
        def __init__(self, status_code: int, content: bytes) -> None:
            self.status_code = status_code
            self.content = content

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"status {self.status_code}")

    class Session:
        def __init__(self) -> None:
            self.urls = []

        def get(self, url: str, timeout: int) -> Response:
            self.urls.append(url)
            if url == "bad":
                return Response(404, b"")
            return Response(200, b"date,tickers\n2020-01-01,AAPL\n")

    session = Session()

    df = fetch_first_csv(session, ["bad", "good"], tmp_path / "history.csv", source_name="sp500")

    assert session.urls == ["bad", "good"]
    assert df["tickers"].tolist() == ["AAPL"]
