import numpy as np
import torch

from fintta.config import FinTTAConfig
from fintta.data import make_synthetic_market, source_training_tensors
from fintta.engine import FinTTAEngine
from fintta.experiment import train_source_model
from fintta.graph import build_signed_graph, graph_reliability
from fintta.model import AdaptableMLP
from fintta.regime import RegimeTracker
from fintta.risk import RiskEstimate, RiskModel


def test_regime_tracker_spawns_and_resets_prior():
    cfg = FinTTAConfig(num_classes=5, tau_cp=0.4, min_effective_assets=1)
    tracker = RegimeTracker(cfg)
    tracker.update(np.array([0.0, 0.0, 0.1, 0.1]))
    info = tracker.update(np.array([7.0, 4.0, 0.9, 0.8]))
    p = torch.softmax(torch.randn(8, 5), dim=-1)
    omega = torch.ones(8)
    prior = tracker.update_label_prior(p, omega, changepoint=True)
    assert info.changepoint
    assert np.isclose(prior.sum(), 1.0)
    assert prior.min() > 0


def test_signed_graph_and_reliability_are_bounded():
    market = make_synthetic_market(n_assets=16, source_days=4, test_days=2, lookback=5, seed=3)
    batch = market.test_batches[0]
    cfg = FinTTAConfig(topk_edges=3)
    graph = build_signed_graph(batch, cfg, "cpu")
    probs = torch.softmax(torch.randn(batch.n_assets, cfg.num_classes), dim=-1)
    omega, disagreement = graph_reliability(probs, graph, batch.liquidity, cfg)
    assert graph.n_edges > 0
    assert torch.all(omega >= 0) and torch.all(omega <= 1)
    assert torch.isfinite(disagreement).all()


def test_risk_weights_downweight_costly_actions():
    cfg = FinTTAConfig(num_classes=5, long_only=True)
    risk = RiskModel(cfg)
    risk.by_regime[0] = RiskEstimate()
    risk.by_regime[0].sigma = 3.0
    risk.by_regime[0].cvar_down = 4.0
    costs = risk.class_costs(np.array([1.0]), [np.array([0.55, 0.25, 0.1, 0.06, 0.04])])
    lam = risk.entropy_weights(costs, torch.device("cpu"))
    assert lam[-1] < lam[2]


def test_online_engine_smoke_prequential(tmp_path):
    market = make_synthetic_market(n_assets=24, source_days=12, test_days=8, lookback=8, seed=11)
    x, y = source_training_tensors(market.source_batches)
    model = train_source_model(AdaptableMLP(market.input_dim, market.num_classes, hidden_dim=32), x, y, epochs=2)
    cfg = FinTTAConfig(num_classes=market.num_classes, min_effective_assets=2, same_batch_adaptation=False)
    engine = FinTTAEngine(model, cfg)
    outs = [engine.step(b, adapt=True) for b in market.test_batches]
    assert len(outs) == 8
    assert all(o.probabilities.shape == (24, 5) for o in outs)
    assert all(torch.isfinite(o.probabilities).all() for o in outs)
    assert len({o.regime for o in outs}) >= 1
    path = tmp_path / "fintta_state.pt"
    engine.save(path)
    restored = FinTTAEngine(model.clone(), cfg)
    restored.load(path)
    out = restored.step(market.test_batches[-1], adapt=False)
    assert out.probabilities.shape == (24, 5)
