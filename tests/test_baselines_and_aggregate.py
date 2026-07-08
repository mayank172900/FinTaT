from __future__ import annotations

import math
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

from fintta.data import make_synthetic_market, source_training_tensors
from fintta.experiment import train_source_model
from fintta.model import AdaptableMLP
from fintta.types import AssetBatch


def _train_source_model(seed: int = 7):
    market = make_synthetic_market(n_assets=12, source_days=6, test_days=8, lookback=4, input_dim=10, seed=seed)
    x_train, y_train = source_training_tensors(market.source_batches)
    model = AdaptableMLP(market.input_dim, market.num_classes, hidden_dim=16, depth=2)
    model = train_source_model(model, x_train, y_train, epochs=2, lr=1e-3)
    source_prior = torch.bincount(y_train, minlength=market.num_classes).float()
    source_prior = (source_prior / source_prior.sum()).clamp_min(1e-6)
    return market, model, source_prior


def _permute_labels(batch: AssetBatch, seed: int) -> AssetBatch:
    rng = np.random.default_rng(seed)
    perm = torch.as_tensor(rng.permutation(batch.n_assets), dtype=torch.long)
    return AssetBatch(
        x=batch.x.clone(),
        asset_ids=list(batch.asset_ids),
        metadata={key: list(values) for key, values in batch.metadata.items()},
        returns_window=None if batch.returns_window is None else np.array(batch.returns_window, copy=True),
        liquidity=None if batch.liquidity is None else batch.liquidity.clone(),
        factor_exposures=None if batch.factor_exposures is None else np.array(batch.factor_exposures, copy=True),
        market_state=None if batch.market_state is None else np.array(batch.market_state, copy=True),
        labels=None if batch.labels is None else batch.labels[perm].clone(),
        forward_returns=None if batch.forward_returns is None else np.array(batch.forward_returns, copy=True),
    )


def _run_stream(engine, batches: list[AssetBatch]) -> list[torch.Tensor]:
    outputs: list[torch.Tensor] = []
    pending_labels: torch.Tensor | None = None
    for batch in batches:
        if pending_labels is not None and hasattr(engine, "observe"):
            engine.observe(pending_labels)
        probs = engine.step(batch)
        outputs.append(probs.detach().cpu())
        pending_labels = batch.labels
    if pending_labels is not None and hasattr(engine, "observe"):
        engine.observe(pending_labels)
    return outputs


def _mean_nll(probabilities: list[torch.Tensor], batches: list[AssetBatch]) -> float:
    values = []
    for probs, batch in zip(probabilities, batches, strict=True):
        labels = batch.labels.detach().cpu().numpy()
        arr = probs.detach().cpu().numpy()
        values.append(float(-np.log(arr[np.arange(labels.size), labels] + 1e-12).mean()))
    return float(np.mean(values))


def test_open_panel_runner_exposes_new_baseline_variants():
    from run_open_panel_experiment import BASELINE_VARIANTS

    assert BASELINE_VARIANTS == [
        "tent_full",
        "eata_style",
        "lame",
        "adaptable_style",
        "online_temp",
        "aci",
    ]


@pytest.mark.parametrize(
    "engine_factory",
    [
        "tent_full",
        "eata_style",
        "lame",
        "adaptable_style",
        "online_temp",
        "aci",
    ],
)
def test_new_baselines_are_leakage_invariant_under_same_day_label_permutation(engine_factory: str):
    from fintta.baselines import (
        ACIWrapper,
        AdaptableStyleEngine,
        EATAStyleEngine,
        LAMEEngine,
        OnlineTempEngine,
        TentFullEngine,
    )

    market, model, source_prior = _train_source_model(seed=11)
    original_batches = market.test_batches[:5]
    target_index = 2
    permuted_batches = list(original_batches)
    permuted_batches[target_index] = _permute_labels(original_batches[target_index], seed=123)

    factories = {
        "tent_full": lambda: TentFullEngine(model.clone()),
        "eata_style": lambda: EATAStyleEngine(model.clone()),
        "lame": lambda: LAMEEngine(model.clone()),
        "adaptable_style": lambda: AdaptableStyleEngine(model.clone(), source_prior=source_prior),
        "online_temp": lambda: OnlineTempEngine(model.clone()),
        "aci": lambda: ACIWrapper(model.clone()),
    }

    original = _run_stream(factories[engine_factory](), original_batches)
    permuted = _run_stream(factories[engine_factory](), permuted_batches)

    assert torch.allclose(original[target_index], permuted[target_index], atol=1e-7), engine_factory


def test_tent_full_updates_only_layernorm_affine_parameters():
    from fintta.baselines import TentFullEngine

    market, model, _ = _train_source_model(seed=13)
    batch = market.test_batches[0]
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    engine = TentFullEngine(model.clone(), lr=1e-2)

    probs = engine.step(batch)

    assert probs.shape == (batch.n_assets, market.num_classes)
    assert torch.allclose(probs.sum(dim=1), torch.ones(batch.n_assets), atol=1e-6)
    assert torch.all(probs >= 0)

    layernorm_param_names = {
        f"{module_name}.{suffix}"
        for module_name, module in engine.model.named_modules()
        if isinstance(module, nn.LayerNorm)
        for suffix in ("weight", "bias")
    }
    after = {name: param.detach().clone() for name, param in engine.model.named_parameters()}
    changed = []
    for name, value in before.items():
        if not torch.allclose(value, after[name]):
            changed.append(name)
            assert name in layernorm_param_names, name
    assert changed, "expected at least one LayerNorm affine parameter to change"


def test_eata_style_diagnostics_expose_selected_count():
    from run_open_panel_experiment import _variant_diagnostics

    from fintta.baselines import EATAStyleEngine

    market = make_synthetic_market(n_assets=8, source_days=3, test_days=1, lookback=4, input_dim=10, seed=15)
    model = AdaptableMLP(market.input_dim, market.num_classes, hidden_dim=12, depth=1)
    engine = EATAStyleEngine(model, entropy_ratio=0.0)

    engine.step(market.test_batches[0])
    diagnostics = _variant_diagnostics("eata_style", engine, t=0)

    assert engine.last_selected_count == 0
    assert diagnostics["selected_count"] == 0.0


@pytest.mark.parametrize("engine_name", ["lame", "adaptable_style"])
def test_output_level_baselines_leave_parameters_untouched_and_return_simplex_probabilities(engine_name: str):
    from fintta.baselines import AdaptableStyleEngine, LAMEEngine

    market, model, source_prior = _train_source_model(seed=17)
    batch = market.test_batches[0]
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    engine = {
        "lame": LAMEEngine(model.clone()),
        "adaptable_style": AdaptableStyleEngine(model.clone(), source_prior=source_prior),
    }[engine_name]

    probs = engine.step(batch)

    after = {name: param.detach().clone() for name, param in engine.model.named_parameters()}
    for name in before:
        assert torch.allclose(before[name], after[name]), name
    assert probs.shape == (batch.n_assets, market.num_classes)
    assert torch.all(probs >= 0)
    assert torch.allclose(probs.sum(dim=1), torch.ones(batch.n_assets), atol=1e-6)


def test_online_temp_reduces_nll_on_miscalibrated_stream():
    from fintta.baselines import OnlineTempEngine

    market, model, _ = _train_source_model(seed=21)
    with torch.no_grad():
        model.log_temperature.fill_(-math.log(3.0))
    frozen_model = model.clone()
    online = OnlineTempEngine(model.clone(), lr=5e-3)

    frozen_probs = []
    online_probs = []
    pending_labels: torch.Tensor | None = None
    for batch in market.test_batches[:10]:
        if pending_labels is not None:
            online.observe(pending_labels)
        frozen_probs.append(torch.softmax(frozen_model(batch.x), dim=-1).detach().cpu())
        online_probs.append(online.step(batch).detach().cpu())
        pending_labels = batch.labels
    if pending_labels is not None:
        online.observe(pending_labels)

    assert _mean_nll(online_probs[3:], market.test_batches[:10][3:]) < _mean_nll(frozen_probs[3:], market.test_batches[:10][3:])


def test_aci_converges_near_target_coverage():
    from fintta.baselines import ACIWrapper

    market, model, _ = _train_source_model(seed=29)
    aci = ACIWrapper(model.clone(), gamma=0.005, target_coverage=0.9)
    pending_labels: torch.Tensor | None = None
    for batch in market.test_batches:
        if pending_labels is not None:
            aci.observe(pending_labels)
        aci.step(batch)
        pending_labels = batch.labels
    if pending_labels is not None:
        aci.observe(pending_labels)

    coverage = np.array([row["coverage"] for row in aci.coverage_history], dtype=np.float64)
    assert coverage.size == len(market.test_batches)
    assert coverage[-5:].mean() == pytest.approx(0.9, abs=0.12)


def test_aggregate_runs_produces_correct_means_and_paired_difference(tmp_path):
    from aggregate_runs import aggregate_runs

    run_a = tmp_path / "seed_001"
    run_b = tmp_path / "seed_002"
    for run_dir, seed_offset in [(run_a, 0), (run_b, 1)]:
        (run_dir / "daily").mkdir(parents=True)
        metrics = pd.DataFrame(
            [
                {"variant": "no_adaptation", "accuracy": 0.50 + 0.05 * seed_offset, "nll": 1.0 + 0.1 * seed_offset, "brier": 0.40 + 0.02 * seed_offset},
                {"variant": "tent_full", "accuracy": 0.60 + 0.05 * seed_offset, "nll": 0.80 + 0.05 * seed_offset, "brier": 0.30 + 0.01 * seed_offset},
            ]
        )
        metrics.to_csv(run_dir / "metrics.csv", index=False)
        pd.DataFrame(
            [
                {"date": "2020-01-01", "n_assets": 2, "nll": 1.0 + 0.1 * seed_offset, "brier": 0.4 + 0.02 * seed_offset, "top1_accuracy": 0.5, "mean_confidence": 0.6, "long_short_return": 0.01},
                {"date": "2020-01-02", "n_assets": 2, "nll": 0.9 + 0.1 * seed_offset, "brier": 0.3 + 0.02 * seed_offset, "top1_accuracy": 0.6, "mean_confidence": 0.7, "long_short_return": 0.02},
            ]
        ).to_csv(run_dir / "daily" / "no_adaptation.csv", index=False)
        pd.DataFrame(
            [
                {"date": "2020-01-01", "n_assets": 2, "nll": 0.7 + 0.05 * seed_offset, "brier": 0.25 + 0.01 * seed_offset, "top1_accuracy": 0.7, "mean_confidence": 0.8, "long_short_return": 0.03},
                {"date": "2020-01-02", "n_assets": 2, "nll": 0.6 + 0.05 * seed_offset, "brier": 0.20 + 0.01 * seed_offset, "top1_accuracy": 0.8, "mean_confidence": 0.9, "long_short_return": 0.04},
            ]
        ).to_csv(run_dir / "daily" / "tent_full.csv", index=False)

    out_dir = tmp_path / "aggregate"
    aggregate_runs([str(tmp_path / "seed_*")], out_dir)

    summary = pd.read_csv(out_dir / "aggregate_metrics.csv")
    paired = pd.read_csv(out_dir / "paired_tests.csv")

    tent = summary.loc[summary["variant"] == "tent_full"].iloc[0]
    assert tent["accuracy_mean"] == pytest.approx(0.625)
    assert tent["nll_mean"] == pytest.approx(0.825)
    assert tent["brier_mean"] == pytest.approx(0.305)

    nll = paired[(paired["variant"] == "tent_full") & (paired["metric"] == "nll")].iloc[0]
    brier = paired[(paired["variant"] == "tent_full") & (paired["metric"] == "brier")].iloc[0]
    assert nll["mean_diff"] < 0
    assert brier["mean_diff"] < 0
    assert nll["win_fraction"] == pytest.approx(1.0)
    assert brier["win_fraction"] == pytest.approx(1.0)
