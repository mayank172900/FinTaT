from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch
import torch.nn.functional as F

from .config import FinTTAConfig
from .data import make_synthetic_market, source_training_tensors
from .engine import FinTTAEngine
from .metrics import classification_metrics, trading_metrics
from .model import AdaptableMLP


def train_source_model(model: AdaptableMLP, x: torch.Tensor, y: torch.Tensor, epochs: int = 25, lr: float = 1e-3) -> AdaptableMLP:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(x.shape[0])
        for idx in perm.split(512):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[idx]), y[idx])
            loss.backward()
            opt.step()
    model.freeze_source_weights()
    return model


@torch.no_grad()
def run_no_adaptation(model: AdaptableMLP, batches, num_classes: int) -> dict[str, float]:
    probs = []
    labels = []
    scores = []
    fwd = []
    asset_ids = []
    exposure = torch.tensor(FinTTAConfig(num_classes=num_classes).ordinal_exposure)
    model.eval()
    for b in batches:
        p = torch.softmax(model(b.x), dim=-1).cpu()
        probs.append(p)
        labels.append(b.labels.cpu())
        scores.append(p @ exposure)
        fwd.append(b.forward_returns)
        asset_ids.append(b.asset_ids)
    out = classification_metrics(probs, labels, num_classes)
    out.update({f"trade_{k}": v for k, v in trading_metrics(scores, fwd, labels, asset_ids=asset_ids).items()})
    return out


def run_fintta(model: AdaptableMLP, batches, config: FinTTAConfig) -> tuple[dict[str, float], list[dict[str, float]]]:
    engine = FinTTAEngine(model, config=config)
    probs = []
    labels = []
    scores = []
    fwd = []
    asset_ids = []
    diagnostics = []
    for t, batch in enumerate(batches):
        out = engine.step(batch, adapt=True)
        probs.append(out.probabilities)
        labels.append(batch.labels.cpu())
        scores.append(out.scores)
        fwd.append(batch.forward_returns)
        asset_ids.append(batch.asset_ids)
        diagnostics.append(
            {
                "t": t,
                "regime": out.regime,
                "shock": out.shock,
                "adapted": float(out.adapted),
                "effective_assets": out.effective_assets,
                **out.diagnostics,
            }
        )
    metrics = classification_metrics(probs, labels, config.num_classes)
    metrics.update({f"trade_{k}": v for k, v in trading_metrics(scores, fwd, labels, asset_ids=asset_ids).items()})
    metrics["regimes_used"] = float(len(set(d["regime"] for d in diagnostics)))
    metrics["adapt_rate"] = float(sum(d["adapted"] for d in diagnostics) / max(len(diagnostics), 1))
    metrics["mean_shock"] = float(sum(d["shock"] for d in diagnostics) / max(len(diagnostics), 1))
    return metrics, diagnostics


def run_fintta_with_source_state(
    model: AdaptableMLP,
    source_batches,
    test_batches,
    config: FinTTAConfig,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    engine = FinTTAEngine(model, config=config)
    engine.warm_start_market_states(source_batches)
    probs = []
    labels = []
    scores = []
    fwd = []
    asset_ids = []
    diagnostics = []
    for t, batch in enumerate(test_batches):
        out = engine.step(batch, adapt=True)
        probs.append(out.probabilities)
        labels.append(batch.labels.cpu())
        scores.append(out.scores)
        fwd.append(batch.forward_returns)
        asset_ids.append(batch.asset_ids)
        diagnostics.append(
            {
                "t": t,
                "regime": out.regime,
                "shock": out.shock,
                "adapted": float(out.adapted),
                "effective_assets": out.effective_assets,
                **out.diagnostics,
            }
        )
    metrics = classification_metrics(probs, labels, config.num_classes)
    metrics.update({f"trade_{k}": v for k, v in trading_metrics(scores, fwd, labels, asset_ids=asset_ids).items()})
    metrics["regimes_used"] = float(len(set(d["regime"] for d in diagnostics)))
    metrics["adapt_rate"] = float(sum(d["adapted"] for d in diagnostics) / max(len(diagnostics), 1))
    metrics["mean_shock"] = float(sum(d["shock"] for d in diagnostics) / max(len(diagnostics), 1))
    return metrics, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FinTTA chronological synthetic smoke experiment.")
    parser.add_argument("--assets", type=int, default=64)
    parser.add_argument("--source-days", type=int, default=120)
    parser.add_argument("--days", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--prequential", action="store_true")
    parser.add_argument("--cold-start-regime", action="store_true", help="Do not warm-start regime state from source-period market summaries.")
    parser.add_argument(
        "--ablation",
        choices=["none", "no-risk", "no-graph", "no-prior", "no-teacher"],
        default="none",
        help="Single FinTTA ablation for conference-style protocol checks.",
    )
    args = parser.parse_args()

    market = make_synthetic_market(n_assets=args.assets, source_days=args.source_days, test_days=args.days, seed=args.seed)
    x_train, y_train = source_training_tensors(market.source_batches)
    source = AdaptableMLP(market.input_dim, market.num_classes, hidden_dim=64, depth=2)
    source = train_source_model(source, x_train, y_train, epochs=args.epochs)

    no_adapt_model = source.clone()
    no_adapt = run_no_adaptation(no_adapt_model, market.test_batches, market.num_classes)

    config = FinTTAConfig(seed=args.seed, num_classes=market.num_classes, same_batch_adaptation=not args.prequential)
    if args.ablation == "no-risk":
        config.rho_lambda = 0.0
        config.risk_temperature = 1e6
    elif args.ablation == "no-graph":
        config.alpha_graph = 0.0
    elif args.ablation == "no-prior":
        config.alpha_prior = 0.0
        config.beta_pi = 0.0
    elif args.ablation == "no-teacher":
        config.alpha_teacher = 0.0
    fintta_model = source.clone()
    if args.cold_start_regime:
        fintta, diagnostics = run_fintta(fintta_model, market.test_batches, config)
    else:
        fintta, diagnostics = run_fintta_with_source_state(fintta_model, market.source_batches, market.test_batches, config)
    payload = {
        "config": asdict(config),
        "no_adaptation": no_adapt,
        "fintta": fintta,
        "diagnostics_head": diagnostics[:5],
        "diagnostics_tail": diagnostics[-5:],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
