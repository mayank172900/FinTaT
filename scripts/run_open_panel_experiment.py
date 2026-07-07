from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fintta.config import FinTTAConfig
from fintta.data import PanelDataset, PanelSpec, source_training_tensors
from fintta.engine import FinTTAEngine
from fintta.experiment import run_no_adaptation, train_source_model
from fintta.metrics import classification_metrics, trading_metrics
from fintta.model import AdaptableMLP


DEFAULT_VARIANTS = [
    "no_adaptation",
    "fintta_prequential",
    "fintta_same_batch",
    "conservative_bias_prequential",
    "conservative_bias_same_batch",
    "calibration_bias_prequential",
    "fintta_no_risk",
    "fintta_no_graph",
    "fintta_no_prior",
    "fintta_no_teacher",
    "tent_lite",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FinTTA experiments on the processed open-data panel.")
    parser.add_argument("--config", default="configs/experiment_2015_2024.yaml")
    parser.add_argument("--output-dir", default="outputs/open_panel")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS), help="Comma-separated variants or 'all'.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-assets-per-day", type=int, default=180, help="Top dollar-volume assets per day; 0 keeps all assets.")
    parser.add_argument("--test-start", default=None)
    parser.add_argument("--test-end", default=None)
    parser.add_argument("--quick", action="store_true", help="Fast smoke run: fewer epochs, assets, and dates.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_yaml(args.config)
    if args.quick:
        args.epochs = min(args.epochs, 2)
        args.max_assets_per_day = min(args.max_assets_per_day, 80) if args.max_assets_per_day else 80
        args.test_start = args.test_start or "2020-01-01"
        args.test_end = args.test_end or "2020-03-31"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnostics").mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    panel_path = Path(config["data"]["panel_path"])
    feature_cols = resolve_feature_columns(panel_path, config)
    factor_cols = config.get("graph", {}).get("factor_columns", [])
    market_cols = config.get("market_state", {}).get("columns", [])
    label_col = config["data"]["label_column"]
    fwd_col = config["data"]["forward_return_column"]
    liquidity_col = config["data"].get("liquidity_column", "liquidity_score")

    needed = sorted(
        set(
            ["date", "asset_id", "ret_1d", "volume", "dollar_volume", "sector", "industry", liquidity_col, label_col, fwd_col]
            + feature_cols
            + factor_cols
            + market_cols
        )
    )
    print(f"loading panel columns={len(needed)} from {panel_path}")
    panel = pq.read_table(panel_path, columns=needed, memory_map=False, use_threads=False).to_pandas()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel[feature_cols] = panel[feature_cols].fillna(0.0)
    panel[label_col] = panel[label_col].fillna(2).astype(int).clip(0, 4)
    panel[fwd_col] = panel[fwd_col].fillna(0.0)
    panel[liquidity_col] = panel[liquidity_col].fillna(1.0).clip(0.0, 1.0)

    source_start = config["data"]["source_start"]
    source_end = config["data"]["source_end"]
    test_start = args.test_start or config["data"]["test_start"]
    test_end = args.test_end or config["data"]["test_end"]
    source_df = slice_dates(panel, source_start, source_end, args.max_assets_per_day)
    test_df = slice_dates(panel, test_start, test_end, args.max_assets_per_day)
    feature_cols, source_df, test_df = standardize_features(feature_cols, source_df, test_df)

    spec = PanelSpec(
        feature_columns=feature_cols,
        label_column=label_col,
        forward_return_column=fwd_col,
        factor_columns=factor_cols,
        market_state_columns=market_cols,
        liquidity_column=liquidity_col,
        lookback=60,
    )
    source_batches = list(PanelDataset(source_df, spec).iter_batches())
    test_batches = list(PanelDataset(test_df, spec).iter_batches())
    print(f"source rows={len(source_df):,} days={len(source_batches):,}; test rows={len(test_df):,} days={len(test_batches):,}")

    x_train, y_train = source_training_tensors(source_batches)
    model = AdaptableMLP(
        input_dim=len(feature_cols),
        num_classes=5,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        adapter_rank=8,
    )
    model = train_source_model_batched(model, x_train, y_train, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    checkpoint_path = output_dir / "checkpoints" / "source_mlp_open.pt"
    torch.save({"model": model.state_dict(), "feature_columns": feature_cols, "args": vars(args)}, checkpoint_path)

    requested = DEFAULT_VARIANTS if args.variants == "all" else [v.strip() for v in args.variants.split(",") if v.strip()]
    rows: list[dict[str, float | str]] = []
    for variant in requested:
        print(f"running variant={variant}")
        if variant == "no_adaptation":
            metrics = run_no_adaptation(model.clone(), test_batches, 5)
            diagnostics = []
        else:
            cfg = variant_config(variant, args.seed)
            metrics, diagnostics = run_engine_variant(model.clone(), source_batches, test_batches, cfg, variant=variant)
        row = {"variant": variant, **metrics}
        rows.append(row)
        with (output_dir / "diagnostics" / f"{variant}.json").open("w", encoding="utf-8") as f:
            json.dump({"variant": variant, "metrics": metrics, "diagnostics": diagnostics[:2000]}, f, indent=2, sort_keys=True)
        pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
        print(json.dumps(row, indent=2, sort_keys=True))

    summary = {
        "config": config,
        "args": vars(args),
        "provenance": build_provenance(args, panel_path),
        "source_rows": len(source_df),
        "test_rows": len(test_df),
        "source_days": len(source_batches),
        "test_days": len(test_batches),
        "feature_count": len(feature_cols),
        "checkpoint": str(checkpoint_path),
        "variants": requested,
        "fintta_config_default": asdict(FinTTAConfig(seed=args.seed, num_classes=5)),
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"wrote {output_dir / 'metrics.csv'}")


def train_source_model_batched(
    model: AdaptableMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
) -> AdaptableMLP:
    if batch_size <= 0:
        return train_source_model(model, x, y, epochs=epochs, lr=lr)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for epoch in range(epochs):
        perm = torch.randperm(x.shape[0])
        losses = []
        for idx in perm.split(batch_size):
            opt.zero_grad(set_to_none=True)
            logits = model(x[idx])
            loss = torch.nn.functional.cross_entropy(logits, y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach()))
        print(f"source epoch {epoch + 1}/{epochs} loss={np.mean(losses):.4f}")
    model.freeze_source_weights()
    return model


def run_engine_variant(
    model: AdaptableMLP,
    source_batches,
    test_batches,
    config: FinTTAConfig,
    *,
    variant: str,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    engine = FinTTAEngine(model, config=config)
    engine.warm_start_market_states(source_batches)
    if variant.startswith("conservative_bias_") or variant.startswith("calibration_bias_"):
        restrict_engine_to_bias_temperature(engine)
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
        if (t + 1) % 100 == 0:
            print(f"  day {t + 1}/{len(test_batches)} regime={out.regime} shock={out.shock:.3f} adapted={out.adapted}")
    metrics = classification_metrics(probs, labels, 5)
    metrics.update({f"trade_{k}": v for k, v in trading_metrics(scores, fwd, labels, asset_ids=asset_ids).items()})
    metrics["regimes_used"] = float(len(set(d["regime"] for d in diagnostics)))
    metrics["adapt_rate"] = float(sum(d["adapted"] for d in diagnostics) / max(len(diagnostics), 1))
    metrics["mean_shock"] = float(sum(d["shock"] for d in diagnostics) / max(len(diagnostics), 1))
    return metrics, diagnostics


def restrict_engine_to_bias_temperature(engine: FinTTAEngine) -> None:
    """Restrict online gradients to final calibration degrees of freedom."""

    allowed = {"logit_bias", "log_temperature", "head.bias"}
    for name, param in engine.model.named_parameters():
        param.requires_grad = name in allowed
    engine.optimizer = engine._make_optimizer()


def variant_config(variant: str, seed: int) -> FinTTAConfig:
    same_batch = not variant.endswith("_prequential")
    cfg = FinTTAConfig(seed=seed, num_classes=5, same_batch_adaptation=same_batch)
    if variant == "fintta_prequential":
        cfg.same_batch_adaptation = False
    elif variant == "fintta_same_batch":
        cfg.same_batch_adaptation = True
    elif variant in {"conservative_bias_prequential", "conservative_bias_same_batch"}:
        cfg.same_batch_adaptation = variant.endswith("_same_batch")
        cfg.lr = 3e-5
        cfg.grad_clip = 0.5
        cfg.confidence_floor = 0.65
        cfg.tau_confidence = 0.04
        cfg.min_effective_assets = 80.0
        cfg.beta_pi = 0.10
        cfg.alpha_prior = 0.05
        cfg.alpha_teacher = 0.05
        cfg.alpha_graph = 0.0
        cfg.alpha_anchor = 0.01
        cfg.lambda_min = 0.80
        cfg.lambda_max = 1.25
        cfg.rho_lambda = 0.5
        cfg.teacher_ema = 0.995
        cfg.stochastic_restore_p = 0.0
        cfg.health_max = 2.0
    elif variant == "calibration_bias_prequential":
        cfg.same_batch_adaptation = False
        cfg.lr = 1e-5
        cfg.grad_clip = 0.25
        cfg.confidence_floor = 0.45
        cfg.tau_confidence = 0.08
        cfg.min_effective_assets = 20.0
        cfg.beta_pi = 0.0
        cfg.alpha_prior = 0.0
        cfg.alpha_graph = 0.0
        cfg.alpha_teacher = 0.10
        cfg.alpha_anchor = 0.05
        cfg.lambda_min = 0.90
        cfg.lambda_max = 1.10
        cfg.rho_lambda = 0.25
        cfg.teacher_ema = 0.995
        cfg.stochastic_restore_p = 0.0
        cfg.health_max = 8.0
    elif variant == "fintta_no_risk":
        cfg.rho_lambda = 0.0
        cfg.risk_temperature = 1e6
    elif variant == "fintta_no_graph":
        cfg.alpha_graph = 0.0
    elif variant == "fintta_no_prior":
        cfg.alpha_prior = 0.0
        cfg.beta_pi = 0.0
    elif variant == "fintta_no_teacher":
        cfg.alpha_teacher = 0.0
    elif variant == "tent_lite":
        cfg.same_batch_adaptation = True
        cfg.rho_lambda = 0.0
        cfg.risk_temperature = 1e6
        cfg.alpha_graph = 0.0
        cfg.alpha_prior = 0.0
        cfg.alpha_teacher = 0.0
        cfg.beta_pi = 0.0
        cfg.confidence_floor = 0.0
    else:
        raise ValueError(f"unknown variant: {variant}")
    return cfg


def resolve_feature_columns(panel_path: Path, config: dict) -> list[str]:
    schema = pq.ParquetFile(panel_path).schema_arrow
    cols = schema.names
    prefixes = config.get("features", {}).get("include_prefixes", ["feat_"])
    excluded = set(config.get("features", {}).get("exclude_columns", []))
    return [c for c in cols if c not in excluded and any(c.startswith(prefix) for prefix in prefixes)]


def slice_dates(frame: pd.DataFrame, start: str, end: str, max_assets_per_day: int) -> pd.DataFrame:
    out = frame[(frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))].copy()
    if max_assets_per_day and max_assets_per_day > 0:
        out["_rank_liq"] = out.groupby("date")["dollar_volume"].rank(method="first", ascending=False)
        out = out[out["_rank_liq"] <= max_assets_per_day].drop(columns=["_rank_liq"])
    return out.sort_values(["date", "asset_id"]).reset_index(drop=True)


def standardize_features(
    feature_cols: list[str],
    source: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    source = source.copy()
    test = test.copy()
    mean = source[feature_cols].mean()
    std = source[feature_cols].std().replace(0.0, 1.0).fillna(1.0)
    source[feature_cols] = ((source[feature_cols] - mean) / std).clip(-10.0, 10.0).fillna(0.0)
    test[feature_cols] = ((test[feature_cols] - mean) / std).clip(-10.0, 10.0).fillna(0.0)
    return feature_cols, source, test


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_provenance(args: argparse.Namespace, panel_path: Path) -> dict[str, Any]:
    return {
        "git": git_provenance(ROOT),
        "command_line": sys.argv[:],
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": package_versions(["fintta", "numpy", "pandas", "pyarrow", "torch", "scikit-learn"]),
        "input_panel": {
            "path": str(panel_path),
            "sha256": file_sha256(panel_path) if panel_path.exists() else None,
        },
        "seeds": {
            "python_random": args.seed,
            "numpy": args.seed,
            "torch": args.seed,
        },
    }


def git_provenance(root: Path) -> dict[str, Any]:
    commit = run_git(root, ["rev-parse", "HEAD"])
    status = run_git(root, ["status", "--porcelain"])
    return {
        "commit": commit,
        "dirty": bool(status),
    }


def run_git(root: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def package_versions(names: list[str]) -> dict[str, str | None]:
    out = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
