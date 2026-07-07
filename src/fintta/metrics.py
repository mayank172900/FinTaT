from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss


def classification_metrics(probabilities: list[torch.Tensor], labels: list[torch.Tensor], num_classes: int) -> dict[str, float]:
    p = torch.cat(probabilities, dim=0).numpy()
    y = torch.cat(labels, dim=0).numpy()
    pred = p.argmax(axis=1)
    onehot = np.eye(num_classes)[y]
    brier = float(np.mean(np.sum((p - onehot) ** 2, axis=1)))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "nll": float(log_loss(y, p, labels=list(range(num_classes)))),
        "brier": brier,
        "ece": expected_calibration_error(p, y),
    }


def daily_long_short_return(
    scores: torch.Tensor | np.ndarray,
    forward_returns: np.ndarray | torch.Tensor,
    previous_weights: dict[str, float] | np.ndarray | None = None,
    asset_ids: list[str] | None = None,
    q: float = 0.2,
    cost_bps: float = 10.0,
) -> tuple[float, dict[str, float] | np.ndarray]:
    s = _as_numpy(scores)
    r = _as_numpy(forward_returns)
    n = len(s)
    if n == 0:
        return 0.0, {} if asset_ids is not None else np.zeros(0, dtype=np.float64)

    k = min(max(1, int(q * n)), n // 2)
    w = np.zeros(n, dtype=np.float64)
    if k > 0:
        long_idx = np.argpartition(s, n - k)[-k:]
        short_idx = np.argpartition(s, k - 1)[:k]
        w[long_idx] = 1.0 / (2 * k)
        w[short_idx] = -1.0 / (2 * k)

    weight_by_asset: dict[str, float] | np.ndarray
    if asset_ids is None:
        weight_by_asset = w.copy()
    else:
        weight_by_asset = {str(asset_id): float(weight) for asset_id, weight in zip(asset_ids, w, strict=True)}

    if previous_weights is None:
        turnover = float(np.abs(w).sum())
    elif isinstance(previous_weights, dict):
        universe = set(previous_weights).union(weight_by_asset)
        turnover = float(sum(abs(weight_by_asset.get(asset_id, 0.0) - previous_weights.get(asset_id, 0.0)) for asset_id in universe))
    else:
        turnover = float(np.abs(w - previous_weights).sum())

    daily_return = float(w @ r - turnover * cost_bps / 10000.0)
    return daily_return, weight_by_asset


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(float)
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins, endpoint=False), np.linspace(0.1, 1.0, bins)):
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)


def trading_metrics(
    scores: list[torch.Tensor],
    forward_returns: list[np.ndarray],
    labels: list[torch.Tensor],
    asset_ids: list[list[str]] | None = None,
    q: float = 0.2,
    cost_bps: float = 10.0,
) -> dict[str, float]:
    daily = []
    fp_buy_losses = []
    prev_w: dict[str, float] | np.ndarray | None = None
    if asset_ids is None:
        asset_ids = [[str(i) for i in range(len(score_t))] for score_t in scores]
    for score_t, ret_t, label_t, ids_t in zip(scores, forward_returns, labels, asset_ids):
        s = _as_numpy(score_t)
        r = _as_numpy(ret_t)
        daily_return, prev_w = daily_long_short_return(score_t, ret_t, prev_w, ids_t, q=q, cost_bps=cost_bps)
        daily.append(daily_return)
        y = label_t.numpy()
        buy = s > np.quantile(s, 0.8)
        mask = buy & np.isin(y, [0, 1])
        if mask.any():
            fp_buy_losses.append(float(np.mean(-r[mask])))
    arr = np.asarray(daily, dtype=np.float64)
    wealth = np.cumprod(1.0 + arr)
    peaks = np.maximum.accumulate(wealth)
    mdd = float(np.max(1.0 - wealth / np.maximum(peaks, 1e-12))) if wealth.size else 0.0
    ann = float(wealth[-1] ** (252.0 / max(len(arr), 1)) - 1.0) if wealth.size else 0.0
    sharpe = float(np.sqrt(252.0) * arr.mean() / (arr.std() + 1e-12)) if arr.size else 0.0
    return {
        "ann_return": ann,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "calmar": ann / (mdd + 1e-12),
        "mean_daily_return": float(arr.mean()) if arr.size else 0.0,
        "tail_loss_5pct": float(np.quantile(arr, 0.05)) if arr.size else 0.0,
        "fp_buy_loss": float(np.mean(fp_buy_losses)) if fp_buy_losses else 0.0,
    }


def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)
