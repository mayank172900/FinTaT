from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def entropy(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return -(p * (p + eps).log()).sum(dim=-1)


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.sum(dim=dim, keepdim=True).clamp_min(eps)


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    return 0.5 * F.kl_div(m.log(), p, reduction="none").sum(-1) + 0.5 * F.kl_div(
        m.log(), q, reduction="none"
    ).sum(-1)


def effective_sample_size(weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return weights.sum().square() / (weights.square().sum() + eps)


def ordinal_reversal(num_classes: int, device: torch.device | None = None) -> torch.Tensor:
    return torch.eye(num_classes, device=device).flip(0)


def weighted_mean(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x * w[:, None]).sum(dim=0) / w.sum().clamp_min(eps)


def topk_softmax(scores: torch.Tensor, k: int) -> torch.Tensor:
    n = scores.numel()
    if n == 0:
        return scores
    k = min(max(k, 1), n)
    vals, idx = torch.topk(scores, k)
    out = torch.zeros_like(scores)
    out[idx] = torch.softmax(vals, dim=0)
    return out


def normalized_entropy(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return entropy(p, eps=eps) / math.log(p.shape[-1])
