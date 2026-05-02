from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import FinTTAConfig
from .utils import weighted_mean


def risk_weighted_entropy(p: torch.Tensor, omega: torch.Tensor, lam: torch.Tensor, config: FinTTAConfig) -> torch.Tensor:
    per = -(lam[None, :] * p * (p + config.epsilon).log()).sum(dim=1)
    return (omega * per).sum() / omega.sum().clamp_min(config.epsilon)


def prior_volume_loss(p: torch.Tensor, omega: torch.Tensor, prior_risk: torch.Tensor, config: FinTTAConfig) -> tuple[torch.Tensor, torch.Tensor]:
    pbar = weighted_mean(p, omega, eps=config.epsilon)
    ce = -(prior_risk * (pbar + config.epsilon).log()).sum()
    floor = (torch.clamp(0.4 * prior_risk - pbar, min=0.0).square()).sum()
    return ce + floor, pbar


def teacher_kl_loss(student_p: torch.Tensor, teacher_p: torch.Tensor, omega: torch.Tensor, config: FinTTAConfig) -> torch.Tensor:
    kl = F.kl_div((student_p + config.epsilon).log(), teacher_p.detach(), reduction="none").sum(dim=1)
    return (omega * kl).sum() / omega.sum().clamp_min(config.epsilon)


def anchor_loss(
    model: torch.nn.Module,
    config: FinTTAConfig,
    source_state: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    terms = []
    for name, param in model.named_parameters():
        if hasattr(model, "is_adaptation_parameter") and model.is_adaptation_parameter(name):
            anchor = 0.0 if source_state is None or name not in source_state else source_state[name].to(param.device)
            terms.append((param - anchor).square().sum())
    if not terms:
        return torch.zeros((), device=next(model.parameters()).device)
    return config.alpha_adapter_l2 * sum(terms)
