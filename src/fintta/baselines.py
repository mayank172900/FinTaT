from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import FinTTAConfig
from .graph import build_signed_graph
from .model import AdaptableMLP
from .types import AssetBatch
from .utils import entropy, normalized_entropy


def _layernorm_affine_names(model: nn.Module) -> set[str]:
    names: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            names.add(f"{module_name}.weight")
            names.add(f"{module_name}.bias")
    return names


def _set_requires_grad(model: nn.Module, allowed: set[str]) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name in allowed


def _freeze_model(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def _raw_logits(model: AdaptableMLP, x: torch.Tensor) -> torch.Tensor:
    x = (x + model.input_shift) * model.input_scale_log.exp().clamp(0.2, 5.0)
    h = model.backbone(x)
    return model.head(h)


def _calibrated_logits(model: AdaptableMLP, raw_logits: torch.Tensor) -> torch.Tensor:
    temperature = model.log_temperature.exp().clamp(0.25, 4.0)
    return (raw_logits + model.logit_bias) / temperature


def _probabilities(model: AdaptableMLP, raw_logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(_calibrated_logits(model, raw_logits), dim=-1)


def _anchor_loss(model: nn.Module, source_state: dict[str, torch.Tensor]) -> torch.Tensor:
    terms = []
    for name, param in model.named_parameters():
        if name in source_state and param.requires_grad:
            terms.append((param - source_state[name].to(param.device)).square().sum())
    if not terms:
        return torch.zeros((), device=next(model.parameters()).device)
    return sum(terms)


def _simplex_renormalize(probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)


class _BaselineEngine:
    def __init__(self, model: AdaptableMLP, *, device: str | torch.device = "cpu", epsilon: float = 1e-6) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.epsilon = epsilon
        self.last_logits: torch.Tensor | None = None
        self.last_probs: torch.Tensor | None = None

    def observe(self, labels: torch.Tensor) -> dict[str, float] | None:
        return None

    def _cache(self, raw_logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        self.last_logits = raw_logits.detach()
        self.last_probs = probs.detach()
        return probs

    def _step_batch(self, batch: AssetBatch) -> AssetBatch:
        return batch.to(self.device)


class TentFullEngine(_BaselineEngine):
    """Faithful Tent baseline from Wang et al. (ICLR 2021), simplified to this repo's LayerNorm affine parameters."""

    def __init__(
        self,
        model: AdaptableMLP,
        *,
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__(model, device=device, epsilon=epsilon)
        self.trainable_names = _layernorm_affine_names(self.model)
        _set_requires_grad(self.model, self.trainable_names)
        self.optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=lr)

    def step(self, batch: AssetBatch) -> torch.Tensor:
        batch = self._step_batch(batch)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        raw_logits = _raw_logits(self.model, batch.x)
        probs = _probabilities(self.model, raw_logits)
        loss = entropy(probs, eps=self.epsilon).mean()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            raw_logits_after = _raw_logits(self.model, batch.x)
            probs_after = _probabilities(self.model, raw_logits_after)
        return self._cache(raw_logits_after, probs_after).cpu()


class EATAStyleEngine(TentFullEngine):
    """EATA-style baseline from Niu et al. (ICML 2022), using entropy filtering and a LayerNorm L2 anchor.

    This keeps the core Tent update, skips the paper's redundancy filter, and anchors only the LayerNorm affine
    parameters back to the source snapshot.
    """

    def __init__(
        self,
        model: AdaptableMLP,
        *,
        lr: float = 1e-3,
        entropy_ratio: float = 0.4,
        anchor_weight: float = 1e-4,
        device: str | torch.device = "cpu",
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__(model, lr=lr, device=device, epsilon=epsilon)
        self.anchor_weight = anchor_weight
        self.entropy_threshold = entropy_ratio * math.log(self.model.head.out_features)
        self.source_state = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if name in self.trainable_names
        }

    def step(self, batch: AssetBatch) -> torch.Tensor:
        batch = self._step_batch(batch)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        raw_logits = _raw_logits(self.model, batch.x)
        probs = _probabilities(self.model, raw_logits)
        ent = entropy(probs, eps=self.epsilon)
        selected = ent < self.entropy_threshold
        loss: torch.Tensor | None = None
        if selected.any():
            loss = ent[selected].mean()
        if self.anchor_weight > 0:
            anchor = _anchor_loss(self.model, self.source_state)
            loss = anchor * self.anchor_weight if loss is None else loss + self.anchor_weight * anchor
        if loss is not None:
            loss.backward()
            self.optimizer.step()
        with torch.no_grad():
            raw_logits_after = _raw_logits(self.model, batch.x)
            probs_after = _probabilities(self.model, raw_logits_after)
        return self._cache(raw_logits_after, probs_after).cpu()


class LAMEEngine(_BaselineEngine):
    """LAME baseline from Boudiaf et al. (CVPR 2022), approximated with a batch Laplacian fixed point.

    We use the positive edges of the existing signed graph as the batch affinity and keep the model frozen.
    """

    def __init__(
        self,
        model: AdaptableMLP,
        *,
        config: FinTTAConfig | None = None,
        num_iters: int = 10,
        smoothness: float = 0.75,
        device: str | torch.device = "cpu",
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__(model, device=device, epsilon=epsilon)
        self.config = config or FinTTAConfig(num_classes=self.model.head.out_features)
        self.num_iters = num_iters
        self.smoothness = smoothness
        _freeze_model(self.model)

    def step(self, batch: AssetBatch) -> torch.Tensor:
        batch = self._step_batch(batch)
        self.model.eval()
        with torch.no_grad():
            raw_logits = _raw_logits(self.model, batch.x)
            base_probs = _probabilities(self.model, raw_logits)
            probs = self._refine_probs(batch, base_probs)
        return self._cache(raw_logits, probs).cpu()

    def _refine_probs(self, batch: AssetBatch, base_probs: torch.Tensor) -> torch.Tensor:
        graph = build_signed_graph(batch, self.config, self.device)
        if graph.n_edges == 0:
            return base_probs
        pos_mask = graph.sign > 0
        if not pos_mask.any():
            return base_probs
        n = batch.n_assets
        affinity = torch.zeros((n, n), device=self.device, dtype=base_probs.dtype)
        affinity.index_put_((graph.src[pos_mask], graph.dst[pos_mask]), graph.weight[pos_mask], accumulate=True)
        affinity = 0.5 * (affinity + affinity.t())
        row_sum = affinity.sum(dim=1, keepdim=True).clamp_min(self.epsilon)
        affinity = affinity / row_sum
        q = base_probs.clamp_min(self.epsilon)
        log_base = q.log()
        for _ in range(self.num_iters):
            smooth = affinity @ q
            q = torch.softmax(log_base + self.smoothness * smooth, dim=-1)
        return _simplex_renormalize(q, eps=self.epsilon)


class AdaptableStyleEngine(_BaselineEngine):
    """AdapTable-style output calibration from arXiv:2407.10784, simplified to prior-mass correction only."""

    def __init__(
        self,
        model: AdaptableMLP,
        *,
        source_prior: torch.Tensor | np.ndarray,
        prior_ema: float = 0.9,
        temperature_gain: float = 1.0,
        device: str | torch.device = "cpu",
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__(model, device=device, epsilon=epsilon)
        prior = torch.as_tensor(source_prior, dtype=torch.float32, device=self.device)
        if prior.ndim != 1 or prior.numel() != self.model.head.out_features:
            raise ValueError("source_prior must match the model class count")
        prior = prior / prior.sum().clamp_min(self.epsilon)
        self.source_prior = prior
        self.target_prior = prior.clone()
        self.prior_ema = prior_ema
        self.temperature_gain = temperature_gain
        _freeze_model(self.model)

    def step(self, batch: AssetBatch) -> torch.Tensor:
        batch = self._step_batch(batch)
        self.model.eval()
        with torch.no_grad():
            raw_logits = _raw_logits(self.model, batch.x)
            base_logits = _calibrated_logits(self.model, raw_logits)
            base_probs = torch.softmax(base_logits, dim=-1)
            batch_entropy = normalized_entropy(base_probs, eps=self.epsilon).mean()
            temperature = torch.clamp(1.0 + self.temperature_gain * (batch_entropy - 0.5), 0.5, 2.5)
            calibrated = torch.softmax(base_logits / temperature, dim=-1)
            batch_prior = calibrated.mean(dim=0)
            self.target_prior = self.prior_ema * self.target_prior + (1.0 - self.prior_ema) * batch_prior
            ratio = (self.target_prior + self.epsilon) / (self.source_prior + self.epsilon)
            probs = calibrated * ratio[None, :]
            probs = _simplex_renormalize(probs, eps=self.epsilon)
        return self._cache(raw_logits, probs).cpu()


class OnlineTempEngine(_BaselineEngine):
    """Delayed-label temperature and bias scaling baseline (temperature scaling after Guo et al., ICML 2017)."""

    def __init__(
        self,
        model: AdaptableMLP,
        *,
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__(model, device=device, epsilon=epsilon)
        _set_requires_grad(self.model, {"logit_bias", "log_temperature"})
        self.optimizer = torch.optim.Adam([self.model.logit_bias, self.model.log_temperature], lr=lr)

    def step(self, batch: AssetBatch) -> torch.Tensor:
        batch = self._step_batch(batch)
        self.model.eval()
        with torch.no_grad():
            raw_logits = _raw_logits(self.model, batch.x)
            probs = _probabilities(self.model, raw_logits)
        return self._cache(raw_logits, probs).cpu()

    def observe(self, labels: torch.Tensor) -> dict[str, float] | None:
        if self.last_logits is None:
            return None
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        labels = labels.to(self.device)
        loss = F.cross_entropy(_calibrated_logits(self.model, self.last_logits), labels)
        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.detach().cpu())}


class ACIWrapper(_BaselineEngine):
    """Adaptive Conformal Inference wrapper from Gibbs & Candès (2021) with delayed labels and no probability calibration."""

    def __init__(
        self,
        model: AdaptableMLP,
        *,
        target_coverage: float = 0.9,
        gamma: float = 0.005,
        device: str | torch.device = "cpu",
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__(model, device=device, epsilon=epsilon)
        _freeze_model(self.model)
        self.target_coverage = target_coverage
        self.target_miscoverage = 1.0 - target_coverage
        self.gamma = gamma
        self.alpha = self.target_miscoverage
        self.min_alpha = 0.001
        self.max_alpha = 0.5
        self.score_history: list[float] = []
        self.coverage_history: list[dict[str, float]] = []
        self.last_set_mask: torch.Tensor | None = None
        self.last_prob_threshold: float = 0.0

    def step(self, batch: AssetBatch) -> torch.Tensor:
        batch = self._step_batch(batch)
        self.model.eval()
        with torch.no_grad():
            raw_logits = _raw_logits(self.model, batch.x)
            probs = _probabilities(self.model, raw_logits)
            prob_threshold = self._prob_threshold()
            set_mask = probs >= prob_threshold
            empty = ~set_mask.any(dim=1)
            if empty.any():
                set_mask[empty] = False
                set_mask[empty, probs[empty].argmax(dim=1)] = True
        self.last_set_mask = set_mask.detach()
        self.last_prob_threshold = float(prob_threshold)
        return self._cache(raw_logits, probs).cpu()

    def observe(self, labels: torch.Tensor) -> dict[str, float] | None:
        if self.last_probs is None or self.last_set_mask is None:
            return None
        labels = labels.to(self.device)
        idx = torch.arange(labels.shape[0], device=self.device)
        coverage = float(self.last_set_mask[idx, labels].float().mean().cpu())
        mean_set_size = float(self.last_set_mask.sum(dim=1).float().mean().cpu())
        scores = (1.0 - self.last_probs[idx, labels]).detach().cpu().tolist()
        self.score_history.extend(float(score) for score in scores)
        error_rate = 1.0 - coverage
        self.alpha = float(np.clip(self.alpha + self.gamma * (self.target_miscoverage - error_rate), self.min_alpha, self.max_alpha))
        row = {
            "coverage": coverage,
            "mean_set_size": mean_set_size,
            "alpha": self.alpha,
            "prob_threshold": self.last_prob_threshold,
        }
        self.coverage_history.append(row)
        return row

    def _prob_threshold(self) -> float:
        if not self.score_history:
            return float(max(0.0, 1.0 - self.alpha))
        threshold = float(np.quantile(np.asarray(self.score_history, dtype=np.float64), 1.0 - self.alpha))
        return float(max(0.0, 1.0 - threshold))


__all__ = [
    "ACIWrapper",
    "AdaptableStyleEngine",
    "EATAStyleEngine",
    "LAMEEngine",
    "OnlineTempEngine",
    "TentFullEngine",
]
