from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn


class AdapterBlock(nn.Module):
    def __init__(self, width: int, rank: int = 8) -> None:
        super().__init__()
        rank = max(1, min(rank, width))
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.up(F.gelu(self.down(h)))


class AdaptableMLP(nn.Module):
    """Frozen MLP backbone with small reversible adaptation parameters."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 5,
        hidden_dim: int = 64,
        depth: int = 2,
        adapter_rank: int = 8,
    ) -> None:
        super().__init__()
        self.input_shift = nn.Parameter(torch.zeros(input_dim))
        self.input_scale_log = nn.Parameter(torch.zeros(input_dim))
        layers: list[nn.Module] = []
        last = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(last, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(AdapterBlock(hidden_dim, rank=adapter_rank))
            last = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(last, num_classes)
        self.logit_bias = nn.Parameter(torch.zeros(num_classes))
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self._adaptation_parameter_names = self._collect_adaptation_parameter_names()

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = (x + self.input_shift) * self.input_scale_log.exp().clamp(0.2, 5.0)
        h = self.backbone(x)
        logits = self.head(h)
        logits = (logits + self.logit_bias) / self.log_temperature.exp().clamp(0.25, 4.0)
        if return_features:
            return logits, h
        return logits

    def freeze_source_weights(self) -> None:
        for name, param in self.named_parameters():
            param.requires_grad = self.is_adaptation_parameter(name)

    def is_adaptation_parameter(self, name: str) -> bool:
        return name in self._adaptation_parameter_names

    def _collect_adaptation_parameter_names(self) -> set[str]:
        names = {"input_shift", "input_scale_log", "logit_bias", "log_temperature", "head.bias"}
        for module_name, module in self.named_modules():
            if isinstance(module, nn.LayerNorm):
                names.add(f"{module_name}.weight")
                names.add(f"{module_name}.bias")
            elif isinstance(module, AdapterBlock):
                names.add(f"{module_name}.down.weight")
                names.add(f"{module_name}.up.weight")
        return names

    def adaptation_state(self) -> dict[str, torch.Tensor]:
        return {n: p.detach().clone() for n, p in self.named_parameters() if self.is_adaptation_parameter(n)}

    def load_adaptation_state(self, state: dict[str, torch.Tensor]) -> None:
        params = dict(self.named_parameters())
        for name, value in state.items():
            if name in params:
                params[name].data.copy_(value.to(params[name].device))

    def zero_adaptation_state(self) -> dict[str, torch.Tensor]:
        return {name: torch.zeros_like(value) for name, value in self.adaptation_state().items()}

    def clone(self) -> AdaptableMLP:
        return copy.deepcopy(self)


class RegimeAdapterBank:
    def __init__(self, model: AdaptableMLP, max_regimes: int, teacher_ema: float = 0.98) -> None:
        self.model = model
        self.max_regimes = max_regimes
        self.teacher_ema = teacher_ema
        base = model.adaptation_state()
        self.source_state = {k: v.clone() for k, v in base.items()}
        self.adapters: dict[int, dict[str, torch.Tensor]] = {0: {k: v.clone() for k, v in base.items()}}
        self.teachers: dict[int, dict[str, torch.Tensor]] = {0: {k: v.clone() for k, v in base.items()}}
        self.active_regime = 0
        model.load_adaptation_state(self.adapters[0])

    def activate(self, regime: int, blend_weights: dict[int, float] | None = None) -> bool:
        created = False
        if regime not in self.adapters:
            created = True
            if blend_weights:
                state = self._blend(blend_weights)
            else:
                state = self.source_state
            self.adapters[regime] = {k: v.clone() for k, v in state.items()}
            self.teachers[regime] = {k: v.clone() for k, v in state.items()}
        self.active_regime = regime
        self.model.load_adaptation_state(self.adapters[regime])
        return created

    def save_active(self) -> None:
        self.adapters[self.active_regime] = self.model.adaptation_state()

    def load_teacher(self) -> None:
        self.model.load_adaptation_state(self.teachers[self.active_regime])

    def load_student(self) -> None:
        self.model.load_adaptation_state(self.adapters[self.active_regime])

    def update_teacher(self) -> None:
        student = self.model.adaptation_state()
        teacher = self.teachers[self.active_regime]
        for k, v in student.items():
            teacher[k] = self.teacher_ema * teacher[k].to(v.device) + (1.0 - self.teacher_ema) * v.detach()

    def snapshot(self) -> dict[str, torch.Tensor]:
        return self.model.adaptation_state()

    def restore_snapshot(self, snapshot: dict[str, torch.Tensor]) -> None:
        self.model.load_adaptation_state(snapshot)
        self.save_active()

    def stochastic_restore(self, p: float) -> None:
        if p <= 0:
            return
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not self.model.is_adaptation_parameter(name):
                    continue
                mask = torch.rand_like(param, dtype=torch.float32) < p
                source = self.source_state[name].to(param.device)
                param.data = torch.where(mask, source, param.data)

    def clip_adapter_norm(self, max_norm: float) -> None:
        if max_norm <= 0:
            return
        with torch.no_grad():
            params = [p for n, p in self.model.named_parameters() if self.model.is_adaptation_parameter(n)]
            norm = torch.sqrt(sum((p.detach() ** 2).sum() for p in params))
            if norm > max_norm:
                scale = max_norm / (norm + 1e-6)
                for p in params:
                    p.mul_(scale)

    def _blend(self, weights: dict[int, float]) -> dict[str, torch.Tensor]:
        available = {regime: weight for regime, weight in weights.items() if regime in self.adapters and weight > 0}
        if not available:
            return self.source_state
        base = {k: torch.zeros_like(v) for k, v in self.source_state.items()}
        total = max(sum(available.values()), 1e-6)
        for regime, weight in available.items():
            for k in base:
                base[k] += self.adapters[regime][k] * (weight / total)
        return base
