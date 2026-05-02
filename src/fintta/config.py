from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FinTTAConfig:
    num_classes: int = 5
    max_regimes: int = 8
    topk_edges: int = 6
    tau_cp: float = 0.6
    eta_slow: float = 0.02
    c_reset: float = 5.0
    m_max: float = 200.0
    epsilon: float = 1e-6
    epsilon_pi: float = 0.02
    lambda_min: float = 0.2
    lambda_max: float = 3.0
    rho_lambda: float = 1.0
    risk_temperature: float = 1.0
    beta_pi: float = 0.7
    confidence_floor: float = 0.35
    tau_confidence: float = 0.08
    tau_graph: float = 0.25
    tau_typicality: float = 8.0
    min_effective_assets: float = 4.0
    alpha_graph: float = 0.25
    alpha_prior: float = 1.0
    alpha_teacher: float = 0.15
    alpha_anchor: float = 1e-3
    alpha_adapter_l2: float = 1e-4
    alpha_sharp: float = 0.0
    eta_dir: float = 0.25
    lr: float = 3e-4
    grad_clip: float = 2.0
    adapter_norm_clip: float = 8.0
    teacher_ema: float = 0.98
    stochastic_restore_p: float = 1e-3
    health_max: float = 8.0
    seed: int = 7
    long_only: bool = False
    same_batch_adaptation: bool = True

    @property
    def ordinal_exposure(self) -> list[float]:
        if self.long_only:
            return [0.0, 0.0, 0.0, 0.5, 1.0][: self.num_classes]
        if self.num_classes == 5:
            return [-1.0, -0.5, 0.0, 0.5, 1.0]
        mid = (self.num_classes - 1) / 2
        return [float((k - mid) / max(mid, 1.0)) for k in range(self.num_classes)]

    @property
    def return_buckets(self) -> list[float]:
        if self.num_classes == 5:
            return [-2.0, -1.0, 0.0, 1.0, 2.0]
        mid = (self.num_classes - 1) / 2
        return [float(k - mid) for k in range(self.num_classes)]
