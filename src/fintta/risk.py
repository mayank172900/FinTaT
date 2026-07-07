from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch

from .config import FinTTAConfig


@dataclass
class RiskEstimate:
    sigma: float = 1.0
    rho: float = 0.0
    cvar_down: float = 1.0
    liquidity_stress: float = 0.0


class RiskModel:
    def __init__(self, config: FinTTAConfig) -> None:
        self.config = config
        self.by_regime: dict[int, RiskEstimate] = {}
        self.sigma_ref = 1.0
        self.cvar_ref = 1.0

    def update(self, regime: int, returns_window: np.ndarray | None, liquidity: torch.Tensor | None) -> RiskEstimate:
        est = self.by_regime.setdefault(regime, RiskEstimate())
        if returns_window is not None and returns_window.size:
            arr = np.asarray(returns_window, dtype=np.float64)
            sigma = float(np.nanmedian(np.sqrt(np.nanmean(arr * arr, axis=1))) + 1e-6)
            flat = arr[np.isfinite(arr)].ravel()
            downside = -flat[flat < np.nanquantile(flat, 0.1)] if flat.size else np.array([1.0])
            cvar = float(np.nanmean(downside) if downside.size else sigma)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                corr = np.corrcoef(np.nan_to_num(arr, nan=0.0))
            upper = corr[np.triu_indices_from(corr, k=1)] if corr.ndim == 2 else np.array([0.0])
            upper = upper[np.isfinite(upper)]
            rho = float(np.nanmean(np.abs(upper))) if upper.size else 0.0
        else:
            sigma, cvar, rho = est.sigma, est.cvar_down, est.rho
        liq_stress = float((1.0 - liquidity.detach().cpu()).mean()) if liquidity is not None else est.liquidity_stress
        sigma = sigma if np.isfinite(sigma) and sigma > 0 else est.sigma
        cvar = cvar if np.isfinite(cvar) and cvar > 0 else est.cvar_down
        rho = rho if np.isfinite(rho) else est.rho
        liq_stress = liq_stress if np.isfinite(liq_stress) else est.liquidity_stress
        eta = 0.08
        est.sigma = (1 - eta) * est.sigma + eta * max(sigma, 1e-6)
        est.cvar_down = (1 - eta) * est.cvar_down + eta * max(cvar, 1e-6)
        est.rho = (1 - eta) * est.rho + eta * max(min(rho, 1.0), 0.0)
        est.liquidity_stress = (1 - eta) * est.liquidity_stress + eta * max(min(liq_stress, 1.0), 0.0)
        self.sigma_ref = (1 - eta) * self.sigma_ref + eta * est.sigma
        self.cvar_ref = (1 - eta) * self.cvar_ref + eta * est.cvar_down
        return est

    def class_costs(
        self,
        posterior: np.ndarray,
        regime_priors: list[np.ndarray],
        regime_ids: list[int] | None = None,
    ) -> torch.Tensor:
        k = self.config.num_classes
        u = np.asarray(self.config.ordinal_exposure, dtype=np.float64)
        b = np.asarray(self.config.return_buckets, dtype=np.float64)
        costs = np.zeros(k, dtype=np.float64)
        ids = regime_ids if regime_ids is not None else list(self.by_regime)
        if not ids:
            ids = [0]
        for idx, regime_id in enumerate(ids[: len(regime_priors)]):
            gamma = posterior[idx] if idx < len(posterior) else 0.0
            pi = regime_priors[idx]
            est = self.by_regime.get(regime_id, RiskEstimate())
            multiplier = (
                1.0
                + est.sigma / max(self.sigma_ref, 1e-6)
                + 0.5 * est.rho
                + est.cvar_down / max(self.cvar_ref, 1e-6)
                + 0.5 * est.liquidity_stress
            )
            for pred in range(k):
                action_cost = 0.0
                for true in range(k):
                    crash = max(0.0, -u[pred] * b[true]) ** 1.4
                    opportunity = 0.15 * max(0.0, u[pred] * b[true])
                    action_cost += pi[true] * multiplier * (crash + opportunity)
                action_cost += 0.03 * abs(u[pred])
                costs[pred] += gamma * action_cost
        if costs.sum() <= 0:
            costs += 1.0
        return torch.tensor(costs, dtype=torch.float32)

    def entropy_weights(self, costs: torch.Tensor, device: torch.device) -> torch.Tensor:
        costs = costs.to(device)
        raw = (costs + self.config.epsilon).pow(-self.config.rho_lambda)
        lam = self.config.num_classes * raw / raw.sum().clamp_min(self.config.epsilon)
        return lam.clamp(self.config.lambda_min, self.config.lambda_max)

    def risk_adjusted_prior(self, prior: torch.Tensor, costs: torch.Tensor, device: torch.device) -> torch.Tensor:
        prior = prior.to(device)
        costs = costs.to(device)
        raw = (prior + self.config.epsilon_pi) * torch.exp(-costs / max(self.config.risk_temperature, 1e-6))
        return raw / raw.sum().clamp_min(self.config.epsilon)

    def to_state(self) -> dict:
        return {
            "sigma_ref": self.sigma_ref,
            "cvar_ref": self.cvar_ref,
            "by_regime": {
                str(regime): {
                    "sigma": estimate.sigma,
                    "rho": estimate.rho,
                    "cvar_down": estimate.cvar_down,
                    "liquidity_stress": estimate.liquidity_stress,
                }
                for regime, estimate in self.by_regime.items()
            },
        }

    @classmethod
    def from_state(cls, config: FinTTAConfig, state: dict) -> RiskModel:
        model = cls(config)
        model.sigma_ref = float(state.get("sigma_ref", 1.0))
        model.cvar_ref = float(state.get("cvar_ref", 1.0))
        model.by_regime = {
            int(regime): RiskEstimate(
                sigma=float(values.get("sigma", 1.0)),
                rho=float(values.get("rho", 0.0)),
                cvar_down=float(values.get("cvar_down", 1.0)),
                liquidity_stress=float(values.get("liquidity_stress", 0.0)),
            )
            for regime, values in state.get("by_regime", {}).items()
        }
        return model
