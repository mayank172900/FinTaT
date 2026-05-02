from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import FinTTAConfig


@dataclass(slots=True)
class RegimeInfo:
    regime: int
    posterior: np.ndarray
    shock: float
    changepoint: bool
    created: bool
    psi: np.ndarray


@dataclass
class _Regime:
    mu: np.ndarray
    cov: np.ndarray
    count: float
    alpha: np.ndarray

    @property
    def pi(self) -> np.ndarray:
        return self.alpha / max(float(self.alpha.sum()), 1e-12)


class RegimeTracker:
    def __init__(self, config: FinTTAConfig, psi_dim: int | None = None) -> None:
        self.config = config
        self.psi_dim = psi_dim
        self.regimes: dict[int, _Regime] = {}
        self.posterior = np.array([1.0], dtype=np.float64)
        self.active = 0
        self.next_id = 0
        if psi_dim is not None:
            self._create_regime(np.zeros(psi_dim, dtype=np.float64))

    def update(self, psi: np.ndarray) -> RegimeInfo:
        psi = np.asarray(psi, dtype=np.float64).ravel()
        psi = np.nan_to_num(psi, nan=0.0, posinf=0.0, neginf=0.0)
        if not self.regimes:
            regime = self._create_regime(psi)
            self.posterior = np.array([1.0], dtype=np.float64)
            self.active = regime
            return RegimeInfo(regime, self.posterior.copy(), 1.0, True, True, psi)

        ids = list(self.regimes)
        distances = np.array([self._mahalanobis(psi, self.regimes[r]) for r in ids])
        log_like = -0.5 * np.log1p(distances)
        prev = self._posterior_by_ids(ids)
        sticky = 0.92
        trans = np.full((len(ids), len(ids)), (1.0 - sticky) / max(len(ids) - 1, 1))
        np.fill_diagonal(trans, sticky)
        pred = prev @ trans if len(ids) > 1 else prev
        raw = np.exp(log_like - log_like.max()) * pred
        posterior = raw / max(raw.sum(), 1e-12)
        best_idx = int(posterior.argmax())
        best_regime = ids[best_idx]
        min_dist = float(distances.min())
        surprise = 1.0 - float(raw.max() / max(raw.sum(), 1e-12))
        bocpd_proxy = 1.0 - float(np.exp(-0.5 * min_dist / max(psi.size, 1)))
        shock = max(bocpd_proxy, surprise if min_dist > 2.5 * max(psi.size, 1) else 0.0)
        changepoint = shock > self.config.tau_cp
        created = False
        spawn_threshold = max(1.5 * max(psi.size, 1), max(psi.size, 1) + 1.0)
        if changepoint and min_dist > spawn_threshold and len(ids) < self.config.max_regimes:
            best_regime = self._create_regime(psi)
            ids = list(self.regimes)
            posterior = np.zeros(len(ids), dtype=np.float64)
            posterior[ids.index(best_regime)] = 1.0
            created = True
        else:
            self._update_emission(best_regime, psi, fast=changepoint)
        self.posterior = self._posterior_array(ids, posterior)
        self.active = best_regime
        return RegimeInfo(best_regime, self.posterior.copy(), shock, changepoint, created, psi)

    def update_label_prior(self, teacher_probs: torch.Tensor, omega: torch.Tensor, changepoint: bool) -> np.ndarray:
        eps = self.config.epsilon
        probs = teacher_probs.detach().cpu()
        weights = omega.detach().cpu()
        k = self.config.num_classes
        q = (weights[:, None] * probs).sum(dim=0).numpy() + eps / k
        q = q / max(q.sum(), eps)
        n_eff = float(weights.sum().square() / (weights.square().sum() + eps))
        regime = self.regimes[self.active]
        if n_eff < self.config.min_effective_assets:
            return self.current_prior()
        if changepoint:
            seed = np.ones(k, dtype=np.float64) / k
            m_jump = min(self.config.m_max, n_eff)
            regime.alpha = self.config.c_reset * seed + m_jump * q
        else:
            target = self.config.c_reset * regime.pi + min(self.config.m_max, n_eff) * q
            regime.alpha = (1.0 - self.config.eta_slow) * regime.alpha + self.config.eta_slow * target
        return self.current_prior()

    def current_prior(self) -> np.ndarray:
        ids = list(self.regimes)
        post = self._posterior_by_ids(ids)
        pi = np.zeros(self.config.num_classes, dtype=np.float64)
        for weight, regime_id in zip(post, ids):
            pi += weight * self.regimes[regime_id].pi
        pi = (1.0 - self.config.epsilon_pi) * pi + self.config.epsilon_pi / self.config.num_classes
        return pi / pi.sum()

    def active_prior(self) -> np.ndarray:
        return self.regimes[self.active].pi

    def regime_ids(self) -> list[int]:
        return list(self.regimes)

    def similarity_weights(self, psi: np.ndarray, temperature: float = 4.0) -> dict[int, float]:
        if not self.regimes:
            return {}
        d = np.array([self._mahalanobis(psi, r) for r in self.regimes.values()])
        w = np.exp(-d / max(temperature, 1e-6))
        ids = list(self.regimes)
        return {ids[i]: float(w[i]) for i in range(len(ids))}

    def _create_regime(self, psi: np.ndarray) -> int:
        rid = self.next_id
        self.next_id += 1
        dim = psi.size
        alpha = np.ones(self.config.num_classes, dtype=np.float64) * self.config.c_reset / self.config.num_classes
        scale = np.maximum(np.abs(psi) * 0.15, 0.05)
        if dim >= 2:
            scale[1] = max(scale[1], 0.005)
        if dim >= 3:
            scale[2:] = np.maximum(scale[2:], 0.04)
        cov = np.diag(scale * scale)
        self.regimes[rid] = _Regime(mu=psi.copy(), cov=cov, count=1.0, alpha=alpha)
        return rid

    def _update_emission(self, rid: int, psi: np.ndarray, fast: bool) -> None:
        r = self.regimes[rid]
        eta = 0.2 if fast else min(0.03, 1.0 / (r.count + 1.0))
        delta = psi - r.mu
        r.mu = (1.0 - eta) * r.mu + eta * psi
        r.cov = (1.0 - eta) * r.cov + eta * (np.outer(delta, delta) + 1e-3 * np.eye(psi.size))
        r.count += 1.0

    def _mahalanobis(self, psi: np.ndarray, regime: _Regime) -> float:
        diff = psi - regime.mu
        cov = regime.cov + 1e-3 * np.eye(regime.cov.shape[0])
        inv = np.linalg.pinv(cov)
        return float(diff.T @ inv @ diff)

    def _posterior_by_ids(self, ids: list[int]) -> np.ndarray:
        old_ids = list(self.regimes)
        if self.posterior.size != len(old_ids):
            return np.ones(len(ids), dtype=np.float64) / len(ids)
        lookup = {rid: self.posterior[i] for i, rid in enumerate(old_ids)}
        arr = np.array([lookup.get(rid, 0.0) for rid in ids], dtype=np.float64)
        if arr.sum() <= 0:
            arr[:] = 1.0 / len(arr)
        return arr / arr.sum()

    def _posterior_array(self, ids: list[int], posterior: np.ndarray) -> np.ndarray:
        full_ids = list(self.regimes)
        lookup = {rid: posterior[i] for i, rid in enumerate(ids)}
        out = np.array([lookup.get(rid, 0.0) for rid in full_ids], dtype=np.float64)
        return out / max(out.sum(), 1e-12)
