from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import FinTTAConfig
from .graph import SignedGraph, build_signed_graph, graph_loss, graph_reliability
from .losses import anchor_loss, prior_volume_loss, risk_weighted_entropy, teacher_kl_loss
from .model import AdaptableMLP, RegimeAdapterBank
from .regime import RegimeInfo, RegimeTracker
from .risk import RiskModel
from .types import AssetBatch
from .utils import effective_sample_size, weighted_mean


@dataclass(slots=True)
class FinTTAOutput:
    probabilities: torch.Tensor
    scores: torch.Tensor
    loss: float
    regime: int
    shock: float
    adapted: bool
    effective_assets: float
    lambda_weights: torch.Tensor
    risk_prior: torch.Tensor
    graph: SignedGraph
    diagnostics: dict[str, float]


@dataclass(slots=True)
class _BatchState:
    info: RegimeInfo
    teacher_p: torch.Tensor
    graph: SignedGraph
    omega: torch.Tensor
    disagreement: torch.Tensor
    lam: torch.Tensor
    pi_risk: torch.Tensor


@dataclass(slots=True)
class _AdaptationResult:
    probabilities: torch.Tensor | None
    loss: float
    adapted: bool


class FinTTAEngine:
    def __init__(self, model: AdaptableMLP, config: FinTTAConfig | None = None, device: str | torch.device = "cpu") -> None:
        self.config = config or FinTTAConfig()
        self.device = torch.device(device)
        torch.manual_seed(self.config.seed)
        self.model = model.to(self.device)
        self.model.freeze_source_weights()
        self.bank = RegimeAdapterBank(self.model, self.config.max_regimes, teacher_ema=self.config.teacher_ema)
        self.regimes = RegimeTracker(self.config)
        self.risk = RiskModel(self.config)
        self.optimizer = self._make_optimizer()

    def step(self, batch: AssetBatch, adapt: bool = True) -> FinTTAOutput:
        batch = batch.to(self.device)
        psi = batch.market_state if batch.market_state is not None else self._market_state(batch)
        if self.config.same_batch_adaptation:
            state = self._update_unsupervised_state(batch, psi)
            if adapt:
                adapt_out = self._adapt_batch(state, batch, return_probabilities=True)
                assert adapt_out.probabilities is not None
                return self._pack_output(
                    adapt_out.probabilities,
                    adapt_out.loss,
                    state.info,
                    adapt_out.adapted,
                    state.omega,
                    state.lam,
                    state.pi_risk,
                    state.graph,
                    state.disagreement,
                )
            probs = self._predict_with_prior(batch, state.omega, state.pi_risk)
            return self._pack_output(
                probs,
                0.0,
                state.info,
                False,
                state.omega,
                state.lam,
                state.pi_risk,
                state.graph,
                state.disagreement,
            )

        prediction_state = self._pre_update_prediction_state(batch, psi)
        probs = self._predict_with_prior(batch, prediction_state.omega, prediction_state.pi_risk)
        update_state = self._update_unsupervised_state(batch, psi)
        adapted = False
        loss = 0.0
        if adapt:
            adapt_out = self._adapt_batch(update_state, batch, return_probabilities=False)
            adapted = adapt_out.adapted
            loss = adapt_out.loss
        return self._pack_output(
            probs,
            loss,
            prediction_state.info,
            adapted,
            prediction_state.omega,
            prediction_state.lam,
            prediction_state.pi_risk,
            prediction_state.graph,
            prediction_state.disagreement,
        )

    def warm_start_market_states(self, batches: list[AssetBatch]) -> None:
        """Initialize regime/risk state from source-period market summaries.

        This method is intended for experiment setup or serialized deployment
        state. It does not update model parameters and does not read labels.
        """

        for batch in batches:
            psi = batch.market_state if batch.market_state is not None else self._market_state(batch.to(self.device))
            info = self.regimes.update(psi)
            self.risk.update(info.regime, batch.returns_window, batch.liquidity)
        self.bank.activate(self.regimes.active)
        self.optimizer = self._make_optimizer()

    def save(self, path: str | Path) -> None:
        """Persist model, regime tracker, risk state, and adapter memory."""

        self.bank.save_active()
        payload = {
            "schema_version": 2,
            "config": asdict(self.config),
            "model": self.model.state_dict(),
            "bank_adapters": self._cpu_bank_state(self.bank.adapters),
            "bank_teachers": self._cpu_bank_state(self.bank.teachers),
            "bank_source": self._cpu_tensor_state(self.bank.source_state),
            "bank_active": self.bank.active_regime,
            "regimes": self.regimes.to_state(),
            "risk": self.risk.to_state(),
        }
        torch.save(payload, Path(path))

    def load(self, path: str | Path) -> None:
        """Load a state saved by `save` into an engine with the same model shape."""

        payload = torch.load(Path(path), map_location=self.device, weights_only=True)
        self.config = FinTTAConfig(**payload["config"])
        self.model.load_state_dict(payload["model"])
        self.model.freeze_source_weights()
        self.regimes = RegimeTracker.from_state(self.config, payload["regimes"])
        self.risk = RiskModel.from_state(self.config, payload["risk"])
        self.bank = RegimeAdapterBank(self.model, self.config.max_regimes, teacher_ema=self.config.teacher_ema)
        self.bank.adapters = self._bank_state_to_device(payload["bank_adapters"])
        self.bank.teachers = self._bank_state_to_device(payload["bank_teachers"])
        self.bank.source_state = self._tensor_state_to_device(payload["bank_source"])
        self.bank.active_regime = int(payload["bank_active"])
        self.bank.activate(self.bank.active_regime)
        self.optimizer = self._make_optimizer()

    def _pre_update_prediction_state(self, batch: AssetBatch, psi: np.ndarray) -> _BatchState:
        teacher_p, teacher_features = self._teacher_prediction(batch)
        graph = build_signed_graph(batch, self.config, self.device)
        typicality = self._typicality(teacher_features.detach())
        omega, disagreement = graph_reliability(teacher_p, graph, batch.liquidity, self.config, typicality=typicality)
        if self.regimes.regimes:
            regime_ids = self.regimes.regime_ids()
            posterior = self.regimes.posterior.copy()
            priors = [self.regimes.regimes[r].pi for r in regime_ids]
            prior_np = self.regimes.current_prior()
            active = self.regimes.active
        else:
            regime_ids = [self.bank.active_regime]
            posterior = np.array([1.0], dtype=np.float64)
            priors = [np.ones(self.config.num_classes, dtype=np.float64) / self.config.num_classes]
            prior_np = priors[0]
            active = self.bank.active_regime
        costs = self.risk.class_costs(posterior, priors, regime_ids=regime_ids)
        costs = torch.nan_to_num(costs, nan=1.0, posinf=1.0, neginf=1.0)
        lam = self.risk.entropy_weights(costs, self.device)
        prior = torch.tensor(prior_np, dtype=torch.float32, device=self.device)
        pi_risk = self.risk.risk_adjusted_prior(prior, costs, self.device)
        info = RegimeInfo(active, posterior, 0.0, False, False, np.asarray(psi, dtype=np.float64).copy())
        return _BatchState(info, teacher_p, graph, omega, disagreement, lam, pi_risk)

    def _update_unsupervised_state(self, batch: AssetBatch, psi: np.ndarray) -> _BatchState:
        info = self.regimes.update(psi)
        if info.changepoint or info.created or info.regime != self.bank.active_regime:
            blend = self.regimes.similarity_weights(info.psi)
            self.bank.activate(info.regime, blend_weights=blend)
            self.optimizer = self._make_optimizer()
        teacher_p, teacher_features = self._teacher_prediction(batch)
        graph = build_signed_graph(batch, self.config, self.device)
        typicality = self._typicality(teacher_features.detach())
        omega, disagreement = graph_reliability(teacher_p, graph, batch.liquidity, self.config, typicality=typicality)
        pi_np = self.regimes.update_label_prior(teacher_p, omega, info.changepoint)
        self.risk.update(info.regime, batch.returns_window, batch.liquidity)
        regime_ids = self.regimes.regime_ids()
        priors = [self.regimes.regimes[r].pi for r in regime_ids]
        costs = self.risk.class_costs(self.regimes.posterior, priors, regime_ids=regime_ids)
        costs = torch.nan_to_num(costs, nan=1.0, posinf=1.0, neginf=1.0)
        lam = self.risk.entropy_weights(costs, self.device)
        prior = torch.tensor(pi_np, dtype=torch.float32, device=self.device)
        pi_risk = self.risk.risk_adjusted_prior(prior, costs, self.device)
        return _BatchState(info, teacher_p, graph, omega, disagreement, lam, pi_risk)

    def _teacher_prediction(self, batch: AssetBatch) -> tuple[torch.Tensor, torch.Tensor]:
        self.bank.load_teacher()
        with torch.no_grad():
            teacher_logits, teacher_features = self.model(batch.x, return_features=True)
            teacher_p = torch.softmax(teacher_logits, dim=-1)
        self.bank.load_student()
        return teacher_p, teacher_features

    def _adapt_batch(
        self,
        state: _BatchState,
        batch: AssetBatch,
        *,
        return_probabilities: bool,
    ) -> _AdaptationResult:
        n_eff = float(effective_sample_size(state.omega, eps=self.config.epsilon))
        graph_health = float(state.disagreement.mean().detach().cpu())
        if n_eff < self.config.min_effective_assets or graph_health > 2.0:
            probs = self._predict_with_prior(batch, state.omega, state.pi_risk) if return_probabilities else None
            return _AdaptationResult(probs, 0.0, False)
        snapshot = self.bank.snapshot()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(batch.x)
        p_uncorr = torch.softmax(logits, dim=-1)
        pbar_minus = weighted_mean(p_uncorr.detach(), state.omega, eps=self.config.epsilon)
        correction = self.config.beta_pi * (
            (state.pi_risk + self.config.epsilon).log() - (pbar_minus + self.config.epsilon).log()
        )
        probs = torch.softmax(logits + correction[None, :], dim=-1)
        l_ent = risk_weighted_entropy(probs, state.omega, state.lam, self.config)
        l_graph = graph_loss(probs, state.teacher_p, state.graph, self.config)
        l_prior, pbar = prior_volume_loss(probs, state.omega, state.pi_risk, self.config)
        l_teacher = teacher_kl_loss(probs, state.teacher_p, state.omega, self.config)
        l_anchor = anchor_loss(self.model, self.config, self.bank.source_state)
        loss = (
            l_ent
            + self.config.alpha_graph * l_graph
            + self.config.alpha_prior * l_prior
            + self.config.alpha_teacher * l_teacher
            + self.config.alpha_anchor * l_anchor
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.config.grad_clip)
        self.optimizer.step()
        self.bank.clip_adapter_norm(self.config.adapter_norm_clip)
        self.bank.stochastic_restore(self.config.stochastic_restore_p)
        probs_after = self._predict_with_prior(batch, state.omega, state.pi_risk)
        health = self._health(probs_after, pbar, state.pi_risk, state.disagreement, n_eff)
        adapted = True
        if health > self.config.health_max:
            self.bank.restore_snapshot(snapshot)
            probs_after = self._predict_with_prior(batch, state.omega, state.pi_risk) if return_probabilities else None
            adapted = False
        else:
            self.bank.save_active()
            self.bank.update_teacher()
            if not return_probabilities:
                probs_after = None
        return _AdaptationResult(probs_after, float(loss.detach().cpu()), adapted)

    @torch.no_grad()
    def _predict_with_prior(self, batch: AssetBatch, omega: torch.Tensor, pi_risk: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        logits = self.model(batch.x)
        p_uncorr = torch.softmax(logits, dim=-1)
        pbar_minus = weighted_mean(p_uncorr, omega, eps=self.config.epsilon)
        correction = self.config.beta_pi * ((pi_risk + self.config.epsilon).log() - (pbar_minus + self.config.epsilon).log())
        return torch.softmax(logits + correction[None, :], dim=-1)

    def _pack_output(
        self,
        probs: torch.Tensor,
        loss: float,
        info: RegimeInfo,
        adapted: bool,
        omega: torch.Tensor,
        lam: torch.Tensor,
        pi_risk: torch.Tensor,
        graph: SignedGraph,
        disagreement: torch.Tensor,
    ) -> FinTTAOutput:
        exposure = torch.tensor(self.config.ordinal_exposure, dtype=probs.dtype, device=probs.device)
        scores = probs @ exposure
        return FinTTAOutput(
            probabilities=probs.detach().cpu(),
            scores=scores.detach().cpu(),
            loss=loss,
            regime=info.regime,
            shock=info.shock,
            adapted=adapted,
            effective_assets=float(effective_sample_size(omega, eps=self.config.epsilon).detach().cpu()),
            lambda_weights=lam.detach().cpu(),
            risk_prior=pi_risk.detach().cpu(),
            graph=graph,
            diagnostics={
                "omega_mean": float(omega.mean().detach().cpu()),
                "graph_disagreement": float(disagreement.mean().detach().cpu()),
                "posterior_entropy": float(-(info.posterior * np.log(info.posterior + 1e-12)).sum()),
            },
        )

    @staticmethod
    def _cpu_tensor_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu().clone() for name, value in state.items()}

    @classmethod
    def _cpu_bank_state(cls, state: dict[int, dict[str, torch.Tensor]]) -> dict[int, dict[str, torch.Tensor]]:
        return {int(regime): cls._cpu_tensor_state(values) for regime, values in state.items()}

    def _tensor_state_to_device(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: value.to(self.device) for name, value in state.items()}

    def _bank_state_to_device(self, state: dict[int, dict[str, torch.Tensor]]) -> dict[int, dict[str, torch.Tensor]]:
        return {int(regime): self._tensor_state_to_device(values) for regime, values in state.items()}

    def _make_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.config.lr, weight_decay=0.0)

    def _market_state(self, batch: AssetBatch) -> np.ndarray:
        if batch.returns_window is None:
            x = batch.x.detach().cpu().numpy()
            return np.array([float(np.log(np.std(x) + 1e-6)), 0.0, 0.0, 0.0], dtype=np.float64)
        r = np.nan_to_num(np.asarray(batch.returns_window, dtype=np.float64), nan=0.0)
        rv = np.sqrt(np.mean(r * r, axis=1) + 1e-8)
        latest = r[:, -1]
        corr = np.corrcoef(r)
        upper = corr[np.triu_indices_from(corr, k=1)] if corr.ndim == 2 else np.array([0.0])
        eig_share = 0.0
        if corr.ndim == 2 and corr.shape[0] > 1:
            vals = np.linalg.eigvalsh(np.nan_to_num(corr, nan=0.0))
            eig_share = float(vals[-1] / max(vals.sum(), 1e-6))
        return np.array(
            [
                np.log(np.median(rv) + 1e-6),
                np.median(np.abs(latest - np.median(latest))),
                float(np.nanmean(np.abs(upper))) if upper.size else 0.0,
                eig_share,
            ],
            dtype=np.float64,
        )

    def _typicality(self, features: torch.Tensor) -> torch.Tensor:
        center = features.mean(dim=0, keepdim=True)
        dist = (features - center).square().mean(dim=1)
        return torch.exp(-dist / self.config.tau_typicality).clamp(0.0, 1.0)

    def _health(
        self,
        probs: torch.Tensor,
        pbar_before: torch.Tensor,
        pi_risk: torch.Tensor,
        disagreement: torch.Tensor,
        n_eff: float,
    ) -> float:
        pbar = probs.mean(dim=0).to(pi_risk.device)
        kl = F.kl_div((pi_risk + self.config.epsilon).log(), pbar, reduction="sum")
        turnover = (pbar - pbar_before.detach()).abs().sum()
        return float((kl + disagreement.mean() + turnover - 0.02 * n_eff).detach().cpu())
