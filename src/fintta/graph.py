from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import FinTTAConfig
from .types import AssetBatch
from .utils import js_divergence, normalized_entropy, ordinal_reversal, topk_softmax


@dataclass(slots=True)
class SignedGraph:
    src: torch.Tensor
    dst: torch.Tensor
    weight: torch.Tensor
    sign: torch.Tensor
    n_nodes: int

    @property
    def n_edges(self) -> int:
        return int(self.src.numel())


def build_signed_graph(batch: AssetBatch, config: FinTTAConfig, device: torch.device | str) -> SignedGraph:
    n = batch.n_assets
    sectors = batch.metadata.get("sector", [""] * n)
    industries = batch.metadata.get("industry", [""] * n)
    corr = _rolling_corr(batch.returns_window, n)
    factors = batch.factor_exposures
    src: list[int] = []
    dst: list[int] = []
    weight: list[float] = []
    sign: list[int] = []
    for i in range(n):
        pos_scores = []
        neg_scores = []
        candidates = []
        for j in range(n):
            if i == j:
                continue
            same_sector = 1.0 if sectors[i] == sectors[j] and sectors[i] != "" else 0.0
            same_industry = 1.0 if industries[i] == industries[j] and industries[i] != "" else 0.0
            rho = float(corr[i, j])
            beta_sim = _cosine(factors[i], factors[j]) if factors is not None else 0.0
            pos = 1.2 * same_sector + 1.6 * same_industry + max(0.0, rho) + 0.5 * max(0.0, beta_sim)
            neg = max(0.0, -rho) + 0.5 * max(0.0, -beta_sim)
            candidates.append(j)
            pos_scores.append(pos)
            neg_scores.append(neg)
        cand_t = torch.tensor(candidates, device=device, dtype=torch.long)
        pos_t = torch.tensor(pos_scores, device=device, dtype=torch.float32)
        neg_t = torch.tensor(neg_scores, device=device, dtype=torch.float32)
        pos_w = topk_softmax(pos_t, config.topk_edges)
        neg_w = topk_softmax(neg_t, max(1, config.topk_edges // 2))
        for local, w in enumerate(pos_w):
            if float(w) > 0:
                src.append(i)
                dst.append(int(cand_t[local]))
                weight.append(float(w))
                sign.append(1)
        for local, w in enumerate(neg_w):
            if float(w) > 0 and float(neg_t[local]) > 0.05:
                src.append(i)
                dst.append(int(cand_t[local]))
                weight.append(float(w))
                sign.append(-1)
    if not src:
        return SignedGraph(
            src=torch.empty(0, dtype=torch.long, device=device),
            dst=torch.empty(0, dtype=torch.long, device=device),
            weight=torch.empty(0, dtype=torch.float32, device=device),
            sign=torch.empty(0, dtype=torch.float32, device=device),
            n_nodes=n,
        )
    return SignedGraph(
        src=torch.tensor(src, dtype=torch.long, device=device),
        dst=torch.tensor(dst, dtype=torch.long, device=device),
        weight=torch.tensor(weight, dtype=torch.float32, device=device),
        sign=torch.tensor(sign, dtype=torch.float32, device=device),
        n_nodes=n,
    )


def graph_loss(student_p: torch.Tensor, teacher_p: torch.Tensor, graph: SignedGraph, config: FinTTAConfig) -> torch.Tensor:
    if graph.n_edges == 0:
        return student_p.sum() * 0.0
    reversal = ordinal_reversal(student_p.shape[-1], student_p.device)
    target = teacher_p[graph.dst].detach()
    target = torch.where((graph.sign < 0)[:, None], target @ reversal, target)
    js = js_divergence(student_p[graph.src], target, eps=config.epsilon)
    values = torch.tensor(config.ordinal_exposure, device=student_p.device, dtype=student_p.dtype)
    m_student = student_p[graph.src] @ values
    m_teacher = teacher_p[graph.dst].detach() @ values
    direction = (m_student - graph.sign * m_teacher).square()
    return (graph.weight * (js + config.eta_dir * direction)).sum() / graph.weight.sum().clamp_min(config.epsilon)


def graph_reliability(
    teacher_p: torch.Tensor,
    graph: SignedGraph,
    liquidity: torch.Tensor | None,
    config: FinTTAConfig,
    typicality: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = graph.n_nodes
    device = teacher_p.device
    disagreement = torch.zeros(n, device=device)
    denom = torch.zeros(n, device=device)
    if graph.n_edges:
        reversal = ordinal_reversal(teacher_p.shape[-1], device)
        neighbor = teacher_p[graph.dst].detach()
        neighbor = torch.where((graph.sign < 0)[:, None], neighbor @ reversal, neighbor)
        edge_js = js_divergence(teacher_p[graph.src].detach(), neighbor, eps=config.epsilon)
        disagreement.scatter_add_(0, graph.src, graph.weight * edge_js)
        denom.scatter_add_(0, graph.src, graph.weight)
        disagreement = disagreement / denom.clamp_min(config.epsilon)
    g = torch.exp(-disagreement / config.tau_graph)
    c = 1.0 - normalized_entropy(teacher_p.detach(), eps=config.epsilon)
    c_gate = torch.sigmoid((c - config.confidence_floor) / config.tau_confidence)
    if typicality is None:
        typicality = torch.ones(n, device=device)
    if liquidity is None:
        liquidity = torch.ones(n, device=device)
    omega = (g * c_gate * typicality * liquidity.to(device)).clamp(0.0, 1.0)
    return omega, disagreement


def _rolling_corr(returns_window: np.ndarray | None, n: int) -> np.ndarray:
    if returns_window is None or returns_window.shape[1] < 3:
        return np.eye(n, dtype=np.float64)
    arr = np.asarray(returns_window, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0)
    corr = np.corrcoef(arr)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)
