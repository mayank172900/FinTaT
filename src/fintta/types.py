from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass(slots=True)
class AssetBatch:
    """One timestamp cross-section.

    `returns_window` must end at the current timestamp and must not include
    future returns. Optional labels are reserved for offline evaluation only.
    """

    x: torch.Tensor
    asset_ids: list[str]
    metadata: dict[str, list[Any]] = field(default_factory=dict)
    returns_window: np.ndarray | None = None
    liquidity: torch.Tensor | None = None
    factor_exposures: np.ndarray | None = None
    market_state: np.ndarray | None = None
    labels: torch.Tensor | None = None
    forward_returns: np.ndarray | None = None

    def to(self, device: torch.device | str) -> "AssetBatch":
        return AssetBatch(
            x=self.x.to(device),
            asset_ids=self.asset_ids,
            metadata=self.metadata,
            returns_window=self.returns_window,
            liquidity=None if self.liquidity is None else self.liquidity.to(device),
            factor_exposures=self.factor_exposures,
            market_state=self.market_state,
            labels=None if self.labels is None else self.labels.to(device),
            forward_returns=self.forward_returns,
        )

    @property
    def n_assets(self) -> int:
        return int(self.x.shape[0])
