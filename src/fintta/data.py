from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .types import AssetBatch

REQUIRED_PANEL_COLUMNS = {"date", "asset_id", "ret_1d", "volume", "sector", "industry"}


@dataclass(slots=True)
class PanelSpec:
    feature_columns: list[str]
    label_column: str | None = None
    forward_return_column: str | None = None
    factor_columns: list[str] | None = None
    market_state_columns: list[str] | None = None
    liquidity_column: str | None = None
    lookback: int = 20


class PanelDataset:
    """Chronological panel wrapper for CRSP-like or vendor-normalized data.

    The loader expects one row per `(date, asset_id)`, with feature columns that
    are already as-of safe. Labels and forward returns are carried only for
    offline evaluation and are never read by `FinTTAEngine.step`.
    """

    def __init__(self, frame: pd.DataFrame, spec: PanelSpec) -> None:
        missing = REQUIRED_PANEL_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"panel is missing required columns: {sorted(missing)}")
        self.frame = frame.copy()
        self.frame["asset_id"] = self.frame["asset_id"].astype(str)
        self.frame = self.frame.sort_values(["date", "asset_id"]).reset_index(drop=True)
        self.spec = spec
        self._validate()

    @classmethod
    def from_path(cls, path: str | Path, spec: PanelSpec) -> PanelDataset:
        path = Path(path)
        if path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame["date"])
        return cls(frame, spec)

    def iter_batches(self, start: str | None = None, end: str | None = None) -> Iterable[AssetBatch]:
        frame = self.frame
        if start:
            frame = frame[frame["date"] >= pd.Timestamp(start)]
        if end:
            frame = frame[frame["date"] <= pd.Timestamp(end)]
        dates = list(frame["date"].drop_duplicates())
        returns_pivot = self.frame.pivot(index="asset_id", columns="date", values="ret_1d").sort_index(axis=1)
        for date in dates:
            day = frame[frame["date"] == date].copy()
            day = day.sort_values("asset_id")
            asset_ids = day["asset_id"].astype(str).tolist()
            x = torch.tensor(day[self.spec.feature_columns].to_numpy(np.float32), dtype=torch.float32)
            labels = None
            if self.spec.label_column and self.spec.label_column in day:
                labels = torch.tensor(day[self.spec.label_column].to_numpy(np.int64), dtype=torch.long)
            fwd = None
            if self.spec.forward_return_column and self.spec.forward_return_column in day:
                fwd = day[self.spec.forward_return_column].to_numpy(np.float64)
            liq = None
            if self.spec.liquidity_column and self.spec.liquidity_column in day:
                liq = torch.tensor(day[self.spec.liquidity_column].to_numpy(np.float32), dtype=torch.float32).clamp(0, 1)
            factors = None
            if self.spec.factor_columns:
                factors = day[self.spec.factor_columns].to_numpy(np.float64)
            market_state = None
            if self.spec.market_state_columns:
                market_state = day[self.spec.market_state_columns].median(numeric_only=True).to_numpy(np.float64)
            window_dates = returns_pivot.columns[returns_pivot.columns <= date][-self.spec.lookback :]
            returns_window = returns_pivot.loc[asset_ids, window_dates].to_numpy(np.float64)
            metadata = {
                "sector": day["sector"].astype(str).tolist(),
                "industry": day["industry"].astype(str).tolist(),
            }
            yield AssetBatch(
                x=x,
                asset_ids=asset_ids,
                metadata=metadata,
                returns_window=returns_window,
                liquidity=liq,
                factor_exposures=factors,
                market_state=market_state,
                labels=labels,
                forward_returns=fwd,
            )

    def _validate(self) -> None:
        if self.frame["date"].is_monotonic_increasing is False:
            raise ValueError("panel dates must be sortable and chronological")
        duplicate = self.frame.duplicated(["date", "asset_id"]).any()
        if duplicate:
            raise ValueError("panel contains duplicate (date, asset_id) rows")
        for col in self.spec.feature_columns:
            if col not in self.frame:
                raise ValueError(f"missing feature column: {col}")
        for col in self.spec.factor_columns or []:
            if col not in self.frame:
                raise ValueError(f"missing factor column: {col}")
        for col in self.spec.market_state_columns or []:
            if col not in self.frame:
                raise ValueError(f"missing market-state column: {col}")


@dataclass(slots=True)
class SyntheticMarket:
    source_batches: list[AssetBatch]
    test_batches: list[AssetBatch]
    input_dim: int
    num_classes: int


def make_synthetic_market(
    *,
    n_assets: int = 64,
    source_days: int = 120,
    test_days: int = 160,
    lookback: int = 20,
    input_dim: int = 16,
    num_classes: int = 5,
    seed: int = 7,
) -> SyntheticMarket:
    """Deterministic nonstationary panel for CI and examples.

    It deliberately has abrupt volatility/correlation regimes and sector
    communities so FinTTA's control flow can be verified without licensed data.
    """

    if input_dim < 10:
        raise ValueError("input_dim must be at least 10 for the core synthetic market features")

    rng = np.random.default_rng(seed)
    sectors = np.array(["tech", "finance", "energy", "health"])
    industries = np.array(["large", "small"])
    asset_sector = sectors[np.arange(n_assets) % len(sectors)]
    asset_industry = np.array([f"{s}_{industries[i % 2]}" for i, s in enumerate(asset_sector)])
    loadings = rng.normal(0, 0.5, size=(n_assets, 4))
    for i, s in enumerate(asset_sector):
        loadings[i, list(sectors).index(s)] += 1.2
    total_days = source_days + test_days + lookback + 2
    returns = np.zeros((n_assets, total_days), dtype=np.float64)
    features = np.zeros((n_assets, total_days, input_dim), dtype=np.float32)
    regimes = []
    for t in range(total_days):
        if t < source_days + lookback:
            regime = 0
            vol, market, crowd = 0.010, 0.0008, 0.25
            sector_alpha = np.array([0.0010, 0.0005, -0.0001, 0.0003])
        elif t < source_days + lookback + test_days * 0.25:
            regime = 2
            vol, market, crowd = 0.035, -0.0035, 0.85
            sector_alpha = np.array([-0.003, -0.004, -0.001, -0.002])
        elif t < source_days + lookback + test_days * 0.55:
            regime = 3
            vol, market, crowd = 0.020, 0.0022, 0.45
            sector_alpha = np.array([0.002, 0.001, 0.000, 0.001])
        elif t < source_days + lookback + test_days * 0.78:
            regime = 4
            vol, market, crowd = 0.024, -0.0010, 0.65
            sector_alpha = np.array([-0.002, 0.0015, 0.0020, -0.0005])
        else:
            regime = 6
            vol, market, crowd = 0.016, 0.0015, 0.35
            sector_alpha = np.array([0.003, -0.0005, 0.0005, 0.0002])
        regimes.append(regime)
        common = rng.normal(market, vol * crowd)
        sector_noise = rng.normal(0, vol, size=4)
        eps = rng.normal(0, vol * (1.0 - 0.4 * crowd), size=n_assets)
        for i in range(n_assets):
            sid = list(sectors).index(asset_sector[i])
            returns[i, t] = common + 0.45 * sector_noise[sid] + sector_alpha[sid] + eps[i]
        if t >= 1:
            lag = returns[:, max(0, t - lookback) : t]
            rv = np.sqrt(np.mean(lag * lag, axis=1) + 1e-6)
            mom = lag[:, -5:].sum(axis=1) if lag.shape[1] >= 5 else lag.sum(axis=1)
            draw = np.minimum.accumulate(lag[:, -min(lookback, lag.shape[1]) :], axis=1).min(axis=1)
            features[:, t, 0] = mom
            features[:, t, 1] = rv
            features[:, t, 2] = draw
            features[:, t, 3:7] = loadings
            features[:, t, 7] = common
            features[:, t, 8] = vol
            features[:, t, 9] = crowd
            features[:, t, 10:] = rng.normal(0, 0.1, size=(n_assets, input_dim - 10))
    labels = _future_bucket_labels(returns, horizon=1, num_classes=num_classes, calibration_end=source_days + lookback)
    batches: list[AssetBatch] = []
    for t in range(lookback, total_days - 1):
        start = t - lookback + 1
        ret_window = returns[:, start : t + 1]
        rv = np.sqrt(np.mean(ret_window * ret_window, axis=1))
        liquidity = torch.tensor(np.clip(1.0 - 10.0 * rv + rng.normal(0, 0.03, size=n_assets), 0.1, 1.0), dtype=torch.float32)
        corr = np.corrcoef(ret_window)
        upper = corr[np.triu_indices_from(corr, k=1)]
        vals = np.linalg.eigvalsh(np.nan_to_num(corr, nan=0.0))
        market_state = np.array(
            [
                np.log(np.median(rv) + 1e-6),
                np.median(np.abs(returns[:, t] - np.median(returns[:, t]))),
                np.nanmean(np.abs(upper)),
                vals[-1] / max(vals.sum(), 1e-6),
            ],
            dtype=np.float64,
        )
        batches.append(
            AssetBatch(
                x=torch.tensor(features[:, t, :], dtype=torch.float32),
                asset_ids=[f"A{i:04d}" for i in range(n_assets)],
                metadata={"sector": asset_sector.tolist(), "industry": asset_industry.tolist()},
                returns_window=ret_window,
                liquidity=liquidity,
                factor_exposures=loadings,
                market_state=market_state,
                labels=torch.tensor(labels[:, t], dtype=torch.long),
                forward_returns=returns[:, t + 1],
            )
        )
    source_batches = batches[:source_days]
    test_batches = batches[source_days : source_days + test_days]
    return SyntheticMarket(source_batches, test_batches, input_dim, num_classes)


def source_training_tensors(batches: list[AssetBatch]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.cat([b.x for b in batches], dim=0)
    y = torch.cat([b.labels for b in batches if b.labels is not None], dim=0)
    return x, y


def _future_bucket_labels(
    returns: np.ndarray,
    horizon: int,
    num_classes: int,
    calibration_end: int | None = None,
) -> np.ndarray:
    fwd = np.roll(returns, -horizon, axis=1)
    labels = np.zeros_like(returns, dtype=np.int64)
    valid = fwd[:, :-horizon]
    if calibration_end is not None:
        end = max(1, min(calibration_end, valid.shape[1]))
        calibration = valid[:, :end]
    else:
        calibration = valid
    cuts = np.quantile(calibration.ravel(), np.linspace(0, 1, num_classes + 1)[1:-1])
    labels[:, :-horizon] = np.digitize(fwd[:, :-horizon], cuts)
    labels[:, -horizon:] = num_classes // 2
    return labels
