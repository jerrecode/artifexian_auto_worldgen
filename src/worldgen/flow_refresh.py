from __future__ import annotations

"""Deterministic policy for deciding when drainage topology must be rebuilt."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

FlowRefreshMode = Literal["interval", "adaptive", "hybrid"]


@dataclass(slots=True, frozen=True)
class FlowRefreshDecision:
    refresh: bool
    reason: str
    iterations_since_refresh: int
    elevation_change_m: float
    land_change_fraction: float
    previous_delta_max_m: float


@dataclass(slots=True)
class FlowRefreshState:
    """Stateful but deterministic drainage-refresh tracker.

    The baseline is updated only after a receiver graph is rebuilt.  Terrain change
    is measured against that baseline, not merely against the previous iteration,
    so many small geomorphic updates cannot accumulate indefinitely without a
    refresh.
    """

    baseline_elevation_km: np.ndarray | None = None
    baseline_land: np.ndarray | None = None
    last_refresh_iteration: int = -1

    def mark_refreshed(self, iteration: int, elevation_km: np.ndarray, land: np.ndarray) -> None:
        self.baseline_elevation_km = np.asarray(elevation_km, dtype=np.float32).copy()
        self.baseline_land = np.asarray(land, dtype=bool).copy()
        self.last_refresh_iteration = int(iteration)


def _weighted_fraction(mask: np.ndarray, weights: np.ndarray | None) -> float:
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return 0.0
    if weights is None:
        return float(np.mean(m))
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != m.shape:
        raise ValueError("weights must match terrain shape")
    total = float(np.sum(w))
    if total <= 0.0:
        return float(np.mean(m))
    return float(np.sum(w[m]) / total)


def _robust_elevation_change_m(
    current_km: np.ndarray,
    baseline_km: np.ndarray,
    *,
    quantile: float = 0.995,
) -> float:
    current = np.asarray(current_km, dtype=np.float64)
    baseline = np.asarray(baseline_km, dtype=np.float64)
    if current.shape != baseline.shape:
        raise ValueError("baseline elevation shape changed")
    delta_m = np.abs(current - baseline).ravel() * 1000.0
    if not delta_m.size:
        return 0.0
    return float(np.quantile(delta_m, float(np.clip(quantile, 0.5, 1.0))))


def decide_flow_refresh(
    state: FlowRefreshState,
    *,
    iteration: int,
    elevation_km: np.ndarray,
    land: np.ndarray,
    mode: FlowRefreshMode = "interval",
    interval: int = 1,
    max_interval: int = 3,
    elevation_threshold_m: float = 4.0,
    land_change_fraction_threshold: float = 2.0e-4,
    delta_threshold_m: float = 2.0,
    previous_delta_max_m: float = 0.0,
    area_weights: np.ndarray | None = None,
) -> FlowRefreshDecision:
    """Return whether a filled terrain/receiver graph should be recomputed.

    ``interval`` exactly represents the legacy fixed-interval policy. ``adaptive``
    reacts to physical terrain/coast/delta change and a deterministic maximum reuse
    interval. ``hybrid`` adds the legacy interval trigger to the adaptive checks.
    """
    mode = str(mode).strip().lower()  # type: ignore[assignment]
    if mode not in {"interval", "adaptive", "hybrid"}:
        raise ValueError("flow refresh mode must be interval, adaptive, or hybrid")
    interval = max(1, int(interval))
    max_interval = max(1, int(max_interval))
    iteration = int(iteration)
    z = np.asarray(elevation_km)
    lm = np.asarray(land, dtype=bool)
    if z.shape != lm.shape:
        raise ValueError("elevation and land masks must have equal shape")

    if state.baseline_elevation_km is None or state.baseline_land is None or state.last_refresh_iteration < 0:
        return FlowRefreshDecision(True, "initial", 0, 0.0, 0.0, float(previous_delta_max_m))

    since = max(0, iteration - int(state.last_refresh_iteration))
    elev_change = _robust_elevation_change_m(z, state.baseline_elevation_km)
    land_change = _weighted_fraction(lm != state.baseline_land, area_weights)
    delta_max = max(0.0, float(previous_delta_max_m))

    if mode == "interval":
        refresh = iteration % interval == 0
        return FlowRefreshDecision(
            refresh,
            "fixed_interval" if refresh else "reuse",
            since,
            elev_change,
            land_change,
            delta_max,
        )

    # Hard upper bound protects against slow accumulated changes below every local
    # threshold and makes runtime cost predictable.
    if since >= max_interval:
        return FlowRefreshDecision(True, "max_interval", since, elev_change, land_change, delta_max)
    if elev_change >= max(0.0, float(elevation_threshold_m)):
        return FlowRefreshDecision(True, "elevation_change", since, elev_change, land_change, delta_max)
    if land_change >= max(0.0, float(land_change_fraction_threshold)):
        return FlowRefreshDecision(True, "coastline_change", since, elev_change, land_change, delta_max)
    if delta_max >= max(0.0, float(delta_threshold_m)):
        return FlowRefreshDecision(True, "delta_aggradation", since, elev_change, land_change, delta_max)
    if mode == "hybrid" and iteration % interval == 0:
        return FlowRefreshDecision(True, "fixed_interval", since, elev_change, land_change, delta_max)
    return FlowRefreshDecision(False, "reuse", since, elev_change, land_change, delta_max)


__all__ = [
    "FlowRefreshMode",
    "FlowRefreshDecision",
    "FlowRefreshState",
    "decide_flow_refresh",
]
