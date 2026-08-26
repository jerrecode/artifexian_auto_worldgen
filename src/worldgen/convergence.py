from __future__ import annotations

"""Convergence accounting for coupled reduced-order Earth-system iterations."""

from dataclasses import dataclass
import math


@dataclass(slots=True, frozen=True)
class ConvergenceThresholds:
    temperature_c: float
    precipitation_mm_year: float
    elevation_m: float | None = None
    required_consecutive: int = 1
    minimum_passes: int = 1

    def __post_init__(self) -> None:
        for name, value in (
            ("temperature_c", self.temperature_c),
            ("precipitation_mm_year", self.precipitation_mm_year),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.elevation_m is not None and (
            not math.isfinite(float(self.elevation_m)) or float(self.elevation_m) < 0.0
        ):
            raise ValueError("elevation_m must be finite and >= 0")
        if int(self.required_consecutive) < 1:
            raise ValueError("required_consecutive must be >= 1")
        if int(self.minimum_passes) < 1:
            raise ValueError("minimum_passes must be >= 1")


@dataclass(slots=True, frozen=True)
class ConvergenceDecision:
    pass_number: int
    eligible: bool
    metrics_available: bool
    converged: bool
    stop: bool
    consecutive_converged: int
    normalized_residual: float
    temperature_ratio: float | None
    precipitation_ratio: float | None
    elevation_ratio: float | None
    reason: str


def _ratio(value: float | None, threshold: float | None) -> float | None:
    if value is None or threshold is None:
        return None
    v = abs(float(value))
    t = float(threshold)
    if not math.isfinite(v):
        return math.inf
    if t > 0.0:
        return v / t
    return 0.0 if v == 0.0 else math.inf


class ConvergenceTracker:
    """Track physical residuals without depending on simulation implementation.

    Pass numbering is one-based. A pass is eligible only once ``minimum_passes``
    have executed and it ran at full fidelity. This prevents low-cost predictor
    passes from terminating an adaptive run before the full numerical model has
    actually been evaluated.
    """

    def __init__(self, thresholds: ConvergenceThresholds):
        self.thresholds = thresholds
        self.consecutive_converged = 0
        self.history: list[ConvergenceDecision] = []

    def evaluate(
        self,
        pass_number: int,
        *,
        temperature_change_c: float | None,
        precipitation_change_mm_year: float | None,
        elevation_change_m: float | None = None,
        full_fidelity: bool = True,
    ) -> ConvergenceDecision:
        pass_number = int(pass_number)
        if pass_number < 1:
            raise ValueError("pass_number must be >= 1")
        t_ratio = _ratio(temperature_change_c, self.thresholds.temperature_c)
        p_ratio = _ratio(
            precipitation_change_mm_year,
            self.thresholds.precipitation_mm_year,
        )
        e_ratio = _ratio(elevation_change_m, self.thresholds.elevation_m)

        required = [t_ratio, p_ratio]
        if self.thresholds.elevation_m is not None:
            required.append(e_ratio)
        metrics_available = all(value is not None for value in required)
        ratios = [float(value) for value in required if value is not None]
        normalized = max(ratios) if ratios else math.inf
        eligible = bool(full_fidelity and pass_number >= self.thresholds.minimum_passes)
        converged = bool(metrics_available and normalized <= 1.0)

        if eligible and converged:
            self.consecutive_converged += 1
        elif eligible:
            self.consecutive_converged = 0

        stop = bool(
            eligible
            and converged
            and self.consecutive_converged >= self.thresholds.required_consecutive
        )
        if not metrics_available:
            reason = "insufficient_history"
        elif not full_fidelity:
            reason = "predictor_pass"
        elif pass_number < self.thresholds.minimum_passes:
            reason = "minimum_passes"
        elif stop:
            reason = "converged"
        elif converged:
            reason = "awaiting_consecutive_passes"
        else:
            reason = "residual_above_tolerance"

        decision = ConvergenceDecision(
            pass_number=pass_number,
            eligible=eligible,
            metrics_available=metrics_available,
            converged=converged,
            stop=stop,
            consecutive_converged=int(self.consecutive_converged),
            normalized_residual=float(normalized),
            temperature_ratio=t_ratio,
            precipitation_ratio=p_ratio,
            elevation_ratio=e_ratio,
            reason=reason,
        )
        self.history.append(decision)
        return decision


__all__ = [
    "ConvergenceThresholds",
    "ConvergenceDecision",
    "ConvergenceTracker",
]
