"""Run-to-run dispersion of the reported mechanism.

Ref: Sec. 2.8 (coefficient of variation of the private mechanism), Sec. 4.9 (twenty
independent initialisations).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

REPORTED_CV_FID: float = 0.0127
REPORTED_CV_DICE: float = 0.0067
REPORTED_CV_DTW: float = 0.0545
INITIALISATIONS: int = 20


@dataclass(frozen=True)
class Dispersion:
    metric: str
    mean: float
    standard_deviation: float

    def coefficient_of_variation(self) -> float:
        if self.mean == 0.0:
            raise ValueError("mean must be non-zero")
        return abs(self.standard_deviation / self.mean)

    def to_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "coefficient_of_variation": self.coefficient_of_variation(),
        }


def dispersion(metric: str, values: Tensor) -> Dispersion:
    if values.dim() != 1 or values.numel() < 2:
        raise ValueError("dispersion needs at least two runs")
    return Dispersion(
        metric=metric,
        mean=float(values.to(torch.float64).mean().item()),
        standard_deviation=float(values.to(torch.float64).std(unbiased=True).item()),
    )


def dispersion_table(columns: dict[str, Tensor]) -> dict[str, dict[str, float]]:
    return {metric: dispersion(metric, values).to_dict() for metric, values in columns.items()}


def worst_initialisation(values: Tensor) -> float:
    """The worst-performing arm is reported as a bound, not averaged into the headline."""
    if values.dim() != 1 or values.numel() < 2:
        raise ValueError("at least two runs are required")
    return float(values.to(torch.float64).max().item())


def beats_unconstrained(worst: float, unconstrained_best_mean: float) -> bool:
    """The weakest private initialisation still sits inside the parity band.

    Ref: Sec. 2.8 (46.2 against 44.9, outside the 0.5-1.0 absolute band).
    """
    return worst <= unconstrained_best_mean


def parity_band_verdict(
    delta: float, absolute_band: tuple[float, float], relative_band: float, reference: float | None = None
) -> str:
    """A difference is a parity result when it sits inside either declared band.

    Ref: Sec. 2.1 (the parity band is absolute and 0.5-1.0 percent relative).
    """
    low, high = absolute_band
    if low <= abs(delta) <= high:
        return "inside_parity_band"
    if reference is not None:
        if reference == 0.0:
            raise ValueError("relative band needs a non-zero reference")
        if abs(delta) / abs(reference) <= relative_band:
            return "inside_parity_band"
    return "outside_parity_band"


def coefficient_of_variation_verdict(measured: float, reported: float, tolerance: float = 0.005) -> str:
    if abs(measured - reported) <= tolerance:
        return "within_tolerance"
    return "outside_tolerance"


def run_to_run_floor(values: Tensor, multiple: float = 1.0) -> float:
    return multiple * float(values.to(torch.float64).std(unbiased=True).item())
