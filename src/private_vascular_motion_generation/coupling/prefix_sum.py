"""Prefix-structured procedure windows and their prefix-sum factorisation.

Ref: Eq. (2), Sec. 4.4, Sec. 4.6, Supplementary Table T7 panel D.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

WINDOW_LADDER: tuple[int, ...] = (4, 8, 16, 32, 64, 128)
REPORTED_COMPOSITION_MULTIPLIER: tuple[float, ...] = (1.42, 1.68, 1.89, 2.07, 2.21, 2.33)
REPORTED_RELATIVE_LATENCY: tuple[float, ...] = (1.10, 1.19, 1.28, 1.37, 1.45, 1.53)
REPORTED_TRUNCATION_RATE: tuple[float, ...] = (0.0, 0.0, 0.2, 0.6, 1.1, 2.4)
REPORTED_CONTACT_F1: tuple[float, ...] = (0.828, 0.832, 0.835, 0.836, 0.834, 0.826)
DEFAULT_WINDOW_LENGTH: int = 16


@dataclass(frozen=True)
class WindowSpec:
    length: int
    micro_steps: int
    uniform: bool = True
    truncated: bool = False

    def __post_init__(self) -> None:
        if self.length < 1 or self.micro_steps < 1:
            raise ValueError("window length and micro-step count must be positive")
        if self.micro_steps % self.length != 0:
            raise ValueError("micro-steps must tile the window uniformly")

    @property
    def stride(self) -> int:
        return self.micro_steps // self.length

    def structural_condition_holds(self) -> bool:
        """The structural condition is a uniform, untruncated prefix window.

        Ref: Sec. 4.6.
        """
        return self.uniform and not self.truncated


@dataclass(frozen=True)
class PrefixFactorisation:
    window: WindowSpec
    group_multiplier: float

    def single_step_charge(self) -> float:
        return 1.0

    def composed_charge(self) -> float:
        return self.group_multiplier

    def saving(self) -> float:
        return 1.0 - 1.0 / self.group_multiplier


def prefix_sum_scan(micro_steps: Tensor) -> Tensor:
    """Exclusive prefix sums over the window axis, the factorisation the window uses.

    Ref: Eq. (2), Sec. 4.4.
    """
    if micro_steps.dim() < 1:
        raise ValueError("micro-steps must carry at least one axis")
    return torch.cumsum(micro_steps, dim=0)


def prefix_sum_inverse(scanned: Tensor) -> Tensor:
    if scanned.dim() < 1:
        raise ValueError("scanned tensor must carry at least one axis")
    first = scanned.select(0, 0).unsqueeze(0)
    rest = scanned[1:] - scanned[:-1]
    return torch.cat([first, rest], dim=0)


def windowed_workload(procedure_gradients: Tensor, window: WindowSpec) -> Tensor:
    """Reshape the strategy matrix so rows are procedures and columns are window micro-steps.

    Ref: Algorithm 1 step 10, Eq. (2).
    """
    if procedure_gradients.dim() != 2:
        raise ValueError("procedure gradients must be P x d")
    rows = int(procedure_gradients.shape[0])
    tiled = procedure_gradients.unsqueeze(1).expand(rows, window.length, procedure_gradients.shape[1])
    return tiled.reshape(rows, window.length * procedure_gradients.shape[1])


def group_multiplier(window: WindowSpec, per_step_multiplier: float = 0.180571) -> float:
    """Polylogarithmic composition cost of a prefix-structured window.

    Ref: Sec. 4.6 (logarithmic, not linear, composition), Supplementary Table T7 panel D.
    """
    if per_step_multiplier <= 0.0:
        raise ValueError("per-step multiplier must be positive")
    return 1.0 + per_step_multiplier * float(torch.log2(torch.tensor(float(window.length))).item())


def truncation_rate(window_lengths: Tensor, aligned_span: int) -> float:
    """Fraction of windows whose length exceeds the aligned span.

    Ref: Supplementary Table T7 panel D.
    """
    if window_lengths.dim() != 1 or window_lengths.numel() == 0:
        raise ValueError("window lengths must be a non-empty vector")
    if aligned_span <= 0:
        raise ValueError("aligned span must be positive")
    return float((window_lengths > aligned_span).to(torch.float64).mean().item())


def fit_log_ladder(window_lengths: tuple[int, ...], values: tuple[float, ...]) -> dict[str, float]:
    """Least-squares fit of a reported ladder against log2 of the window length."""
    if len(window_lengths) != len(values) or len(values) < 3:
        raise ValueError("the ladder needs at least three paired rungs")
    if window_lengths != tuple(sorted(window_lengths)):
        raise ValueError("window lengths must be increasing")
    x = torch.tensor([float(torch.log2(torch.tensor(float(k))).item()) for k in window_lengths], dtype=torch.float64)
    y = torch.tensor(values, dtype=torch.float64)
    design = torch.stack([x, torch.ones_like(x)], dim=1)
    solution = torch.linalg.lstsq(design, y.unsqueeze(1)).solution.squeeze(1)
    prediction = design @ solution
    residual = y - prediction
    total = float(((y - y.mean()) ** 2).sum().item())
    r_squared = 1.0 - float((residual**2).sum().item()) / total
    return {
        "slope": float(solution[0].item()),
        "intercept": float(solution[1].item()),
        "r_squared": r_squared,
        "max_abs_residual": float(residual.abs().max().item()),
    }


def composition_multiplier_ladder() -> dict[str, object]:
    """Monotonicity and polylogarithmic shape of the reported window ladder."""
    costs = torch.tensor(REPORTED_COMPOSITION_MULTIPLIER, dtype=torch.float64)
    latency = torch.tensor(REPORTED_RELATIVE_LATENCY, dtype=torch.float64)
    truncation = torch.tensor(REPORTED_TRUNCATION_RATE, dtype=torch.float64)
    monotone_costs = bool(torch.all(costs[1:] >= costs[:-1]).item())
    monotone_latency = bool(torch.all(latency[1:] >= latency[:-1]).item())
    monotone_truncation = bool(torch.all(truncation[1:] >= truncation[:-1]).item())
    return {
        "cost_ladder_monotone": monotone_costs,
        "latency_ladder_monotone": monotone_latency,
        "truncation_ladder_monotone": monotone_truncation,
        "cost_fit": fit_log_ladder(WINDOW_LADDER, REPORTED_COMPOSITION_MULTIPLIER),
        "latency_fit": fit_log_ladder(WINDOW_LADDER, REPORTED_RELATIVE_LATENCY),
    }
