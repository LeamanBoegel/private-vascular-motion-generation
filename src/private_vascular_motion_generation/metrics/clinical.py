"""Clinical endpoints, reader validity and interval construction.

Ref: Sec. 4.10, Table 5, Table 6, Supplementary Table SE1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ProportionEstimate:
    successes: int
    denominator: int
    point: float
    low: float
    high: float

    def to_dict(self) -> dict[str, object]:
        return {
            "successes": self.successes,
            "denominator": self.denominator,
            "point": self.point,
            "low": self.low,
            "high": self.high,
        }


def wilson_interval(successes: int, denominator: int, level: float = 0.95) -> ProportionEstimate:
    """Wilson score interval; every reported proportion carries its own denominator.

    Ref: Table 6 caption.
    """
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if not 0 <= successes <= denominator:
        raise ValueError("successes must lie within the denominator")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")
    z = float(torch.special.ndtri(torch.tensor(0.5 + level / 2.0)).item())
    point = successes / denominator
    denominator_term = 1.0 + z**2 / denominator
    centre = (point + z**2 / (2.0 * denominator)) / denominator_term
    half = z * math.sqrt(point * (1.0 - point) / denominator + z**2 / (4.0 * denominator**2)) / denominator_term
    return ProportionEstimate(
        successes=successes,
        denominator=denominator,
        point=point,
        low=max(0.0, centre - half),
        high=min(1.0, centre + half),
    )


def task_success(flags: Tensor, level: float = 0.95) -> ProportionEstimate:
    if flags.dim() != 1 or flags.numel() == 0:
        raise ValueError("success flags must be a non-empty vector")
    successes = int((flags > 0.5).sum().item())
    return wilson_interval(successes, int(flags.numel()), level)


def cohen_kappa(first: Tensor, second: Tensor, categories: int) -> float:
    """Chance-corrected agreement between two raters over a categorical scale.

    Ref: Sec. 4.10, Supplementary Table SE1 (inter-reader kappa).
    """
    if first.shape != second.shape or first.dim() != 1:
        raise ValueError("ratings must be aligned one-dimensional vectors")
    if categories < 2:
        raise ValueError("at least two categories are required")
    total = int(first.numel())
    if total == 0:
        raise ValueError("empty rating vector")
    observed = float((first == second).to(torch.float64).mean().item())
    expected = 0.0
    for category in range(categories):
        first_share = float((first == category).to(torch.float64).mean().item())
        second_share = float((second == category).to(torch.float64).mean().item())
        expected += first_share * second_share
    if expected >= 1.0:
        raise ValueError("degenerate marginals leave no chance correction")
    return (observed - expected) / (1.0 - expected)


def discrimination_accuracy(scores: Tensor, labels: Tensor, threshold: float = 0.5) -> float:
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must align")
    predicted = (scores >= threshold).to(torch.float64)
    return float((predicted == labels.to(torch.float64)).to(torch.float64).mean().item())


def three_way_reader_contrast(
    clinician_scores: Tensor, model_scores: Tensor, fused_scores: Tensor, labels: Tensor
) -> dict[str, float]:
    """Clinician alone, model alone and clinician with model.

    Ref: Sec. 4.10.
    """
    if not (clinician_scores.shape == model_scores.shape == fused_scores.shape == labels.shape):
        raise ValueError("all four vectors must share a shape")
    return {
        "clinician_alone": discrimination_accuracy(clinician_scores, labels),
        "model_alone": discrimination_accuracy(model_scores, labels),
        "clinician_with_model": discrimination_accuracy(fused_scores, labels),
    }


def reader_agreement_gain(kappa_alone: float, kappa_with_model: float) -> float:
    return kappa_with_model - kappa_alone
