"""DeLong construction for correlated ROC areas.

Ref: Sec. 2.8, Supplementary Table SE1 (nonparametric DeLong construction rather than
the normal approximation).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

DEFAULT_LEVEL: float = 0.95
NO_ADVANTAGE_REFERENCE: float = 0.500


@dataclass(frozen=True)
class AreaEstimate:
    auc: float
    low: float
    high: float
    standard_error: float
    level: float

    def includes_no_advantage(self) -> bool:
        """An interval covering 0.500 is a retained null result.

        Ref: Sec. 2.2 (the 95 percent interval of the private arm contains the no-advantage
        reference).
        """
        return self.low <= NO_ADVANTAGE_REFERENCE <= self.high

    def to_dict(self) -> dict[str, object]:
        return {
            "auc": self.auc,
            "low": self.low,
            "high": self.high,
            "standard_error": self.standard_error,
            "level": self.level,
            "includes_no_advantage": self.includes_no_advantage(),
        }


def _placement_matrix(scores: Tensor, labels: Tensor) -> Tensor:
    positives = scores[labels > 0.5].to(torch.float64)
    negatives = scores[labels <= 0.5].to(torch.float64)
    if positives.numel() == 0 or negatives.numel() == 0:
        raise ValueError("both classes must be represented")
    comparisons = positives.unsqueeze(1) - negatives.unsqueeze(0)
    ties = (comparisons == 0).to(torch.float64)
    return (comparisons > 0).to(torch.float64) + 0.5 * ties


def _centred_covariance(first: Tensor, second: Tensor) -> float:
    first_centred = first - first.mean()
    second_centred = second - second.mean()
    denominator = max(1, int(first.numel()) - 1)
    return float((first_centred * second_centred).sum().item() / denominator)


def _structural_components(scores: Tensor, labels: Tensor) -> tuple[Tensor, Tensor, float]:
    placements = _placement_matrix(scores, labels)
    return placements.mean(dim=1), placements.mean(dim=0), float(placements.mean().item())


def auc(scores: Tensor, labels: Tensor) -> float:
    """Mann-Whitney estimate of the area under the ROC curve."""
    _v10, _v01, area = _structural_components(scores, labels)
    return area


def delong_variance(scores: Tensor, labels: Tensor) -> float:
    v10, v01, _area = _structural_components(scores, labels)
    s10 = _centred_covariance(v10, v10)
    s01 = _centred_covariance(v01, v01)
    return s10 / int(v10.numel()) + s01 / int(v01.numel())


def delong_interval(scores: Tensor, labels: Tensor, level: float = DEFAULT_LEVEL) -> AreaEstimate:
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")
    _v10, _v01, area = _structural_components(scores, labels)
    standard_error = delong_variance(scores, labels) ** 0.5
    z = float(torch.special.ndtri(torch.tensor(0.5 + level / 2.0)).item())
    return AreaEstimate(
        auc=area,
        low=max(0.0, area - z * standard_error),
        high=min(1.0, area + z * standard_error),
        standard_error=standard_error,
        level=level,
    )


def delong_paired_test(
    first_scores: Tensor, second_scores: Tensor, labels: Tensor, level: float = DEFAULT_LEVEL
) -> dict[str, float]:
    """Paired DeLong comparison of two areas on the same records.

    Ref: Supplementary Table SE1 (DeLong paired, z = 9.0, delta AUC 0.114).
    """
    if not (first_scores.shape == second_scores.shape == labels.shape):
        raise ValueError("paired scores and labels must align")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")
    first_v10, first_v01, first_area = _structural_components(first_scores, labels)
    second_v10, second_v01, second_area = _structural_components(second_scores, labels)
    variance = _centred_covariance(first_v10 - second_v10, first_v10 - second_v10) / int(
        first_v10.numel()
    ) + _centred_covariance(first_v01 - second_v01, first_v01 - second_v01) / int(first_v01.numel())
    if variance <= 0.0:
        raise ValueError("degenerate paired variance")
    difference = first_area - second_area
    z_statistic = difference / variance**0.5
    return {
        "delta_auc": difference,
        "z": z_statistic,
        "p_value": float(2.0 * (1.0 - _normal_cdf(abs(z_statistic)))),
        "level": level,
    }


def _normal_cdf(value: float) -> float:
    return float(0.5 * (1.0 + torch.erf(torch.tensor(value / 2.0**0.5, dtype=torch.float64)).item()))
