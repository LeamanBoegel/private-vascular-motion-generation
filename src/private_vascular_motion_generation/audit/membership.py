"""Membership-inference auditing of the released substrate.

Ref: Sec. 2.2, Sec. 4.9 (attack-based audit rather than a distance proxy).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..analysis.delong import AreaEstimate, delong_interval, delong_paired_test

REPORTED_CAPS_AUDIT: float = 0.503
REPORTED_NON_PRIVATE_AUDIT: float = 0.617


@dataclass(frozen=True)
class MembershipAudit:
    label: str
    estimate: AreaEstimate

    def includes_no_advantage(self) -> bool:
        return self.estimate.includes_no_advantage()

    def to_dict(self) -> dict[str, object]:
        payload = self.estimate.to_dict()
        payload["label"] = self.label
        return payload


def loss_scores(model: nn.Module, features: Tensor, targets: Tensor) -> Tensor:
    """Per-record negative log-likelihood used as the attack statistic."""
    if features.dim() != 2 or targets.dim() != 1:
        raise ValueError("features must be N x d and targets one-dimensional")
    logits = model(features)
    if logits.dim() != 1:
        logits = logits.squeeze(-1)
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("model output and targets must align")
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")


def frequency_calibrated_scores(scores: Tensor, features: Tensor, bins: int = 8) -> Tensor:
    """Frequency-calibrated attack statistic for medical diffusion models.

    Ref: Sec. 2.2 (the frequency-calibrated variant of the attack).
    """
    if scores.dim() != 1 or features.dim() != 2 or features.shape[0] != scores.shape[0]:
        raise ValueError("scores and features must align")
    if bins < 2:
        raise ValueError("at least two bins are required")
    magnitudes = features.norm(dim=1)
    edges = torch.quantile(magnitudes, torch.linspace(0.0, 1.0, bins + 1))
    calibrated = scores.clone()
    for index in range(bins):
        low = edges[index]
        high = edges[index + 1]
        if index == bins - 1:
            member = (magnitudes >= low) & (magnitudes <= high)
        else:
            member = (magnitudes >= low) & (magnitudes < high)
        if int(member.sum().item()) == 0:
            continue
        block = calibrated[member]
        spread = block.std(unbiased=False)
        centre = block.mean()
        if spread <= 0.0:
            calibrated[member] = block - centre
        else:
            calibrated[member] = (block - centre) / spread
    return calibrated


def member_nonmember_auc(member_scores: Tensor, nonmember_scores: Tensor) -> AreaEstimate:
    """AUC of the attack statistic, oriented so that no-advantage sits at 0.500."""
    if member_scores.dim() != 1 or nonmember_scores.dim() != 1:
        raise ValueError("score vectors must be one-dimensional")
    scores = torch.cat([member_scores, nonmember_scores])
    labels = torch.cat([torch.ones_like(member_scores), torch.zeros_like(nonmember_scores)])
    return delong_interval(scores, labels)


def audit_gap(non_private: AreaEstimate, private: AreaEstimate) -> dict[str, float]:
    """The non-private reference fails the no-advantage condition while the private one passes.

    Ref: Sec. 2.2.
    """
    return {
        "delta_auc": non_private.auc - private.auc,
        "non_private_includes_no_advantage": float(non_private.includes_no_advantage()),
        "private_includes_no_advantage": float(private.includes_no_advantage()),
    }


def paired_audit_test(member_scores: Tensor, nonmember_scores: Tensor, labels: Tensor) -> dict[str, float]:
    if labels.dim() != 1:
        raise ValueError("labels must be one-dimensional")
    if member_scores.shape != nonmember_scores.shape or member_scores.shape != labels.shape:
        raise ValueError("paired audit vectors must align")
    return delong_paired_test(member_scores, nonmember_scores, labels)


def audit_is_attack_based() -> dict[str, str]:
    """A distance proxy cannot serve as the main evidence.

    Ref: Sec. 2.2.
    """
    return {
        "primary_evidence": "membership-inference attack",
        "distance_proxy_role": "informative quantity only",
    }
