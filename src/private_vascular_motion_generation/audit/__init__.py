"""Attack-based privacy auditing of the released substrate.

Ref: Sec. 2.2, Table 2.
"""

from __future__ import annotations

from ..analysis.delong import NO_ADVANTAGE_REFERENCE
from .distance import (
    dcr_against_marginals,
    dcr_summary,
    distance_to_closest_record,
    duplicate_fraction,
)
from .membership import (
    REPORTED_CAPS_AUDIT,
    REPORTED_NON_PRIVATE_AUDIT,
    MembershipAudit,
    audit_gap,
    audit_is_attack_based,
    frequency_calibrated_scores,
    loss_scores,
    member_nonmember_auc,
    paired_audit_test,
)

__all__ = [
    "NO_ADVANTAGE_REFERENCE",
    "REPORTED_CAPS_AUDIT",
    "REPORTED_NON_PRIVATE_AUDIT",
    "MembershipAudit",
    "audit_gap",
    "audit_is_attack_based",
    "dcr_against_marginals",
    "dcr_summary",
    "distance_to_closest_record",
    "duplicate_fraction",
    "frequency_calibrated_scores",
    "loss_scores",
    "member_nonmember_auc",
    "paired_audit_test",
]
