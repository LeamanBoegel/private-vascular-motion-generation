"""Pairwise component interactions of the ablation grid.

Ref: Table 4 panel (b), Sec. 2.5, Sec. 2.9.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

SUPER_ADDITIVE_THRESHOLD: float = 1.0
PREREQUISITE_BAND: tuple[float, float] = (0.98, 1.02)


class PairwiseRow(TypedDict):
    left: str
    right: str
    drop_both: float
    drop_left_plus_right: float
    interaction_ratio: float
    relation: str


class PairwiseReport(TypedDict, total=False):
    rows: list[PairwiseRow]
    all_super_additive: bool
    max_interaction_ratio: float
    max_pair: str
    prerequisite_pairs: list[PairwiseRow]
    prerequisites_are_not_synergy: bool


@dataclass(frozen=True)
class PairwiseInteraction:
    left: str
    right: str
    drop_both: float
    drop_left_plus_right: float

    def interaction_ratio(self) -> float:
        """IR = dAB / (dA + dB); above one is super-additive.

        Ref: Table 4 caption.
        """
        if self.drop_left_plus_right <= 0.0:
            raise ValueError("the additive denominator must be positive")
        return self.drop_both / self.drop_left_plus_right

    def relation(self) -> str:
        ratio = self.interaction_ratio()
        if PREREQUISITE_BAND[0] <= ratio <= PREREQUISITE_BAND[1]:
            return "prerequisite, not synergy"
        if ratio > SUPER_ADDITIVE_THRESHOLD:
            return "super-additive"
        return "sub-additive"

    def to_dict(self) -> PairwiseRow:
        return {
            "left": self.left,
            "right": self.right,
            "drop_both": self.drop_both,
            "drop_left_plus_right": self.drop_left_plus_right,
            "interaction_ratio": self.interaction_ratio(),
            "relation": self.relation(),
        }


def component_drop(full_value: float, ablated_value: float) -> float:
    """The larger the drop in contact-event F1, the more the component carries."""
    return full_value - ablated_value


def pairwise_from_full(
    full_contact_f1: float,
    left_ablated: float,
    right_ablated: float,
    both_ablated: float,
    left_label: str,
    right_label: str,
) -> PairwiseInteraction:
    return PairwiseInteraction(
        left=left_label,
        right=right_label,
        drop_both=component_drop(full_contact_f1, both_ablated),
        drop_left_plus_right=component_drop(full_contact_f1, left_ablated)
        + component_drop(full_contact_f1, right_ablated),
    )


def report_pairwise(interactions: list[PairwiseInteraction]) -> PairwiseReport:
    if not interactions:
        raise ValueError("no pairwise interaction supplied")
    return {
        "rows": [interaction.to_dict() for interaction in interactions],
        "all_super_additive": all(interaction.relation() == "super-additive" for interaction in interactions),
        "max_interaction_ratio": max(interaction.interaction_ratio() for interaction in interactions),
        "max_pair": max(interactions, key=lambda item: item.interaction_ratio()).left,
    }


def largest_pair_is_design_pair(interactions: list[PairwiseInteraction], expected: tuple[str, str]) -> bool:
    """The shaped-noise factorial is the largest interaction of the design.

    Ref: Sec. 2.5.
    """
    if not interactions:
        raise ValueError("no pairwise interaction supplied")
    largest = max(interactions, key=lambda item: item.interaction_ratio())
    return {largest.left, largest.right} == set(expected)


def prerequisites_are_not_synergy(interactions: list[PairwiseInteraction]) -> bool:
    """The estimator supplies the subspace but adds nothing beyond it.

    Ref: Sec. 2.5.
    """
    estimator_rows = [
        interaction
        for interaction in interactions
        if "estimator" in interaction.left or "estimator" in interaction.right
    ]
    if not estimator_rows:
        raise ValueError("no estimator interaction supplied")
    return all(row.relation() == "prerequisite, not synergy" for row in estimator_rows)
