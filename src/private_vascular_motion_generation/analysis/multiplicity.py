"""Multiplicity control across the pre-specified and exploratory comparison families.

Ref: Sec. 2.8, Sec. 4.9, Supplementary Table SE1.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_ALPHA: float = 0.05
PRELIMINARY_PRIMARIES: int = 3
EXPLORATORY_FAMILIES: tuple[str, ...] = (
    "strata",
    "ablation",
    "classical_comparators",
    "factorial_contrasts",
    "reader_study",
)


@dataclass(frozen=True)
class AdjustedComparison:
    label: str
    statistic: float
    p_uncorrected: float
    p_corrected: float
    effect_size: str
    correction: str
    passes: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "statistic": self.statistic,
            "p_uncorrected": self.p_uncorrected,
            "p_corrected": self.p_corrected,
            "effect_size": self.effect_size,
            "correction": self.correction,
            "passes": self.passes,
        }


def holm_bonferroni(labels: list[str], p_values: list[float], alpha: float = DEFAULT_ALPHA) -> list[AdjustedComparison]:
    """Step-down Holm correction over the pre-specified primaries.

    Ref: Sec. 2.8, Sec. 4.9.
    """
    if len(labels) != len(p_values):
        raise ValueError("labels and p-values must align")
    if not p_values:
        raise ValueError("no comparison supplied")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    running_maximum = 0.0
    outcomes: list[AdjustedComparison | None] = [None] * len(p_values)
    for rank, index in enumerate(order):
        scaled = min(1.0, p_values[index] * (len(p_values) - rank))
        running_maximum = max(running_maximum, scaled)
        outcomes[index] = AdjustedComparison(
            label=labels[index],
            statistic=0.0,
            p_uncorrected=p_values[index],
            p_corrected=running_maximum,
            effect_size="",
            correction="holm",
            passes=running_maximum <= alpha,
        )
    return [outcome for outcome in outcomes if outcome is not None]


def benjamini_hochberg(
    labels: list[str], p_values: list[float], alpha: float = DEFAULT_ALPHA
) -> list[AdjustedComparison]:
    """Step-up Benjamini-Hochberg control at the declared false discovery rate.

    Ref: Sec. 2.8, Sec. 4.9.
    """
    if len(labels) != len(p_values):
        raise ValueError("labels and p-values must align")
    if not p_values:
        raise ValueError("no comparison supplied")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    total = len(p_values)
    order = sorted(range(total), key=lambda index: p_values[index])
    running_minimum = 1.0
    outcomes: list[AdjustedComparison | None] = [None] * total
    for rank in range(total - 1, -1, -1):
        index = order[rank]
        scaled = min(1.0, p_values[index] * total / (rank + 1))
        running_minimum = min(running_minimum, scaled)
        outcomes[index] = AdjustedComparison(
            label=labels[index],
            statistic=0.0,
            p_uncorrected=p_values[index],
            p_corrected=running_minimum,
            effect_size="",
            correction="benjamini_hochberg",
            passes=running_minimum <= alpha,
        )
    return [outcome for outcome in outcomes if outcome is not None]


def exploratory_inventory_size() -> int:
    """Twenty-three exploratory comparisons across seven examined families.

    Ref: Sec. 2.8.
    """
    return 23


def declare_families() -> dict[str, int]:
    """The exploratory inventory grouped the way the design states it.

    Ref: Sec. 2.8.
    """
    return {
        "per_site_and_per_region_strata": 5,
        "anatomy_complexity_and_subgroups": 7,
        "access_route_and_operator_strata": 2,
        "reproduced_reference_ablations": 3,
        "classical_comparators": 2,
        "factorial_contrasts": 3,
        "reader_study": 1,
    }


def inventory_total() -> int:
    return sum(declare_families().values())
