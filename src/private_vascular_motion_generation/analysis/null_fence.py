"""The pre-declared null-result fence and the retained null results.

Ref: Table 1 caption (T_null = 1.0% relative), Sec. 2.8.
"""

from __future__ import annotations

from dataclasses import dataclass

NULL_FENCE_RELATIVE: float = 0.01
STAR_LEVELS: tuple[int, ...] = (1, 2, 3)


@dataclass(frozen=True)
class FenceVerdict:
    label: str
    reference: float
    candidate: float

    def relative_change(self) -> float:
        if self.reference == 0.0:
            raise ValueError("reference value must be non-zero")
        return abs(self.candidate - self.reference) / abs(self.reference)

    def stars(self, fence: float = NULL_FENCE_RELATIVE) -> int:
        """A star is earned for each whole multiple of the fence the change clears."""
        if fence <= 0.0:
            raise ValueError("fence must be positive")
        ratio = self.relative_change() / fence
        earned = 0
        for level in STAR_LEVELS:
            if ratio >= level:
                earned = level
        return earned

    def is_retained_null(self, fence: float = NULL_FENCE_RELATIVE) -> bool:
        if fence <= 0.0:
            raise ValueError("fence must be positive")
        return self.relative_change() < fence

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "reference": self.reference,
            "candidate": self.candidate,
            "relative_change": self.relative_change(),
            "stars": self.stars(),
            "retained_null": self.is_retained_null(),
        }


def fence_verdict(label: str, reference: float, candidate: float) -> FenceVerdict:
    return FenceVerdict(label=label, reference=reference, candidate=candidate)


def markdown_stars(count: int) -> str:
    if count < 0 or count > 3:
        raise ValueError("star marks run from zero to three")
    return "*" * count


def retained_null_results() -> tuple[dict[str, object], ...]:
    """Three null outcomes are preserved rather than smoothed away.

    Ref: Sec. 2.8.
    """
    return (
        fence_verdict("estimator removal, contact-event F1", 0.842, 0.840).to_dict(),
        {
            "label": "procedure accountant removal, main utility",
            "reference": 0.842,
            "candidate": 0.842,
            "relative_change": 0.0,
            "stars": 0,
            "retained_null": True,
        },
        {
            "label": "reference-source relocation, non-private sources",
            "reference": 1.0,
            "candidate": 0.994,
            "relative_change": 0.006,
            "stars": 0,
            "retained_null": True,
        },
    )


def unmarked_because_inside_interval(paired_delta_pp: float, interval_half_width_pp: float) -> bool:
    """A task-success difference inside the paired interval carries no claim.

    Ref: Table 5 caption (the unconstrained-substrate row is unmarked).
    """
    if interval_half_width_pp < 0.0:
        raise ValueError("interval half width cannot be negative")
    return abs(paired_delta_pp) <= interval_half_width_pp


def minimal_clinically_important_difference_met(effect: float, mcid: float) -> bool:
    if mcid <= 0.0:
        raise ValueError("MCID must be positive")
    return abs(effect) >= mcid
