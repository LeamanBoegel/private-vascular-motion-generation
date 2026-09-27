"""The full privacy tuple reported alongside every mechanism row.

Ref: Sec. 4.6 ("a budget is recorded in an incomplete manner when it is reported without the tuple").
"""

from __future__ import annotations

from dataclasses import dataclass, field

TUPLE_FIELDS: tuple[str, ...] = (
    "epsilon",
    "delta",
    "accountant",
    "order",
    "steps_times_sampling",
    "clipping_norm",
    "sigma_parallel",
    "sigma_perpendicular",
    "rank",
    "dimension",
)


@dataclass(frozen=True)
class PrivacyTuple:
    epsilon: float
    delta: float
    accountant: str
    order: float
    steps_times_sampling: float
    clipping_norm: float
    sigma_parallel: float
    sigma_perpendicular: float
    rank: int
    dimension: int
    rho: float | None = None
    privacy_unit: str = "procedure"
    composition_unit: str = "procedure_window"
    notes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.accountant not in ("renyi", "rdp"):
            raise ValueError("the accountant of record is the Renyi accountant")
        if self.privacy_unit not in ("procedure", "example"):
            raise ValueError("privacy unit must be a procedure or an example")
        if self.clipping_norm <= 0.0:
            raise ValueError("clipping norm must be positive")
        if self.steps_times_sampling <= 0.0:
            raise ValueError("steps x q must be positive")
        if self.rank < 0 or self.rank > self.dimension:
            raise ValueError("rank must lie in [0, d]")
        if self.rho is not None and not 0.0 <= self.rho <= 1.0:
            raise ValueError("rho must lie in [0, 1]")

    def as_ordered_items(self) -> tuple[tuple[str, object], ...]:
        payload: dict[str, object] = {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "accountant": self.accountant,
            "order": self.order,
            "steps_times_sampling": self.steps_times_sampling,
            "clipping_norm": self.clipping_norm,
            "sigma_parallel": self.sigma_parallel,
            "sigma_perpendicular": self.sigma_perpendicular,
            "rank": self.rank,
            "dimension": self.dimension,
        }
        return tuple((name, payload[name]) for name in TUPLE_FIELDS)

    def to_dict(self) -> dict[str, object]:
        payload = dict(self.as_ordered_items())
        payload["rho"] = self.rho
        payload["privacy_unit"] = self.privacy_unit
        payload["composition_unit"] = self.composition_unit
        payload["notes"] = dict(self.notes)
        return payload

    def weaker_unit(self) -> str:
        return "example" if self.privacy_unit == "procedure" else "procedure"


def tuple_is_complete(candidate: dict[str, object]) -> bool:
    return all(name in candidate and candidate[name] is not None for name in TUPLE_FIELDS)


def bare_budget_warning(candidate: dict[str, object]) -> str | None:
    """A budget without its accountant, unit and delta is not interpretable.

    Ref: Sec. 4.6.
    """
    missing = [name for name in TUPLE_FIELDS if name not in candidate]
    if not missing:
        return None
    return "reported without the tuple: missing " + ", ".join(missing)


def conservative_ratio(strong: float, weak: float) -> dict[str, object]:
    """The quotient of a procedure-level budget by an example-level budget is a lower reading.

    Ref: Sec. 2.2, Table 2 caption.
    """
    if strong <= 0.0 or weak <= 0.0:
        raise ValueError("budgets must be positive")
    return {"quotient": weak / strong, "reading": "conservative_lower_bound"}
