"""Verification vocabulary shared by the manuscript layer and the execution layer.

Ref: Sec. 2.8 (pre-specified primary criterion), Sec. 4.9 (multiplicity families).
"""

from __future__ import annotations

from dataclasses import dataclass, field

STATUSES: tuple[str, ...] = ("PASS", "FAIL", "NOT_RUN", "BLOCKED")
MANUSCRIPT_FAMILY: str = "manuscript_arithmetic"
EXECUTION_FAMILY: str = "code_execution"


@dataclass(frozen=True)
class CheckResult:
    identifier: str
    family: str
    description: str
    source: str
    code_ref: str
    status: str
    observed: object | None = None
    expected: object | None = None
    tolerance: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status}")
        if self.family not in (MANUSCRIPT_FAMILY, EXECUTION_FAMILY):
            raise ValueError(f"unknown family {self.family}")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.identifier,
            "family": self.family,
            "description": self.description,
            "source": self.source,
            "code_ref": self.code_ref,
            "status": self.status,
            "observed": self.observed,
            "expected": self.expected,
            "tolerance": self.tolerance,
            "note": self.note,
        }


@dataclass
class CheckLedger:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    def extend(self, results: list[CheckResult]) -> None:
        self.results.extend(results)

    def by_family(self, family: str) -> list[CheckResult]:
        return [result for result in self.results if result.family == family]

    def counts(self) -> dict[str, int]:
        tally = dict.fromkeys(STATUSES, 0)
        for result in self.results:
            tally[result.status] += 1
        return tally

    def family_status(self, family: str) -> str:
        subset = self.by_family(family)
        if not subset:
            return "NOT_RUN"
        if any(result.status == "FAIL" for result in subset):
            return "FAIL"
        if any(result.status == "BLOCKED" for result in subset):
            return "PARTIALLY_BLOCKED"
        if any(result.status == "NOT_RUN" for result in subset):
            return "PARTIALLY_NOT_RUN"
        return "PASS"

    def overall_status(self) -> str:
        """The release verdict is driven by the execution family; the manuscript family is reported beside it.

        Ref: Sec. 2.8 (retained null results are kept rather than smoothed away).
        """
        execution = self.family_status(EXECUTION_FAMILY)
        manuscript = self.family_status(MANUSCRIPT_FAMILY)
        if execution == "FAIL":
            return "FAIL"
        if execution == "NOT_RUN":
            return "UNVERIFIED"
        if execution.startswith("PARTIALLY") or manuscript != "PASS":
            return "PARTIALLY_VERIFIED"
        return "PASS"

    def discrepancies(self) -> list[dict[str, object]]:
        return [result.to_dict() for result in self.results if result.status == "FAIL"]

    def to_dict(self) -> dict[str, object]:
        return {
            "counts": self.counts(),
            "manuscript_family_status": self.family_status(MANUSCRIPT_FAMILY),
            "execution_family_status": self.family_status(EXECUTION_FAMILY),
            "overall_status": self.overall_status(),
            "discrepancies": self.discrepancies(),
            "results": [result.to_dict() for result in self.results],
        }


def close_enough(observed: float, expected: float, tolerance: float) -> bool:
    return abs(observed - expected) <= tolerance


def relative_close_enough(observed: float, expected: float, tolerance: float) -> bool:
    if expected == 0.0:
        raise ValueError("relative comparison needs a non-zero reference")
    return abs(observed - expected) / abs(expected) <= tolerance
