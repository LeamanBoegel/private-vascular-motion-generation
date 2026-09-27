"""Site-stratified procedure-level partitioning.

Ref: Sec. 4.2, Sec. 4.9, Table 6.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

SITE_TOTALS: Mapping[str, int] = {"A": 2214, "B": 1986, "C": 1642}
SITE_HELD_OUT: Mapping[str, int] = {"A": 456, "B": 410, "C": 338}
REGION_OF_SITE: Mapping[str, str] = {"A": "I", "B": "I", "C": "II"}
SITE_OF_REGION: Mapping[str, tuple[str, ...]] = {"I": ("A", "B"), "II": ("C",)}
CLINICAL_ARMS: Mapping[str, int] = {"retrospective": 4106, "prospective": 1736}
FORCE_COVERAGE_BY_SITE: Mapping[str, float] = {"A": 0.71, "B": 0.64, "C": 0.49}
FULL_COHORT_SIZE: int = 5842
HELD_OUT_SIZE: int = 1204
ANATOMY_TERTILES: Mapping[str, int] = {"simple": 412, "intermediate": 458, "complex": 334}


@dataclass(frozen=True)
class SiteStratum:
    site: str
    region: str
    accrued: int
    held_out: int
    force_coverage: float

    def development(self) -> int:
        return self.accrued - self.held_out


@dataclass
class CohortPartition:
    development: tuple[str, ...]
    held_out: tuple[str, ...]
    per_site_held_out: Mapping[str, int]

    def __len__(self) -> int:
        return len(self.development) + len(self.held_out)

    def region_mix(self) -> dict[str, int]:
        mix = {"I": 0, "II": 0}
        for procedure_id in self.held_out:
            mix[REGION_OF_SITE[procedure_id.split("::")[0]]] += 1
        return mix

    def is_disjoint(self) -> bool:
        return not (set(self.development) & set(self.held_out))


def site_strata() -> tuple[SiteStratum, ...]:
    return tuple(
        SiteStratum(
            site=site,
            region=REGION_OF_SITE[site],
            accrued=SITE_TOTALS[site],
            held_out=SITE_HELD_OUT[site],
            force_coverage=FORCE_COVERAGE_BY_SITE[site],
        )
        for site in ("A", "B", "C")
    )


def cohort_identifiers(site: str, count: int) -> list[str]:
    return [f"{site}::{index:05d}" for index in range(count)]


def stratified_procedure_split(
    per_site_ids: Mapping[str, Sequence[str]],
    held_out_per_site: Mapping[str, int],
    seed: int,
) -> CohortPartition:
    """Allocate whole procedures so that no procedure straddles a partition.

    Ref: Sec. 4.2 ("a process is referred to as a split unit ... only once present in the partition").
    """
    generator = torch.Generator().manual_seed(seed)
    development: list[str] = []
    held_out: list[str] = []
    observed: dict[str, int] = {}
    for site in sorted(per_site_ids):
        ids = list(per_site_ids[site])
        wanted = int(held_out_per_site[site])
        if wanted > len(ids):
            raise ValueError(f"site {site} cannot supply {wanted} held-out procedures")
        order = torch.randperm(len(ids), generator=generator).tolist()
        shuffled = [ids[index] for index in order]
        held_out.extend(shuffled[:wanted])
        development.extend(shuffled[wanted:])
        observed[site] = wanted
    return CohortPartition(
        development=tuple(sorted(development)),
        held_out=tuple(sorted(held_out)),
        per_site_held_out=observed,
    )


def default_partition(seed: int = 20260101) -> CohortPartition:
    per_site = {site: cohort_identifiers(site, SITE_TOTALS[site]) for site in SITE_TOTALS}
    return stratified_procedure_split(per_site, SITE_HELD_OUT, seed)


def anatomy_tertile_assignment(complexity_score: torch.Tensor, counts: Mapping[str, int] | None = None) -> list[str]:
    """Tertile labels sized to the held-out stratum counts of the protocol.

    Ref: Table 6 panel (b).
    """
    if complexity_score.dim() != 1:
        raise ValueError("complexity scores must be one-dimensional")
    sizing = dict(ANATOMY_TERTILES if counts is None else counts)
    total = sum(sizing.values())
    if complexity_score.numel() != total:
        raise ValueError(f"expected {total} held-out procedures, got {complexity_score.numel()}")
    order = torch.argsort(complexity_score, stable=True).tolist()
    labels = ["intermediate"] * total
    cursor = 0
    for label in ("simple", "intermediate", "complex"):
        span = sizing[label]
        for index in order[cursor : cursor + span]:
            labels[index] = label
        cursor += span
    return labels


def region_pool(sites: Sequence[str]) -> dict[str, list[str]]:
    pooled: dict[str, list[str]] = {"I": [], "II": []}
    for site in sites:
        pooled[REGION_OF_SITE[site]].append(site)
    return pooled
