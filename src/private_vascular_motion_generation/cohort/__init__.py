"""Private multi-centre cohort and the auxiliary public layer.

Ref: Sec. 4.2, Sec. 4.1.
"""

from __future__ import annotations

from .force_channel import (
    ContactEvent,
    ForceChannelCoverage,
    contacts_per_thousand_frames,
    count_contact_events,
)
from .frames import DiagnosticFramePlan, select_diagnostic_frames
from .partition import (
    ANATOMY_TERTILES,
    CLINICAL_ARMS,
    FULL_COHORT_SIZE,
    HELD_OUT_SIZE,
    REGION_OF_SITE,
    SITE_HELD_OUT,
    SITE_OF_REGION,
    SITE_TOTALS,
    CohortPartition,
    SiteStratum,
    anatomy_tertile_assignment,
    cohort_identifiers,
    default_partition,
    region_pool,
    site_strata,
    stratified_procedure_split,
)
from .procedure import ProcedureRecord, ProcedureTensorSpec, stack_procedures
from .reference import (
    ExpertTrajectoryClip,
    ReferenceCorpus,
    ReferenceCorpusSpec,
    load_reference_corpus,
)

__all__ = [
    "ANATOMY_TERTILES",
    "CLINICAL_ARMS",
    "FULL_COHORT_SIZE",
    "HELD_OUT_SIZE",
    "REGION_OF_SITE",
    "SITE_HELD_OUT",
    "SITE_OF_REGION",
    "SITE_TOTALS",
    "CohortPartition",
    "ContactEvent",
    "DiagnosticFramePlan",
    "ExpertTrajectoryClip",
    "ForceChannelCoverage",
    "ProcedureRecord",
    "ProcedureTensorSpec",
    "ReferenceCorpus",
    "ReferenceCorpusSpec",
    "SiteStratum",
    "anatomy_tertile_assignment",
    "cohort_identifiers",
    "contacts_per_thousand_frames",
    "count_contact_events",
    "default_partition",
    "load_reference_corpus",
    "region_pool",
    "select_diagnostic_frames",
    "site_strata",
    "stack_procedures",
    "stratified_procedure_split",
]
