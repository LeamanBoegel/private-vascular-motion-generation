"""Utility, geometry, motion, safety and clinical endpoints.

Ref: Sec. 4.10, Tables 1, 5 and 6.
"""

from __future__ import annotations

from .clinical import (
    ProportionEstimate,
    cohen_kappa,
    discrimination_accuracy,
    reader_agreement_gain,
    task_success,
    three_way_reader_contrast,
    wilson_interval,
)
from .fidelity import (
    fidelity_report,
    frechet_distance,
    polynomial_mmd,
    structural_similarity,
)
from .safety import (
    CONTACT_EVENT_MCID,
    CROSS_REGION_BOUND_PP,
    NON_INFERIORITY_MARGIN_PP,
    PREVALENCE_FORCE_REFERENCE_N,
    SafetyRow,
    clears_mcid,
    contact_event_reduction,
    cross_region_bound_holds,
    force_envelope_margin,
    non_inferiority_margin,
    per_site_spread,
    safety_from_rollouts,
    within_non_inferiority_margin,
)
from .segmentation import dice_by_stratum, dice_overlap, geometry_report, percentile_hausdorff
from .trajectory import (
    PATH_EFFICIENCY_ARTIFACT_CEILING,
    average_displacement_error,
    deviation_from_centreline,
    dynamic_time_warping,
    final_displacement_error,
    motion_report,
    path_efficiency_is_artifact,
    path_length,
    path_length_efficiency,
    trajectory_smoothness,
)

__all__ = [
    "CONTACT_EVENT_MCID",
    "CROSS_REGION_BOUND_PP",
    "NON_INFERIORITY_MARGIN_PP",
    "PATH_EFFICIENCY_ARTIFACT_CEILING",
    "PREVALENCE_FORCE_REFERENCE_N",
    "ProportionEstimate",
    "SafetyRow",
    "average_displacement_error",
    "clears_mcid",
    "cohen_kappa",
    "contact_event_reduction",
    "cross_region_bound_holds",
    "deviation_from_centreline",
    "dice_by_stratum",
    "dice_overlap",
    "discrimination_accuracy",
    "dynamic_time_warping",
    "fidelity_report",
    "final_displacement_error",
    "force_envelope_margin",
    "frechet_distance",
    "geometry_report",
    "motion_report",
    "non_inferiority_margin",
    "path_efficiency_is_artifact",
    "path_length",
    "path_length_efficiency",
    "per_site_spread",
    "percentile_hausdorff",
    "polynomial_mmd",
    "reader_agreement_gain",
    "safety_from_rollouts",
    "structural_similarity",
    "task_success",
    "three_way_reader_contrast",
    "trajectory_smoothness",
    "wilson_interval",
    "within_non_inferiority_margin",
]
