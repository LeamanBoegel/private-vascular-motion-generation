"""Safety and procedural-efficiency endpoints on the held-out trajectories.

Ref: Sec. 4.10, Table 5, Supplementary Table SC1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..cohort.force_channel import (
    contacts_per_procedure,
    count_contact_events,
    force_duration_integral,
    peak_force_envelope,
)

PREVALENCE_FORCE_REFERENCE_N: float = 0.76
CONTACT_EVENT_MCID: float = 0.5
NON_INFERIORITY_MARGIN_PP: float = 3.0
CROSS_REGION_BOUND_PP: float = 5.0


@dataclass(frozen=True)
class SafetyRow:
    label: str
    task_success: float
    contacts_per_procedure: float
    analysed_frames_per_procedure: float
    peak_force_n: float
    force_duration_integral_ns: float
    fluoroscopy_time_min: float
    contrast_volume_ml: float
    navigation_time_min: float
    path_efficiency: float

    def contacts_per_thousand_frames(self) -> float:
        return self.contacts_per_procedure * 1000.0 / self.analysed_frames_per_procedure

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "task_success": self.task_success,
            "contacts_per_procedure": self.contacts_per_procedure,
            "contacts_per_1000_frames": self.contacts_per_thousand_frames(),
            "analysed_frames_per_procedure": self.analysed_frames_per_procedure,
            "peak_force_n": self.peak_force_n,
            "force_duration_integral_ns": self.force_duration_integral_ns,
            "fluoroscopy_time_min": self.fluoroscopy_time_min,
            "contrast_volume_ml": self.contrast_volume_ml,
            "navigation_time_min": self.navigation_time_min,
            "path_efficiency": self.path_efficiency,
        }


def force_envelope_margin(peak_force_n: float, reference_n: float = PREVALENCE_FORCE_REFERENCE_N) -> float:
    """Relative reduction of the peak contact force against the perforation magnitude.

    Ref: Sec. 2.6 (0.41 N is about 46% below 0.76 N).
    """
    if reference_n <= 0.0:
        raise ValueError("reference force must be positive")
    return 1.0 - peak_force_n / reference_n


def contact_event_reduction(private_contacts: float, standard_contacts: float) -> float:
    if standard_contacts <= 0.0:
        raise ValueError("standard-of-care contact count must be positive")
    return standard_contacts - private_contacts


def clears_mcid(reduction: float, mcid: float = CONTACT_EVENT_MCID) -> bool:
    """A clinical endpoint is judged against its minimum clinically important difference.

    Ref: Table 5 caption.
    """
    return reduction >= mcid


def non_inferiority_margin(standard_success: float, candidate_success: float) -> float:
    return (standard_success - candidate_success) * 100.0


def within_non_inferiority_margin(delta_pp: float, margin_pp: float = NON_INFERIORITY_MARGIN_PP) -> bool:
    return abs(delta_pp) <= margin_pp


def cross_region_bound_holds(
    best_site_success: float, worst_site_success: float, bound_pp: float = CROSS_REGION_BOUND_PP
) -> bool:
    return (best_site_success - worst_site_success) * 100.0 <= bound_pp


def per_site_spread(site_success: dict[str, float]) -> float:
    if not site_success:
        raise ValueError("at least one site is required")
    values = list(site_success.values())
    return (max(values) - min(values)) * 100.0


def safety_from_rollouts(
    label: str,
    forces: Tensor,
    successful: Tensor,
    fluoroscopy_min: float,
    contrast_ml: float,
    navigation_min: float,
    path_efficiency: float,
) -> SafetyRow:
    """Assemble one safety row from per-procedure force traces and success flags."""
    if forces.dim() != 2:
        raise ValueError("forces must be procedures x frames")
    if successful.dim() != 1 or successful.shape[0] != forces.shape[0]:
        raise ValueError("success flags must carry one entry per procedure")
    counts = torch.tensor([len(count_contact_events(trace)) for trace in forces], dtype=torch.float64)
    return SafetyRow(
        label=label,
        task_success=float(successful.to(torch.float64).mean().item()),
        contacts_per_procedure=contacts_per_procedure(counts),
        analysed_frames_per_procedure=float(forces.shape[1]),
        peak_force_n=peak_force_envelope(forces.reshape(-1)),
        force_duration_integral_ns=force_duration_integral(forces.reshape(-1)),
        fluoroscopy_time_min=fluoroscopy_min,
        contrast_volume_ml=contrast_ml,
        navigation_time_min=navigation_min,
        path_efficiency=path_efficiency,
    )
