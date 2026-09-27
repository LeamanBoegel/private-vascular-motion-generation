"""Force-channel coverage and wall-contact event counting.

Ref: Sec. 4.1, Sec. 4.10, Tables 5 and 6.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

CONTACT_FORCE_THRESHOLD_N: float = 0.05
CONTACT_DEBOUNCE_SAMPLES: int = 2
FRAME_NORMALISATION_STRIDE: int = 1000


@dataclass(frozen=True)
class ContactEvent:
    onset: int
    offset: int
    peak_force_n: float

    @property
    def duration(self) -> int:
        return self.offset - self.onset


@dataclass(frozen=True)
class ForceChannelCoverage:
    site: str
    covered: int
    accrued: int

    @property
    def fraction(self) -> float:
        if self.accrued <= 0:
            raise ValueError("accrued denominator must be positive")
        return self.covered / self.accrued


def count_contact_events(
    force_trace: Tensor,
    threshold_n: float = CONTACT_FORCE_THRESHOLD_N,
    debounce_samples: int = CONTACT_DEBOUNCE_SAMPLES,
) -> tuple[ContactEvent, ...]:
    """Threshold the axial-force channel and merge runs shorter than the debounce rule.

    Ref: Sec. 4.10 ("counted only on the force-channel-covered subset after the
    pre-specified threshold and debounce rule").
    """
    if force_trace.dim() != 1:
        raise ValueError("force trace must be one-dimensional")
    if debounce_samples < 1:
        raise ValueError("debounce must span at least one sample")
    if force_trace.numel() == 0:
        return ()

    active = force_trace.abs() >= threshold_n
    events: list[ContactEvent] = []
    index = 0
    total = int(active.numel())
    while index < total:
        if not bool(active[index]):
            index += 1
            continue
        onset = index
        while index < total and bool(active[index]):
            index += 1
        offset = index
        if offset - onset < debounce_samples:
            continue
        peak = float(force_trace[onset:offset].abs().max().item())
        events.append(ContactEvent(onset=onset, offset=offset, peak_force_n=peak))
    return tuple(events)


def contacts_per_procedure(events_per_procedure: Tensor) -> float:
    if events_per_procedure.dim() != 1 or events_per_procedure.numel() == 0:
        raise ValueError("expected one contact count per procedure")
    return float(events_per_procedure.mean().item())


def contacts_per_thousand_frames(total_events: int, analysed_frames: int) -> float:
    """Secondary normalisation, never a rescaling of the per-procedure column.

    Ref: Sec. 4.10.
    """
    if analysed_frames <= 0:
        raise ValueError("analysed frame denominator must be positive")
    return total_events * FRAME_NORMALISATION_STRIDE / analysed_frames


def force_duration_integral(force_trace: Tensor, sample_rate_hz: float = 15.0) -> float:
    """Trapezoidal integral of the axial force over the procedure.

    Ref: Table 5, Supplementary Table SC1.
    """
    if force_trace.dim() != 1:
        raise ValueError("force trace must be one-dimensional")
    if force_trace.numel() < 2:
        return 0.0
    dt = 1.0 / sample_rate_hz
    return float(torch.trapz(force_trace.abs(), dx=dt).item())


def peak_force_envelope(force_trace: Tensor, envelope_quantile: float = 1.0) -> float:
    if not 0.0 < envelope_quantile <= 1.0:
        raise ValueError("envelope quantile must lie in (0, 1]")
    if force_trace.numel() == 0:
        raise ValueError("empty force trace")
    if envelope_quantile == 1.0:
        return float(force_trace.abs().max().item())
    return float(torch.quantile(force_trace.abs(), envelope_quantile).item())


def coverage_from_mask(coverage_mask: Tensor, site: str) -> ForceChannelCoverage:
    if coverage_mask.dim() != 1:
        raise ValueError("coverage mask must be one-dimensional")
    covered = int(coverage_mask.sum().item())
    return ForceChannelCoverage(site=site, covered=covered, accrued=int(coverage_mask.numel()))
