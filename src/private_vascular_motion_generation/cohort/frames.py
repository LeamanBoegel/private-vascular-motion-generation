"""Diagnostic-frame selection over the acquired angiographic run.

Ref: Sec. 4.10 (pre-opacification mask, peak-opacification frame per injection run,
tip-load transient frames at about 0.25 Hz, event marking on the 15 Hz motion stream).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

TRANSIENT_SAMPLE_RATE_HZ: float = 0.25
MOTION_STREAM_RATE_HZ: float = 15.0


@dataclass(frozen=True)
class DiagnosticFramePlan:
    pre_opacification: tuple[int, ...]
    peak_opacification: tuple[int, ...]
    transient: tuple[int, ...]

    def mask(self, frames: int) -> Tensor:
        mask = torch.zeros(frames, dtype=torch.bool)
        for index in self.union():
            if not 0 <= index < frames:
                raise IndexError(f"diagnostic frame {index} outside the acquired window")
            mask[index] = True
        return mask

    def union(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.pre_opacification) | set(self.peak_opacification) | set(self.transient)))

    def analysed_frame_count(self) -> int:
        return len(self.union())


def select_diagnostic_frames(
    intensity: Tensor,
    injection_runs: Sequence[tuple[int, int]],
    tip_load: Tensor,
    transient_quantile: float = 0.99,
    pre_opacification_offset: int = 1,
) -> DiagnosticFramePlan:
    """Pick the mask frame, the peak-opacification frame and the load-transient frames.

    Ref: Sec. 4.10.
    """
    if intensity.dim() != 1:
        raise ValueError("intensity must be a single-channel frame series")
    if tip_load.shape != intensity.shape:
        raise ValueError("tip-load trace must align with the frame series")
    if not injection_runs:
        raise ValueError("at least one injection run is required")

    pre: list[int] = []
    peak: list[int] = []
    for start, stop in injection_runs:
        if stop <= start:
            raise ValueError("injection run must span at least one frame")
        if start - pre_opacification_offset < 0 or stop > intensity.numel():
            raise ValueError("injection run outside the acquired window")
        window = intensity[start:stop]
        pre.append(start - pre_opacification_offset)
        peak.append(start + int(torch.argmax(window).item()))

    threshold = torch.quantile(tip_load, transient_quantile)
    candidates = torch.nonzero(tip_load >= threshold, as_tuple=False).flatten().tolist()
    hold = max(1, int(round(MOTION_STREAM_RATE_HZ / TRANSIENT_SAMPLE_RATE_HZ)))
    transient = [index for index in candidates if index % hold == 0]
    if not transient:
        transient = [candidates[0]]

    return DiagnosticFramePlan(
        pre_opacification=tuple(sorted(pre)),
        peak_opacification=tuple(sorted(peak)),
        transient=tuple(sorted(transient)),
    )


def transient_ladder(frames: int, rate_hz: float = TRANSIENT_SAMPLE_RATE_HZ) -> Tensor:
    if frames <= 0:
        raise ValueError("frames must be positive")
    stride = max(1, int(round(MOTION_STREAM_RATE_HZ / rate_hz)))
    return torch.arange(0, frames, stride, dtype=torch.long)


def frame_axis_alignment(appearance_frames: int, motion_samples: int, tolerance_ms: float) -> bool:
    """Check the appearance/motion tolerance used by the perturbation family SC1.

    Ref: Supplementary Table SC1 (temporal misalignment in milliseconds).
    """
    expected = motion_samples / MOTION_STREAM_RATE_HZ * 1000.0
    observed = appearance_frames / MOTION_STREAM_RATE_HZ * 1000.0
    return abs(observed - expected) <= tolerance_ms
