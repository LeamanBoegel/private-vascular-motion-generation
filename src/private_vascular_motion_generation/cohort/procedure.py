"""Synchronised procedure tensors: appearance frames paired with six-channel device motion.

Ref: Sec. 4.3 (P = (X, A)), Sec. 4.2 (channel semantics).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

MOTION_CHANNELS: tuple[str, ...] = (
    "tip_x",
    "tip_y",
    "tip_z",
    "insertion_depth",
    "rotation",
    "axial_force",
)
FORCE_CHANNEL_INDEX: int = MOTION_CHANNELS.index("axial_force")
MOTION_DIM: int = len(MOTION_CHANNELS)


@dataclass(frozen=True)
class ProcedureTensorSpec:
    height: int
    width: int
    frames: int
    motion_dim: int = MOTION_DIM
    motion_rate_hz: float = 15.0
    channels: tuple[str, ...] = MOTION_CHANNELS

    def __post_init__(self) -> None:
        if self.motion_dim != MOTION_DIM:
            raise ValueError(f"motion_dim must be {MOTION_DIM}, got {self.motion_dim}")
        if tuple(self.channels) != MOTION_CHANNELS:
            raise ValueError("motion channel order is fixed by the protocol")
        if min(self.height, self.width, self.frames) <= 0:
            raise ValueError("appearance extent must be positive")

    @property
    def appearance_shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, self.frames)

    def covers(self, appearance: Tensor, motion: Tensor) -> bool:
        return tuple(appearance.shape) == self.appearance_shape and tuple(motion.shape) == (
            MOTION_DIM,
            self.frames,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "height": self.height,
            "width": self.width,
            "frames": self.frames,
            "motion_dim": self.motion_dim,
            "motion_rate_hz": self.motion_rate_hz,
            "channels": list(self.channels),
        }


@dataclass
class ProcedureRecord:
    procedure_id: str
    site: str
    region: str
    appearance: Tensor
    motion: Tensor
    force_channel_present: bool
    injection_runs: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    transient_frames: tuple[int, ...] = field(default_factory=tuple)
    anatomy_complexity: str = "intermediate"
    access_route: str = "transfemoral"
    operator_stratum: str = "experienced"
    clinical_arm: str = "retrospective"

    def __post_init__(self) -> None:
        if self.appearance.dim() != 3:
            raise ValueError("appearance must be H x W x T")
        if self.motion.dim() != 2 or self.motion.shape[0] != MOTION_DIM:
            raise ValueError("motion must be 6 x T")
        if self.appearance.shape[2] != self.motion.shape[1]:
            raise ValueError("appearance and motion must share the frame axis")
        if self.force_channel_present and not bool(torch.isfinite(self.motion[FORCE_CHANNEL_INDEX]).all()):
            raise ValueError("declared force channel carries non-finite samples")

    @property
    def frames(self) -> int:
        return int(self.appearance.shape[2])

    @property
    def appearance_extent(self) -> tuple[int, int]:
        return (int(self.appearance.shape[0]), int(self.appearance.shape[1]))

    def tip_trajectory(self) -> Tensor:
        return self.motion[:3].transpose(0, 1).contiguous()

    def force_trace(self) -> Tensor:
        if not self.force_channel_present:
            raise ValueError("procedure has no force channel")
        return self.motion[FORCE_CHANNEL_INDEX].contiguous()

    def kinematic_channels(self) -> Tensor:
        return self.motion[:FORCE_CHANNEL_INDEX].contiguous()

    def describe(self) -> dict[str, object]:
        return {
            "procedure_id": self.procedure_id,
            "site": self.site,
            "region": self.region,
            "frames": self.frames,
            "appearance_extent": list(self.appearance_extent),
            "force_channel_present": self.force_channel_present,
            "injection_runs": [list(run) for run in self.injection_runs],
            "anatomy_complexity": self.anatomy_complexity,
            "access_route": self.access_route,
            "operator_stratum": self.operator_stratum,
            "clinical_arm": self.clinical_arm,
        }


def stack_procedures(records: Sequence[ProcedureRecord]) -> tuple[Tensor, Tensor]:
    """Collate a procedure-level batch onto the shared frame axis.

    Ref: Sec. 4.3.
    """
    if not records:
        raise ValueError("empty procedure batch")
    frames = {record.frames for record in records}
    if len(frames) != 1:
        raise ValueError("a procedure-level batch must share the frame axis")
    extent = {record.appearance_extent for record in records}
    if len(extent) != 1:
        raise ValueError("a procedure-level batch must share the appearance extent")
    appearance = torch.stack([record.appearance for record in records], dim=0)
    motion = torch.stack([record.motion for record in records], dim=0)
    return appearance, motion
