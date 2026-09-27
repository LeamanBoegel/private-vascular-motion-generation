"""Collation of procedure records into the paired training batch.

Ref: Sec. 4.3 (X in H x W x T and A in 6 x T), Algorithm 1 step 4.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ..branches.pairing import ProcedureBatch
from ..cohort.procedure import ProcedureRecord
from ..cohort.reference import ExpertTrajectoryClip
from ..coupling.latent import CouplingLatentSpec, latent_from_embeddings

VESSEL_INTENSITY_QUANTILE: float = 0.82
ENDPOINT_CLAMP: float = 1e-3


@dataclass(frozen=True)
class BatchSpec:
    frames: int
    height: int
    width: int
    latent_dim: int

    def __post_init__(self) -> None:
        if min(self.frames, self.height, self.width, self.latent_dim) < 1:
            raise ValueError("batch extents must be positive")


def vessel_mask(appearance: Tensor, quantile: float = VESSEL_INTENSITY_QUANTILE) -> Tensor:
    """Stand-in segmentation target from the maximum-intensity projection.

    Ref: Sec. 4.10. The expert annotations of the cohort stay private, so the mechanism
    runs against this projection and every cohort-level Dice value stays blocked.
    """
    if appearance.dim() != 3:
        raise ValueError("appearance must be H x W x T")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie in (0, 1)")
    projection = appearance.max(dim=2).values
    threshold = torch.quantile(projection.reshape(-1), quantile)
    return (projection > threshold).to(appearance.dtype).unsqueeze(0)


def endpoint_from_trajectory(trajectory: Tensor) -> Tensor:
    if trajectory.dim() != 2 or trajectory.shape[1] != 3:
        raise ValueError("trajectory must be frames x 3")
    return trajectory[-1, :2].clamp(ENDPOINT_CLAMP, 1.0 - ENDPOINT_CLAMP)


def identity_embeddings(records: Sequence[ProcedureRecord], latent_dim: int, seed: int) -> Tensor:
    """Deterministic per-procedure identity codes for the shared latent channel."""
    return identity_embeddings_for(len(records), latent_dim, seed)


def batch_from_records(
    records: Sequence[ProcedureRecord],
    spec: BatchSpec,
    seed: int,
    latent_spec: CouplingLatentSpec | None = None,
) -> ProcedureBatch:
    if not records:
        raise ValueError("at least one procedure is required")
    frames = torch.stack([record.appearance.permute(2, 0, 1) for record in records], dim=0).to(torch.float32)
    if frames.shape[1] != spec.frames:
        raise ValueError("procedure frame count disagrees with the batch specification")
    masks = torch.stack([vessel_mask(record.appearance) for record in records], dim=0).to(torch.float32)
    motion = torch.stack([record.motion for record in records], dim=0).to(torch.float32)
    endpoints = torch.stack([endpoint_from_trajectory(record.tip_trajectory()) for record in records], dim=0).to(
        torch.float32
    )
    codes = identity_embeddings(records, spec.latent_dim, seed).to(torch.float32)
    if latent_spec is not None:
        codes = latent_from_embeddings(codes, latent_spec).codes
    return ProcedureBatch(frames=frames, mask=masks, motion=motion, endpoints=endpoints, latent_codes=codes)


def batch_from_clips(
    clips: Sequence[ExpertTrajectoryClip],
    spec: BatchSpec,
    seed: int,
    latent_spec: CouplingLatentSpec | None = None,
) -> ProcedureBatch:
    if not clips:
        raise ValueError("at least one reference clip is required")
    frames = torch.stack([clip.appearance.permute(2, 0, 1) for clip in clips], dim=0).to(torch.float32)
    masks = torch.stack([vessel_mask(clip.appearance) for clip in clips], dim=0).to(torch.float32)
    motion = torch.stack([clip.motion for clip in clips], dim=0).to(torch.float32)
    endpoints = torch.stack([endpoint_from_trajectory(clip.motion[:3].transpose(0, 1)) for clip in clips], dim=0).to(
        torch.float32
    )
    codes = identity_embeddings_for(len(clips), spec.latent_dim, seed).to(torch.float32)
    if latent_spec is not None:
        codes = latent_from_embeddings(codes, latent_spec).codes
    return ProcedureBatch(frames=frames, mask=masks, motion=motion, endpoints=endpoints, latent_codes=codes)


def identity_embeddings_for(count: int, latent_dim: int, seed: int) -> Tensor:
    if count < 1:
        raise ValueError("count must be positive")
    generator = torch.Generator().manual_seed(seed)
    codes = torch.randn(count, latent_dim, generator=generator)
    normalised: Tensor = codes / codes.norm(dim=1, keepdim=True).clamp_min(torch.finfo(codes.dtype).tiny)
    return normalised


class RoundSampler:
    """Draws a procedure-level batch at the declared sampling rate.

    Ref: Algorithm 1 step 4.
    """

    def __init__(self, batches: Sequence[ProcedureBatch], sampling_rate: float, seed: int) -> None:
        if not batches:
            raise ValueError("the sampler needs at least one registered batch")
        if not 0.0 < sampling_rate <= 1.0:
            raise ValueError("sampling rate must lie in (0, 1]")
        self.batches = list(batches)
        self.sampling_rate = sampling_rate
        self.generator = torch.Generator().manual_seed(seed)

    def batch_size(self) -> int:
        pool = sum(batch.size() for batch in self.batches)
        return max(1, int(round(self.sampling_rate * pool)))

    def draw(self) -> ProcedureBatch:
        pool = sum(batch.size() for batch in self.batches)
        assembled = _flatten(self.batches)
        indices = torch.randperm(pool, generator=self.generator)[: self.batch_size()]
        chosen = [assembled[index] for index in indices.tolist()]
        return _concatenate(chosen)


def _flatten(batches: Sequence[ProcedureBatch]) -> list[ProcedureBatch]:
    singles: list[ProcedureBatch] = []
    for batch in batches:
        for index in range(batch.size()):
            singles.append(
                ProcedureBatch(
                    frames=batch.frames[index : index + 1],
                    mask=batch.mask[index : index + 1],
                    motion=batch.motion[index : index + 1],
                    endpoints=batch.endpoints[index : index + 1],
                    latent_codes=batch.latent_codes[index : index + 1],
                )
            )
    return singles


def _concatenate(batches: Sequence[ProcedureBatch]) -> ProcedureBatch:
    if not batches:
        raise ValueError("cannot concatenate an empty batch list")
    return ProcedureBatch(
        frames=torch.cat([batch.frames for batch in batches], dim=0),
        mask=torch.cat([batch.mask for batch in batches], dim=0),
        motion=torch.cat([batch.motion for batch in batches], dim=0),
        endpoints=torch.cat([batch.endpoints for batch in batches], dim=0),
        latent_codes=torch.cat([batch.latent_codes for batch in batches], dim=0),
    )
