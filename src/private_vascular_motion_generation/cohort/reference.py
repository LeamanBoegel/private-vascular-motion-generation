"""The non-private reference layer: public angiographic collections and open-simulator trajectories.

Ref: Sec. 4.2 (auxiliary public layer), Algorithm 2 (reference corpora), Sec. 4.4 (estimator-disjointness).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch
from torch import Tensor

from .procedure import MOTION_DIM

REFERENCE_SOURCE_KINDS: tuple[str, ...] = ("public_angiography", "open_simulator")
MANIFEST_NAME = "manifest.json"
CLIP_NAME = "clip_{index:05d}.pt"


class ManifestClipEntry(TypedDict, total=False):
    index: int
    source_kind: str


@dataclass(frozen=True)
class ReferenceCorpusSpec:
    root: Path
    clip_frames: int
    appearance_extent: tuple[int, int]
    source_kinds: tuple[str, ...] = REFERENCE_SOURCE_KINDS


@dataclass(frozen=True)
class ExpertTrajectoryClip:
    clip_id: str
    source_kind: str
    appearance: Tensor
    motion: Tensor

    def frames(self) -> int:
        return int(self.appearance.shape[2])


@dataclass
class ReferenceCorpus:
    spec: ReferenceCorpusSpec
    clips: tuple[ExpertTrajectoryClip, ...]

    def __len__(self) -> int:
        return len(self.clips)

    def batches(self, batch_size: int) -> Iterator[tuple[Tensor, Tensor]]:
        if batch_size <= 0:
            raise ValueError("reference batch size must be positive")
        for start in range(0, len(self.clips), batch_size):
            chunk = self.clips[start : start + batch_size]
            appearance = torch.stack([clip.appearance for clip in chunk], dim=0)
            motion = torch.stack([clip.motion for clip in chunk], dim=0)
            yield appearance, motion

    def source_mix(self) -> dict[str, int]:
        mix = dict.fromkeys(self.spec.source_kinds, 0)
        for clip in self.clips:
            mix[clip.source_kind] = mix.get(clip.source_kind, 0) + 1
        return mix

    def aligned(self) -> bool:
        return all(
            clip.frames() == self.spec.clip_frames
            and (int(clip.appearance.shape[0]), int(clip.appearance.shape[1])) == self.spec.appearance_extent
            and tuple(clip.motion.shape) == (MOTION_DIM, self.spec.clip_frames)
            for clip in self.clips
        )


def _clip_from_payload(payload: dict[str, object], clip_id: str, source_kind: str) -> ExpertTrajectoryClip:
    appearance = payload.get("appearance")
    motion = payload.get("motion")
    if not isinstance(appearance, Tensor) or not isinstance(motion, Tensor):
        raise TypeError(f"reference clip {clip_id} is missing tensor payloads")
    return ExpertTrajectoryClip(
        clip_id=clip_id,
        source_kind=source_kind,
        appearance=appearance.to(torch.float32),
        motion=motion.to(torch.float32),
    )


def load_reference_corpus(spec: ReferenceCorpusSpec, limit: int | None = None) -> ReferenceCorpus:
    """Read the staged reference layer from disk; no network access is performed here.

    Ref: Sec. 4.2, Data availability.
    """
    manifest_path = spec.root / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"reference manifest not staged at {MANIFEST_NAME}; run scripts/prepare_reference_corpus.sh"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries: Sequence[ManifestClipEntry] = manifest.get("clips", [])
    if not entries:
        raise ValueError("reference manifest lists no clips")

    clips: list[ExpertTrajectoryClip] = []
    for entry in entries:
        index = int(entry["index"])
        source_kind = str(entry.get("source_kind", spec.source_kinds[0]))
        if source_kind not in spec.source_kinds:
            raise ValueError(f"unexpected reference source kind {source_kind}")
        clip_path = spec.root / CLIP_NAME.format(index=index)
        if not clip_path.exists():
            raise FileNotFoundError(f"reference clip payload missing for index {index}")
        payload = torch.load(clip_path, weights_only=True)
        if not isinstance(payload, dict):
            raise TypeError("reference clip payload must be a mapping")
        clips.append(_clip_from_payload(payload, clip_id=f"ref-{index:05d}", source_kind=source_kind))
        if limit is not None and len(clips) >= limit:
            break

    corpus = ReferenceCorpus(spec=spec, clips=tuple(clips))
    if not corpus.aligned():
        raise ValueError("reference corpus is not aligned to the declared extent")
    return corpus


def assert_estimator_disjointness(reference: ReferenceCorpus, private_ids: Sequence[str]) -> None:
    """The shaping estimator must never see a private record.

    Ref: Sec. 4.4 (estimator-disjointness), Algorithm 1 step 1.
    """
    overlap = {clip.clip_id for clip in reference.clips} & set(private_ids)
    if overlap:
        raise ValueError(f"reference corpus overlaps the private cohort on {sorted(overlap)[:4]}")


def repeated_reference_views(corpus: ReferenceCorpus, repetitions: int) -> list[ExpertTrajectoryClip]:
    """Deterministic re-presentation of the same reference clips for the estimator pass.

    Ref: Algorithm 2 (multiple minibatches over the reference corpora).
    """
    if repetitions < 1:
        raise ValueError("repetitions must be at least one")
    views: list[ExpertTrajectoryClip] = []
    for round_index in range(repetitions):
        views.extend(
            ExpertTrajectoryClip(
                clip_id=f"{clip.clip_id}#r{round_index}",
                source_kind=clip.source_kind,
                appearance=clip.appearance,
                motion=clip.motion,
            )
            for clip in corpus.clips
        )
    return views
