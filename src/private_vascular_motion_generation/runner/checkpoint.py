"""Atomic checkpointing; every checkpoint carries its seed and privacy tuple.

Ref: Sec. 4.6 (the tuple travels with the released artifact).
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from ..accounting.tuple import PrivacyTuple
from .seeding import SeedState


@dataclass(frozen=True)
class CheckpointMeta:
    step: int
    round_index: int
    seed: int
    initialisations: int
    privacy_tuple: dict[str, object]
    config_digest: str


def _write_atomic(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=str(target.parent), prefix=".ckpt-", suffix=".tmp")
    scratch = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        scratch.replace(target)
    except BaseException:
        if scratch.exists():
            scratch.unlink()
        raise


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimiser: torch.optim.Optimizer,
    meta: CheckpointMeta,
) -> Path:
    payload = {
        "model": model.state_dict(),
        "optimiser": optimiser.state_dict(),
        "meta": asdict(meta),
    }
    with tempfile.SpooledTemporaryFile() as buffer:
        torch.save(payload, buffer)
        buffer.seek(0)
        _write_atomic(path, buffer.read())
    return path


def load_checkpoint(path: Path, model: nn.Module, optimiser: torch.optim.Optimizer | None = None) -> CheckpointMeta:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not staged at {path.name}")
    payload = torch.load(path, weights_only=True)
    if not isinstance(payload, dict) or "meta" not in payload:
        raise TypeError("checkpoint payload is malformed")
    model.load_state_dict(payload["model"])
    if optimiser is not None and "optimiser" in payload:
        optimiser.load_state_dict(payload["optimiser"])
    raw = payload["meta"]
    return CheckpointMeta(
        step=int(raw["step"]),
        round_index=int(raw["round_index"]),
        seed=int(raw["seed"]),
        initialisations=int(raw["initialisations"]),
        privacy_tuple=dict(raw["privacy_tuple"]),
        config_digest=str(raw["config_digest"]),
    )


def seed_state_from(meta: CheckpointMeta) -> SeedState:
    return SeedState(seed=meta.seed, initialisations=meta.initialisations)


def privacy_tuple_from(meta: CheckpointMeta) -> PrivacyTuple | None:
    """Rebuild the accounting statement, or None when the checkpoint predates one.

    A checkpoint written before the accountant has converted carries no tuple, and an absent
    statement must stay absent rather than be reported as a zero budget.
    """
    if not meta.privacy_tuple:
        return None
    return PrivacyTuple(**meta.privacy_tuple)  # type: ignore[arg-type]


def config_digest(config: Mapping[str, object]) -> str:
    import hashlib

    serialised = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()[:16]


def digest_matches(meta: CheckpointMeta, config: Mapping[str, object]) -> bool:
    return meta.config_digest == config_digest(config)
