"""Single seeding utility shared by every entry point.

Ref: Sec. 4.9 (three independent training occasions plus twenty independent initialisations).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch

DEFAULT_SEED: int = 20260101
SEED_COUNT: int = 3
INITIALISATION_COUNT: int = 20


@dataclass(frozen=True)
class SeedState:
    seed: int
    initialisations: int = INITIALISATION_COUNT

    def offsets(self) -> tuple[int, ...]:
        return tuple(self.seed + offset for offset in range(self.initialisations))

    def training_seeds(self) -> tuple[int, ...]:
        return tuple(self.seed + 1000 * offset for offset in range(SEED_COUNT))


def set_seed(seed: int, deterministic: bool = True) -> SeedState:
    """Seed every generator the pipeline uses and return the state for checkpointing."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return SeedState(seed=seed)


def generator_for(seed: int, stream: int) -> torch.Generator:
    """Independent generator per stream so that a stream never consumes another's draws."""
    if stream < 0:
        raise ValueError("stream index must be non-negative")
    return torch.Generator().manual_seed(seed + 7919 * (stream + 1))


def restore_seed(state: SeedState) -> None:
    set_seed(state.seed)


def worker_init(worker_id: int) -> None:
    base = torch.initial_seed() % (2**31)
    np.random.seed((base + worker_id) % (2**32 - 1))
    random.seed(base + worker_id)


def same_seed_state(left: SeedState, right: SeedState) -> bool:
    return left.seed == right.seed and left.initialisations == right.initialisations
