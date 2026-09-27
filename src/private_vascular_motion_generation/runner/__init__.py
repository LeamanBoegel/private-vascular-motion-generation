"""Seeding, configuration, optimisation, checkpointing and the private training loop.

Ref: Algorithm 1, Sec. 4.4, Sec. 4.6, Sec. 4.9.
"""

from __future__ import annotations

from .batching import BatchSpec, RoundSampler, batch_from_clips, batch_from_records, vessel_mask
from .checkpoint import (
    CheckpointMeta,
    config_digest,
    digest_matches,
    load_checkpoint,
    privacy_tuple_from,
    save_checkpoint,
    seed_state_from,
)
from .config import ConfigError, StudyConfig, apply_overrides, deep_merge, load_yaml, resolve_study
from .distributed import (
    DistributedPlan,
    PrecisionPolicy,
    autocast_context,
    broadcast_scalar,
    build_grad_scaler,
    initialise_process_group,
    is_distributed_launch,
    shard_workload,
    uses_grad_scaler,
    wrap_model,
)
from .logging_utils import LoggingConfig, configure_logging, get_logger, log_metrics, summarise_round
from .optim import (
    OptimisationPlan,
    apply_learning_rate,
    build_adamw,
    build_optimiser,
    clip_gradients,
    learning_rate_at,
    schedule_ladder,
)
from .seeding import (
    DEFAULT_SEED,
    INITIALISATION_COUNT,
    SEED_COUNT,
    SeedState,
    generator_for,
    restore_seed,
    same_seed_state,
    set_seed,
    worker_init,
)
from .trainer import CAPSTrainer, RoundRecord, TrainingRun, TrainingSpec, accumulate_rdp_for_run

__all__ = [
    "DEFAULT_SEED",
    "INITIALISATION_COUNT",
    "SEED_COUNT",
    "BatchSpec",
    "CAPSTrainer",
    "CheckpointMeta",
    "ConfigError",
    "DistributedPlan",
    "LoggingConfig",
    "OptimisationPlan",
    "PrecisionPolicy",
    "RoundRecord",
    "RoundSampler",
    "SeedState",
    "StudyConfig",
    "TrainingRun",
    "TrainingSpec",
    "accumulate_rdp_for_run",
    "apply_learning_rate",
    "apply_overrides",
    "autocast_context",
    "batch_from_clips",
    "batch_from_records",
    "broadcast_scalar",
    "build_adamw",
    "build_grad_scaler",
    "build_optimiser",
    "clip_gradients",
    "config_digest",
    "configure_logging",
    "deep_merge",
    "digest_matches",
    "generator_for",
    "get_logger",
    "initialise_process_group",
    "is_distributed_launch",
    "learning_rate_at",
    "load_checkpoint",
    "load_yaml",
    "log_metrics",
    "privacy_tuple_from",
    "resolve_study",
    "restore_seed",
    "same_seed_state",
    "save_checkpoint",
    "schedule_ladder",
    "seed_state_from",
    "set_seed",
    "shard_workload",
    "summarise_round",
    "uses_grad_scaler",
    "vessel_mask",
    "worker_init",
    "wrap_model",
]
