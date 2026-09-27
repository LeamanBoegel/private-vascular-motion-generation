"""Assembly of a runnable mechanism from a resolved study configuration.

Ref: Sec. 4.4, Sec. 4.5, Algorithm 1.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from ..branches.adapter import PrivateCouplingAdapter
from ..branches.appearance import AppearanceBranch
from ..branches.motion import FlowMatchingActionHead
from ..branches.pairing import BranchPair, ProcedureBatch
from ..coupling.head import CouplingHead, CouplingHeadSpec
from ..estimator.reference_estimator import ReferenceShapingEstimator
from ..substrate import SubstrateSpec
from .batching import BatchSpec, RoundSampler, batch_from_records
from .config import StudyConfig
from .optim import OptimisationPlan
from .seeding import set_seed
from .trainer import CAPSTrainer, TrainingSpec

Record = TypeVar("Record")


@dataclass
class Session:
    config: StudyConfig
    pair: BranchPair
    head: CouplingHead
    estimator: ReferenceShapingEstimator
    batch_spec: BatchSpec
    plan: OptimisationPlan
    training: TrainingSpec
    substrate: SubstrateSpec
    checkpoint_dir: Path | None = None
    notes: dict[str, str] = field(default_factory=dict)

    def trainer(self, seed: int) -> CAPSTrainer:
        return CAPSTrainer(
            pair=self.pair,
            head=self.head,
            covariance=self.config.covariance(),
            plan=self.plan,
            training=self.training,
            estimator=self.estimator,
            order_grid=self.config.order_grid(),
            seed=seed,
            checkpoint_dir=self.checkpoint_dir,
        )

    def make_batch(self, records: Sequence[object], seed: int) -> ProcedureBatch:
        return batch_from_records(records, self.batch_spec, seed)  # type: ignore[arg-type]

    def sampler(self, batches: Sequence[ProcedureBatch], seed: int) -> RoundSampler:
        return RoundSampler(list(batches), self.training.sampling_rate, seed)


def build_session(config: StudyConfig, checkpoint_dir: Path | None = None) -> Session:
    appearance = AppearanceBranch(config.appearance_config())
    motion_config = config.section("branch", "motion")
    motion = FlowMatchingActionHead(**motion_config)
    adapter = PrivateCouplingAdapter(config.adapter_spec())
    pair = BranchPair(
        appearance=appearance,
        motion=motion,
        adapter=adapter,
        latent_spec=config.latent_spec(),
        pairing=config.pairing_spec(),
    )
    head = CouplingHead(spec=CouplingHeadSpec(rank=config.estimator_spec().rank))
    estimator = ReferenceShapingEstimator(config.estimator_spec(), config.pairing_spec())
    run_section = config.section("run")
    batch_spec = BatchSpec(
        frames=int(config.section("branch", "appearance")["in_frames"]),
        height=int(run_section["height"]),
        width=int(run_section["width"]),
        latent_dim=int(config.section("coupling")["latent_dim"]),
    )
    training = TrainingSpec(
        rounds=int(config.section("optimiser")["rounds"]),
        delta=float(config.section("substrate")["delta"]),
        clipping_norm=float(config.section("substrate")["clipping_norm"]),
        sampling_rate=float(config.section("substrate")["sampling_rate"]),
        checkpoint_every=int(run_section.get("checkpoint_every", 0)),
    )
    return Session(
        config=config,
        pair=pair,
        head=head,
        estimator=estimator,
        batch_spec=batch_spec,
        plan=config.optimisation_plan(),
        training=training,
        substrate=config.substrate_spec(),
        checkpoint_dir=checkpoint_dir,
    )


def seed_session(config: StudyConfig, seed: int | None = None) -> None:
    set_seed(config.seed() if seed is None else seed)


def split_reference_and_private(records: Sequence[Record], reference_size: int) -> tuple[list[Record], list[Record]]:
    """Hold the reference corpora strictly apart from the private cohort.

    Ref: Sec. 4.4 (estimator-disjointness), Algorithm 1 step 1.
    """
    if reference_size < 1:
        raise ValueError("the reference split must carry at least one procedure")
    if len(records) <= reference_size:
        raise ValueError("the cohort is too small to keep the reference disjoint")
    return list(records[:reference_size]), list(records[reference_size:])
