"""Schema-compatible cohort fixture used by the mechanism tests and smoke studies.

Ref: Sec. 4.3 (tensor contract). The private multi-centre cohort of Sec. 4.2 is not
exportable, so every cohort-level number in the manuscript is blocked; this fixture
supplies the same tensor contract for mechanism-level execution only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict

import torch
from torch import Tensor

from .force_channel import CONTACT_FORCE_THRESHOLD_N
from .procedure import MOTION_DIM, ProcedureRecord

COMPLEXITY_TO_SCORE: Mapping[str, float] = {"simple": 0.0, "intermediate": 0.55, "complex": 1.0}
ACCESS_ROUTES: tuple[str, ...] = ("transradial", "transfemoral")
OPERATOR_STRATA: tuple[str, ...] = ("experienced", "early_career")


class FixtureShapeReport(TypedDict):
    procedures: int
    appearance_shape: list[int]
    motion_shape: list[int]
    expected_motion_dim: int
    force_covered: int


@dataclass(frozen=True)
class FixtureSpec:
    height: int = 32
    width: int = 32
    frames: int = 16
    catheter_radius: float = 0.18
    lesion_strength: float = 0.35
    noise_scale: float = 0.05


@dataclass
class CohortFixture:
    records: tuple[ProcedureRecord, ...]
    spec: FixtureSpec
    seed: int

    def by_site(self, site: str) -> tuple[ProcedureRecord, ...]:
        return tuple(record for record in self.records if record.site == site)

    def force_covered(self) -> tuple[ProcedureRecord, ...]:
        return tuple(record for record in self.records if record.force_channel_present)

    def identifiers(self) -> tuple[str, ...]:
        return tuple(record.procedure_id for record in self.records)


def _tip_path(generator: torch.Generator, frames: int, access_route: str) -> Tensor:
    start = torch.rand(3, generator=generator) * 0.2 + 0.05
    if access_route == "transradial":
        start[0] += 0.35
    curvature = (torch.rand(3, generator=generator) - 0.5) * 0.25
    progress = torch.linspace(0.0, 1.0, frames).unsqueeze(1)
    path = start.unsqueeze(0) + progress * (0.75 + curvature)
    wobble = torch.sin(progress * torch.pi * 2.0) * 0.02
    path = path + wobble
    return path.transpose(0, 1).contiguous()


def _render_appearance(
    spec: FixtureSpec,
    path: Tensor,
    complexity: str,
    transient_frames: Sequence[int],
    generator: torch.Generator,
) -> Tensor:
    height, width, frames = spec.height, spec.width, spec.frames
    yy = torch.arange(height, dtype=torch.float32).view(height, 1, 1)
    xx = torch.arange(width, dtype=torch.float32).view(1, width, 1)
    frames_axis = torch.arange(frames, dtype=torch.float32).view(1, 1, frames)
    phase = torch.rand(1, generator=generator) * 0.4 + 0.8
    vessel = torch.exp(-((yy - (0.5 + 0.18 * torch.sin(phase * frames_axis / frames))) ** 2) / (2.0 * 0.045**2))
    vessel = vessel.expand(height, width, frames) * torch.exp(-((xx - 0.5) ** 2) / (2.0 * 0.55**2))
    lesion = spec.lesion_strength * (COMPLEXITY_TO_SCORE[complexity] + 0.15)
    stenosis = 1.0 - lesion * torch.exp(-((frames_axis / frames - 0.45) ** 2) / (2.0 * 0.08**2))
    device = torch.zeros(height, width, frames)
    for frame in range(frames):
        centre_y = float(path[1, frame].clamp(0.0, 0.999).item()) * (height - 1)
        centre_x = float(path[0, frame].clamp(0.0, 0.999).item()) * (width - 1)
        radial = (yy - centre_y) ** 2 + (xx - centre_x) ** 2
        device = device + torch.exp(-radial / (2.0 * spec.catheter_radius**2)) * (frames_axis == frame)
    load = torch.zeros(frames)
    for index in transient_frames:
        if 0 <= index < frames:
            load[index] = 1.0
    contrast = 0.5 + 0.5 * torch.sin(frames_axis / frames * torch.pi)
    field = vessel * stenosis * contrast + 0.7 * device + 0.25 * load
    field = field + torch.randn(height, width, frames, generator=generator) * spec.noise_scale
    return field.clamp(0.0, 1.5).contiguous()


def _motion_payload(
    path: Tensor,
    complexity: str,
    coverage: float,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, bool]:
    frames = int(path.shape[1])
    insertion = torch.linspace(0.0, 1.0, frames) * (0.8 + 0.15 * COMPLEXITY_TO_SCORE[complexity])
    rotation = torch.cumsum(torch.randn(frames, generator=generator) * 0.03, dim=0)
    rotation = rotation - rotation[0]
    force = torch.zeros(frames)
    contacts = max(1, int(round(1.4 + 1.4 * COMPLEXITY_TO_SCORE[complexity])))
    for contact in range(contacts):
        centre = int((contact + 1) * frames / (contacts + 1))
        span = torch.arange(frames) - centre
        pulse = torch.exp(-(span**2) / (2.0 * 1.4**2))
        amplitude = CONTACT_FORCE_THRESHOLD_N * (6.0 + 4.0 * COMPLEXITY_TO_SCORE[complexity])
        force = force + pulse * amplitude
    force = force + torch.rand(frames, generator=generator) * 0.02
    force_covered = bool(torch.rand(1, generator=generator).item() < coverage)
    motion = torch.cat([path, insertion.view(1, frames), rotation.view(1, frames), force.view(1, frames)], dim=0)
    return motion.contiguous(), force, force_covered


def build_fixture(
    spec: FixtureSpec,
    per_site: Mapping[str, int],
    coverage_by_site: Mapping[str, float],
    complexity_by_site: Mapping[str, tuple[str, ...]],
    access_route_cycle: Sequence[str],
    operator_cycle: Sequence[str],
    seed: int,
) -> CohortFixture:
    generator = torch.Generator().manual_seed(seed)
    records: list[ProcedureRecord] = []
    for site in sorted(per_site):
        complexities = complexity_by_site[site]
        for index in range(per_site[site]):
            complexity = complexities[index % len(complexities)]
            access_route = access_route_cycle[index % len(access_route_cycle)]
            operator = operator_cycle[index % len(operator_cycle)]
            path = _tip_path(generator, spec.frames, access_route)
            transient = tuple(sorted({int(spec.frames * 0.25), int(spec.frames * 0.75)}))
            appearance = _render_appearance(spec, path, complexity, transient, generator)
            motion, _force, covered = _motion_payload(path, complexity, coverage_by_site[site], generator)
            run_span = max(2, spec.frames // 4)
            records.append(
                ProcedureRecord(
                    procedure_id=f"{site}::{index:05d}",
                    site=site,
                    region="I" if site in ("A", "B") else "II",
                    appearance=appearance,
                    motion=motion,
                    force_channel_present=covered,
                    injection_runs=((1, 1 + run_span), (spec.frames // 2, spec.frames // 2 + run_span)),
                    transient_frames=transient,
                    anatomy_complexity=complexity,
                    access_route=access_route,
                    operator_stratum=operator,
                    clinical_arm="retrospective" if index % 3 else "prospective",
                )
            )
    return CohortFixture(records=tuple(records), spec=spec, seed=seed)


def default_fixture(seed: int = 20260101, per_site: Mapping[str, int] | None = None) -> CohortFixture:
    sizing = dict(per_site) if per_site is not None else {"A": 24, "B": 20, "C": 16}
    coverage = {"A": 0.71, "B": 0.64, "C": 0.49}
    complexity = {
        "A": ("simple", "intermediate", "complex"),
        "B": ("intermediate", "simple", "complex"),
        "C": ("complex", "intermediate", "simple"),
    }
    return build_fixture(
        spec=FixtureSpec(),
        per_site=sizing,
        coverage_by_site=coverage,
        complexity_by_site=complexity,
        access_route_cycle=ACCESS_ROUTES,
        operator_cycle=OPERATOR_STRATA,
        seed=seed,
    )


def fixture_motion_tensor(fixture: CohortFixture) -> Tensor:
    if not fixture.records:
        raise ValueError("fixture carries no procedures")
    return torch.stack([record.motion for record in fixture.records], dim=0)


def fixture_appearance_tensor(fixture: CohortFixture) -> Tensor:
    if not fixture.records:
        raise ValueError("fixture carries no procedures")
    return torch.stack([record.appearance for record in fixture.records], dim=0)


def fixture_shape_report(fixture: CohortFixture) -> FixtureShapeReport:
    motion = fixture_motion_tensor(fixture)
    appearance = fixture_appearance_tensor(fixture)
    return {
        "procedures": len(fixture.records),
        "appearance_shape": list(appearance.shape),
        "motion_shape": list(motion.shape),
        "expected_motion_dim": MOTION_DIM,
        "force_covered": len(fixture.force_covered()),
    }
