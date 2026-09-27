"""Shared fixtures for the release test suite.

Ref: Sec. 4.9 (the reported cells are the mean over three independent training occasions).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from private_vascular_motion_generation.cohort import fixture as fixture_module
from private_vascular_motion_generation.runner.config import StudyConfig, resolve_study
from private_vascular_motion_generation.runner.seeding import set_seed
from private_vascular_motion_generation.runner.session import Session, build_session

RELEASE_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def release_root() -> Path:
    return RELEASE_ROOT


@pytest.fixture(scope="session")
def smoke_config(release_root: Path) -> StudyConfig:
    return resolve_study("_smoke", root=release_root / "configs" / "study")


@pytest.fixture()
def smoke_session(smoke_config: StudyConfig) -> Session:
    set_seed(smoke_config.seed())
    return build_session(smoke_config)


@pytest.fixture()
def small_fixture():
    set_seed(101)
    return fixture_module.default_fixture(seed=101, per_site={"A": 6, "B": 4, "C": 4})


@pytest.fixture(scope="session")
def torch_generator() -> torch.Generator:
    return torch.Generator().manual_seed(20260101)
