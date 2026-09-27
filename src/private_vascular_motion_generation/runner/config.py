"""Composition of study configuration from layered YAML fragments and CLI overrides.

Ref: Sec. 4.9 (the reported cells are the mean over three independent training occasions).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..accounting.orders import DEFAULT_ORDER_GRID, OrderGrid
from ..branches.adapter import AdapterSpec
from ..branches.appearance import AppearanceConfig
from ..branches.decoder import JointDecoderConfig
from ..branches.pairing import PairingSpec
from ..coupling.latent import CouplingLatentSpec
from ..coupling.prefix_sum import WindowSpec
from ..estimator.reference_estimator import EstimatorSpec
from ..geometry.shaped_covariance import ShapedCovariance
from ..substrate import SubstrateSpec
from .optim import OptimisationPlan
from .seeding import DEFAULT_SEED

STUDY_ROOT = Path("configs/study")
FRAGMENT_ROOTS: Mapping[str, Path] = {
    "scheme": Path("configs/scheme"),
    "cohort": Path("configs/cohort"),
    "runner": Path("configs/runner"),
}


class ConfigError(ValueError):
    """Raised when a study file is missing a required fragment."""


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"configuration fragment is not staged at {path.name}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError(f"{path.name} must hold a mapping")
    return payload


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _coerce(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(config: Mapping[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply dotted key=value overrides, matching the CLI surface."""
    resolved = copy.deepcopy(dict(config))
    for override in overrides:
        if "=" not in override:
            raise ConfigError(f"override must be key=value, got {override}")
        key, _, raw = override.partition("=")
        keys = key.strip().split(".")
        cursor: dict[str, Any] = resolved
        for step in keys[:-1]:
            nested = cursor.get(step)
            if not isinstance(nested, dict):
                nested = {}
                cursor[step] = nested
            cursor = nested
        cursor[keys[-1]] = _coerce(raw.strip())
    return resolved


@dataclass(frozen=True)
class StudyConfig:
    name: str
    payload: dict[str, Any]
    source: str

    def section(self, *keys: str) -> dict[str, Any]:
        cursor: Any = self.payload
        for key in keys:
            if not isinstance(cursor, dict) or key not in cursor:
                raise ConfigError(f"missing configuration section {'/'.join(keys)}")
            cursor = cursor[key]
        if not isinstance(cursor, dict):
            raise ConfigError(f"section {'/'.join(keys)} must be a mapping")
        return cursor

    def value(self, *keys: str, default: Any = None) -> Any:
        cursor: Any = self.payload
        for key in keys:
            if not isinstance(cursor, dict) or key not in cursor:
                if default is None:
                    raise ConfigError(f"missing configuration key {'/'.join(keys)}")
                return default
            cursor = cursor[key]
        return cursor

    def appearance_config(self) -> AppearanceConfig:
        section = self.section("branch", "appearance")
        return AppearanceConfig(**section)

    def adapter_spec(self) -> AdapterSpec:
        section = self.section("substrate")
        return AdapterSpec(dimension=int(section["dimension"]), micro_steps=int(section["window_length"]))

    def latent_spec(self) -> CouplingLatentSpec:
        section = self.section("coupling")
        return CouplingLatentSpec(dimension=int(section["latent_dim"]))

    def pairing_spec(self) -> PairingSpec:
        section = self.section("coupling")
        return PairingSpec(
            shared_latent=bool(section.get("shared_latent", True)),
            coupling_head=bool(section.get("coupling_head", True)),
            window_micro_steps=int(self.section("substrate")["window_length"]),
        )

    def estimator_spec(self) -> EstimatorSpec:
        section = self.section("estimator")
        return EstimatorSpec(
            reference_batch_size=int(section["reference_batch_size"]),
            rank=int(self.section("substrate")["rank"]),
            max_minibatches=section.get("max_minibatches"),
        )

    def substrate_spec(self) -> SubstrateSpec:
        section = self.section("substrate")
        return SubstrateSpec(
            dimension=int(section["dimension"]),
            window_length=int(section["window_length"]),
            sigma_parallel=float(section["sigma_parallel"]),
            sigma_perpendicular=float(section["sigma_perpendicular"]),
            rank=int(section["rank"]),
            clipping_norm=float(section["clipping_norm"]),
            sampling_rate=float(section["sampling_rate"]),
            rounds=int(section["rounds"]),
            delta=float(section["delta"]),
        )

    def covariance(self) -> ShapedCovariance:
        spec = self.substrate_spec()
        return ShapedCovariance(
            sigma_parallel=spec.sigma_parallel,
            sigma_perpendicular=spec.sigma_perpendicular,
            rank=spec.rank,
            dimension=spec.dimension,
        )

    def window_spec(self) -> WindowSpec:
        section = self.section("substrate")
        return WindowSpec(
            length=int(section["window_length"]),
            micro_steps=int(section["window_length"]),
            uniform=bool(section.get("uniform_window", True)),
            truncated=bool(section.get("truncated_window", False)),
        )

    def decoder_config(self) -> JointDecoderConfig:
        section = self.section("branch", "decoder")
        return JointDecoderConfig(**section)

    def optimisation_plan(self) -> OptimisationPlan:
        return OptimisationPlan(**self.section("optimiser"))

    def order_grid(self) -> OrderGrid:
        section = self.section("accounting").get("order_grid")
        if not section:
            return DEFAULT_ORDER_GRID
        orders = tuple(float(order) for order in section)
        return OrderGrid(orders)

    def seed(self) -> int:
        return int(self.value("run", "seed", default=DEFAULT_SEED))


def resolve_study(
    name: str,
    overrides: Sequence[str] | None = None,
    root: Path = STUDY_ROOT,
    depth: int = 0,
) -> StudyConfig:
    """Layer a study file over its named base and over the referenced fragments.

    Ref: Sec. 4.9 (each reported cell is one study arm on the same protocol).
    """
    if depth > 4:
        raise ConfigError("study inheritance is too deep")
    path = root / f"{name}.yaml"
    payload = load_yaml(path)
    parent = payload.get("base")
    if isinstance(parent, str):
        inherited = resolve_study(parent, None, root, depth + 1).payload
        merged: dict[str, Any] = deep_merge(inherited, payload)
    else:
        merged = dict(payload)
    for fragment_key, fragment_root in FRAGMENT_ROOTS.items():
        fragment = merged.get(fragment_key)
        if isinstance(fragment, str):
            merged = deep_merge(load_yaml(fragment_root / f"{fragment}.yaml"), merged)
    if overrides:
        merged = apply_overrides(merged, overrides)
    return StudyConfig(name=name, payload=merged, source=str(path))
