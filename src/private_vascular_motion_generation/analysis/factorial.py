"""Resolution x conditioning factorial and its contrasts.

Ref: Table 3, Sec. 2.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch
from torch import Tensor

CONDITIONING_ON_RESOLUTION_1 = "on_1x"
CONDITIONING_ON_RESOLUTION_2 = "on_2x"
CONDITIONING_OFF_RESOLUTION_1 = "off_1x"
CONDITIONING_OFF_RESOLUTION_2 = "off_2x"
CELLS: tuple[str, ...] = (
    CONDITIONING_ON_RESOLUTION_1,
    CONDITIONING_ON_RESOLUTION_2,
    CONDITIONING_OFF_RESOLUTION_1,
    CONDITIONING_OFF_RESOLUTION_2,
)
CONDITIONING_CELLS: tuple[str, ...] = (CONDITIONING_ON_RESOLUTION_1, CONDITIONING_ON_RESOLUTION_2)
UNCONDITIONED_CELLS: tuple[str, ...] = (CONDITIONING_OFF_RESOLUTION_1, CONDITIONING_OFF_RESOLUTION_2)
REFERENCE_RESOLUTION_CELLS: tuple[str, ...] = (CONDITIONING_ON_RESOLUTION_1, CONDITIONING_OFF_RESOLUTION_1)
DOUBLED_RESOLUTION_CELLS: tuple[str, ...] = (CONDITIONING_ON_RESOLUTION_2, CONDITIONING_OFF_RESOLUTION_2)


@dataclass(frozen=True)
class FactorialCell:
    label: str
    fid: float
    dtw: float
    contact_event_f1: float
    rho: float
    epsilon_at_parity: float


class ContrastRow(TypedDict):
    metric: str
    conditioning_main_effect: float
    resolution_main_effect: float
    interaction: float
    dominance_ratio: float


@dataclass(frozen=True)
class FactorialContrast:
    metric: str
    conditioning_main_effect: float
    resolution_main_effect: float
    interaction: float

    def dominance_ratio(self) -> float:
        if self.resolution_main_effect == 0.0:
            raise ValueError("resolution main effect is zero")
        return abs(self.conditioning_main_effect) / abs(self.resolution_main_effect)

    def to_dict(self) -> ContrastRow:
        return {
            "metric": self.metric,
            "conditioning_main_effect": self.conditioning_main_effect,
            "resolution_main_effect": self.resolution_main_effect,
            "interaction": self.interaction,
            "dominance_ratio": self.dominance_ratio(),
        }


def _cell_value(cells: dict[str, FactorialCell], labels: tuple[str, ...], metric: str) -> float:
    values = [float(getattr(cells[label], metric)) for label in labels]
    return sum(values) / len(values)


def cell_is_complete(cells: dict[str, FactorialCell]) -> bool:
    return all(label in cells for label in CELLS)


def conditioning_main_effect(cells: dict[str, FactorialCell], metric: str) -> float:
    """Mean over the conditioning-on cells minus the mean over the conditioning-off cells."""
    return _cell_value(cells, CONDITIONING_CELLS, metric) - _cell_value(cells, UNCONDITIONED_CELLS, metric)


def resolution_main_effect(cells: dict[str, FactorialCell], metric: str) -> float:
    """Mean over the doubled-resolution cells minus the mean over the reference-resolution cells."""
    return _cell_value(cells, DOUBLED_RESOLUTION_CELLS, metric) - _cell_value(cells, REFERENCE_RESOLUTION_CELLS, metric)


def interaction(cells: dict[str, FactorialCell], metric: str) -> float:
    """Difference of the resolution effects across the two conditioning levels."""
    on_effect = float(getattr(cells[CONDITIONING_ON_RESOLUTION_2], metric)) - float(
        getattr(cells[CONDITIONING_ON_RESOLUTION_1], metric)
    )
    off_effect = float(getattr(cells[CONDITIONING_OFF_RESOLUTION_2], metric)) - float(
        getattr(cells[CONDITIONING_OFF_RESOLUTION_1], metric)
    )
    return off_effect - on_effect


def contrast(cells: dict[str, FactorialCell], metric: str) -> FactorialContrast:
    return FactorialContrast(
        metric=metric,
        conditioning_main_effect=conditioning_main_effect(cells, metric),
        resolution_main_effect=resolution_main_effect(cells, metric),
        interaction=interaction(cells, metric),
    )


def full_contrast_table(cells: dict[str, FactorialCell]) -> dict[str, ContrastRow]:
    if not cell_is_complete(cells):
        raise ValueError("all four factorial cells are required")
    return {metric: contrast(cells, metric).to_dict() for metric in ("fid", "dtw", "contact_event_f1")}


def interaction_resolvable(interaction_value: float, run_to_run_sd: float, multiple: float = 1.0) -> bool:
    """An interaction smaller than the run-to-run floor carries no significance claim.

    Ref: Table 3 caption (0.009 is about 0.9 run-to-run standard deviations).
    """
    if run_to_run_sd < 0.0:
        raise ValueError("run-to-run spread cannot be negative")
    return abs(interaction_value) >= multiple * run_to_run_sd


def rho_invariance_across_resolution(cells: dict[str, FactorialCell]) -> dict[str, float]:
    """rho moves with conditioning and not with resolution.

    Ref: Sec. 2.3.
    """
    if not cell_is_complete(cells):
        raise ValueError("all four factorial cells are required")
    return {
        "on_1x_minus_on_2x": cells[CONDITIONING_ON_RESOLUTION_1].rho - cells[CONDITIONING_ON_RESOLUTION_2].rho,
        "off_1x_minus_off_2x": cells[CONDITIONING_OFF_RESOLUTION_1].rho - cells[CONDITIONING_OFF_RESOLUTION_2].rho,
        "conditioning_shift": (cells[CONDITIONING_ON_RESOLUTION_1].rho + cells[CONDITIONING_ON_RESOLUTION_2].rho) / 2.0
        - (cells[CONDITIONING_OFF_RESOLUTION_1].rho + cells[CONDITIONING_OFF_RESOLUTION_2].rho) / 2.0,
    }


def budget_moves_with_conditioning(cells: dict[str, FactorialCell]) -> bool:
    """Removing conditioning moves the budget at which a fidelity level is reached.

    Ref: Sec. 2.3.
    """
    if not cell_is_complete(cells):
        raise ValueError("all four factorial cells are required")
    return all(
        cells[on].epsilon_at_parity < cells[off].epsilon_at_parity
        for on, off in zip(CONDITIONING_CELLS, UNCONDITIONED_CELLS)
    )


def vectorise(cells: dict[str, FactorialCell], metric: str) -> Tensor:
    return torch.tensor([float(getattr(cells[label], metric)) for label in CELLS], dtype=torch.float64)
