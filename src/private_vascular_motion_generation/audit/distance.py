"""Distance to the closest record: an informative diagnostic, never the location definition.

Ref: Sec. 2.2.
"""

from __future__ import annotations

import torch
from torch import Tensor


def distance_to_closest_record(generated: Tensor, real: Tensor, batch_size: int = 256, metric: str = "l2") -> Tensor:
    """Minimum distance from each generated record to the real cohort."""
    if generated.dim() < 2 or real.dim() < 2:
        raise ValueError("records must carry a leading record axis")
    if generated.shape[1:] != real.shape[1:]:
        raise ValueError("generated and real records must share their per-record extent")
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    flat_generated = generated.reshape(generated.shape[0], -1).to(torch.float64)
    flat_real = real.reshape(real.shape[0], -1).to(torch.float64)
    minima = torch.empty(flat_generated.shape[0], dtype=torch.float64)
    for start in range(0, flat_generated.shape[0], batch_size):
        block = flat_generated[start : start + batch_size]
        distances = _pairwise(block, flat_real, metric)
        minima[start : start + block.shape[0]] = distances.min(dim=1).values
    return minima


def _pairwise(left: Tensor, right: Tensor, metric: str) -> Tensor:
    if metric == "l2":
        return torch.cdist(left, right)
    if metric == "l1":
        return torch.cdist(left, right, p=1)
    raise ValueError("unsupported distance metric")


def dcr_summary(values: Tensor) -> dict[str, float]:
    if values.dim() != 1 or values.numel() == 0:
        raise ValueError("distance vector must be non-empty")
    return {
        "min": float(values.min().item()),
        "median": float(values.median().item()),
        "mean": float(values.mean().item()),
        "max": float(values.max().item()),
    }


def duplicate_fraction(values: Tensor, tolerance: float) -> float:
    """Fraction of generated records sitting within tolerance of a real record."""
    if values.dim() != 1 or values.numel() == 0:
        raise ValueError("distance vector must be non-empty")
    if tolerance < 0.0:
        raise ValueError("tolerance cannot be negative")
    return float((values <= tolerance).to(torch.float64).mean().item())


def dcr_against_marginals(
    generated_appearance: Tensor,
    generated_motion: Tensor,
    real_appearance: Tensor,
    real_motion: Tensor,
) -> dict[str, float]:
    """The diagnostic is reported per modality and never promoted to the main evidence.

    Ref: Sec. 2.2.
    """
    appearance = dcr_summary(distance_to_closest_record(generated_appearance, real_appearance))
    motion = dcr_summary(distance_to_closest_record(generated_motion, real_motion))
    return {f"appearance_{key}": value for key, value in appearance.items()} | {
        f"motion_{key}": value for key, value in motion.items()
    }
