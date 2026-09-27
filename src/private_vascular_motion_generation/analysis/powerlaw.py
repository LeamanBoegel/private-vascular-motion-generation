"""Power-law fitting and residual diagnostics for the substrate-size scaling panel.

Ref: Supplementary Table SC2 (F1 proportional to N_sub^alpha, residual in units of 1e-3).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

RESIDUAL_UNITS: float = 1e-3
MIN_FIT_POINTS: int = 3


@dataclass(frozen=True)
class PowerLawFit:
    exponent: float
    intercept: float
    r_squared: float

    def predict(self, sizes: Tensor) -> Tensor:
        if sizes.dim() != 1:
            raise ValueError("sizes must be one-dimensional")
        return torch.exp(self.intercept + self.exponent * torch.log(sizes.to(torch.float64)))

    def residuals(self, sizes: Tensor, observed: Tensor) -> Tensor:
        return observed.to(torch.float64) - self.predict(sizes)


def fit_power_law(sizes: Tensor, values: Tensor) -> PowerLawFit:
    """Least-squares fit of log F1 on log N."""
    if sizes.dim() != 1 or values.dim() != 1 or sizes.numel() != values.numel():
        raise ValueError("sizes and values must be aligned vectors")
    if sizes.numel() < MIN_FIT_POINTS:
        raise ValueError("a power law needs at least three points")
    if bool((sizes <= 0).any().item()) or bool((values <= 0).any().item()):
        raise ValueError("log-log fitting requires strictly positive entries")
    x = torch.log(sizes.to(torch.float64))
    y = torch.log(values.to(torch.float64))
    design = torch.stack([x, torch.ones_like(x)], dim=1)
    solution = torch.linalg.lstsq(design, y.unsqueeze(1)).solution.squeeze(1)
    prediction = design @ solution
    total = float(((y - y.mean()) ** 2).sum().item())
    residual = float(((y - prediction) ** 2).sum().item())
    return PowerLawFit(
        exponent=float(solution[0].item()),
        intercept=float(solution[1].item()),
        r_squared=1.0 - residual / total if total > 0.0 else 0.0,
    )


def residuals_in_units(fit: PowerLawFit, sizes: Tensor, observed: Tensor, units: float = RESIDUAL_UNITS) -> Tensor:
    return fit.residuals(sizes, observed) / units


def systematic_at_size(residuals: Tensor, sizes: Tensor, size: int) -> bool:
    """A consistent sign at one substrate size marks a departure of the fit.

    Ref: Supplementary Table SC2 caption.
    """
    selected = residuals[sizes == size]
    if selected.numel() == 0:
        raise ValueError(f"no residual is recorded at substrate size {size}")
    sign = torch.sign(selected)
    return bool(torch.all(sign == sign[0]).item())


def scaling_is_monotone(sizes: Tensor, values: Tensor) -> bool:
    order = torch.argsort(sizes)
    ordered = values[order]
    return bool(torch.all(ordered[1:] >= ordered[:-1]).item())


def sample_efficiency_exponent(sizes: Tensor, values: Tensor) -> dict[str, float]:
    """Fractionally improving contact-event F1 with substrate volume."""
    fit = fit_power_law(sizes, values)
    return {
        "exponent": fit.exponent,
        "intercept": fit.intercept,
        "r_squared": fit.r_squared,
        "gain_600_to_19200": float(values[-1].item() - values[0].item()),
    }


def budget_floor_reached(sizes: Tensor, budgets: Tensor, floor: float = 2.0) -> Tensor:
    """Substrate volumes at which the parity budget stops improving."""
    if sizes.shape != budgets.shape:
        raise ValueError("sizes and budgets must align")
    return sizes[budgets <= floor + 1e-9]


def reference_corpus_saturation(reference_sizes: Tensor, values: Tensor, tolerance: float = 1e-3) -> bool:
    """The reference-corpus panel saturates: the last two rungs are within tolerance.

    Ref: Supplementary Table T7 panel C.
    """
    if reference_sizes.numel() < 2:
        raise ValueError("at least two rungs are required")
    return abs(float(values[-1].item() - values[-2].item())) <= tolerance
