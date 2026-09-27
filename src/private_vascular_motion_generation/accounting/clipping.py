"""Per-procedure single-norm clipping applied before any noise is drawn.

Ref: Sec. 4.4 ("clipping is done once during the procedure ... before any noise is generated"),
Algorithm 1 step 8.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

DEFAULT_CLIPPING_NORM: float = 0.85
MARGINAL_CLIPPING_NORM: float = 1.00


@dataclass(frozen=True)
class ClipOutcome:
    clipped: Tensor
    scale: float
    observed_norm: float

    @property
    def was_clipped(self) -> bool:
        return self.scale < 1.0


def clip_procedure_gradient(gradient: Tensor, clipping_norm: float) -> ClipOutcome:
    """G_p <- G_p * min{1, C / ||G_p||}, taken once at a single norm.

    Ref: Algorithm 1 step 8.
    """
    if clipping_norm <= 0.0:
        raise ValueError("clipping norm must be positive")
    observed = float(torch.linalg.norm(gradient.reshape(-1)).item())
    if observed == 0.0:
        return ClipOutcome(clipped=gradient.clone(), scale=1.0, observed_norm=0.0)
    scale = min(1.0, clipping_norm / observed)
    return ClipOutcome(clipped=gradient * scale, scale=scale, observed_norm=observed)


def clip_workload(gradients: Tensor, clipping_norm: float) -> tuple[Tensor, Tensor]:
    """Clip each procedure once, then accumulate.

    Ref: Algorithm 1 steps 5-10.
    """
    if gradients.dim() < 2:
        raise ValueError("workload must carry a leading procedure axis")
    flat = gradients.reshape(gradients.shape[0], -1)
    norms = torch.linalg.norm(flat, dim=1)
    scales = torch.clamp(clipping_norm / norms.clamp_min(torch.finfo(flat.dtype).tiny), max=1.0)
    scaled = gradients * scales.reshape((-1,) + (1,) * (gradients.dim() - 1))
    return scaled, scales


def no_post_noise_clipping(noisy: Tensor, raw: Tensor, tolerance: float = 1e-6) -> bool:
    """Clipping the noisy update would invalidate the accountant's charge, so it must not happen.

    Ref: Sec. 4.6.
    """
    if noisy.shape != raw.shape:
        raise ValueError("noise and raw workload must share a shape")
    return bool(torch.allclose(noisy, raw, atol=tolerance, rtol=0.0))


def clipping_norm_report(clipping_norm: float, unit: str) -> dict[str, object]:
    if unit not in ("procedure", "example"):
        raise ValueError("privacy unit must be a procedure or an example")
    return {
        "clipping_norm": clipping_norm,
        "privacy_unit": unit,
        "per_procedure_single_norm": unit == "procedure",
    }


def compare_clipping_norms(joint_norm: float, marginal_norm: float) -> dict[str, float]:
    """The marginal rows use C = 1.00 and the joint rows C = 0.85 in the frontier table.

    Ref: Table 2 caption.
    """
    if joint_norm <= 0.0 or marginal_norm <= 0.0:
        raise ValueError("clipping norms must be positive")
    return {"ratio": marginal_norm / joint_norm, "joint": joint_norm, "marginal": marginal_norm}
