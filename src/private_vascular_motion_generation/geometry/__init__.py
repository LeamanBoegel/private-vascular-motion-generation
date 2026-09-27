"""Noise geometry: projectors, the coupling-shaped covariance and its concentration measure.

Ref: Eq. (1), Eq. (4), Sec. 4.4, Sec. 4.6.
"""

from __future__ import annotations

from .concentration import (
    concentration_from_gradients,
    off_coupling_energy,
    on_coupling_energy,
    residual_energy_trace,
)
from .projector import (
    CouplingProjector,
    complement_projector,
    projector_from_directions,
    projector_residual,
)
from .shaped_covariance import (
    ShapedCovariance,
    covariance_from_projector,
    normalised_sensitivity,
    normalised_sensitivity_isotropic,
    sample_shaped_noise,
)

__all__ = [
    "CouplingProjector",
    "ShapedCovariance",
    "complement_projector",
    "concentration_from_gradients",
    "covariance_from_projector",
    "normalised_sensitivity",
    "normalised_sensitivity_isotropic",
    "off_coupling_energy",
    "on_coupling_energy",
    "projector_from_directions",
    "projector_residual",
    "residual_energy_trace",
    "sample_shaped_noise",
]
