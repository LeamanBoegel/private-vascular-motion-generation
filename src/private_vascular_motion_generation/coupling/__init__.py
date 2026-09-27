"""Coupling between the appearance and motion branches of the paired procedure record.

Ref: Sec. 4.4, Algorithm 1, Algorithm 2.
"""

from __future__ import annotations

from .cca import (
    CanonicalSolution,
    CovarianceTriple,
    canonical_orthonormality,
    coupling_loss,
    generalised_eigen_residual,
    inverse_sqrt,
    paired_covariances,
    solve_canonical,
)
from .head import (
    DEFAULT_COUPLING_RANK,
    CouplingHead,
    CouplingHeadSpec,
    canonical_spectrum_report,
    head_removal_contrast,
)
from .latent import (
    LATENT_DIM,
    CouplingLatent,
    CouplingLatentSpec,
    draw_latent,
    encode_procedure_identity,
    latent_from_embeddings,
)
from .prefix_sum import (
    DEFAULT_WINDOW_LENGTH,
    REPORTED_COMPOSITION_MULTIPLIER,
    REPORTED_CONTACT_F1,
    REPORTED_RELATIVE_LATENCY,
    REPORTED_TRUNCATION_RATE,
    WINDOW_LADDER,
    PrefixFactorisation,
    WindowSpec,
    composition_multiplier_ladder,
    fit_log_ladder,
    group_multiplier,
    prefix_sum_inverse,
    prefix_sum_scan,
    truncation_rate,
    windowed_workload,
)

__all__ = [
    "DEFAULT_COUPLING_RANK",
    "DEFAULT_WINDOW_LENGTH",
    "LATENT_DIM",
    "REPORTED_COMPOSITION_MULTIPLIER",
    "REPORTED_CONTACT_F1",
    "REPORTED_RELATIVE_LATENCY",
    "REPORTED_TRUNCATION_RATE",
    "WINDOW_LADDER",
    "CanonicalSolution",
    "CouplingHead",
    "CouplingHeadSpec",
    "CouplingLatent",
    "CouplingLatentSpec",
    "CovarianceTriple",
    "PrefixFactorisation",
    "WindowSpec",
    "canonical_orthonormality",
    "canonical_spectrum_report",
    "composition_multiplier_ladder",
    "coupling_loss",
    "draw_latent",
    "encode_procedure_identity",
    "fit_log_ladder",
    "generalised_eigen_residual",
    "group_multiplier",
    "head_removal_contrast",
    "inverse_sqrt",
    "latent_from_embeddings",
    "paired_covariances",
    "prefix_sum_inverse",
    "prefix_sum_scan",
    "solve_canonical",
    "truncation_rate",
    "windowed_workload",
]
