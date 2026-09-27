"""The public-reference shaping estimator.

Ref: Algorithm 2, Sec. 4.7.
"""

from __future__ import annotations

from .reference_estimator import (
    REFERENCE_BATCH_SIZE,
    EstimatorResult,
    EstimatorSpec,
    ReferenceShapingEstimator,
    cost_estimate,
    subspace_from_private_records,
)

__all__ = [
    "REFERENCE_BATCH_SIZE",
    "EstimatorResult",
    "EstimatorSpec",
    "ReferenceShapingEstimator",
    "cost_estimate",
    "subspace_from_private_records",
]
