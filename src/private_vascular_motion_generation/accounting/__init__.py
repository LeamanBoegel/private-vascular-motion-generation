"""Rényi accounting for the coupling-shaped Gaussian mechanism.

Ref: Eq. (3), Eq. (4), Sec. 4.6, Algorithm 1 step 13.
"""

from __future__ import annotations

from .clipping import ClipOutcome, clip_procedure_gradient, clip_workload, clipping_norm_report
from .orders import DEFAULT_ORDER_GRID, OrderGrid, select_order
from .renyi import (
    PREAMPLIFICATION_TERM,
    PrivacyBudget,
    composed_rdp,
    convert_to_delta,
    convert_to_epsilon,
    epsilon_at_parity,
    monotone_in_noise,
    prefix_amplification_term,
    rdp_binomial_integer,
    rdp_from_noise_multiplier,
    rdp_sampled_gaussian,
    rdp_shaped_mechanism,
)
from .tuple import TUPLE_FIELDS, PrivacyTuple, bare_budget_warning, conservative_ratio, tuple_is_complete

__all__ = [
    "DEFAULT_ORDER_GRID",
    "MARGINAL_CLIPPING_NORM",
    "PREAMPLIFICATION_TERM",
    "TUPLE_FIELDS",
    "ClipOutcome",
    "OrderGrid",
    "PrivacyBudget",
    "PrivacyTuple",
    "bare_budget_warning",
    "clip_procedure_gradient",
    "clip_workload",
    "clipping_norm_report",
    "compare_clipping_norms",
    "composed_rdp",
    "conservative_ratio",
    "convert_to_delta",
    "convert_to_epsilon",
    "epsilon_at_parity",
    "monotone_in_noise",
    "no_post_noise_clipping",
    "prefix_amplification_term",
    "rdp_binomial_integer",
    "rdp_from_noise_multiplier",
    "rdp_sampled_gaussian",
    "rdp_shaped_mechanism",
    "select_order",
    "tuple_is_complete",
]
