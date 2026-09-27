"""Resampling tests used by the pre-specified and exploratory comparisons.

Ref: Supplementary Table SE1 (paired Wilcoxon over twenty initialisations, paired
permutation over the held-out procedures, paired bootstrap over readers).
"""

from __future__ import annotations

import torch
from torch import Tensor


def paired_permutation_test(first: Tensor, second: Tensor, permutations: int, seed: int) -> dict[str, float]:
    """Two-sided sign-flip permutation test on paired differences."""
    if first.shape != second.shape or first.dim() != 1:
        raise ValueError("paired vectors must be aligned and one-dimensional")
    if permutations < 1:
        raise ValueError("at least one permutation is required")
    differences = (first - second).to(torch.float64)
    observed = float(differences.mean().item())
    spread = float(differences.std(unbiased=True).item())
    generator = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (permutations, differences.numel()), generator=generator).to(torch.float64)
    signs = signs * 2.0 - 1.0
    null = (signs * differences.unsqueeze(0)).mean(dim=1)
    extreme = int((null.abs() >= abs(observed)).sum().item())
    z_statistic = observed / (spread / differences.numel() ** 0.5) if spread > 0.0 else 0.0
    return {
        "mean_difference": observed,
        "z": float(z_statistic),
        "p_value": (extreme + 1.0) / (permutations + 1.0),
    }


def wilcoxon_signed_rank(first: Tensor, second: Tensor) -> dict[str, float]:
    """Exact-tail-free signed-rank statistic on the paired differences."""
    if first.shape != second.shape or first.dim() != 1:
        raise ValueError("paired vectors must be aligned and one-dimensional")
    differences = (first - second).to(torch.float64)
    nonzero = differences[differences != 0]
    if nonzero.numel() == 0:
        raise ValueError("all paired differences are zero")
    ranks = torch.argsort(torch.argsort(nonzero.abs())).to(torch.float64) + 1.0
    positive = float(ranks[nonzero > 0].sum().item())
    negative = float(ranks[nonzero < 0].sum().item())
    statistic = min(positive, negative)
    total = nonzero.numel()
    mean = total * (total + 1) / 4.0
    variance = total * (total + 1) * (2 * total + 1) / 24.0
    z_statistic = (statistic - mean) / variance**0.5
    p_value = float(1.0 + torch.erf(torch.tensor(z_statistic / 2.0**0.5, dtype=torch.float64)).item())
    return {"w": statistic, "z": z_statistic, "p_value": min(1.0, max(0.0, p_value))}


def paired_bootstrap_delta(
    first: Tensor, second: Tensor, resamples: int, seed: int, level: float = 0.95
) -> dict[str, float]:
    """Percentile interval for the paired difference, used for the reader study."""
    if first.shape != second.shape or first.dim() != 1:
        raise ValueError("paired vectors must be aligned and one-dimensional")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")
    differences = (first - second).to(torch.float64)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(0, differences.numel(), (resamples, differences.numel()), generator=generator)
    draws = differences[indices].mean(dim=1)
    tail = (1.0 - level) / 2.0
    return {
        "delta": float(differences.mean().item()),
        "low": float(torch.quantile(draws, tail).item()),
        "high": float(torch.quantile(draws, 1.0 - tail).item()),
        "p_value": float(
            2.0 * min((draws <= 0).to(torch.float64).mean().item(), (draws >= 0).to(torch.float64).mean().item())
        ),
    }


def two_proportion_z(
    successes_first: int, total_first: int, successes_second: int, total_second: int
) -> dict[str, float]:
    """Pooled two-proportion z test on two independent strata."""
    if min(total_first, total_second) <= 0:
        raise ValueError("both denominators must be positive")
    if not (0 <= successes_first <= total_first and 0 <= successes_second <= total_second):
        raise ValueError("success counts must lie within their denominators")
    first_rate = successes_first / total_first
    second_rate = successes_second / total_second
    pooled = (successes_first + successes_second) / (total_first + total_second)
    standard_error = (pooled * (1.0 - pooled) * (1.0 / total_first + 1.0 / total_second)) ** 0.5
    if standard_error == 0.0:
        raise ValueError("degenerate pooled standard error")
    z_statistic = (first_rate - second_rate) / standard_error
    p_value = float(1.0 + torch.erf(torch.tensor(-abs(z_statistic) / 2.0**0.5, dtype=torch.float64)).item())
    return {
        "rate_first": first_rate,
        "rate_second": second_rate,
        "z": z_statistic,
        "p_value": min(1.0, p_value),
        "delta_pp": (first_rate - second_rate) * 100.0,
    }
