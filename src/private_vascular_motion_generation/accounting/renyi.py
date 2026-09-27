"""Rényi differential privacy of the sub-sampled coupling-shaped Gaussian mechanism.

Ref: Eq. (3), Eq. (4), Sec. 4.6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .orders import DEFAULT_ORDER_GRID, OrderGrid

QUADRATURE_NODES: int = 96
PREAMPLIFICATION_TERM: float = 0.0


@dataclass(frozen=True)
class PrivacyBudget:
    epsilon: float
    delta: float
    order: float
    accountant: str = "renyi"

    def __post_init__(self) -> None:
        if self.epsilon < 0.0:
            raise ValueError("epsilon cannot be negative")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must lie in (0, 1)")
        if self.order <= 1.0:
            raise ValueError("order must exceed one")

    def conservative_union(self, other: PrivacyBudget) -> PrivacyBudget:
        if not math.isclose(self.delta, other.delta):
            raise ValueError("budgets must share a delta to be unioned")
        return PrivacyBudget(
            epsilon=max(self.epsilon, other.epsilon),
            delta=self.delta,
            order=self.order if self.epsilon >= other.epsilon else other.order,
            accountant=self.accountant,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "order": self.order,
            "accountant": self.accountant,
        }


def _hermite_nodes(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.hermite.hermgauss(nodes)
    return x, w


def rdp_binomial_integer(order: int, sampling_rate: float, noise_multiplier: float) -> float:
    """Closed form for integer Rényi orders under the sub-sampled Gaussian mechanism.

    Ref: Eq. (3) (per-round term).
    """
    if order < 2:
        raise ValueError("integer closed form requires order >= 2")
    if not 0.0 < sampling_rate <= 1.0:
        raise ValueError("sampling rate must lie in (0, 1]")
    if noise_multiplier <= 0.0:
        raise ValueError("noise multiplier must be positive")
    if sampling_rate == 1.0:
        return order / (2.0 * noise_multiplier**2)
    logs = []
    for count in range(order + 1):
        log_binom = math.lgamma(order + 1) - math.lgamma(count + 1) - math.lgamma(order - count + 1)
        log_term = (
            log_binom
            + count * math.log(sampling_rate)
            + (order - count) * math.log1p(-sampling_rate)
            + count * (count - 1) / (2.0 * noise_multiplier**2)
        )
        logs.append(log_term)
    peak = max(logs)
    accumulated = peak + math.log(sum(math.exp(value - peak) for value in logs))
    return accumulated / (order - 1)


def rdp_sampled_gaussian(order: float, sampling_rate: float, noise_multiplier: float) -> float:
    """Expectation form, valid at any real order above one.

    Ref: Eq. (3).
    """
    if order <= 1.0:
        raise ValueError("order must exceed one")
    if not 0.0 < sampling_rate <= 1.0:
        raise ValueError("sampling rate must lie in (0, 1]")
    if noise_multiplier <= 0.0:
        raise ValueError("noise multiplier must be positive")
    if sampling_rate == 1.0:
        return order / (2.0 * noise_multiplier**2)
    nodes, weights = _hermite_nodes(QUADRATURE_NODES)
    z = np.sqrt(2.0) * noise_multiplier * nodes
    log_ratio = (2.0 * z - 1.0) / (2.0 * noise_multiplier**2)
    ratio = np.clip(np.exp(np.clip(log_ratio, -700.0, 700.0)), 0.0, None)
    integrand = (1.0 - sampling_rate + sampling_rate * ratio) ** order
    expectation = float(np.sum(weights * integrand) / math.sqrt(math.pi))
    if expectation <= 0.0 or not math.isfinite(expectation):
        raise ValueError("quadrature produced a degenerate divergence")
    return math.log(expectation) / (order - 1)


def rdp_from_noise_multiplier(
    order: float, sampling_rate: float, noise_multiplier: float, use_closed_form: bool = True
) -> float:
    if use_closed_form and float(order).is_integer() and order >= 2:
        return rdp_binomial_integer(int(order), sampling_rate, noise_multiplier)
    return rdp_sampled_gaussian(order, sampling_rate, noise_multiplier)


def rdp_shaped_mechanism(
    order: float,
    sampling_rate: float,
    sigma_parallel: float,
    sigma_perpendicular: float,
    rho: float,
    clipping_norm: float,
    use_closed_form: bool = True,
) -> float:
    """Per-round charge after converting the shaped geometry to a noise multiplier.

    Ref: Eq. (4) (u is the noise-normalised sensitivity), Eq. (3).
    """
    from ..geometry.shaped_covariance import normalised_sensitivity

    if clipping_norm <= 0.0:
        raise ValueError("clipping norm must be positive")
    u = normalised_sensitivity(sigma_parallel, sigma_perpendicular, rho)
    noise_multiplier = 1.0 / (clipping_norm * u)
    return rdp_from_noise_multiplier(order, sampling_rate, noise_multiplier, use_closed_form)


def prefix_amplification_term(window_length: int, order: float) -> float:
    """O(log^2 k / alpha) slack carried by the prefix-structured window.

    Ref: Eq. (3) (final term), Sec. 4.6.
    """
    if window_length < 1:
        raise ValueError("window length must be positive")
    if order <= 1.0:
        raise ValueError("order must exceed one")
    return float(math.log2(window_length + 1) ** 2 / order)


def composed_rdp(
    grid: OrderGrid,
    rounds: int,
    sampling_rate: float,
    noise_multiplier: float,
    use_closed_form: bool = True,
) -> Tensor:
    """Rényi composition is additive at a fixed order.

    Ref: Eq. (3) (sum over the K rounds).
    """
    if rounds < 1:
        raise ValueError("rounds must be at least one")
    if not 0.0 < sampling_rate <= 1.0:
        raise ValueError("sampling rate must lie in (0, 1]")
    per_round = [
        rdp_from_noise_multiplier(order, sampling_rate, noise_multiplier, use_closed_form) for order in grid.orders
    ]
    return torch.tensor(per_round, dtype=torch.float64) * rounds


def convert_to_epsilon(rdp_vector: Tensor, grid: OrderGrid, delta: float, window_length: int = 1) -> PrivacyBudget:
    """epsilon = min_alpha [ eps_alpha + log(1/delta)/(alpha - 1) ].

    Ref: Eq. (3).
    """
    if rdp_vector.dim() != 1 or rdp_vector.numel() != len(grid.orders):
        raise ValueError("RDP vector must align with the order grid")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    orders = grid.as_tensor()
    slack = torch.tensor(
        [prefix_amplification_term(window_length, float(order)) for order in grid.orders],
        dtype=torch.float64,
    )
    converted = rdp_vector + math.log(1.0 / delta) / (orders - 1.0) + slack + PREAMPLIFICATION_TERM
    order, epsilon = grid.best(converted)
    return PrivacyBudget(epsilon=epsilon, delta=delta, order=order)


def convert_to_delta(rdp_vector: Tensor, grid: OrderGrid, epsilon: float) -> PrivacyBudget:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    orders = grid.as_tensor()
    deltas = torch.exp((orders - 1.0) * (rdp_vector - epsilon))
    index = int(torch.argmin(deltas).item())
    return PrivacyBudget(epsilon=epsilon, delta=float(deltas[index].item()), order=float(orders[index].item()))


def epsilon_at_parity(
    grid: OrderGrid,
    rounds: int,
    sampling_rate: float,
    noise_multiplier: float,
    delta: float,
    use_closed_form: bool = True,
) -> PrivacyBudget:
    vector = composed_rdp(grid, rounds, sampling_rate, noise_multiplier, use_closed_form)
    return convert_to_epsilon(vector, grid, delta)


def default_budget(rounds: int, sampling_rate: float, noise_multiplier: float, delta: float = 1e-5) -> PrivacyBudget:
    return epsilon_at_parity(DEFAULT_ORDER_GRID, rounds, sampling_rate, noise_multiplier, delta)


def monotone_in_noise(
    grid: OrderGrid, rounds: int, sampling_rate: float, low: float, high: float, delta: float
) -> bool:
    """More noise must never buy less privacy at a fixed order grid.

    Ref: Eq. (3).
    """
    grid_points = torch.linspace(low, high, 12, dtype=torch.float64).tolist()
    budgets = [epsilon_at_parity(grid, rounds, sampling_rate, multiplier, delta).epsilon for multiplier in grid_points]
    return all(later <= earlier + 1e-9 for earlier, later in zip(budgets, budgets[1:]))


def structure_versus_concentration(sigma_parallel: float, sigma_perpendicular: float, rho: float) -> dict[str, object]:
    """Separate the structural condition (bound) from the concentration condition (gain).

    Ref: Sec. 4.6, Sec. 2.2.
    """
    from ..geometry.shaped_covariance import anisotropy_admissible, normalised_sensitivity

    admissible = anisotropy_admissible(sigma_parallel, sigma_perpendicular)
    if not admissible:
        raise ValueError("the declared multipliers do not satisfy the admissibility premise")
    return {
        "structural_condition": "uniform, untruncated prefix window",
        "concentration_condition_gain_rho": rho,
        "normalised_sensitivity": normalised_sensitivity(sigma_parallel, sigma_perpendicular, rho),
        "isotropic_degradation_holds_guarantee": True,
    }
