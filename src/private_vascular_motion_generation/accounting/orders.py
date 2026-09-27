"""Rényi-order grids over which the accountant minimises its reported budget.

Ref: Sec. 4.6 ("alpha represents the Rényi order ... whose epsilon is the lowest
obtainable in the order grid evaluated by the accountant").
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

DEFAULT_INTEGER_ORDERS: tuple[int, ...] = (*tuple(range(2, 65)), 96, 128, 192, 256)


@dataclass(frozen=True)
class OrderGrid:
    orders: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.orders) < 2:
            raise ValueError("an order grid must carry at least two orders")
        if any(order <= 1.0 for order in self.orders):
            raise ValueError("Rényi orders must exceed one")
        if list(self.orders) != sorted(self.orders):
            raise ValueError("order grid must be increasing")

    def as_tensor(self) -> Tensor:
        return torch.tensor(self.orders, dtype=torch.float64)

    def best(self, budgets: Tensor) -> tuple[float, float]:
        if budgets.dim() != 1 or budgets.numel() != len(self.orders):
            raise ValueError("budget vector must align with the order grid")
        index = int(torch.argmin(budgets).item())
        return float(self.orders[index]), float(budgets[index].item())


DEFAULT_ORDER_GRID = OrderGrid(DEFAULT_INTEGER_ORDERS)


def interpolated_grid(low: float, high: float, count: int) -> OrderGrid:
    if count < 2:
        raise ValueError("count must be at least two")
    if low <= 1.0 or high <= low:
        raise ValueError("order range must be increasing and above one")
    steps = torch.linspace(low, high, count, dtype=torch.float64).tolist()
    return OrderGrid(tuple(steps))


def integer_grid(orders: Sequence[int]) -> OrderGrid:
    return OrderGrid(tuple(float(order) for order in orders))


def select_order(grid: OrderGrid, budgets: Tensor) -> tuple[float, float]:
    return grid.best(budgets)
