"""Precision policy and distributed wrapping kept behind explicit switches.

Ref: Sec. 4.4 (the private pass runs on the frozen backbone), Algorithm 1 step 2.
"""

from __future__ import annotations

import contextlib
import os
from contextlib import AbstractContextManager
from dataclasses import dataclass

import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel

ACCELERATOR: str = "cuda"


@dataclass(frozen=True)
class PrecisionPolicy:
    enabled: bool = True
    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if self.dtype not in ("float16", "bfloat16"):
            raise ValueError("autocast dtype must be float16 or bfloat16")

    def autocast(self) -> AbstractContextManager[None]:
        if not self.enabled or not torch.cuda.is_available():
            return contextlib.nullcontext()
        resolved = torch.bfloat16 if self.dtype == "bfloat16" else torch.float16
        return torch.autocast(device_type=ACCELERATOR, dtype=resolved)


def autocast_context(policy: PrecisionPolicy) -> AbstractContextManager[None]:
    return policy.autocast()


def uses_grad_scaler(policy: PrecisionPolicy) -> bool:
    return policy.enabled and policy.dtype == "float16" and torch.cuda.is_available()


def build_grad_scaler(policy: PrecisionPolicy) -> GradScaler:
    return GradScaler(ACCELERATOR, enabled=uses_grad_scaler(policy))


@dataclass(frozen=True)
class DistributedPlan:
    backend: str = "nccl"
    world_size: int = 1
    enabled: bool = False

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError("world size must be positive")


def is_distributed_launch() -> bool:
    return (
        all(key in os.environ for key in ("RANK", "WORLD_SIZE", "MASTER_ADDR"))
        and int(os.environ.get("WORLD_SIZE", "1")) > 1
    )


def initialise_process_group(plan: DistributedPlan) -> bool:
    if not plan.enabled or not is_distributed_launch():
        return False
    from torch import distributed

    if distributed.is_initialized():
        return True
    distributed.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"]),
    )
    return True


def wrap_model(model: nn.Module, plan: DistributedPlan, device_index: int = 0) -> nn.Module:
    """DDP wrap when launched distributed, otherwise return the module unchanged."""
    if not initialise_process_group(plan):
        return model
    device = torch.device(ACCELERATOR, device_index) if torch.cuda.is_available() else torch.device("cpu")
    wrapped: nn.Module = DistributedDataParallel(
        model.to(device), device_ids=[device_index] if torch.cuda.is_available() else None
    )
    return wrapped


def barrier(plan: DistributedPlan) -> None:
    if not plan.enabled:
        return
    from torch import distributed

    if distributed.is_initialized():
        distributed.barrier()


def broadcast_scalar(value: float, plan: DistributedPlan) -> float:
    if not plan.enabled:
        return value
    from torch import distributed

    if not distributed.is_initialized():
        return value
    tensor = torch.tensor([value], dtype=torch.float64)
    distributed.all_reduce(tensor, op=distributed.ReduceOp.AVG)
    return float(tensor.item())


def shard_workload(workload: torch.Tensor, plan: DistributedPlan) -> torch.Tensor:
    """Split the procedure axis across ranks; the aggregate is unchanged by the split."""
    if not plan.enabled or plan.world_size == 1:
        return workload
    total = int(workload.shape[0])
    if total < plan.world_size:
        raise ValueError("fewer procedures than ranks")
    per_rank = total // plan.world_size
    rank = int(os.environ.get("RANK", "0"))
    return workload[rank * per_rank : (rank + 1) * per_rank]
