"""Process-group setup and the few collectives the trainer needs.

Kept deliberately small: everything here works unchanged on one process, so
the single-GPU resume test and the four-GPU smoke run the same trainer code.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

__all__ = ["DistributedContext", "all_reduce_mean", "all_reduce_sum", "gather_objects"]


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    distributed: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @classmethod
    def from_environment(cls, *, timeout_minutes: int = 30) -> DistributedContext:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        distributed = world_size > 1

        if torch.cuda.is_available():
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

        if distributed and not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if device.type == "cuda" else "gloo",
                timeout=datetime.timedelta(minutes=timeout_minutes),
            )
        return cls(rank, local_rank, world_size, device, distributed)

    def barrier(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.barrier()

    def shutdown(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()


def all_reduce_sum(value: float, context: DistributedContext) -> float:
    if not context.distributed:
        return float(value)
    tensor = torch.tensor([value], dtype=torch.float64, device=context.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def all_reduce_mean(value: float, context: DistributedContext) -> float:
    if not context.distributed:
        return float(value)
    return all_reduce_sum(value, context) / context.world_size


def gather_objects(payload, context: DistributedContext) -> list:
    """Collect one Python object per rank on every rank."""
    if not context.distributed:
        return [payload]
    bucket = [None] * context.world_size
    dist.all_gather_object(bucket, payload)
    return bucket
