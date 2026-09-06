"""Optimizer, schedule, and the accounting one training step depends on."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from metabo_sllm.training.distributed import DistributedContext
from metabo_sllm.training.metrics import RatioMeter, batch_workload, gradient_norm
from metabo_sllm.training.trainer import build_optimizer, build_scheduler

SINGLE = DistributedContext(
    rank=0, local_rank=0, world_size=1, device=torch.device("cpu"), distributed=False
)


class PartlyFrozen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen = nn.Linear(8, 8)
        self.trainable = nn.Linear(8, 8)
        for parameter in self.frozen.parameters():
            parameter.requires_grad_(False)


# ---------------------------------------------------------------- optimizer


def test_optimizer_only_sees_trainable_parameters():
    model = PartlyFrozen()
    optimizer = build_optimizer(
        model, learning_rate=1e-4, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8, fused=False
    )

    grouped = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert grouped == {id(p) for p in model.trainable.parameters()}
    assert not any(id(p) in grouped for p in model.frozen.parameters())


def test_optimizer_rejects_a_fully_frozen_model():
    model = PartlyFrozen()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with pytest.raises(ValueError, match="no trainable parameters"):
        build_optimizer(
            model, learning_rate=1e-4, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8, fused=False
        )


def test_fused_request_falls_back_without_cuda():
    model = PartlyFrozen()
    optimizer = build_optimizer(
        model, learning_rate=1e-4, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8, fused=True
    )

    assert isinstance(optimizer, torch.optim.AdamW)


# ---------------------------------------------------------------- scheduler


def _schedule(total: int = 100, warmup_ratio: float = 0.1, min_lr_ratio: float = 0.1):
    model = PartlyFrozen()
    optimizer = build_optimizer(
        model, learning_rate=1.0, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8, fused=False
    )
    scheduler = build_scheduler(
        optimizer, total_steps=total, warmup_ratio=warmup_ratio, min_lr_ratio=min_lr_ratio
    )
    rates = []
    for _ in range(total):
        rates.append(scheduler.get_last_lr()[0])
        scheduler.step()
    return rates


def test_learning_rate_warms_up_then_decays():
    rates = _schedule()
    warmup = 10

    assert rates[0] < rates[warmup - 1]
    assert rates[warmup - 1] == pytest.approx(1.0)
    assert all(a >= b - 1e-12 for a, b in zip(rates[warmup:], rates[warmup + 1 :], strict=False))


def test_learning_rate_never_falls_below_the_floor():
    rates = _schedule(min_lr_ratio=0.1)

    assert min(rates) >= 0.1 - 1e-9
    assert rates[-1] == pytest.approx(0.1, abs=0.02)


def test_cosine_midpoint_is_halfway_down():
    rates = _schedule(total=100, warmup_ratio=0.0, min_lr_ratio=0.0)

    assert rates[50] == pytest.approx(0.5 * (1 + math.cos(math.pi * 0.5)), abs=0.02)


def test_scheduler_state_round_trips():
    model = PartlyFrozen()
    optimizer = build_optimizer(
        model, learning_rate=1.0, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8, fused=False
    )
    scheduler = build_scheduler(optimizer, total_steps=50, warmup_ratio=0.1, min_lr_ratio=0.1)
    for _ in range(20):
        scheduler.step()
    expected = scheduler.get_last_lr()[0]

    fresh_optimizer = build_optimizer(
        PartlyFrozen(), learning_rate=1.0, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8, fused=False
    )
    fresh = build_scheduler(fresh_optimizer, total_steps=50, warmup_ratio=0.1, min_lr_ratio=0.1)
    fresh.load_state_dict(scheduler.state_dict())

    assert fresh.get_last_lr()[0] == pytest.approx(expected)


# ------------------------------------------------------------------ metrics


def test_ratio_meter_weights_by_denominator_not_by_batch():
    meter = RatioMeter()
    meter.add(1, 1)  # a tiny batch, perfect
    meter.add(0, 99)  # a large batch, all wrong

    # averaging the two batch means would give 0.5; the honest answer is 0.01
    assert meter.reduced(SINGLE) == pytest.approx(1 / 100)


def test_ratio_meter_reports_none_without_data():
    assert RatioMeter().reduced(SINGLE) is None


def test_batch_workload_counts_the_padded_tensors_correctly():
    batch = {
        "target_peak_mask": torch.tensor([[True, True, False], [True, False, False]]),
        "candidate_mask": torch.tensor([[True, True], [True, False]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        "full_peak_mask": torch.tensor([[True, True], [True, True]]),
    }

    assert batch_workload(batch) == {
        "batch_spectra": 2,
        "batch_targets": 3,
        "batch_candidates": 3,
        "batch_tokens": 5,
        "batch_peaks": 4,
    }


def test_gradient_norm_matches_torch():
    model = PartlyFrozen()
    for parameter in model.trainable.parameters():
        parameter.grad = torch.full_like(parameter, 0.5)
    parameters = [p for p in model.parameters() if p.requires_grad]

    expected = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1e9)

    assert gradient_norm(parameters) == pytest.approx(float(expected), rel=1e-6)


def test_gradient_norm_ignores_parameters_without_gradients():
    model = PartlyFrozen()

    assert gradient_norm(list(model.parameters())) == pytest.approx(0.0)


# --------------------------------------------------------- microbatch weights


def test_microbatch_weights_sum_to_one():
    """A step's microbatches are weighted by their share of its spectra."""
    sizes = [2.0, 16.0, 5.0]
    total = sum(sizes)
    weights = [size / total for size in sizes]

    assert sum(weights) == pytest.approx(1.0)
    assert weights[1] > weights[0], "the larger microbatch must count for more"


def test_equal_microbatches_reduce_to_flat_averaging():
    sizes = [8.0, 8.0]
    weights = [size / sum(sizes) for size in sizes]

    assert weights == [pytest.approx(0.5), pytest.approx(0.5)]
