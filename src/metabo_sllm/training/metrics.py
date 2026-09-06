"""Training metrics, reduced correctly across ranks.

The quantities that say whether the model is learning -- how often the greedy
fragment lands in its target's bag, whether matched slots are actually marked
present -- are ratios, so they are reduced as (numerator, denominator) sums
rather than as an average of averages, which would weight a rank's small batch
the same as another rank's large one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from metabo_sllm.training.distributed import DistributedContext, all_reduce_sum

__all__ = ["RatioMeter", "StepMetrics", "batch_workload", "slot_metrics"]


@dataclass
class RatioMeter:
    """A numerator/denominator pair that survives an all-reduce intact."""

    numerator: float = 0.0
    denominator: float = 0.0

    def add(self, numerator: float, denominator: float) -> None:
        self.numerator += float(numerator)
        self.denominator += float(denominator)

    def reduced(self, context: DistributedContext) -> float | None:
        numerator = all_reduce_sum(self.numerator, context)
        denominator = all_reduce_sum(self.denominator, context)
        return numerator / denominator if denominator > 0 else None


@dataclass
class StepMetrics:
    """Everything logged for one optimizer step."""

    losses: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    ratios: dict = field(default_factory=dict)


def batch_workload(batch: dict) -> dict:
    """Actual spectra/targets/candidates/tokens in a collated batch."""
    return {
        "batch_spectra": int(batch["target_peak_mask"].shape[0]),
        "batch_targets": int(batch["target_peak_mask"].sum()),
        "batch_candidates": int(batch["candidate_mask"].sum()),
        "batch_tokens": int(batch["attention_mask"].sum()),
        "batch_peaks": int(batch["full_peak_mask"].sum()),
    }


@torch.no_grad()
def slot_metrics(model, batch: dict, outputs, assignment) -> dict:
    """Greedy identity accuracy, presence recall, and slot occupancy.

    Returned as raw counts so the caller can reduce them across ranks before
    turning them into ratios.
    """
    from metabo_sllm.rendering.spectrum import greedy_identity_metrics

    identity = greedy_identity_metrics(model, batch, outputs, assignment)
    presence = outputs.presence
    active = int((presence > 0.5).sum())
    spectra = int(presence.shape[0])

    if len(assignment):
        matched_presence = presence[assignment.batch_index, assignment.slot_index]
        recalled = int((matched_presence > 0.5).sum())
        matched = int(len(assignment))
    else:
        recalled = matched = 0

    hits = identity["argmax_in_candidate_bag"]
    pairs = identity["matched_pairs"]
    return {
        "bag_hits": (hits * pairs) if hits is not None else 0.0,
        "bag_pairs": float(pairs),
        "presence_recalled": float(recalled),
        "presence_targets": float(matched),
        "active_slots": float(active),
        "active_spectra": float(spectra),
        "unique_candidate_accuracy": identity["unique_candidate_accuracy"],
        "ambiguous_bag_hit": identity["ambiguous_bag_hit"],
    }


def gradient_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum().item())
    return float(np.sqrt(total))
