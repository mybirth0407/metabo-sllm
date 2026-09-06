"""Candidate-bag likelihood and slot-to-peak assignment.

An ambiguous peak has several formulas that fit its mass, and the data does not
say which one is right.  Picking one would teach the model an arbitrary answer,
so a slot is scored by the *total* probability it puts on the peak's bag --
any member counts as correct.

Slots are then assigned to target peaks by rectangular Hungarian matching on a
detached cost, so the assignment is a discrete decision and gradients flow only
through the losses evaluated at the chosen pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

__all__ = ["Assignment", "bag_log_probability", "hungarian_assign", "pairwise_huber"]

# Large but finite, so masked entries never turn into inf - inf = NaN.
NEG = -1.0e9


def bag_log_probability(
    candidate_log_prob: torch.Tensor,
    candidate_to_peak: torch.Tensor,
    candidate_mask: torch.Tensor,
    num_targets: int,
) -> torch.Tensor:
    """``log sum_k p(candidate k)`` per (slot, target), ``[B, S, T]``.

    Args:
        candidate_log_prob: ``[B, S, C]`` log-probability of each candidate
            under each slot.
        candidate_to_peak: ``[B, C]`` target index each candidate belongs to.
        candidate_mask: ``[B, C]`` real-candidate flag.
        num_targets: ``T``.
    """
    batch, slots, _ = candidate_log_prob.shape
    scores = torch.clamp(candidate_log_prob, min=NEG)
    scores = scores.masked_fill(~candidate_mask.unsqueeze(1), NEG)
    index = candidate_to_peak.unsqueeze(1).expand(batch, slots, -1)

    base = torch.full(
        (batch, slots, num_targets), NEG, device=scores.device, dtype=scores.dtype
    )
    shift = base.scatter_reduce(2, index, scores, reduce="amax", include_self=True).detach()
    centred = (scores - shift.gather(2, index)).exp() * candidate_mask.unsqueeze(1).to(
        scores.dtype
    )
    total = torch.zeros_like(base).scatter_add(2, index, centred)
    return shift + torch.log(total + 1e-30)


def pairwise_huber(
    predicted: torch.Tensor, target: torch.Tensor, *, delta: float = 1.0
) -> torch.Tensor:
    """Huber distance between every slot and every target, ``[B, S, T]``."""
    difference = predicted.unsqueeze(2) - target.unsqueeze(1)
    absolute = difference.abs()
    return torch.where(
        absolute <= delta, 0.5 * difference.pow(2), delta * (absolute - 0.5 * delta)
    )


@dataclass(frozen=True)
class Assignment:
    """Matched pairs across a batch, as flat index tensors."""

    batch_index: torch.Tensor
    slot_index: torch.Tensor
    target_index: torch.Tensor

    def __len__(self) -> int:
        return int(self.batch_index.shape[0])


def hungarian_assign(cost: torch.Tensor, target_mask: torch.Tensor) -> Assignment:
    """One-to-one slot/target assignment minimising ``cost``.

    ``cost`` is detached before the solve: the assignment is a discrete choice,
    not something to differentiate through.  Spectra with no targets contribute
    nothing.
    """
    detached = cost.detach().to(torch.float64).cpu().numpy()
    valid = target_mask.detach().cpu().numpy()
    slots = detached.shape[1]
    most = int(valid.sum(axis=1).max()) if valid.size else 0
    if most > slots:
        raise ValueError(
            f"{most} targets cannot be matched one-to-one against {slots} slots; "
            "the collator must cap targets at the slot count"
        )

    batches: list[int] = []
    slots: list[int] = []
    targets: list[int] = []
    for batch in range(detached.shape[0]):
        columns = np.flatnonzero(valid[batch])
        if columns.size == 0:
            continue
        rows, chosen = linear_sum_assignment(detached[batch][:, columns])
        batches.extend([batch] * rows.size)
        slots.extend(rows.tolist())
        targets.extend(columns[chosen].tolist())

    device = cost.device
    return Assignment(
        batch_index=torch.as_tensor(batches, dtype=torch.long, device=device),
        slot_index=torch.as_tensor(slots, dtype=torch.long, device=device),
        target_index=torch.as_tensor(targets, dtype=torch.long, device=device),
    )
