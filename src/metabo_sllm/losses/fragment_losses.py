"""Losses over matched fragment slots.

Identity is learned from the candidate bag, presence and intensity from the
assignment, and the spectrum term ties the predicted amplitudes back to the
observed peaks through a cosine -- the same quantity the field reports, and
one that does not reward simply shrinking every prediction the way an L1 term
on a sparse target would.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from metabo_sllm.losses.matching import Assignment

__all__ = ["LossWeights", "compute_losses", "presence_loss", "spectrum_cosine_loss"]


@dataclass(frozen=True)
class LossWeights:
    identity: float = 1.0
    presence: float = 0.2
    intensity: float = 1.0
    spectrum: float = 1.0


@dataclass
class LossOutput:
    total: torch.Tensor
    identity: torch.Tensor
    presence: torch.Tensor
    intensity: torch.Tensor
    spectrum: torch.Tensor
    extras: dict = field(default_factory=dict)


def presence_loss(presence_logits: torch.Tensor, assignment: Assignment) -> torch.Tensor:
    """Balanced BCE: matched slots are positives, everything else negative.

    Positives and negatives are averaged separately and then weighted equally,
    because at most ``T <= 64`` of the slots are ever positive and a plain mean
    would let the model win by predicting absence everywhere.
    """
    target = torch.zeros_like(presence_logits)
    if len(assignment):
        target[assignment.batch_index, assignment.slot_index] = 1.0
    elementwise = F.binary_cross_entropy_with_logits(
        presence_logits, target, reduction="none"
    )
    positive = target > 0.5
    negative = ~positive
    terms = []
    if positive.any():
        terms.append(elementwise[positive].mean())
    if negative.any():
        terms.append(elementwise[negative].mean())
    if not terms:
        return presence_logits.sum() * 0.0
    return torch.stack(terms).mean()


def spectrum_cosine_loss(
    predicted_intensity: torch.Tensor,
    assignment: Assignment,
    target_peak_indices: torch.Tensor,
    full_peak_intensities: torch.Tensor,
    full_peak_mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """``1 - cos`` between the rendered spectrum and the full transformed one.

    Matched slots are scattered onto the peaks they were assigned to; every
    other observed peak is predicted as zero, so peaks outside the slot budget
    still cost something.
    """
    rendered = torch.zeros_like(full_peak_intensities)
    if len(assignment):
        peaks = target_peak_indices[assignment.batch_index, assignment.target_index]
        values = predicted_intensity[assignment.batch_index, assignment.slot_index]
        rendered = rendered.index_put((assignment.batch_index, peaks), values, accumulate=True)
    rendered = rendered * full_peak_mask.to(rendered.dtype)
    target = full_peak_intensities * full_peak_mask.to(full_peak_intensities.dtype)

    numerator = (rendered * target).sum(dim=1)
    denominator = rendered.norm(dim=1) * target.norm(dim=1) + eps
    return (1.0 - numerator / denominator).mean()


def compute_losses(
    *,
    matched_bag_nll: torch.Tensor,
    presence_logits: torch.Tensor,
    predicted_intensity: torch.Tensor,
    target_intensities: torch.Tensor,
    target_peak_indices: torch.Tensor,
    target_peak_mask: torch.Tensor,
    full_peak_intensities: torch.Tensor,
    full_peak_mask: torch.Tensor,
    assignment: Assignment,
    weights: LossWeights,
    huber_delta: float = 1.0,
) -> LossOutput:
    """Identity + presence + intensity + spectrum, averaged over matched pairs.

    ``matched_bag_nll`` is one value per matched pair, already restricted to the
    slot Hungarian chose for each target.
    """
    device = presence_logits.device
    zero = presence_logits.sum() * 0.0

    if len(assignment):
        identity = matched_bag_nll.mean()
        matched_pred = predicted_intensity[assignment.batch_index, assignment.slot_index]
        matched_target = target_intensities[assignment.batch_index, assignment.target_index]
        intensity = F.huber_loss(
            matched_pred, matched_target, delta=huber_delta, reduction="mean"
        )
    else:
        identity = zero
        intensity = zero

    presence = presence_loss(presence_logits, assignment)
    spectrum = spectrum_cosine_loss(
        predicted_intensity,
        assignment,
        target_peak_indices,
        full_peak_intensities,
        full_peak_mask,
    )

    total = (
        weights.identity * identity
        + weights.presence * presence
        + weights.intensity * intensity
        + weights.spectrum * spectrum
    )
    _ = (target_peak_mask, device)
    return LossOutput(
        total=total,
        identity=identity,
        presence=presence,
        intensity=intensity,
        spectrum=spectrum,
        extras={"matched_pairs": len(assignment)},
    )
