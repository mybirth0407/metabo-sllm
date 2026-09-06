"""Scoring fragment candidates without holding the whole cross-product.

Matching needs a bag likelihood for *every* (slot, target) pair, but the loss
only ever reads the pairs Hungarian selected.  Scoring all of them with
gradients attached keeps ``64 x C`` formula-decoder activations alive; on a
spectrum with 1,902 candidates that is roughly half a million sequences.

So the work is split:

``matching_cost_no_grad``  scores everything under ``no_grad``, in candidate
                           chunks, and keeps only the ``[B, 64, T]`` cost.
``matched_bag_nll``        rescores just the matched slot against its own
                           target's bag, with gradients on.

Neither pass drops a candidate: every bag member is scored in both, and the
chunked combination is an exact log-sum-exp, not an approximation.  The
gradient graph shrinks from ``B x 64 x C`` to ``sum_j |B_j|``.
"""

from __future__ import annotations

import torch

from metabo_sllm.losses.matching import NEG, pairwise_huber

__all__ = [
    "candidate_pairs_for_assignment",
    "matched_bag_nll",
    "matching_cost_no_grad",
    "reference_candidate_log_prob",
]

_LOG_EPS = 1e-30


def _formula_log_prob_grid(
    formula_decoder,
    slots: torch.Tensor,
    batch: dict,
    start: int,
    stop: int,
) -> torch.Tensor:
    """``log p(F_c | slot_r)`` for one candidate chunk, ``[B, S, stop - start]``."""
    batch_size, num_slots, slot_dim = slots.shape
    elements = batch["precursor_element_ids"].shape[1]
    width = stop - start
    flat = batch_size * num_slots * width
    shape = (batch_size, num_slots, width, elements)

    slot_flat = (
        slots.unsqueeze(2)
        .expand(batch_size, num_slots, width, slot_dim)
        .reshape(flat, slot_dim)
    )
    element_ids = (
        batch["precursor_element_ids"].view(batch_size, 1, 1, elements).expand(shape).reshape(flat, elements)
    )
    element_counts = (
        batch["precursor_element_counts"].view(batch_size, 1, 1, elements).expand(shape).reshape(flat, elements)
    )
    element_mask = (
        batch["precursor_element_mask"].view(batch_size, 1, 1, elements).expand(shape).reshape(flat, elements)
    )
    target_counts = (
        batch["candidate_formula_counts"][:, start:stop, :]
        .reshape(batch_size, 1, width, elements)
        .expand(shape)
        .reshape(flat, elements)
    )
    return formula_decoder.log_prob(
        slot_flat, element_ids, element_counts, target_counts, element_mask
    ).view(batch_size, num_slots, width)


def _chunk_candidate_scores(
    formula_decoder,
    ion_log_prob: torch.Tensor,
    slots: torch.Tensor,
    batch: dict,
    start: int,
    stop: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Candidate scores and their real-candidate mask for one chunk."""
    batch_size, num_slots = slots.shape[0], slots.shape[1]
    width = stop - start
    formula = _formula_log_prob_grid(formula_decoder, slots, batch, start, stop)
    ion_index = (
        batch["candidate_ion_states"][:, start:stop].unsqueeze(1).expand(batch_size, num_slots, width)
    )
    score = formula + ion_log_prob.gather(2, ion_index)
    mask = batch["candidate_mask"][:, start:stop].unsqueeze(1)
    return torch.clamp(score, min=NEG).masked_fill(~mask, NEG), mask


@torch.no_grad()
def matching_cost_no_grad(
    formula_decoder,
    ion_head,
    slots: torch.Tensor,
    contribution: torch.Tensor,
    batch: dict,
    *,
    candidate_chunk_size: int = 64,
    match_intensity_weight: float = 0.25,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Full ``[B, 64, T]`` matching cost, built without a gradient graph.

    Candidates are streamed in chunks and folded into a running log-sum-exp, so
    the ``[B, 64, C]`` score grid never exists in one piece and is never
    returned.  Every bag member still contributes.
    """
    slots = slots.detach()
    contribution = contribution.detach()
    batch_size, num_slots = slots.shape[0], slots.shape[1]
    candidates = batch["candidate_ion_states"].shape[1]
    targets = batch["target_peak_mask"].shape[1]
    ion_log_prob = ion_head(slots, batch["admissible_ion_mask"]).detach()

    shape = (batch_size, num_slots, targets)
    running_max = torch.full(shape, NEG, device=slots.device, dtype=slots.dtype)
    running_sum = torch.zeros(shape, device=slots.device, dtype=slots.dtype)
    index_all = batch["candidate_to_peak"].unsqueeze(1).expand(batch_size, num_slots, candidates)

    step = max(1, int(candidate_chunk_size))
    for start in range(0, candidates, step):
        stop = min(start + step, candidates)
        score, mask = _chunk_candidate_scores(
            formula_decoder, ion_log_prob, slots, batch, start, stop
        )
        index = index_all[:, :, start:stop]
        chunk_max = torch.full(shape, NEG, device=slots.device, dtype=slots.dtype).scatter_reduce(
            2, index, score, reduce="amax", include_self=True
        )
        centred = (score - chunk_max.gather(2, index)).exp() * mask.to(score.dtype)
        chunk_sum = torch.zeros(shape, device=slots.device, dtype=slots.dtype).scatter_add(
            2, index, centred
        )
        # exact running log-sum-exp across chunks
        merged = torch.maximum(running_max, chunk_max)
        running_sum = running_sum * (running_max - merged).exp() + chunk_sum * (
            chunk_max - merged
        ).exp()
        running_max = merged

    bag_nll = -(running_max + torch.log(running_sum + _LOG_EPS))
    return bag_nll + match_intensity_weight * pairwise_huber(
        contribution, batch["target_intensities"], delta=huber_delta
    )


def candidate_pairs_for_assignment(batch: dict, assignment) -> tuple[torch.Tensor, ...]:
    """Flatten the candidates each matched target owns.

    Returns ``(batch_index, candidate_index, slot_index, pair_index)`` where
    ``pair_index`` says which matched pair a candidate belongs to.  The length
    is ``sum_j |B_j|`` -- one entry per bag member of a matched target, and
    nothing for the other 63 slots.
    """
    device = batch["candidate_mask"].device
    pairs = len(assignment)
    targets = batch["target_peak_mask"].shape[1]
    lookup = torch.full(
        (batch["candidate_mask"].shape[0], targets), -1, dtype=torch.long, device=device
    )
    if pairs:
        lookup[assignment.batch_index, assignment.target_index] = torch.arange(
            pairs, dtype=torch.long, device=device
        )

    batch_index, candidate_index = batch["candidate_mask"].nonzero(as_tuple=True)
    pair_index = lookup[batch_index, batch["candidate_to_peak"][batch_index, candidate_index]]
    keep = pair_index >= 0
    batch_index = batch_index[keep]
    candidate_index = candidate_index[keep]
    pair_index = pair_index[keep]
    slot_index = (
        assignment.slot_index[pair_index]
        if pairs
        else torch.zeros(0, dtype=torch.long, device=device)
    )
    return batch_index, candidate_index, slot_index, pair_index


def matched_bag_nll(
    formula_decoder,
    ion_head,
    slots: torch.Tensor,
    batch: dict,
    assignment,
    *,
    candidate_chunk_size: int = 256,
) -> torch.Tensor:
    """``-log sum_{(F,s) in B_j} p(F, s | z_pi(j))`` for each matched pair, ``[M]``.

    Only the matched slot is scored against its target's bag, so the gradient
    graph covers ``sum_j |B_j|`` candidate evaluations instead of ``B x 64 x C``.
    The bag itself is untouched: every member is included.
    """
    pairs = len(assignment)
    if pairs == 0:
        return slots.new_zeros(0)

    batch_index, candidate_index, slot_index, pair_index = candidate_pairs_for_assignment(
        batch, assignment
    )
    ion_log_prob = ion_head(slots, batch["admissible_ion_mask"])

    total = int(batch_index.shape[0])
    scores: list[torch.Tensor] = []
    step = max(1, int(candidate_chunk_size))
    for start in range(0, total, step):
        window = slice(start, min(start + step, total))
        rows = batch_index[window]
        columns = candidate_index[window]
        slots_here = slot_index[window]
        formula = formula_decoder.log_prob(
            slots[rows, slots_here],
            batch["precursor_element_ids"][rows],
            batch["precursor_element_counts"][rows],
            batch["candidate_formula_counts"][rows, columns],
            batch["precursor_element_mask"][rows],
        )
        ion = ion_log_prob[rows, slots_here, batch["candidate_ion_states"][rows, columns]]
        scores.append(torch.clamp(formula + ion, min=NEG))

    flat = (
        torch.cat(scores)
        if scores
        else torch.zeros(0, device=slots.device, dtype=slots.dtype)
    )
    shift = (
        torch.full((pairs,), NEG, device=slots.device, dtype=flat.dtype)
        .scatter_reduce(0, pair_index, flat, reduce="amax", include_self=True)
        .detach()
    )
    centred = (flat - shift[pair_index]).exp()
    total_mass = torch.zeros(pairs, device=slots.device, dtype=flat.dtype).scatter_add(
        0, pair_index, centred
    )
    return -(shift + torch.log(total_mass + _LOG_EPS))


def reference_candidate_log_prob(
    formula_decoder,
    ion_head,
    slots: torch.Tensor,
    batch: dict,
    *,
    candidate_chunk_size: int = 128,
) -> torch.Tensor:
    """Full ``[B, 64, C]`` candidate log-probability.

    Kept for equivalence tests and for the explicit debug flag only; the
    training path must not call it, which is the whole point of the two-pass
    split.
    """
    batch_size, num_slots = slots.shape[0], slots.shape[1]
    candidates = batch["candidate_ion_states"].shape[1]
    ion_log_prob = ion_head(slots, batch["admissible_ion_mask"])
    chunks = []
    step = max(1, int(candidate_chunk_size))
    for start in range(0, candidates, step):
        stop = min(start + step, candidates)
        score, _ = _chunk_candidate_scores(
            formula_decoder, ion_log_prob, slots, batch, start, stop
        )
        chunks.append(score)
    if not chunks:
        return slots.new_zeros(batch_size, num_slots, 0)
    return torch.cat(chunks, dim=2)
