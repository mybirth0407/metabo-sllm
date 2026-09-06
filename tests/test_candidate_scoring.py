"""The two-pass scorer must be the old full scorer, only cheaper.

Every test here compares against ``reference_candidate_log_prob`` -- the
original all-slots-by-all-candidates path -- so a memory optimisation cannot
quietly become a different loss.
"""

from __future__ import annotations

import pytest
import torch

from metabo_sllm.chem.candidates import ION_STATE_TO_ID, ION_STATE_VOCABULARY
from metabo_sllm.losses.candidate_scoring import (
    candidate_pairs_for_assignment,
    matched_bag_nll,
    matching_cost_no_grad,
    reference_candidate_log_prob,
)
from metabo_sllm.losses.matching import bag_log_probability, hungarian_assign, pairwise_huber
from metabo_sllm.model.formula_decoder import StructuredFormulaDecoder
from metabo_sllm.model.heads import IonStateHead

SLOT_DIM = 24
SLOTS = 8
TARGETS = 4
ELEMENTS = 3
WEIGHT = 0.25
DELTA = 1.0


def build_parts():
    torch.manual_seed(0)
    decoder = StructuredFormulaDecoder(
        SLOT_DIM, hidden_dim=32, num_layers=2, num_heads=4, max_count=8, dropout=0.0
    ).eval()
    ion_head = IonStateHead(SLOT_DIM, hidden_dim=16, dropout=0.0).eval()
    return decoder, ion_head


def make_batch(batch_size: int = 2, candidates: int = 9, seed: int = 0) -> dict:
    generator = torch.Generator().manual_seed(seed)
    counts = torch.randint(0, 3, (batch_size, candidates, ELEMENTS), generator=generator)
    counts[:, :, 0].clamp_(min=1)  # no candidate is the empty formula
    # spread candidates over the targets, leaving one target with a single member
    to_peak = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2, 3]])[:, :candidates].expand(
        batch_size, candidates
    )
    admissible = torch.zeros(batch_size, len(ION_STATE_VOCABULARY), dtype=torch.bool)
    admissible[:, ION_STATE_TO_ID["protonated"]] = True
    admissible[:, ION_STATE_TO_ID["sodiated"]] = True
    return {
        "precursor_element_ids": torch.tensor([[1, 6, 8]]).expand(batch_size, ELEMENTS),
        "precursor_element_counts": torch.tensor([[4, 3, 2]]).expand(batch_size, ELEMENTS),
        "precursor_element_mask": torch.ones(batch_size, ELEMENTS, dtype=torch.bool),
        "candidate_formula_counts": counts,
        "candidate_ion_states": torch.zeros(batch_size, candidates, dtype=torch.long),
        "candidate_to_peak": to_peak.contiguous(),
        "candidate_mask": torch.ones(batch_size, candidates, dtype=torch.bool),
        "admissible_ion_mask": admissible,
        "target_peak_mask": torch.ones(batch_size, TARGETS, dtype=torch.bool),
        "target_intensities": torch.rand(batch_size, TARGETS, generator=generator),
        "target_peak_indices": torch.arange(TARGETS).expand(batch_size, TARGETS).contiguous(),
    }


def reference_cost(decoder, ion_head, slots, contribution, batch):
    """The pre-optimisation path, kept only as a test oracle."""
    scores = reference_candidate_log_prob(decoder, ion_head, slots, batch)
    bag = bag_log_probability(
        scores, batch["candidate_to_peak"], batch["candidate_mask"], TARGETS
    )
    return -bag + WEIGHT * pairwise_huber(
        contribution, batch["target_intensities"], delta=DELTA
    )


def slots_and_gates(seed: int = 0, batch_size: int = 2, requires_grad: bool = False):
    generator = torch.Generator().manual_seed(seed + 100)
    slots = torch.randn(batch_size, SLOTS, SLOT_DIM, generator=generator)
    slots.requires_grad_(requires_grad)
    contribution = torch.rand(batch_size, SLOTS, generator=generator)
    return slots, contribution


# ------------------------------------------------------------------ pass one


def test_matching_cost_matches_the_full_scorer():
    decoder, ion_head = build_parts()
    batch = make_batch()
    slots, contribution = slots_and_gates()

    two_pass = matching_cost_no_grad(
        decoder,
        ion_head,
        slots,
        contribution,
        batch,
        candidate_chunk_size=4,
        match_intensity_weight=WEIGHT,
        huber_delta=DELTA,
    )
    with torch.no_grad():
        expected = reference_cost(decoder, ion_head, slots, contribution, batch)

    assert two_pass.shape == (2, SLOTS, TARGETS)
    assert torch.max(torch.abs(two_pass - expected)) <= 1e-6


def test_matching_cost_carries_no_gradient():
    decoder, ion_head = build_parts()
    batch = make_batch()
    slots, contribution = slots_and_gates(requires_grad=True)

    cost = matching_cost_no_grad(
        decoder, ion_head, slots, contribution, batch, candidate_chunk_size=4
    )

    assert not cost.requires_grad
    assert cost.grad_fn is None


def test_hungarian_assignment_is_unchanged():
    decoder, ion_head = build_parts()
    batch = make_batch()
    slots, contribution = slots_and_gates()

    two_pass = matching_cost_no_grad(
        decoder,
        ion_head,
        slots,
        contribution,
        batch,
        candidate_chunk_size=3,
        match_intensity_weight=WEIGHT,
        huber_delta=DELTA,
    )
    with torch.no_grad():
        expected = reference_cost(decoder, ion_head, slots, contribution, batch)

    a = hungarian_assign(two_pass, batch["target_peak_mask"])
    b = hungarian_assign(expected, batch["target_peak_mask"])

    assert a.batch_index.tolist() == b.batch_index.tolist()
    assert a.slot_index.tolist() == b.slot_index.tolist()
    assert a.target_index.tolist() == b.target_index.tolist()


@pytest.mark.parametrize("chunk", [1, 2, 5, 64])
def test_matching_cost_is_chunk_invariant(chunk):
    decoder, ion_head = build_parts()
    batch = make_batch()
    slots, contribution = slots_and_gates()

    baseline = matching_cost_no_grad(
        decoder, ion_head, slots, contribution, batch, candidate_chunk_size=1000
    )
    chunked = matching_cost_no_grad(
        decoder, ion_head, slots, contribution, batch, candidate_chunk_size=chunk
    )

    assert torch.max(torch.abs(baseline - chunked)) <= 1e-6


# ------------------------------------------------------------------ pass two


def test_matched_bag_nll_matches_the_full_scorer():
    decoder, ion_head = build_parts()
    batch = make_batch()
    slots, contribution = slots_and_gates()

    cost = matching_cost_no_grad(decoder, ion_head, slots, contribution, batch)
    assignment = hungarian_assign(cost, batch["target_peak_mask"])

    matched = matched_bag_nll(decoder, ion_head, slots, batch, assignment, candidate_chunk_size=3)
    with torch.no_grad():
        scores = reference_candidate_log_prob(decoder, ion_head, slots, batch)
        full = -bag_log_probability(
            scores, batch["candidate_to_peak"], batch["candidate_mask"], TARGETS
        )
        expected = full[
            assignment.batch_index, assignment.slot_index, assignment.target_index
        ]

    assert matched.shape == expected.shape
    assert torch.max(torch.abs(matched - expected)) <= 1e-6


def test_identity_loss_matches_the_full_scorer():
    decoder, ion_head = build_parts()
    batch = make_batch()
    slots, contribution = slots_and_gates()
    cost = matching_cost_no_grad(decoder, ion_head, slots, contribution, batch)
    assignment = hungarian_assign(cost, batch["target_peak_mask"])

    new_loss = matched_bag_nll(decoder, ion_head, slots, batch, assignment).mean()
    with torch.no_grad():
        scores = reference_candidate_log_prob(decoder, ion_head, slots, batch)
        full = -bag_log_probability(
            scores, batch["candidate_to_peak"], batch["candidate_mask"], TARGETS
        )
        old_loss = full[
            assignment.batch_index, assignment.slot_index, assignment.target_index
        ].mean()

    assert abs(float(new_loss) - float(old_loss)) <= 1e-6


@pytest.mark.parametrize("chunk", [1, 4, 1000])
def test_matched_bag_nll_is_chunk_invariant(chunk):
    decoder, ion_head = build_parts()
    batch = make_batch()
    slots, contribution = slots_and_gates()
    cost = matching_cost_no_grad(decoder, ion_head, slots, contribution, batch)
    assignment = hungarian_assign(cost, batch["target_peak_mask"])

    baseline = matched_bag_nll(decoder, ion_head, slots, batch, assignment, candidate_chunk_size=1000)
    chunked = matched_bag_nll(decoder, ion_head, slots, batch, assignment, candidate_chunk_size=chunk)

    assert torch.max(torch.abs(baseline - chunked)) <= 1e-6


def test_only_matched_slot_candidate_pairs_enter_the_graph():
    decoder, ion_head = build_parts()
    batch = make_batch()
    slots, contribution = slots_and_gates()
    cost = matching_cost_no_grad(decoder, ion_head, slots, contribution, batch)
    assignment = hungarian_assign(cost, batch["target_peak_mask"])

    rows, columns, slot_index, pair_index = candidate_pairs_for_assignment(batch, assignment)

    # one entry per bag member of a matched target, and nothing more
    assert rows.shape == columns.shape == slot_index.shape == pair_index.shape
    assert int(rows.shape[0]) == int(batch["candidate_mask"].sum())
    # every candidate keeps the slot Hungarian gave its target
    for pair, slot in zip(pair_index.tolist(), slot_index.tolist(), strict=True):
        assert slot == int(assignment.slot_index[pair])
    # far fewer than the full cross-product
    assert rows.shape[0] < batch["candidate_mask"].shape[0] * SLOTS * batch["candidate_mask"].shape[1]


# ----------------------------------------------------------------- gradients


def _gradients(chunk_matching: int, chunk_matched: int, use_reference: bool):
    decoder, ion_head = build_parts()
    batch = make_batch()
    slots, contribution = slots_and_gates(requires_grad=True)

    cost = matching_cost_no_grad(
        decoder,
        ion_head,
        slots,
        contribution,
        batch,
        candidate_chunk_size=chunk_matching,
        match_intensity_weight=WEIGHT,
        huber_delta=DELTA,
    )
    assignment = hungarian_assign(cost, batch["target_peak_mask"])

    if use_reference:
        scores = reference_candidate_log_prob(decoder, ion_head, slots, batch)
        full = -bag_log_probability(
            scores, batch["candidate_to_peak"], batch["candidate_mask"], TARGETS
        )
        loss = full[
            assignment.batch_index, assignment.slot_index, assignment.target_index
        ].mean()
    else:
        loss = matched_bag_nll(
            decoder, ion_head, slots, batch, assignment, candidate_chunk_size=chunk_matched
        ).mean()

    loss.backward()
    parameter_grads = {
        name: p.grad.detach().clone()
        for name, p in decoder.named_parameters()
        if p.grad is not None
    }
    return float(loss), slots.grad.detach().clone(), parameter_grads


def test_slot_and_decoder_gradients_match_the_full_scorer():
    reference_loss, reference_slot_grad, reference_params = _gradients(64, 256, True)
    two_pass_loss, two_pass_slot_grad, two_pass_params = _gradients(4, 3, False)

    assert abs(reference_loss - two_pass_loss) <= 1e-6
    assert torch.max(torch.abs(reference_slot_grad - two_pass_slot_grad)) <= 1e-5
    assert set(reference_params) == set(two_pass_params)
    for name, grad in reference_params.items():
        assert torch.max(torch.abs(grad - two_pass_params[name])) <= 1e-5, name


def test_gradients_are_chunk_invariant():
    _, slot_a, params_a = _gradients(2, 1, False)
    _, slot_b, params_b = _gradients(1000, 1000, False)

    assert torch.max(torch.abs(slot_a - slot_b)) <= 1e-5
    for name, grad in params_a.items():
        assert torch.max(torch.abs(grad - params_b[name])) <= 1e-5, name


# --------------------------------------------------------------- zero targets


def _bag_nll_from_cost(cost, contribution, batch, assignment):
    """Undo the Huber term so the matching cost's own bag NLL is comparable."""
    bag = cost - WEIGHT * pairwise_huber(
        contribution, batch["target_intensities"], delta=DELTA
    )
    return bag[assignment.batch_index, assignment.slot_index, assignment.target_index]


def test_two_passes_agree_in_train_mode():
    """Pass 1 decides the assignment, Pass 2 supplies the loss.

    They must see the same distribution, or the model is matched against one
    sample and trained on another. That is only true while the formula decoder
    has no dropout, which is why its default is 0.0.
    """
    decoder, ion_head = build_parts()
    decoder.train()
    ion_head.train()
    batch = make_batch()
    slots, contribution = slots_and_gates()

    cost = matching_cost_no_grad(
        decoder,
        ion_head,
        slots,
        contribution,
        batch,
        candidate_chunk_size=4,
        match_intensity_weight=WEIGHT,
        huber_delta=DELTA,
    )
    assignment = hungarian_assign(cost, batch["target_peak_mask"])
    matched = matched_bag_nll(decoder, ion_head, slots, batch, assignment, candidate_chunk_size=3)
    from_cost = _bag_nll_from_cost(cost, contribution, batch, assignment)

    assert decoder.training and ion_head.training
    assert torch.max(torch.abs(matched - from_cost)) <= 1e-6


def test_formula_decoder_defaults_to_no_dropout():
    decoder = StructuredFormulaDecoder(SLOT_DIM, hidden_dim=16, num_layers=1, num_heads=4)

    dropouts = [
        module.p for module in decoder.modules() if isinstance(module, torch.nn.Dropout)
    ]
    assert dropouts, "expected the transformer body to contain dropout modules"
    assert all(p == 0.0 for p in dropouts)


def test_train_mode_scoring_is_repeatable():
    decoder, ion_head = build_parts()
    decoder.train()
    batch = make_batch()
    slots, contribution = slots_and_gates()
    cost = matching_cost_no_grad(decoder, ion_head, slots, contribution, batch)
    assignment = hungarian_assign(cost, batch["target_peak_mask"])

    first = matched_bag_nll(decoder, ion_head, slots, batch, assignment)
    second = matched_bag_nll(decoder, ion_head, slots, batch, assignment)

    assert torch.max(torch.abs(first - second)) <= 1e-6


def test_spectrum_without_targets_scores_nothing():
    decoder, ion_head = build_parts()
    batch = make_batch()
    batch["target_peak_mask"] = torch.zeros(2, TARGETS, dtype=torch.bool)
    slots, contribution = slots_and_gates()

    cost = matching_cost_no_grad(decoder, ion_head, slots, contribution, batch)
    assignment = hungarian_assign(cost, batch["target_peak_mask"])
    matched = matched_bag_nll(decoder, ion_head, slots, batch, assignment)

    assert torch.isfinite(cost).all()
    assert len(assignment) == 0
    assert matched.shape == (0,)


def test_masked_candidates_are_ignored_by_both_passes():
    decoder, ion_head = build_parts()
    batch = make_batch()
    batch["candidate_mask"][:, -3:] = False
    slots, contribution = slots_and_gates()

    two_pass = matching_cost_no_grad(
        decoder,
        ion_head,
        slots,
        contribution,
        batch,
        candidate_chunk_size=2,
        match_intensity_weight=WEIGHT,
        huber_delta=DELTA,
    )
    with torch.no_grad():
        expected = reference_cost(decoder, ion_head, slots, contribution, batch)

    assert torch.max(torch.abs(two_pass - expected)) <= 1e-6
