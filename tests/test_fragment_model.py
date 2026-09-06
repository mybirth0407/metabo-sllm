"""Slot decoder, heads, and the loss pipeline wired end to end.

The Qwen backbone is not loaded here -- that is what the one-batch smoke does.
These tests drive the same code path with a stand-in memory tensor so the
maths is checked in under a second.
"""

from __future__ import annotations

import pytest
import torch

from metabo_sllm.chem.candidates import ION_STATE_TO_ID, ION_STATE_VOCABULARY
from metabo_sllm.losses.fragment_losses import (
    LossWeights,
    compute_losses,
    presence_loss,
    spectrum_cosine_loss,
)
from metabo_sllm.losses.matching import (
    Assignment,
    bag_log_probability,
    hungarian_assign,
    pairwise_huber,
)
from metabo_sllm.model.formula_decoder import StructuredFormulaDecoder
from metabo_sllm.model.heads import IntensityHead, IonStateHead, PresenceHead
from metabo_sllm.model.qwen_encoder import MissingBackboneError, QwenEncoder
from metabo_sllm.model.slot_decoder import SlotDecoder

SLOTS = 64
SLOT_DIM = 32
MEMORY_DIM = 48


def make_slot_decoder() -> SlotDecoder:
    torch.manual_seed(0)
    return SlotDecoder(
        MEMORY_DIM,
        num_slots=SLOTS,
        hidden_dim=SLOT_DIM,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
    ).eval()


# ------------------------------------------------------------------ slot decoder


def test_slot_decoder_shape():
    decoder = make_slot_decoder()
    memory = torch.randn(3, 11, MEMORY_DIM)
    mask = torch.ones(3, 11, dtype=torch.long)

    slots = decoder(memory, mask)

    assert slots.shape == (3, SLOTS, SLOT_DIM)
    assert torch.isfinite(slots).all()


def test_slot_decoder_ignores_padded_tokens():
    decoder = make_slot_decoder()
    memory = torch.randn(1, 8, MEMORY_DIM)
    mask = torch.zeros(1, 8, dtype=torch.long)
    mask[0, :5] = 1

    first = decoder(memory, mask)
    altered = memory.clone()
    altered[0, 5:] = 99.0
    second = decoder(altered, mask)

    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-5)


def test_learned_queries_start_out_different():
    decoder = make_slot_decoder()
    queries = decoder.queries

    assert queries.shape == (SLOTS, SLOT_DIM)
    assert not torch.allclose(queries[0], queries[1])


# ------------------------------------------------------------------------ heads


def test_ion_head_gives_inadmissible_states_zero_probability():
    torch.manual_seed(0)
    head = IonStateHead(SLOT_DIM, hidden_dim=16).eval()
    slots = torch.randn(2, SLOTS, SLOT_DIM)
    admissible = torch.zeros(2, len(ION_STATE_VOCABULARY), dtype=torch.bool)
    admissible[0, ION_STATE_TO_ID["protonated"]] = True
    admissible[1, ION_STATE_TO_ID["deprotonated"]] = True
    admissible[1, ION_STATE_TO_ID["chloride_retained"]] = True

    log_prob = head(slots, admissible)
    probabilities = log_prob.exp()

    assert log_prob.shape == (2, SLOTS, len(ION_STATE_VOCABULARY))
    torch.testing.assert_close(
        probabilities.sum(-1), torch.ones(2, SLOTS), rtol=1e-5, atol=1e-5
    )
    assert float(probabilities[0, :, ION_STATE_TO_ID["sodiated"]].sum()) == 0.0
    assert float(probabilities[1, :, ION_STATE_TO_ID["protonated"]].sum()) == 0.0


def test_presence_and_intensity_head_shapes():
    torch.manual_seed(0)
    slots = torch.randn(2, SLOTS, SLOT_DIM)

    presence = PresenceHead(SLOT_DIM, hidden_dim=16).eval()(slots)
    intensity = IntensityHead(SLOT_DIM, hidden_dim=16).eval()(slots)

    assert presence.shape == (2, SLOTS)
    assert intensity.shape == (2, SLOTS)
    assert (intensity >= 0).all()


# ------------------------------------------------------------------------ loss


def test_presence_loss_balances_positives_and_negatives():
    logits = torch.zeros(1, SLOTS)
    assignment = Assignment(
        batch_index=torch.zeros(2, dtype=torch.long),
        slot_index=torch.tensor([0, 1]),
        target_index=torch.tensor([0, 1]),
    )

    value = presence_loss(logits, assignment)

    # both groups sit at logit 0 -> log 2 each, averaged
    assert float(value) == pytest.approx(torch.log(torch.tensor(2.0)).item(), abs=1e-5)


def test_presence_loss_without_positives_is_finite():
    logits = torch.zeros(1, SLOTS)
    empty = Assignment(
        batch_index=torch.zeros(0, dtype=torch.long),
        slot_index=torch.zeros(0, dtype=torch.long),
        target_index=torch.zeros(0, dtype=torch.long),
    )

    assert torch.isfinite(presence_loss(logits, empty))


def test_spectrum_loss_is_zero_for_a_perfect_rendering():
    full = torch.tensor([[0.6, 0.8, 0.0, 0.0]])
    mask = torch.tensor([[True, True, True, True]])
    peaks = torch.tensor([[0, 1] + [0] * (SLOTS - 2)])
    intensity = torch.zeros(1, SLOTS)
    intensity[0, 0] = 0.6
    intensity[0, 1] = 0.8
    assignment = Assignment(
        batch_index=torch.zeros(2, dtype=torch.long),
        slot_index=torch.tensor([0, 1]),
        target_index=torch.tensor([0, 1]),
    )

    loss = spectrum_cosine_loss(intensity, assignment, peaks, full, mask)

    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_spectrum_loss_penalises_peaks_left_outside_the_slots():
    # the model nails peak 0 but peak 1 has no slot, so cosine cannot reach 1
    full = torch.tensor([[0.6, 0.8, 0.0, 0.0]])
    mask = torch.tensor([[True, True, True, True]])
    peaks = torch.zeros(1, SLOTS, dtype=torch.long)
    intensity = torch.zeros(1, SLOTS)
    intensity[0, 0] = 0.6
    assignment = Assignment(
        batch_index=torch.zeros(1, dtype=torch.long),
        slot_index=torch.tensor([0]),
        target_index=torch.tensor([0]),
    )

    loss = spectrum_cosine_loss(intensity, assignment, peaks, full, mask)

    assert float(loss) == pytest.approx(1.0 - 0.6, abs=1e-5)


def test_compute_losses_runs_with_no_targets():
    weights = LossWeights()
    output = compute_losses(
        matched_bag_nll=torch.zeros(0),
        presence_logits=torch.zeros(1, SLOTS, requires_grad=True),
        predicted_intensity=torch.zeros(1, SLOTS),
        target_intensities=torch.zeros(1, 1),
        target_peak_indices=torch.zeros(1, 1, dtype=torch.long),
        target_peak_mask=torch.zeros(1, 1, dtype=torch.bool),
        full_peak_intensities=torch.zeros(1, 3),
        full_peak_mask=torch.ones(1, 3, dtype=torch.bool),
        assignment=Assignment(
            batch_index=torch.zeros(0, dtype=torch.long),
            slot_index=torch.zeros(0, dtype=torch.long),
            target_index=torch.zeros(0, dtype=torch.long),
        ),
        weights=weights,
    )

    assert torch.isfinite(output.total)
    output.total.backward()


# --------------------------------------------------------------- end to end


def test_pipeline_forward_and_backward_with_a_stand_in_memory():
    torch.manual_seed(0)
    batch, targets, candidates, elements = 2, 3, 5, 3
    decoder = make_slot_decoder()
    formula = StructuredFormulaDecoder(
        SLOT_DIM, hidden_dim=32, num_layers=1, num_heads=4, max_count=16, dropout=0.0
    )
    ion_head = IonStateHead(SLOT_DIM, hidden_dim=16)
    presence_head = PresenceHead(SLOT_DIM, hidden_dim=16)
    intensity_head = IntensityHead(SLOT_DIM, hidden_dim=16)

    memory = torch.randn(batch, 7, MEMORY_DIM)
    attention = torch.ones(batch, 7, dtype=torch.long)
    element_ids = torch.tensor([[1, 6, 8]]).expand(batch, elements)
    element_counts = torch.tensor([[6, 2, 1]]).expand(batch, elements)
    element_mask = torch.ones(batch, elements, dtype=torch.bool)
    candidate_counts = torch.randint(0, 2, (batch, candidates, elements))
    candidate_counts[:, :, 0].clamp_(min=1)  # candidates are never the empty formula
    candidate_ions = torch.zeros(batch, candidates, dtype=torch.long)
    candidate_to_peak = torch.tensor([[0, 0, 1, 2, 2]]).expand(batch, candidates).contiguous()
    candidate_mask = torch.ones(batch, candidates, dtype=torch.bool)
    admissible = torch.zeros(batch, len(ION_STATE_VOCABULARY), dtype=torch.bool)
    admissible[:, ION_STATE_TO_ID["protonated"]] = True

    slots = decoder(memory, attention)
    presence_logits = presence_head(slots)
    intensity = intensity_head(slots)
    ion_log_prob = ion_head(slots, admissible)

    shape = (batch, SLOTS, candidates, elements)
    flat = batch * SLOTS * candidates
    formula_log_prob = formula.log_prob(
        slots.unsqueeze(2).expand(batch, SLOTS, candidates, SLOT_DIM).reshape(flat, SLOT_DIM),
        element_ids.view(batch, 1, 1, elements).expand(shape).reshape(flat, elements),
        element_counts.view(batch, 1, 1, elements).expand(shape).reshape(flat, elements),
        candidate_counts.view(batch, 1, candidates, elements).expand(shape).reshape(flat, elements),
        element_mask.view(batch, 1, 1, elements).expand(shape).reshape(flat, elements),
    ).view(batch, SLOTS, candidates)

    candidate_log_prob = formula_log_prob + ion_log_prob.gather(
        2, candidate_ions.unsqueeze(1).expand(batch, SLOTS, candidates)
    )
    bag_nll = -bag_log_probability(
        candidate_log_prob, candidate_to_peak, candidate_mask, targets
    )
    target_intensities = torch.rand(batch, targets)
    cost = bag_nll + 0.25 * pairwise_huber(intensity, target_intensities)
    target_mask = torch.ones(batch, targets, dtype=torch.bool)
    assignment = hungarian_assign(cost, target_mask)

    matched = bag_nll[
        assignment.batch_index, assignment.slot_index, assignment.target_index
    ]
    losses = compute_losses(
        matched_bag_nll=matched,
        presence_logits=presence_logits,
        predicted_intensity=intensity,
        target_intensities=target_intensities,
        target_peak_indices=torch.arange(targets).expand(batch, targets).contiguous(),
        target_peak_mask=target_mask,
        full_peak_intensities=torch.rand(batch, 6),
        full_peak_mask=torch.ones(batch, 6, dtype=torch.bool),
        assignment=assignment,
        weights=LossWeights(),
    )

    assert slots.shape == (batch, SLOTS, SLOT_DIM)
    assert presence_logits.shape == (batch, SLOTS)
    assert intensity.shape == (batch, SLOTS)
    assert bag_nll.shape == (batch, SLOTS, targets)
    assert len(assignment) == batch * targets
    for value in (losses.total, losses.identity, losses.presence, losses.intensity, losses.spectrum):
        assert torch.isfinite(value)

    losses.total.backward()
    assert decoder.queries.grad is not None
    assert torch.isfinite(decoder.queries.grad).all()


def test_bag_over_every_non_empty_formula_has_probability_one():
    """The bag NLL must be scored against the non-empty distribution.

    Collect every non-empty subformula of a tiny precursor into one bag: if the
    decoder normalises over non-empty formulas, that bag holds all the mass and
    its NLL is zero. If the empty formula still carried probability, it would
    not.
    """
    torch.manual_seed(0)
    decoder = StructuredFormulaDecoder(
        SLOT_DIM, hidden_dim=32, num_layers=1, num_heads=4, max_count=8, dropout=0.0
    ).eval()
    elements = torch.tensor([[1, 6]])
    precursor = torch.tensor([[2, 1]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    members = [(h, c) for h in range(3) for c in range(2) if h + c > 0]
    size = len(members)
    slots = torch.randn(1, SLOT_DIM)

    log_prob = decoder.log_prob(
        slots.expand(size, SLOT_DIM),
        elements.expand(size, 2),
        precursor.expand(size, 2),
        torch.tensor(members),
        mask.expand(size, 2),
    ).view(1, 1, size)

    bag = bag_log_probability(
        log_prob,
        torch.zeros(1, size, dtype=torch.long),
        torch.ones(1, size, dtype=torch.bool),
        num_targets=1,
    )

    assert float(bag[0, 0, 0]) == pytest.approx(0.0, abs=1e-5)
    assert float(-bag[0, 0, 0]) == pytest.approx(0.0, abs=1e-5)


def test_production_output_carries_no_full_candidate_tensor():
    """The [B, 64, C] grid must not be reachable from a normal forward."""
    import dataclasses

    from metabo_sllm.model.fragment_latent_model import ModelConfig, ModelOutput

    names = {field.name for field in dataclasses.fields(ModelOutput)}

    assert "candidate_log_prob" not in names
    assert "bag_nll" not in names
    assert "cost" not in names
    assert ModelConfig(model_name_or_path="unused").return_full_candidate_scores is False


def test_encoder_refuses_a_missing_backbone(tmp_path):
    with pytest.raises(MissingBackboneError):
        QwenEncoder(tmp_path / "does-not-exist")


def test_encoder_refuses_a_directory_without_weights(tmp_path):
    (tmp_path / "config.json").write_text("{}")

    with pytest.raises(MissingBackboneError, match="no model weights"):
        QwenEncoder(tmp_path)
