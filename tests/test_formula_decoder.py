"""The formula decoder must never be able to emit a formula the precursor cannot supply."""

from __future__ import annotations

import pytest
import torch

from metabo_sllm.model.formula_decoder import StructuredFormulaDecoder

SLOT_DIM = 32


def build(max_count: int = 16, max_elements: int = 8) -> StructuredFormulaDecoder:
    torch.manual_seed(0)
    return StructuredFormulaDecoder(
        SLOT_DIM,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        max_count=max_count,
        max_elements=max_elements,
        dropout=0.0,
    ).eval()


# C2H6O with elements ordered by atomic number: H(1), C(6), O(8)
ELEMENTS = torch.tensor([[1, 6, 8]])
PRECURSOR = torch.tensor([[6, 2, 1]])
MASK = torch.tensor([[True, True, True]])


def test_counts_above_the_precursor_have_exactly_zero_probability():
    decoder = build()
    slots = torch.randn(1, SLOT_DIM)
    previous = torch.tensor([[decoder.bos_count, 0, 0]])

    logits = decoder.logits(slots, ELEMENTS, PRECURSOR, previous, MASK)
    probabilities = torch.softmax(logits, dim=-1)

    for position, limit in enumerate(PRECURSOR[0].tolist()):
        assert torch.isinf(logits[0, position, limit + 1 :]).all()
        assert float(probabilities[0, position, limit + 1 :].sum()) == 0.0
        assert float(probabilities[0, position, : limit + 1].sum()) == pytest.approx(1.0, abs=1e-5)


def test_teacher_forced_log_probability_matches_a_manual_walk():
    decoder = build()
    slots = torch.randn(2, SLOT_DIM)
    elements = ELEMENTS.expand(2, -1)
    precursor = PRECURSOR.expand(2, -1)
    mask = MASK.expand(2, -1)
    targets = torch.tensor([[4, 1, 0], [6, 2, 1]])

    combined = decoder.log_prob(slots, elements, precursor, targets, mask)

    manual = torch.zeros(2)
    for row in range(2):
        previous = [decoder.bos_count, *targets[row, :-1].tolist()]
        logits = decoder.logits(
            slots[row : row + 1],
            elements[row : row + 1],
            precursor[row : row + 1],
            torch.tensor([previous]),
            mask[row : row + 1],
        )
        log_probs = torch.log_softmax(logits, dim=-1)
        for step, value in enumerate(targets[row].tolist()):
            manual[row] += log_probs[0, step, value]

    torch.testing.assert_close(combined, manual, rtol=1e-5, atol=1e-5)


def test_padded_elements_do_not_contribute():
    decoder = build()
    slots = torch.randn(1, SLOT_DIM)
    elements = torch.tensor([[1, 6, 0]])
    precursor = torch.tensor([[6, 2, 0]])
    mask = torch.tensor([[True, True, False]])
    targets = torch.tensor([[4, 1, 0]])

    padded = decoder.log_prob(slots, elements, precursor, targets, mask)
    short = decoder.log_prob(
        slots, elements[:, :2], precursor[:, :2], targets[:, :2], mask[:, :2]
    )

    torch.testing.assert_close(padded, short, rtol=1e-5, atol=1e-5)


def enumerate_subformulas(precursor: torch.Tensor):
    """Every subformula of a precursor, empty one included."""
    ranges = [range(int(value) + 1) for value in precursor[0]]
    for h in ranges[0]:
        for c in ranges[1]:
            for o in ranges[2]:
                yield (h, c, o)


def test_non_empty_formula_probabilities_sum_to_one():
    decoder = build()
    slots = torch.randn(1, SLOT_DIM)

    total = 0.0
    for combination in enumerate_subformulas(PRECURSOR):
        if sum(combination) == 0:
            continue
        target = torch.tensor([combination])
        total += float(decoder.log_prob(slots, ELEMENTS, PRECURSOR, target, MASK).exp())

    assert total == pytest.approx(1.0, abs=1e-4)


def test_empty_formula_has_exactly_zero_probability():
    decoder = build()
    slots = torch.randn(4, SLOT_DIM)
    empty = torch.zeros(4, 3, dtype=torch.long)

    log_prob = decoder.log_prob(
        slots, ELEMENTS.expand(4, -1), PRECURSOR.expand(4, -1), empty, MASK.expand(4, -1)
    )

    assert torch.isneginf(log_prob).all()
    assert float(log_prob.exp().sum()) == 0.0


def test_single_element_precursor_cannot_emit_zero():
    decoder = build()
    slots = torch.randn(2, SLOT_DIM)
    elements = torch.tensor([[6], [6]])
    precursor = torch.tensor([[4], [4]])
    mask = torch.ones(2, 1, dtype=torch.bool)

    logits = decoder.logits(
        slots, elements, precursor, torch.full((2, 1), decoder.bos_count), mask
    )
    probabilities = torch.softmax(logits, dim=-1)

    assert float(probabilities[:, 0, 0].sum()) == 0.0
    assert float(probabilities[0, 0, 1:5].sum()) == pytest.approx(1.0, abs=1e-5)


def test_zero_allowed_on_the_last_element_once_something_is_present():
    decoder = build()
    slots = torch.randn(1, SLOT_DIM)
    # H already took 2 atoms, so O may still take 0
    previous = torch.tensor([[decoder.bos_count, 2, 1]])

    logits = decoder.logits(slots, ELEMENTS, PRECURSOR, previous, MASK)

    assert torch.isfinite(logits[0, 2, 0])


def test_greedy_and_teacher_forcing_share_the_same_constraint():
    decoder = build()
    slots = torch.randn(6, SLOT_DIM)
    elements = ELEMENTS.expand(6, -1)
    precursor = PRECURSOR.expand(6, -1)
    mask = MASK.expand(6, -1)

    counts = decoder.greedy_decode(slots, elements, precursor, mask)
    log_prob = decoder.log_prob(slots, elements, precursor, counts, mask)

    assert (counts.sum(dim=1) > 0).all()
    assert torch.isfinite(log_prob).all()


def test_precursor_with_a_zero_count_element_is_rejected():
    decoder = build()
    slots = torch.randn(1, SLOT_DIM)
    precursor = torch.tensor([[6, 0, 1]])

    with pytest.raises(ValueError, match="no non-empty subformula"):
        decoder.log_prob(slots, ELEMENTS, precursor, torch.tensor([[1, 0, 1]]), MASK)


def test_greedy_decode_stays_inside_the_precursor():
    decoder = build()
    slots = torch.randn(16, SLOT_DIM)
    elements = ELEMENTS.expand(16, -1)
    precursor = PRECURSOR.expand(16, -1)
    mask = MASK.expand(16, -1)

    counts = decoder.greedy_decode(slots, elements, precursor, mask)

    assert (counts <= precursor).all()
    assert (counts >= 0).all()


def test_greedy_decode_never_returns_the_empty_formula():
    decoder = build()
    # push every count logit towards zero so the natural argmax would be all-zero
    with torch.no_grad():
        decoder.count_head.weight.zero_()
        decoder.count_head.bias.zero_()
        decoder.count_head.bias[0] = 10.0
    slots = torch.randn(8, SLOT_DIM)

    counts = decoder.greedy_decode(
        slots, ELEMENTS.expand(8, -1), PRECURSOR.expand(8, -1), MASK.expand(8, -1)
    )

    assert (counts.sum(dim=1) > 0).all()


def test_precursor_count_above_the_vocabulary_fails_instead_of_clipping():
    decoder = build(max_count=8)
    slots = torch.randn(1, SLOT_DIM)
    precursor = torch.tensor([[99, 2, 1]])

    with pytest.raises(ValueError, match="refusing to clip"):
        decoder.log_prob(slots, ELEMENTS, precursor, torch.tensor([[1, 1, 1]]), MASK)


def test_too_many_elements_fails():
    decoder = build(max_elements=2)
    slots = torch.randn(1, SLOT_DIM)

    with pytest.raises(ValueError, match="max_elements"):
        decoder.log_prob(slots, ELEMENTS, PRECURSOR, torch.tensor([[1, 1, 1]]), MASK)


def test_log_probability_is_differentiable_through_the_slot():
    decoder = build()
    slots = torch.randn(2, SLOT_DIM, requires_grad=True)
    value = decoder.log_prob(
        slots, ELEMENTS.expand(2, -1), PRECURSOR.expand(2, -1),
        torch.tensor([[4, 1, 0], [2, 2, 1]]), MASK.expand(2, -1)
    ).sum()
    value.backward()

    assert slots.grad is not None
    assert torch.isfinite(slots.grad).all()
