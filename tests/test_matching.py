"""Bag likelihood and slot/target assignment."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from metabo_sllm.losses.matching import (
    bag_log_probability,
    hungarian_assign,
    pairwise_huber,
)


def test_bag_log_probability_is_a_logsumexp_over_the_bag():
    # one spectrum, two slots, three candidates: peaks 0, 0, 1
    log_prob = torch.tensor([[[-1.0, -2.0, -3.0], [-0.5, -4.0, -0.25]]])
    to_peak = torch.tensor([[0, 0, 1]])
    mask = torch.ones(1, 3, dtype=torch.bool)

    bag = bag_log_probability(log_prob, to_peak, mask, num_targets=2)

    expected = torch.tensor(
        [
            [
                [torch.logsumexp(torch.tensor([-1.0, -2.0]), 0), torch.tensor(-3.0)],
                [torch.logsumexp(torch.tensor([-0.5, -4.0]), 0), torch.tensor(-0.25)],
            ]
        ]
    )
    torch.testing.assert_close(bag, expected, rtol=1e-5, atol=1e-5)


def test_a_single_candidate_bag_reduces_to_its_own_log_probability():
    log_prob = torch.tensor([[[-1.5, -9.0]]])
    to_peak = torch.tensor([[0, 1]])
    mask = torch.ones(1, 2, dtype=torch.bool)

    bag = bag_log_probability(log_prob, to_peak, mask, num_targets=2)

    torch.testing.assert_close(bag[0, 0], torch.tensor([-1.5, -9.0]), rtol=1e-5, atol=1e-5)


def test_masked_candidates_are_ignored():
    log_prob = torch.tensor([[[-1.0, 0.0]]])
    to_peak = torch.tensor([[0, 0]])
    mask = torch.tensor([[True, False]])

    bag = bag_log_probability(log_prob, to_peak, mask, num_targets=1)

    torch.testing.assert_close(bag[0, 0, 0], torch.tensor(-1.0), rtol=1e-5, atol=1e-4)


def test_bag_log_probability_is_differentiable():
    log_prob = torch.tensor([[[-1.0, -2.0]]], requires_grad=True)
    to_peak = torch.tensor([[0, 0]])
    mask = torch.ones(1, 2, dtype=torch.bool)

    bag_log_probability(log_prob, to_peak, mask, num_targets=1).sum().backward()

    assert log_prob.grad is not None
    assert torch.isfinite(log_prob.grad).all()
    # gradient is the softmax over the bag
    torch.testing.assert_close(
        log_prob.grad[0, 0], torch.softmax(torch.tensor([-1.0, -2.0]), 0), rtol=1e-5, atol=1e-5
    )


def test_empty_bag_stays_finite():
    log_prob = torch.zeros(1, 2, 1)
    to_peak = torch.zeros(1, 1, dtype=torch.long)
    mask = torch.zeros(1, 1, dtype=torch.bool)

    bag = bag_log_probability(log_prob, to_peak, mask, num_targets=2)

    assert torch.isfinite(bag).all()


# ------------------------------------------------------------------ hungarian


def test_assignment_is_one_to_one_and_optimal():
    cost = torch.tensor([[[0.0, 5.0, 5.0], [5.0, 0.0, 5.0], [5.0, 5.0, 0.0], [9.0, 9.0, 9.0]]])
    mask = torch.ones(1, 3, dtype=torch.bool)

    assignment = hungarian_assign(cost, mask)

    assert len(assignment) == 3
    assert sorted(assignment.target_index.tolist()) == [0, 1, 2]
    assert len(set(assignment.slot_index.tolist())) == 3
    pairs = dict(zip(assignment.target_index.tolist(), assignment.slot_index.tolist(), strict=True))
    assert pairs == {0: 0, 1: 1, 2: 2}


def test_assignment_is_invariant_to_target_permutation():
    torch.manual_seed(0)
    cost = torch.rand(1, 8, 5)
    mask = torch.ones(1, 5, dtype=torch.bool)

    base = hungarian_assign(cost, mask)
    base_pairs = set(zip(base.slot_index.tolist(), base.target_index.tolist(), strict=True))

    permutation = torch.tensor([3, 0, 4, 1, 2])
    permuted = hungarian_assign(cost[:, :, permutation], mask)
    restored = {
        (slot, int(permutation[target]))
        for slot, target in zip(
            permuted.slot_index.tolist(), permuted.target_index.tolist(), strict=True
        )
    }

    assert restored == base_pairs


def test_padded_targets_are_never_matched():
    cost = torch.zeros(1, 4, 4)
    cost[0, :, 2:] = -100.0  # padded columns look attractive but must be skipped
    mask = torch.tensor([[True, True, False, False]])

    assignment = hungarian_assign(cost, mask)

    assert sorted(assignment.target_index.tolist()) == [0, 1]


def test_spectrum_without_targets_is_skipped():
    cost = torch.zeros(2, 4, 3)
    mask = torch.tensor([[True, True, False], [False, False, False]])

    assignment = hungarian_assign(cost, mask)

    assert assignment.batch_index.tolist() == [0, 0]


def test_assignment_does_not_carry_gradient():
    cost = torch.rand(1, 4, 2, requires_grad=True)
    mask = torch.ones(1, 2, dtype=torch.bool)

    assignment = hungarian_assign(cost, mask)

    assert not assignment.slot_index.is_floating_point()
    assert cost.grad is None


def test_more_targets_than_available_slots_is_rejected_by_shape():
    cost = torch.rand(1, 2, 5)
    mask = torch.ones(1, 5, dtype=torch.bool)

    with pytest.raises(ValueError):
        hungarian_assign(cost, mask)


def test_pairwise_huber_matches_the_elementwise_formula():
    predicted = torch.tensor([[0.0, 2.0]])
    target = torch.tensor([[0.5, 3.0]])

    matrix = pairwise_huber(predicted, target, delta=1.0)

    expected = torch.nn.functional.huber_loss(
        predicted.unsqueeze(2).expand(1, 2, 2),
        target.unsqueeze(1).expand(1, 2, 2),
        delta=1.0,
        reduction="none",
    )
    torch.testing.assert_close(matrix, expected, rtol=1e-6, atol=1e-6)
    assert matrix.shape == (1, 2, 2)


def test_hungarian_beats_a_greedy_choice():
    cost = torch.tensor([[[1.0, 2.0], [2.0, 100.0]]])
    mask = torch.ones(1, 2, dtype=torch.bool)

    assignment = hungarian_assign(cost, mask)
    total = sum(
        float(cost[0, slot, target])
        for slot, target in zip(
            assignment.slot_index.tolist(), assignment.target_index.tolist(), strict=True
        )
    )

    assert total == pytest.approx(4.0)  # greedy would pick 1.0 then 100.0
    assert np.isfinite(total)
