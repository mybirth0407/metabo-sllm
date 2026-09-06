"""Tests for the fragment-supervision mask and the fixed-slot ordering."""

from __future__ import annotations

import numpy as np
import pytest

from metabo_sllm.data.supervision import (
    FIXED_SLOT_COUNT,
    supervision_mask,
    supervision_rank,
    top_slot_mask,
)


def test_mask_needs_a_candidate_and_enough_decimals():
    counts = np.array([0, 1, 3, 1, 0])
    decimals = np.array([4, 4, 1, 2, 0])

    assert supervision_mask(counts, decimals).tolist() == [False, True, False, True, False]


def test_low_precision_peaks_are_never_supervised():
    counts = np.array([5, 5, 5])
    decimals = np.array([0, 1, 2])

    assert supervision_mask(counts, decimals).tolist() == [False, False, True]


def test_peaks_without_candidates_are_never_supervised():
    counts = np.array([0, 0])
    decimals = np.array([4, 4])

    assert not supervision_mask(counts, decimals).any()


def test_mask_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        supervision_mask(np.array([1, 2]), np.array([4]))


def test_rank_orders_by_intensity_then_mz_then_index():
    mask = np.array([True, True, True, True])
    mzs = np.array([300.0, 100.0, 200.0, 400.0])
    intensities = np.array([10.0, 50.0, 10.0, 50.0])

    rank = supervision_rank(mask, mzs, intensities)

    # intensity 50 first (m/z 100 before 400), then intensity 10 (200 before 300)
    order = np.argsort(rank)
    assert order.tolist() == [1, 3, 2, 0]
    assert rank.dtype == np.int32


def test_rank_breaks_full_ties_by_original_index():
    mask = np.array([True, True, True])
    mzs = np.array([100.0, 100.0, 100.0])
    intensities = np.array([5.0, 5.0, 5.0])

    assert supervision_rank(mask, mzs, intensities).tolist() == [0, 1, 2]


def test_unsupervised_peaks_get_minus_one_and_do_not_consume_ranks():
    mask = np.array([False, True, False, True])
    mzs = np.array([100.0, 200.0, 300.0, 400.0])
    intensities = np.array([99.0, 1.0, 99.0, 2.0])

    rank = supervision_rank(mask, mzs, intensities)

    assert rank.tolist() == [-1, 1, -1, 0]


def test_rank_is_empty_when_nothing_is_supervised():
    mask = np.zeros(3, dtype=bool)
    rank = supervision_rank(mask, np.arange(3.0), np.arange(3.0))

    assert rank.tolist() == [-1, -1, -1]
    assert not top_slot_mask(rank).any()


def test_top_slot_mask_keeps_exactly_the_slot_count():
    count = FIXED_SLOT_COUNT + 25
    mask = np.ones(count, dtype=bool)
    mzs = np.arange(count, dtype=np.float64)
    intensities = np.arange(count, dtype=np.float64)[::-1].copy()

    rank = supervision_rank(mask, mzs, intensities)
    slots = top_slot_mask(rank)

    assert int(slots.sum()) == FIXED_SLOT_COUNT
    # highest intensity is peak 0, so the first 64 peaks fill the slots
    assert slots[:FIXED_SLOT_COUNT].all()
    assert not slots[FIXED_SLOT_COUNT:].any()


def test_top_slot_mask_keeps_everything_when_under_the_slot_count():
    mask = np.array([True, False, True])
    rank = supervision_rank(mask, np.array([1.0, 2.0, 3.0]), np.array([1.0, 9.0, 2.0]))

    assert top_slot_mask(rank).tolist() == [True, False, True]


def test_rank_reproduces_itself():
    rng = np.random.default_rng(0)
    mzs = rng.uniform(50, 900, 500)
    intensities = rng.integers(0, 1000, 500).astype(np.float64)
    mask = rng.random(500) > 0.3

    first = supervision_rank(mask, mzs, intensities)
    second = supervision_rank(mask, mzs, intensities)

    assert np.array_equal(first, second)
    assert set(first[mask].tolist()) == set(range(int(mask.sum())))
