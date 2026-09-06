"""One prediction may fit several bags; it must still count only once."""

from __future__ import annotations

import pytest

from metabo_sllm.evaluation.fragment_metrics import (
    match_predictions_to_targets,
    score_fragments,
    summarise_fragments,
)

P = ("C6H10O5", "protonated")
Q = ("C6H12O6", "protonated")
R = ("CH4", "protonated")


# ------------------------------------------------------------------ matching


def test_a_prediction_matching_two_bags_is_used_once():
    matches = match_predictions_to_targets([P], [{P}, {P}])

    assert len(matches) == 1


def test_matching_maximises_the_number_of_covered_targets():
    # a greedy pass could pair P with the first bag and leave Q unmatched
    matches = match_predictions_to_targets([P, Q], [{P, Q}, {Q}])

    assert len(matches) == 2
    assert {column for _, column in matches} == {0, 1}


def test_unmatched_predictions_and_targets_are_left_alone():
    matches = match_predictions_to_targets([R], [{P}, {Q}])

    assert matches == []


def test_empty_inputs_are_safe():
    assert match_predictions_to_targets([], [{P}]) == []
    assert match_predictions_to_targets([P], []) == []
    assert match_predictions_to_targets([], []) == []


def test_ion_state_is_part_of_the_identity():
    assert match_predictions_to_targets([("C6H10O5", "sodiated")], [{P}]) == []


# -------------------------------------------------------------------- scores


def test_all_targets_hit():
    diagnostic = score_fragments("uid", [P, Q], [{P}, {Q}], [10.0, 5.0], active_slots=2)

    assert diagnostic.matched == 2
    assert diagnostic.unique_targets == 2
    assert diagnostic.unique_matched == 2
    assert diagnostic.matched_intensity == pytest.approx(15.0)


def test_ambiguous_and_unique_targets_are_counted_separately():
    diagnostic = score_fragments("uid", [P], [{P, Q}, {R}], [3.0, 7.0])

    assert diagnostic.ambiguous_targets == 1
    assert diagnostic.ambiguous_matched == 1
    assert diagnostic.unique_targets == 1
    assert diagnostic.unique_matched == 0
    assert diagnostic.matched_intensity == pytest.approx(3.0)


def test_wrong_predictions_score_nothing():
    diagnostic = score_fragments("uid", [R], [{P}], [1.0])

    assert diagnostic.matched == 0
    assert diagnostic.matched_intensity == pytest.approx(0.0)


def test_spectrum_without_targets_is_safe():
    diagnostic = score_fragments("uid", [P], [], [])

    assert diagnostic.targets == 0
    assert diagnostic.matched == 0


# ------------------------------------------------------------------- summary


def test_summary_pools_counts_rather_than_averaging_ratios():
    small = score_fragments("a", [P], [{P}], [1.0])  # 1/1
    large = score_fragments("b", [R], [{P}, {Q}, {P}], [1.0, 1.0, 1.0])  # 0/3

    summary = summarise_fragments([small, large])

    # averaging per-spectrum rates would give 0.5; pooled is 1/4
    assert summary["fragment_bag_hit"] == pytest.approx(0.25)
    assert summary["targets"] == 4
    assert summary["predictions"] == 2


def test_summary_reports_precision_and_duplicates():
    diagnostic = score_fragments(
        "a", [P, R], [{P}], [1.0], active_slots=5, duplicate_identity_slots=3
    )
    summary = summarise_fragments([diagnostic])

    assert summary["active_slot_precision"] == pytest.approx(0.5)
    assert summary["active_slots_mean"] == pytest.approx(5.0)
    assert summary["duplicate_identity_slots"] == 3
    assert summary["duplicate_active_formula_rate"] == pytest.approx(3 / 5)


def test_summary_of_nothing_is_safe():
    assert summarise_fragments([]) == {"spectra": 0}


def test_summary_returns_none_rather_than_dividing_by_zero():
    summary = summarise_fragments([score_fragments("a", [], [], [])])

    assert summary["fragment_bag_hit"] is None
    assert summary["active_slot_precision"] is None
