"""Binning and cos@K must match ms-pred's definitions exactly."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metabo_sllm.evaluation.spectrum_metrics import (
    EVALUATION_SPACES,
    PRIMARY_SPACE,
    BinningConfig,
    bin_experimental,
    bin_index,
    bin_prediction,
    cosine_at_k,
    observation_in_space,
    score_prediction,
    summarise_scores,
)

CONFIG = BinningConfig()


# ------------------------------------------------------------------- binning


@pytest.mark.parametrize("mz", [0.0, 1.0, 55.5, 180.0634, 999.9, 1499.0])
def test_bin_index_matches_ms_pred(mz):
    assert int(bin_index(np.array([mz]))[0]) == math.floor(mz * 14999 / 1500) + 1


def test_bin_indices_stay_in_range():
    index = bin_index(np.linspace(0.0, CONFIG.upper_limit, 4000, endpoint=False))

    assert index.min() >= 0
    assert index.max() < CONFIG.num_bins


def test_experimental_bins_take_the_maximum():
    binned = bin_experimental(np.array([100.0, 100.001, 100.002]), np.array([1.0, 7.0, 3.0]))

    assert binned.sum() == pytest.approx(7.0)
    assert int((binned > 0).sum()) == 1


def test_predicted_bins_add_up():
    binned, dropped = bin_prediction(np.array([100.0, 100.001]), np.array([1.0, 3.0]))

    assert dropped == 0
    assert binned.max() == pytest.approx(1.0)  # normalised by the peak
    assert int((binned > 0).sum()) == 1


def test_out_of_range_predictions_are_dropped_and_counted():
    binned, dropped = bin_prediction(np.array([100.0, 1e9, -5.0]), np.array([1.0, 5.0, 5.0]))

    assert dropped == 2
    assert int((binned > 0).sum()) == 1


def test_min_intensity_threshold_prunes_dust():
    config = BinningConfig(min_pred_intensity=1e-2)
    binned, _ = bin_prediction(np.array([100.0, 200.0]), np.array([1.0, 1e-6]), config)

    assert int((binned > 0).sum()) == 1


def test_empty_prediction_bins_to_zeros():
    binned, dropped = bin_prediction(np.array([]), np.array([]))

    assert binned.shape == (CONFIG.num_bins,)
    assert binned.sum() == 0.0
    assert dropped == 0


# -------------------------------------------------------------------- cosine


def test_identical_spectra_score_one():
    spectrum = np.zeros(CONFIG.num_bins)
    spectrum[[10, 20, 30]] = [1.0, 0.5, 0.25]

    assert cosine_at_k(spectrum, spectrum, 100) == pytest.approx(1.0)


def test_disjoint_spectra_score_zero():
    prediction = np.zeros(CONFIG.num_bins)
    prediction[10] = 1.0
    experimental = np.zeros(CONFIG.num_bins)
    experimental[500] = 1.0

    assert cosine_at_k(prediction, experimental, 100) == pytest.approx(0.0)


def test_cosine_is_scale_invariant():
    prediction = np.zeros(CONFIG.num_bins)
    prediction[[3, 9]] = [1.0, 2.0]
    experimental = np.zeros(CONFIG.num_bins)
    experimental[[3, 9]] = [10.0, 20.0]

    assert cosine_at_k(prediction, experimental, 100) == pytest.approx(1.0)
    assert cosine_at_k(prediction * 1e6, experimental, 100) == pytest.approx(1.0)


def test_top_k_keeps_only_the_strongest_predicted_bins():
    prediction = np.zeros(CONFIG.num_bins)
    prediction[[1, 2, 3]] = [1.0, 0.5, 0.25]
    experimental = np.zeros(CONFIG.num_bins)
    experimental[3] = 1.0  # only the weakest predicted bin is real

    assert cosine_at_k(prediction, experimental, 1) == pytest.approx(0.0)
    assert cosine_at_k(prediction, experimental, 3) > 0.0


def test_cos_at_20_never_exceeds_cos_at_100_when_prediction_is_pruned():
    rng = np.random.default_rng(0)
    prediction = np.zeros(CONFIG.num_bins)
    experimental = np.zeros(CONFIG.num_bins)
    bins = rng.choice(CONFIG.num_bins, 150, replace=False)
    prediction[bins] = rng.random(150)
    experimental[bins] = rng.random(150)

    assert cosine_at_k(prediction, experimental, 20) <= cosine_at_k(prediction, experimental, 100) + 1e-12


def test_zero_prediction_scores_zero_without_dividing_by_zero():
    experimental = np.zeros(CONFIG.num_bins)
    experimental[5] = 1.0

    assert cosine_at_k(np.zeros(CONFIG.num_bins), experimental, 100) == 0.0


def test_zero_experimental_scores_zero():
    prediction = np.zeros(CONFIG.num_bins)
    prediction[5] = 1.0

    assert cosine_at_k(prediction, np.zeros(CONFIG.num_bins), 100) == 0.0


# --------------------------------------------------------------------- score


MZ = np.array([100.0, 200.0, 300.0])
OBSERVED = np.array([1.0, 0.5, 0.25])


def test_both_spaces_are_always_reported():
    score = score_prediction("uid", MZ, OBSERVED, MZ, OBSERVED)

    assert set(score.cosine) == set(EVALUATION_SPACES)


def test_an_unnamed_space_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="unknown intensity space"):
        observation_in_space(OBSERVED, "sqrt")


def test_a_square_root_prediction_scores_one_in_the_primary_space():
    """The intensity head emits sqrt(y); that is where a perfect model is perfect."""
    score = score_prediction("uid", MZ, np.sqrt(OBSERVED), MZ, OBSERVED)

    assert score.cosine[PRIMARY_SPACE][20] == pytest.approx(1.0)
    assert score.cosine[PRIMARY_SPACE][100] == pytest.approx(1.0)
    assert score.cosine["legacy_raw"][100] < 1.0
    assert not score.zero_prediction
    assert score.out_of_range == 0


def test_a_raw_prediction_scores_one_only_in_the_legacy_space():
    """And the mirror image, so neither space can silently stand for the other."""
    score = score_prediction("uid", MZ, OBSERVED, MZ, OBSERVED)

    assert score.cosine["legacy_raw"][100] == pytest.approx(1.0)
    assert score.cosine[PRIMARY_SPACE][100] < 1.0


def test_duplicate_bins_are_counted():
    mz = np.array([100.0, 100.0005, 300.0])
    score = score_prediction(
        "uid", mz, np.array([1.0, 1.0, 1.0]), np.array([100.0]), np.array([1.0])
    )

    assert score.predicted_peaks == 3
    assert score.predicted_bins == 2
    assert score.duplicate_bin_collisions == 1


def test_empty_prediction_is_flagged_not_crashed():
    score = score_prediction(
        "uid", np.array([]), np.array([]), np.array([100.0]), np.array([1.0])
    )

    assert score.zero_prediction
    assert all(score.cosine[space][100] == 0.0 for space in EVALUATION_SPACES)


def test_summary_reports_the_distribution():
    mz = np.array([100.0])
    good = score_prediction("a", mz, np.array([1.0]), mz, np.array([1.0]))
    bad = score_prediction("b", np.array([]), np.array([]), mz, np.array([1.0]))

    summary = summarise_scores([good, bad])

    assert summary["spectra"] == 2
    assert summary["primary_space"] == PRIMARY_SPACE
    for space in EVALUATION_SPACES:
        assert summary[space]["cos@100"]["mean"] == pytest.approx(0.5)
        assert set(summary[space]["cos@100"]) == {"mean", "median", "p10", "p90"}
    # No bare ``cos@K``: a summary never leaves its space implicit, and an
    # older file that does is distinguishable from a newer one at a glance.
    assert "cos@100" not in summary
    assert summary["zero_prediction_spectra"] == 1


def test_summary_of_nothing_is_safe():
    assert summarise_scores([]) == {"spectra": 0}
