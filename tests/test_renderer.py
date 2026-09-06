"""Evaluation binning follows ms-pred exactly.

Binning belongs to inference and evaluation only. Training's spectrum loss
scores predictions at observed peak indices and never bins anything, so the two
must not be conflated.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from metabo_sllm.rendering.spectrum import BinningConfig, bin_index, bin_spectrum


def test_canonical_ms_pred_settings():
    config = BinningConfig()

    assert config.num_bins == 15000
    assert config.upper_limit == 1500.0
    assert config.ppm_tolerance == 20.0
    assert not hasattr(config, "provisional_upper_limit")


@pytest.mark.parametrize("mz", [0.0, 1.0, 55.5, 180.0634, 999.9, 1499.0])
def test_bin_index_matches_the_ms_pred_formula(mz):
    expected = math.floor(mz * (15000 - 1) / 1500) + 1

    assert int(bin_index(np.array([mz]))[0]) == expected


def test_bin_index_is_one_based_at_zero():
    assert int(bin_index(np.array([0.0]))[0]) == 1


def test_peaks_beyond_the_upper_limit_are_dropped_not_clamped():
    config = BinningConfig()
    inside = bin_spectrum(np.array([100.0]), np.array([1.0]), config)
    outside = bin_spectrum(np.array([100.0, 1e9]), np.array([1.0, 5.0]), config)

    np.testing.assert_allclose(inside, outside)
    assert outside.sum() == pytest.approx(1.0)


def test_bins_accumulate_duplicate_masses():
    config = BinningConfig()
    binned = bin_spectrum(np.array([100.0, 100.0, 100.001]), np.array([1.0, 2.0, 3.0]), config)

    assert binned.sum() == pytest.approx(6.0)
    assert int((binned > 0).sum()) == 1


def test_binned_vector_has_the_declared_length():
    binned = bin_spectrum(np.array([10.0]), np.array([1.0]))

    assert binned.shape == (15000,)
    assert np.all(np.isfinite(binned))


def test_every_bin_index_used_is_in_range():
    config = BinningConfig()
    mz = np.linspace(0.0, config.upper_limit, 5000, endpoint=False)
    index = bin_index(mz, config)

    assert int(index.min()) >= 0
    assert int(index.max()) < config.num_bins


def test_empty_spectrum_bins_to_zeros():
    binned = bin_spectrum(np.array([]), np.array([]))

    assert binned.shape == (15000,)
    assert binned.sum() == 0.0
