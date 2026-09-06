"""Tests for the graph-free fragment-formula candidate policy."""

from __future__ import annotations

import numpy as np
import pytest

from metabo_sllm.chem.candidates import (
    CHLORIDE_RETAINED,
    DEFAULT_PPM,
    HEAVY_SUBFORMULA_CAP,
    PROTONATED,
    SODIATED,
    UnsupportedChargeError,
    channels_for_adduct,
    generate_candidates,
    is_low_precision,
    mass_tolerance,
    match_spectrum,
    tolerance_array,
)
from metabo_sllm.chem.formula import (
    EnumerationCapExceeded,
    SubformulaTable,
    element_mass,
    formula_to_string,
    parse_formula,
)

GLUCOSE = "C6H12O6"


def neutral_mass(counts: dict[str, int]) -> float:
    return sum(element_mass(symbol) * n for symbol, n in counts.items())


def table_for(formula: str) -> SubformulaTable:
    return SubformulaTable(parse_formula(formula), heavy_cap=HEAVY_SUBFORMULA_CAP)


# --------------------------------------------------------------------------- ion policy


@pytest.mark.parametrize(
    ("adduct", "expected"),
    [
        ("[M+H]+", ("protonated",)),
        ("[M-H]-", ("deprotonated",)),
        ("[M-H2O+H]+", ("protonated",)),
        ("[M-H-CO2]-", ("deprotonated",)),
        ("[M+Na]+", ("protonated", "sodiated")),
        ("[M+K]+", ("protonated", "potassiated")),
        ("[M+Cl]-", ("deprotonated", "chloride_retained")),
    ],
)
def test_channels_for_adduct(adduct, expected):
    assert tuple(c.name for c in channels_for_adduct(adduct)) == expected


@pytest.mark.parametrize("adduct", ["[M+H3N+H]+", "[M+NH4]+"])
def test_ammonium_is_not_retained(adduct):
    assert tuple(c.name for c in channels_for_adduct(adduct)) == ("protonated",)


@pytest.mark.parametrize("adduct", ["[M+CHO2]-", "[M+HCOO]-"])
def test_formate_is_not_retained(adduct):
    assert tuple(c.name for c in channels_for_adduct(adduct)) == ("deprotonated",)


def test_no_radical_channels_are_ever_returned():
    for adduct in ("[M+H]+", "[M-H]-", "[M+Na]+", "[M+K]+", "[M+Cl]-"):
        names = {c.name for c in channels_for_adduct(adduct)}
        assert not any("radical" in name for name in names)


@pytest.mark.parametrize("adduct", ["[M+2H]2+", "[M-2H]2-", "[M+3H]3+", "[M+H]++", "[M-H]--"])
def test_multiple_charge_is_refused_not_guessed(adduct):
    with pytest.raises(UnsupportedChargeError, match="charge"):
        channels_for_adduct(adduct)


@pytest.mark.parametrize("adduct", ["", "   ", "[M+H]", "M+H"])
def test_unreadable_polarity_is_refused(adduct):
    with pytest.raises(UnsupportedChargeError):
        channels_for_adduct(adduct)


def test_channel_offsets_reproduce_known_ion_masses():
    glucose = neutral_mass({"C": 6, "H": 12, "O": 6})
    assert glucose == pytest.approx(180.06339, abs=1e-4)
    assert glucose + PROTONATED.mz_offset == pytest.approx(181.07066, abs=1e-4)
    assert glucose + SODIATED.mz_offset == pytest.approx(203.05261, abs=1e-4)
    assert glucose + CHLORIDE_RETAINED.mz_offset == pytest.approx(215.03278, abs=1e-4)


# --------------------------------------------------------------------------- tolerance


def test_tolerance_uses_the_wider_of_ppm_and_rounding():
    # four decimals: the 10 ppm window dominates
    assert mass_tolerance(300.0, 4) == pytest.approx(300.0 * DEFAULT_PPM * 1e-6)
    # two decimals at low m/z: the rounding half-width dominates
    assert mass_tolerance(100.0, 2) == pytest.approx(0.005)


def test_low_precision_peaks_use_ppm_alone():
    assert is_low_precision(0)
    assert is_low_precision(1)
    assert not is_low_precision(2)
    assert mass_tolerance(100.0, 1) == pytest.approx(100.0 * DEFAULT_PPM * 1e-6)
    assert mass_tolerance(100.0, 0) == pytest.approx(100.0 * DEFAULT_PPM * 1e-6)


def test_tolerance_array_matches_the_scalar_rule():
    mzs = np.array([100.0, 300.0, 100.0, 50.0])
    decimals = np.array([1, 4, 2, 0])
    np.testing.assert_allclose(
        tolerance_array(mzs, decimals),
        [mass_tolerance(float(m), int(d)) for m, d in zip(mzs, decimals, strict=True)],
    )


# --------------------------------------------------------------------------- matching


def test_known_subformula_is_found_with_zero_error():
    table = table_for(GLUCOSE)
    target = {"C": 6, "H": 10, "O": 5}
    mz = neutral_mass(target) + PROTONATED.mz_offset

    candidates = generate_candidates(table, mz, 4, (PROTONATED,))
    exact = [c for c in candidates if c.neutral_formula == "C6H10O5"]

    assert len(exact) == 1
    assert exact[0].ion_state == "protonated"
    assert exact[0].theoretical_mz == pytest.approx(mz, abs=1e-9)
    assert exact[0].error_da == pytest.approx(0.0, abs=1e-9)
    assert exact[0].error_ppm == pytest.approx(0.0, abs=1e-3)


def test_empty_formula_is_never_a_candidate():
    table = table_for(GLUCOSE)
    match = match_spectrum(
        table, np.array([PROTONATED.mz_offset]), np.array([4]), (PROTONATED,)
    )

    assert int(match.peak_candidate_counts[0]) == 0
    assert match.unique_candidates == 0


def test_hydrogen_is_searched_over_its_whole_range():
    table = table_for(GLUCOSE)
    for hydrogen in (0, 1, 6, 12):
        counts = {"C": 6, "H": hydrogen}
        mz = neutral_mass(counts) + PROTONATED.mz_offset
        formulas = {c.neutral_formula for c in generate_candidates(table, mz, 4, (PROTONATED,))}
        assert formula_to_string(counts) in formulas


def test_candidates_never_exceed_precursor_element_counts():
    precursor = parse_formula("C3H6O2")
    table = SubformulaTable(precursor, heavy_cap=1_000)

    for mz in np.linspace(15.0, 130.0, 60):
        for candidate in generate_candidates(table, float(mz), 4, (PROTONATED,)):
            for symbol, count in parse_formula(candidate.neutral_formula).items():
                assert symbol in precursor
                assert count <= precursor[symbol]


def test_sodiated_channel_reached_only_through_the_adduct_policy():
    table = table_for(GLUCOSE)
    mz = neutral_mass({"C": 6, "H": 12, "O": 6}) + SODIATED.mz_offset

    protonated_only = generate_candidates(table, mz, 4, channels_for_adduct("[M+H]+"))
    with_sodium = generate_candidates(table, mz, 4, channels_for_adduct("[M+Na]+"))

    assert "sodiated" not in {c.ion_state for c in protonated_only}
    assert "C6H12O6" in {c.neutral_formula for c in with_sodium if c.ion_state == "sodiated"}


def test_single_peak_count_equals_unique_candidate_count():
    table = table_for(GLUCOSE)
    mz = neutral_mass({"C": 6, "H": 10, "O": 5}) + PROTONATED.mz_offset
    match = match_spectrum(table, np.array([mz]), np.array([4]), (PROTONATED,))

    assert match.unique_candidates == int(match.peak_candidate_counts[0])


def test_unique_candidates_never_exceed_the_peak_total():
    table = table_for(GLUCOSE)
    mzs = np.array([100.0399, 145.0495, 163.0601, 181.0707])
    decimals = np.array([4, 4, 4, 4])
    match = match_spectrum(table, mzs, decimals, channels_for_adduct("[M+H]+"))

    assert match.peak_candidate_counts.shape == mzs.shape
    assert match.unique_candidates <= int(match.peak_candidate_counts.sum())


def test_wider_tolerance_never_loses_candidates():
    table = table_for(GLUCOSE)
    mz = neutral_mass({"C": 6, "H": 10, "O": 5}) + PROTONATED.mz_offset
    tight = match_spectrum(table, np.array([mz]), np.array([4]), (PROTONATED,), ppm=1.0)
    loose = match_spectrum(table, np.array([mz]), np.array([4]), (PROTONATED,), ppm=100.0)

    assert int(loose.peak_candidate_counts[0]) >= int(tight.peak_candidate_counts[0]) >= 1


def test_peak_candidate_cap_fails_instead_of_truncating():
    table = table_for(GLUCOSE)
    mz = neutral_mass({"C": 6, "H": 10, "O": 5}) + PROTONATED.mz_offset

    with pytest.raises(EnumerationCapExceeded, match="per-peak cap"):
        match_spectrum(table, np.array([mz]), np.array([4]), (PROTONATED,), peak_cap=0)


def test_empty_spectrum_matches_nothing():
    table = table_for(GLUCOSE)
    match = match_spectrum(
        table, np.empty(0), np.empty(0, dtype=np.int64), (PROTONATED,)
    )

    assert match.peak_candidate_counts.shape == (0,)
    assert match.unique_candidates == 0
