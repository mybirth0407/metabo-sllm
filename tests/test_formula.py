"""Tests for formula parsing, monoisotopic masses and subformula enumeration."""

from __future__ import annotations

import itertools
from math import prod

import numpy as np
import pytest

from metabo_sllm.chem.formula import (
    EnumerationCapExceeded,
    FormulaError,
    SubformulaTable,
    element_mass,
    formula_to_string,
    parse_formula,
)


def test_parse_simple_formula():
    assert parse_formula("C6H12O6") == {"C": 6, "H": 12, "O": 6}


def test_parse_two_letter_elements_and_implicit_one():
    assert parse_formula("CHClBrNa") == {"C": 1, "H": 1, "Cl": 1, "Br": 1, "Na": 1}


def test_parse_repeated_element_accumulates():
    assert parse_formula("CH3CH2OH") == {"C": 2, "H": 6, "O": 1}


@pytest.mark.parametrize("text", ["", "6C", "C6H12O6!", "Xx4", "C0", "C-1"])
def test_parse_rejects_bad_formulas(text):
    with pytest.raises(FormulaError):
        parse_formula(text)


def test_hill_notation():
    assert formula_to_string({"O": 6, "H": 12, "C": 6}) == "C6H12O6"
    assert formula_to_string({"C": 1, "H": 4}) == "CH4"
    assert formula_to_string({"S": 1, "H": 2, "O": 4}) == "H2O4S"  # no carbon -> alphabetical
    assert formula_to_string({"C": 2, "H": 0}) == "C2"
    assert formula_to_string({}) == ""


def test_monoisotopic_masses_match_known_values():
    assert element_mass("C") == pytest.approx(12.0, abs=1e-9)
    assert element_mass("H") == pytest.approx(1.007825, abs=1e-5)
    assert element_mass("O") == pytest.approx(15.994915, abs=1e-5)
    assert element_mass("Cl") == pytest.approx(34.968853, abs=1e-5)


def test_unknown_element_rejected():
    with pytest.raises(FormulaError):
        element_mass("Xx")


def test_table_shape_and_sorted_masses():
    table = SubformulaTable(parse_formula("C2H4O"), heavy_cap=1_000)

    assert table.heavy_symbols == ("C", "O")
    assert table.heavy_counts == (2, 1)
    assert table.heavy_size == 3 * 2
    assert table.max_hydrogen == 4
    assert table.total_size == 6 * 5
    assert table.heavy_masses.shape == (6,)
    assert np.all(np.diff(table.heavy_masses) >= 0)
    assert table.heavy_masses[0] == 0.0  # the empty heavy subformula


def test_every_subformula_is_enumerated_exactly_once():
    precursor = parse_formula("C2H4OS")
    table = SubformulaTable(precursor, heavy_cap=1_000)

    decoded = set()
    for heavy_index in range(table.heavy_size):
        for hydrogen in range(table.max_hydrogen + 1):
            counts = table.decode(heavy_index, hydrogen)
            for symbol, value in counts.items():
                assert 0 <= value <= precursor[symbol]
            decoded.add(formula_to_string(counts))

    expected = set()
    ranges = [range(precursor[s] + 1) for s in ("C", "H", "O", "S")]
    for combination in itertools.product(*ranges):
        counts = dict(zip(("C", "H", "O", "S"), combination, strict=True))
        expected.add(formula_to_string(counts))

    assert decoded == expected
    assert len(decoded) == prod(precursor[s] + 1 for s in ("C", "H", "O", "S"))


def test_decoded_mass_matches_element_sum():
    table = SubformulaTable(parse_formula("C3H6NO2"), heavy_cap=10_000)

    for heavy_index in (0, 1, table.heavy_size // 2, table.heavy_size - 1):
        for hydrogen in (0, 3, table.max_hydrogen):
            counts = table.decode(heavy_index, hydrogen)
            expected = sum(element_mass(s) * n for s, n in counts.items())
            assert table.mass(heavy_index, hydrogen) == pytest.approx(expected, abs=1e-9)


def test_formula_without_hydrogen():
    table = SubformulaTable(parse_formula("CCl4"), heavy_cap=1_000)

    assert table.max_hydrogen == 0
    assert table.total_size == table.heavy_size == 2 * 5


def test_heavy_cap_is_enforced_not_truncated():
    with pytest.raises(EnumerationCapExceeded, match="exceed the cap"):
        SubformulaTable(parse_formula("C10H10O10"), heavy_cap=100)
