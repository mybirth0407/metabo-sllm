"""The encoder's view of a molecule must not vary between runs or machines."""

from __future__ import annotations

import math

import pytest

from metabo_sllm.model.input_formatter import (
    FIELD_ORDER,
    MISSING,
    format_molecular_input,
    format_row,
)

ROW = {
    "smiles": "CCO",
    "formula": "C2H6O",
    "adduct": "[M+H]+",
    "collision_energy": 20.0,
    "instrument": "Orbitrap",
    "precursor_mz": 47.049141,
}


def test_field_order_and_layout():
    text = format_row(ROW)

    assert text == (
        "SMILES=CCO | formula=C2H6O | adduct=[M+H]+ | collision_energy=20.00 | "
        "instrument=Orbitrap | precursor_mz=47.0491"
    )
    assert [part.split("=", 1)[0] for part in text.split(" | ")] == list(FIELD_ORDER)


def test_formatting_is_deterministic():
    assert format_row(ROW) == format_row(dict(ROW))
    assert format_row(ROW) == format_row({**ROW})


def test_number_formatting_is_fixed_width():
    text = format_molecular_input("C", "C", "[M+H]+", 2, "QTOF", 100.0)

    assert "collision_energy=2.00" in text
    assert "precursor_mz=100.0000" in text


def test_equal_floats_with_different_spellings_agree():
    a = format_molecular_input("C", "C", "[M+H]+", 20, "QTOF", 47.04914100)
    b = format_molecular_input("C", "C", "[M+H]+", 20.0, "QTOF", 47.049141)

    assert a == b


@pytest.mark.parametrize("value", [None, float("nan"), "", "   "])
def test_missing_text_uses_one_fixed_token(value):
    text = format_molecular_input(value, "C", "[M+H]+", 1, "QTOF", 10.0)

    assert text.startswith(f"SMILES={MISSING} |")


@pytest.mark.parametrize("value", [None, float("nan"), math.inf, "abc"])
def test_missing_numbers_use_one_fixed_token(value):
    text = format_molecular_input("C", "C", "[M+H]+", value, "QTOF", value)

    assert f"collision_energy={MISSING}" in text
    assert f"precursor_mz={MISSING}" in text


def test_absent_keys_are_treated_as_missing():
    text = format_row({"smiles": "CCO"})

    assert text.count(MISSING) == len(FIELD_ORDER) - 1


def test_whitespace_is_stripped():
    assert "SMILES=CCO |" in format_molecular_input("  CCO  ", "C", "[M+H]+", 1, "QTOF", 1.0)
