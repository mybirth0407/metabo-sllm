"""Deterministic molecular text for the language-model encoder.

The encoder sees a molecule only as this string, so its exact shape is part of
the model contract: the same row must produce the same text on every machine
and every run.  Field order is fixed, missing values get one fixed token, and
numbers are rendered with a fixed number of decimals rather than through
``str(float)``, whose output varies with the value's binary representation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

__all__ = [
    "COLLISION_ENERGY_DECIMALS",
    "FIELD_ORDER",
    "MISSING",
    "PRECURSOR_MZ_DECIMALS",
    "format_molecular_input",
    "format_row",
]

FIELD_ORDER = (
    "SMILES",
    "formula",
    "adduct",
    "collision_energy",
    "instrument",
    "precursor_mz",
)
MISSING = "NA"
SEPARATOR = " | "
COLLISION_ENERGY_DECIMALS = 2
PRECURSOR_MZ_DECIMALS = 4

# Row keys backing each field, in FIELD_ORDER.
_ROW_KEYS = ("smiles", "formula", "adduct", "collision_energy", "instrument", "precursor_mz")


def _text(value: object) -> str:
    if value is None:
        return MISSING
    if isinstance(value, float) and math.isnan(value):
        return MISSING
    text = str(value).strip()
    return text if text else MISSING


def _number(value: object, decimals: int) -> str:
    if value is None:
        return MISSING
    try:
        number = float(value)
    except (TypeError, ValueError):
        return MISSING
    if not math.isfinite(number):
        return MISSING
    return f"{number:.{decimals}f}"


def format_molecular_input(
    smiles: object,
    formula: object,
    adduct: object,
    collision_energy: object,
    instrument: object,
    precursor_mz: object,
) -> str:
    """Render one spectrum's conditioning text.

    ``SMILES=... | formula=... | adduct=... | collision_energy=... |
    instrument=... | precursor_mz=...``
    """
    values = (
        _text(smiles),
        _text(formula),
        _text(adduct),
        _number(collision_energy, COLLISION_ENERGY_DECIMALS),
        _text(instrument),
        _number(precursor_mz, PRECURSOR_MZ_DECIMALS),
    )
    return SEPARATOR.join(
        f"{name}={value}" for name, value in zip(FIELD_ORDER, values, strict=True)
    )


def format_row(row: Mapping[str, object]) -> str:
    """Render a dataset row; absent keys are treated as missing."""
    return format_molecular_input(*(row.get(key) for key in _ROW_KEYS))
