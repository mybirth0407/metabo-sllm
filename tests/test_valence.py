"""The two hard valence bounds, and only those.

They must reject formulas no neutral molecule can have -- a negative RDBE, or
more monovalent atoms than the heavy atoms can carry -- and nothing else: a
hydrogen-poor formula with a large RDBE is unlikely, but it is not impossible,
and the filter is not allowed to have opinions.
"""

from __future__ import annotations

import pytest

from metabo_sllm.chem.formula import hydrogen_excess, is_valence_plausible, parse_formula, rdbe


@pytest.mark.parametrize(
    ("formula", "expected"),
    [("C6H6", 4.0), ("CH4", 0.0), ("C2H6O", 0.0), ("C5H5N", 4.0), ("CH3Cl", 0.0), ("C2H10", -2.0)],
)
def test_rdbe(formula, expected):
    assert rdbe(parse_formula(formula)) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("formula", "plausible"),
    [
        ("CH4", True),
        ("C2H6O", True),
        ("C22H6O", True),  # hydrogen-poor, RDBE 20: implausible-looking, not impossible
        ("CH5", False),  # one hydrogen over saturation
        ("C2H10", False),  # negative RDBE
        ("CHCl5", False),  # halogens count as monovalent
        ("H2", True),  # RDBE 0, at saturation
        ("H3", False),  # RDBE -1/2
    ],
)
def test_hard_bounds(formula, plausible):
    assert is_valence_plausible(parse_formula(formula)) is plausible


def test_hydrogen_excess_is_zero_at_saturation():
    assert hydrogen_excess(parse_formula("C3H8")) == 0
    assert hydrogen_excess(parse_formula("C3H9")) == 1
    assert hydrogen_excess(parse_formula("CH5N")) == 0  # methylamine: nitrogen carries one more
