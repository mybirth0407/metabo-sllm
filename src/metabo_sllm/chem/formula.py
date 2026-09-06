"""Molecular formulas and exhaustive subformula enumeration.

Everything here works on *element counts only*.  No molecular graph is built,
no bonds are consulted, and no external fragment annotation (MAGMa or
otherwise) is read.  A candidate fragment formula is any non-empty subformula
of the precursor: ``0 <= F[e] <= precursor[e]`` for every element, hydrogen
included.  There is no atom-subset restriction and no +-6H rule -- the
hydrogen count is part of the search space, so a separate hydrogen shift would
be redundant.

Enumeration is split into a *heavy* part (everything except H) and hydrogen.
The heavy subformulas are enumerated once per precursor formula and their
masses sorted; hydrogen is then handled analytically, because for a target
mass and a fixed hydrogen count the required heavy mass is a single interval
that a binary search resolves.  That keeps the cost proportional to
``n_hydrogen * log(n_heavy)`` per peak instead of the full product.

RDKit is used only as a periodic table for monoisotopic masses.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from math import prod

import numpy as np
from rdkit.Chem import GetPeriodicTable

__all__ = [
    "ELECTRON_MASS",
    "EnumerationCapExceeded",
    "FormulaError",
    "SubformulaTable",
    "element_mass",
    "formula_to_string",
    "parse_formula",
]

# Element symbol followed by an optional count, e.g. "C", "C6", "Cl2".
_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")

# CODATA electron rest mass in u.  Ion m/z has to account for the missing or
# extra electron, which is well above a 10 ppm window at low m/z.
ELECTRON_MASS = 0.000548579909065

_PERIODIC_TABLE = GetPeriodicTable()
_MASS_CACHE: dict[str, float] = {}


class FormulaError(ValueError):
    """Raised when a formula string cannot be parsed or uses an unknown element."""


class EnumerationCapExceeded(RuntimeError):
    """Raised when a subformula enumeration would exceed its configured cap.

    Never caught internally to truncate a search: a formula too large to
    enumerate is reported, not silently reduced.
    """


def element_mass(symbol: str) -> float:
    """Monoisotopic mass of ``symbol`` (mass of its most common isotope)."""
    cached = _MASS_CACHE.get(symbol)
    if cached is not None:
        return cached
    try:
        mass = float(_PERIODIC_TABLE.GetMostCommonIsotopeMass(symbol))
    except (RuntimeError, ValueError) as exc:
        raise FormulaError(f"unknown element {symbol!r}") from exc
    if mass <= 0.0:
        raise FormulaError(f"element {symbol!r} has no monoisotopic mass")
    _MASS_CACHE[symbol] = mass
    return mass


def parse_formula(text: str) -> dict[str, int]:
    """Parse ``"C6H12O6"`` into ``{"C": 6, "H": 12, "O": 6}``.

    Raises:
        FormulaError: on empty input, trailing junk, an unknown element, or a
            zero/negative count.
    """
    if not text:
        raise FormulaError("empty formula")
    counts: dict[str, int] = {}
    position = 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        if match is None or match.start() != position or not match.group(1):
            raise FormulaError(f"cannot parse formula {text!r} at offset {position}")
        symbol, digits = match.group(1), match.group(2)
        count = int(digits) if digits else 1
        if count <= 0:
            raise FormulaError(f"formula {text!r} gives element {symbol!r} a count of {count}")
        element_mass(symbol)  # validates the symbol
        counts[symbol] = counts.get(symbol, 0) + count
        position = match.end()
    if not counts:
        raise FormulaError(f"formula {text!r} has no elements")
    return counts


def formula_to_string(counts: dict[str, int]) -> str:
    """Render element counts in Hill notation (C, then H, then alphabetical)."""
    present = {symbol: n for symbol, n in counts.items() if n > 0}
    if not present:
        return ""
    ordered: list[str] = []
    if "C" in present:
        ordered.append("C")
        if "H" in present:
            ordered.append("H")
        ordered.extend(sorted(s for s in present if s not in ("C", "H")))
    else:
        ordered.extend(sorted(present))
    parts = []
    for symbol in ordered:
        n = present[symbol]
        parts.append(symbol if n == 1 else f"{symbol}{n}")
    return "".join(parts)


class SubformulaTable:
    """All heavy-element subformulas of one precursor formula, sorted by mass.

    The hydrogen dimension is deliberately *not* materialised; callers pair a
    hydrogen count with a mass interval over :attr:`heavy_masses`.

    Attributes:
        heavy_masses: ascending monoisotopic masses of every heavy subformula,
            including the empty one at index 0 (mass 0.0).
        max_hydrogen: hydrogen count of the precursor; the search range is
            ``0..max_hydrogen`` inclusive.
    """

    __slots__ = (
        "precursor",
        "formula",
        "heavy_symbols",
        "heavy_counts",
        "heavy_size",
        "heavy_masses",
        "max_hydrogen",
        "_order",
        "_radix",
        "_string_cache",
    )

    def __init__(self, precursor: dict[str, int], *, heavy_cap: int) -> None:
        if not precursor:
            raise FormulaError("empty precursor formula")
        self.precursor = dict(precursor)
        self.formula = formula_to_string(precursor)
        self.max_hydrogen = int(precursor.get("H", 0))

        heavy = OrderedDict(
            (symbol, int(count))
            for symbol, count in sorted(precursor.items())
            if symbol != "H" and count > 0
        )
        self.heavy_symbols = tuple(heavy)
        self.heavy_counts = tuple(heavy.values())
        self.heavy_size = prod(count + 1 for count in self.heavy_counts) if heavy else 1
        if self.heavy_size > heavy_cap:
            raise EnumerationCapExceeded(
                f"{self.formula}: {self.heavy_size:,} heavy subformulas exceed the cap "
                f"of {heavy_cap:,}"
            )

        # Mixed-radix expansion; the last element varies fastest, which
        # _decode_heavy relies on.
        masses = np.zeros(1, dtype=np.float64)
        for symbol, count in heavy.items():
            steps = np.arange(count + 1, dtype=np.float64) * element_mass(symbol)
            masses = (masses[:, None] + steps[None, :]).ravel()
        order = np.argsort(masses, kind="stable")
        self.heavy_masses = masses[order]
        self.heavy_masses.setflags(write=False)
        self._order = order.astype(np.int64, copy=False)
        self._radix = tuple(count + 1 for count in self.heavy_counts)
        # Collision-energy siblings of one molecule hit the same subformulas
        # repeatedly, so rendering each formula string once pays for itself.
        self._string_cache: dict[tuple[int, int], str] = {}

    @property
    def total_size(self) -> int:
        """Number of subformulas including hydrogen, empty formula included."""
        return self.heavy_size * (self.max_hydrogen + 1)

    def decode(self, heavy_index: int, hydrogen: int) -> dict[str, int]:
        """Element counts for a sorted heavy index paired with a hydrogen count."""
        counts = self._decode_heavy(int(heavy_index))
        if hydrogen:
            counts["H"] = int(hydrogen)
        return counts

    def _decode_heavy(self, heavy_index: int) -> dict[str, int]:
        raw = int(self._order[heavy_index])
        counts: dict[str, int] = {}
        for symbol, radix in zip(reversed(self.heavy_symbols), reversed(self._radix), strict=True):
            raw, remainder = divmod(raw, radix)
            if remainder:
                counts[symbol] = remainder
        return dict(reversed(list(counts.items())))

    def formula_string(self, heavy_index: int, hydrogen: int) -> str:
        """Hill-notation formula for ``(heavy_index, hydrogen)``, memoised."""
        cache_key = (int(heavy_index), int(hydrogen))
        cached = self._string_cache.get(cache_key)
        if cached is None:
            cached = formula_to_string(self.decode(*cache_key))
            self._string_cache[cache_key] = cached
        return cached

    def mass(self, heavy_index: int, hydrogen: int) -> float:
        """Monoisotopic mass of the subformula at ``(heavy_index, hydrogen)``."""
        return float(self.heavy_masses[heavy_index]) + hydrogen * element_mass("H")
