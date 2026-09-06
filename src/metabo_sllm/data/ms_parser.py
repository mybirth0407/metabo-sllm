"""Contract-first parser for NIST23 ``.ms`` records.

A ``.ms`` record is a text blob made of a metadata header followed by one or
more ``>collision <energy>`` blocks; every non-empty line inside a block is a
``"<m/z> <intensity>"`` peak.

The parser deliberately does **not** sort, merge, filter or normalise peaks.
Materialising the raw spectrum has to stay reversible; binning and
normalisation belong to a separate, versioned downstream stage.  Anything that
violates the format is reported as :class:`MsParseError` rather than silently
repaired, so corruption is caught at build time instead of leaking into a
training set.

Alongside each m/z the parser records how many decimal places the *source text*
carried (``mz_decimal_places``).  Converting to float loses that, yet it is the
only evidence of how finely the instrument reported the peak, and the mass
tolerance used downstream depends on it.

Parsing happens on ``bytes``: only the collision-energy token and the peak
tokens are decoded (as ASCII), so a header carrying non-UTF-8 compound names
can never break a record.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

import numpy as np

__all__ = ["CollisionBlock", "MsParseError", "parse_ms"]

_COLLISION_TAG = b">collision"
_MAX_ENERGIES_IN_MESSAGE = 12
# mz_decimal_places is stored as int8 downstream.
_MAX_DECIMAL_PLACES = 127


class MsParseError(ValueError):
    """Raised when a ``.ms`` record violates the parser contract."""


@dataclass(frozen=True, slots=True)
class CollisionBlock:
    """One ``>collision`` block of a ``.ms`` record.

    ``mzs``, ``intensities`` and ``mz_decimal_places`` are read-only arrays of
    equal length, in the exact order the peaks appear in the file.
    """

    collision_index: int
    collision_energy: float
    collision_energy_raw: str
    mzs: np.ndarray
    intensities: np.ndarray
    mz_decimal_places: np.ndarray

    def __len__(self) -> int:
        return int(self.mzs.shape[0])


def parse_ms(
    content: bytes,
    *,
    expected_energies: Sequence[object] | None = None,
    source: str = "<bytes>",
) -> list[CollisionBlock]:
    """Parse one ``.ms`` record into its collision blocks.

    Args:
        content: raw record bytes.
        expected_energies: if given, the collision-energy tokens found in the
            record must equal ``[str(x) for x in expected_energies]`` in both
            count and order.  Used to cross-check a record against ``labels.tsv``.
        source: label used in error messages (typically the HDF5 key).

    Returns:
        Blocks in file order, ``collision_index`` numbered from 0.

    Raises:
        MsParseError: on any contract violation.
    """
    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise MsParseError(f"{source}: expected bytes, got {type(content).__name__}")

    text = bytes(content).replace(b"\r\n", b"\n").replace(b"\r", b"\n")

    blocks: list[CollisionBlock] = []
    started = False
    energy_raw = ""
    energy = 0.0
    mzs: list[float] = []
    intensities: list[float] = []
    decimals: list[int] = []

    for lineno, line in enumerate(text.split(b"\n"), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        if stripped[:1] in (b">", b"#"):
            fields = stripped.split()
            if fields[0] != _COLLISION_TAG:
                if started:
                    raise _error(source, lineno, stripped, "metadata line inside a collision block")
                continue  # header line, before the first block
            if len(fields) != 2:
                raise _error(source, lineno, stripped, "expected '>collision <energy>'")
            token = _decode(fields[1], source, lineno, stripped)
            value = _to_float(token, source, lineno, stripped, "collision energy")
            if not math.isfinite(value):
                raise _error(source, lineno, stripped, "collision energy is not finite")
            if started:
                blocks.append(_freeze(len(blocks), energy, energy_raw, mzs, intensities, decimals))
            started, energy_raw, energy = True, token, value
            mzs, intensities, decimals = [], [], []
            continue

        if not started:
            raise _error(source, lineno, stripped, "peak line before the first '>collision' block")

        fields = stripped.split()
        if len(fields) != 2:
            raise _error(
                source,
                lineno,
                stripped,
                f"expected 2 whitespace-separated values, got {len(fields)}",
            )
        mz_token = _decode(fields[0], source, lineno, stripped)
        mz = _to_float(mz_token, source, lineno, stripped, "m/z")
        intensity = _to_float(
            _decode(fields[1], source, lineno, stripped), source, lineno, stripped, "intensity"
        )
        if not math.isfinite(mz):
            raise _error(source, lineno, stripped, "m/z is not finite")
        if not math.isfinite(intensity):
            raise _error(source, lineno, stripped, "intensity is not finite")
        if mz <= 0.0:
            raise _error(source, lineno, stripped, f"m/z must be positive, got {mz!r}")
        if intensity < 0.0:
            raise _error(
                source, lineno, stripped, f"intensity must be non-negative, got {intensity!r}"
            )
        mzs.append(mz)
        intensities.append(intensity)
        decimals.append(_decimal_places(mz_token, source, lineno, stripped))

    if started:
        blocks.append(_freeze(len(blocks), energy, energy_raw, mzs, intensities, decimals))

    if expected_energies is not None:
        found = [block.collision_energy_raw for block in blocks]
        wanted = [str(item) for item in expected_energies]
        if found != wanted:
            raise MsParseError(
                f"{source}: collision-energy mismatch - "
                f"expected {len(wanted)} {_brief(wanted)}, found {len(found)} {_brief(found)}"
            )

    return blocks


def _decimal_places(token: str, source: str, lineno: int, line: bytes) -> int:
    """Decimal places carried by the source text, from the Decimal exponent.

    ``"100"`` -> 0, ``"100.1"`` -> 1, ``"100.10"`` -> 2, ``"100.1000"`` -> 4.
    """
    try:
        exponent = Decimal(token).as_tuple().exponent
    except InvalidOperation:
        raise _error(source, lineno, line, f"m/z is not a decimal literal: {token!r}") from None
    if not isinstance(exponent, int):  # NaN/Infinity carry a string exponent
        raise _error(source, lineno, line, f"m/z has no decimal exponent: {token!r}")
    places = max(0, -exponent)
    if places > _MAX_DECIMAL_PLACES:
        raise _error(
            source, lineno, line, f"m/z has {places} decimal places (max {_MAX_DECIMAL_PLACES})"
        )
    return places


def _freeze(
    index: int,
    energy: float,
    energy_raw: str,
    mzs: list[float],
    intensities: list[float],
    decimals: list[int],
) -> CollisionBlock:
    mz_array = np.asarray(mzs, dtype=np.float64)
    intensity_array = np.asarray(intensities, dtype=np.float32)
    decimal_array = np.asarray(decimals, dtype=np.int8)
    for array in (mz_array, intensity_array, decimal_array):
        array.setflags(write=False)
    return CollisionBlock(
        collision_index=index,
        collision_energy=energy,
        collision_energy_raw=energy_raw,
        mzs=mz_array,
        intensities=intensity_array,
        mz_decimal_places=decimal_array,
    )


def _decode(token: bytes, source: str, lineno: int, line: bytes) -> str:
    try:
        return token.decode("ascii")
    except UnicodeDecodeError:
        raise _error(source, lineno, line, "non-ASCII token") from None


def _to_float(token: str, source: str, lineno: int, line: bytes, what: str) -> float:
    try:
        return float(token)
    except ValueError:
        raise _error(source, lineno, line, f"{what} is not a number: {token!r}") from None


def _error(source: str, lineno: int, line: bytes, reason: str) -> MsParseError:
    shown = line[:120].decode("ascii", "replace")
    return MsParseError(f"{source}:{lineno}: {reason} - line {shown!r}")


def _brief(values: list[str]) -> str:
    if len(values) <= _MAX_ENERGIES_IN_MESSAGE:
        return repr(values)
    head = values[:_MAX_ENERGIES_IN_MESSAGE]
    return f"{head!r}[+{len(values) - _MAX_ENERGIES_IN_MESSAGE} more]"
