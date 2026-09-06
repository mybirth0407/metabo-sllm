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

Parsing happens on ``bytes``: only the collision-energy token and the peak
tokens are decoded (as ASCII), so a header carrying non-UTF-8 compound names
can never break a record.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["CollisionBlock", "MsParseError", "parse_ms"]

_COLLISION_TAG = b">collision"
_MAX_ENERGIES_IN_MESSAGE = 12


class MsParseError(ValueError):
    """Raised when a ``.ms`` record violates the parser contract."""


@dataclass(frozen=True, slots=True)
class CollisionBlock:
    """One ``>collision`` block of a ``.ms`` record.

    ``mzs`` and ``intensities`` are read-only arrays of equal length, in the
    exact order the peaks appear in the file.
    """

    collision_index: int
    collision_energy: float
    collision_energy_raw: str
    mzs: np.ndarray
    intensities: np.ndarray

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
                blocks.append(_freeze(len(blocks), energy, energy_raw, mzs, intensities))
            started, energy_raw, energy = True, token, value
            mzs, intensities = [], []
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
        mz = _to_float(_decode(fields[0], source, lineno, stripped), source, lineno, stripped, "m/z")
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

    if started:
        blocks.append(_freeze(len(blocks), energy, energy_raw, mzs, intensities))

    if expected_energies is not None:
        found = [block.collision_energy_raw for block in blocks]
        wanted = [str(item) for item in expected_energies]
        if found != wanted:
            raise MsParseError(
                f"{source}: collision-energy mismatch - "
                f"expected {len(wanted)} {_brief(wanted)}, found {len(found)} {_brief(found)}"
            )

    return blocks


def _freeze(
    index: int, energy: float, energy_raw: str, mzs: list[float], intensities: list[float]
) -> CollisionBlock:
    mz_array = np.asarray(mzs, dtype=np.float64)
    intensity_array = np.asarray(intensities, dtype=np.float32)
    mz_array.setflags(write=False)
    intensity_array.setflags(write=False)
    return CollisionBlock(
        collision_index=index,
        collision_energy=energy,
        collision_energy_raw=energy_raw,
        mzs=mz_array,
        intensities=intensity_array,
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
