"""Fragment-formula candidates from a precursor formula alone.

A candidate is a ``(neutral subformula, ion state)`` pair whose theoretical m/z
falls inside the tolerance window of an observed peak.  No molecular graph,
bond list, or external fragment annotation is consulted anywhere in this
module -- the only inputs are the precursor's element counts and the peak's m/z.

Ion-state policy (deliberately fixed; it is not tuned against validation
results):

* positive adduct -> protonated, negative adduct -> deprotonated
* an ``Na`` in the adduct adds a sodiated channel, ``K`` a potassiated one
* a negative ``Cl`` adduct adds a chloride-retained channel
* ammonium and formate are *not* retained as fragment adducts
* no radical channels
* anything carrying more than one charge is refused rather than guessed at
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from metabo_sllm.chem.formula import (
    ELECTRON_MASS,
    EnumerationCapExceeded,
    SubformulaTable,
    element_mass,
    formula_to_string,
)

__all__ = [
    "BOUNDARY_EPS",
    "CHLORIDE_RETAINED",
    "Candidate",
    "DEFAULT_PPM",
    "DEPROTONATED",
    "HEAVY_SUBFORMULA_CAP",
    "IonChannel",
    "PEAK_CANDIDATE_CAP",
    "POTASSIATED",
    "PROTONATED",
    "SODIATED",
    "SpectrumEdges",
    "SpectrumMatch",
    "UnsupportedChargeError",
    "channels_for_adduct",
    "decode_key",
    "generate_candidates",
    "is_low_precision",
    "key_components",
    "mass_tolerance",
    "match_edges",
    "match_spectrum",
    "theoretical_mz",
    "tolerance_array",
]

DEFAULT_PPM = 10.0
HEAVY_SUBFORMULA_CAP = 4_000_000
PEAK_CANDIDATE_CAP = 100_000
# Only wide enough to absorb float64 round-off at the window edge.
BOUNDARY_EPS = 1e-9

_ELEMENT_TOKEN = re.compile(r"[A-Z][a-z]?")
_CHARGE_SUFFIX = re.compile(r"(\d*)\s*([+-])\s*$")


class UnsupportedChargeError(ValueError):
    """Raised for an adduct whose charge state this policy refuses to guess at."""


@dataclass(frozen=True, slots=True)
class IonChannel:
    """How a neutral fragment formula is turned into an observed m/z."""

    name: str
    charge: int
    mz_offset: float


_HYDROGEN_MASS = element_mass("H")
_PROTON_MASS = _HYDROGEN_MASS - ELECTRON_MASS

PROTONATED = IonChannel("protonated", 1, _PROTON_MASS)
DEPROTONATED = IonChannel("deprotonated", -1, -_PROTON_MASS)
SODIATED = IonChannel("sodiated", 1, element_mass("Na") - ELECTRON_MASS)
POTASSIATED = IonChannel("potassiated", 1, element_mass("K") - ELECTRON_MASS)
CHLORIDE_RETAINED = IonChannel("chloride_retained", -1, element_mass("Cl") + ELECTRON_MASS)


def channels_for_adduct(adduct: str) -> tuple[IonChannel, ...]:
    """Ion channels to search for a labelled adduct such as ``"[M+Na]+"``.

    Raises:
        UnsupportedChargeError: if the polarity cannot be read, or the adduct
            carries more than one charge.
    """
    text = (adduct or "").strip()
    if not text:
        raise UnsupportedChargeError("empty adduct")
    if text.endswith(("++", "--")):
        raise UnsupportedChargeError(f"adduct {adduct!r} carries more than one charge")
    match = _CHARGE_SUFFIX.search(text)
    if match is None:
        raise UnsupportedChargeError(f"cannot read polarity from adduct {adduct!r}")
    magnitude = int(match.group(1)) if match.group(1) else 1
    if magnitude != 1:
        raise UnsupportedChargeError(f"adduct {adduct!r} carries charge magnitude {magnitude}")

    positive = match.group(2) == "+"
    channels = [PROTONATED if positive else DEPROTONATED]
    tokens = set(_ELEMENT_TOKEN.findall(text))
    if "Na" in tokens:
        channels.append(SODIATED)
    if "K" in tokens:
        channels.append(POTASSIATED)
    if not positive and "Cl" in tokens:
        channels.append(CHLORIDE_RETAINED)
    return tuple(channels)


def is_low_precision(decimals: int) -> bool:
    """True when the reported m/z has too few decimals for a rounding window."""
    return decimals <= 1


def mass_tolerance(mz: float, decimals: int, ppm: float = DEFAULT_PPM) -> float:
    """Half-width of the mass window for one peak, in daltons.

    At two or more decimals the reported m/z has a rounding half-width of
    ``0.5 * 10**-decimals``, which dominates the ppm window at low m/z; below
    that the printed value is too coarse for a rounding term to mean anything,
    so the ppm window is used alone and the peak is flagged low precision.
    """
    ppm_window = mz * ppm * 1e-6
    if decimals >= 2:
        return max(ppm_window, 0.5 * 10.0 ** (-decimals))
    return ppm_window


def tolerance_array(
    mzs: np.ndarray, decimals: np.ndarray, ppm: float = DEFAULT_PPM
) -> np.ndarray:
    """Vectorised :func:`mass_tolerance`."""
    mzs = np.asarray(mzs, dtype=np.float64)
    decimals = np.asarray(decimals, dtype=np.int64)
    ppm_window = mzs * (ppm * 1e-6)
    rounding = 0.5 * np.power(10.0, -decimals.astype(np.float64))
    return np.where(decimals >= 2, np.maximum(ppm_window, rounding), ppm_window)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One matched ``(neutral formula, ion state)`` pair for a peak."""

    neutral_formula: str
    ion_state: str
    theoretical_mz: float
    error_da: float
    error_ppm: float


@dataclass(frozen=True, slots=True)
class SpectrumMatch:
    """Match result for one spectrum.

    Attributes:
        peak_candidate_counts: candidates found per peak, in peak order.
        candidate_keys: sorted, de-duplicated candidate keys across the whole
            spectrum; decode with :func:`decode_key`.
    """

    peak_candidate_counts: np.ndarray
    candidate_keys: np.ndarray

    @property
    def unique_candidates(self) -> int:
        return int(self.candidate_keys.shape[0])


def _key_strides(table: SubformulaTable) -> tuple[int, int]:
    hydrogen_stride = table.max_hydrogen + 1
    return hydrogen_stride, table.heavy_size * hydrogen_stride


def decode_key(
    table: SubformulaTable, channels: tuple[IonChannel, ...], key: int
) -> tuple[IonChannel, dict[str, int], float]:
    """Decode a candidate key into its channel, element counts and neutral mass."""
    hydrogen_stride, channel_stride = _key_strides(table)
    channel_index, rest = divmod(int(key), channel_stride)
    heavy_index, hydrogen = divmod(rest, hydrogen_stride)
    return (
        channels[channel_index],
        table.decode(heavy_index, hydrogen),
        table.mass(heavy_index, hydrogen),
    )


def _expand_ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Concatenate ``range(start, start + length)`` for every pair, vectorised."""
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    ends = np.cumsum(lengths)
    offsets = np.arange(total, dtype=np.int64) - np.repeat(ends - lengths, lengths)
    return np.repeat(starts, lengths) + offsets


@dataclass(frozen=True, slots=True)
class SpectrumEdges:
    """Every ``(peak, candidate)`` pair for one spectrum.

    Attributes:
        peak_candidate_counts: candidates found per peak, in peak order.
        edge_peak_index: peak index of each edge.
        edge_key: candidate key of each edge; decode with :func:`decode_key`.
    """

    peak_candidate_counts: np.ndarray
    edge_peak_index: np.ndarray
    edge_key: np.ndarray


def _match_blocks(
    table: SubformulaTable,
    mzs: np.ndarray,
    decimals: np.ndarray,
    channels: tuple[IonChannel, ...],
    ppm: float,
    peak_cap: int,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Per-channel heavy-index ranges for every ``(peak, hydrogen count)`` cell.

    Hydrogen is handled analytically: for a fixed hydrogen count the admissible
    heavy mass is one contiguous interval of :attr:`SubformulaTable.heavy_masses`,
    located with two binary searches.  The empty formula is never a candidate.

    Counts are computed before any candidate is materialised so the per-peak cap
    is checked on the true total, never on a truncated list.
    """
    n_peaks = int(mzs.shape[0])
    counts = np.zeros(n_peaks, dtype=np.int64)
    if n_peaks == 0 or not channels:
        return counts, []

    tolerances = tolerance_array(mzs, decimals, ppm)
    heavy = table.heavy_masses
    hydrogen_grid = np.arange(table.max_hydrogen + 1, dtype=np.float64) * _HYDROGEN_MASS

    blocks: list[tuple[np.ndarray, np.ndarray]] = []
    for channel in channels:
        neutral = mzs - channel.mz_offset
        low = (neutral - tolerances - BOUNDARY_EPS)[:, None] - hydrogen_grid[None, :]
        high = (neutral + tolerances + BOUNDARY_EPS)[:, None] - hydrogen_grid[None, :]
        start = np.searchsorted(heavy, low, side="left")
        stop = np.searchsorted(heavy, high, side="right")
        length = np.maximum(stop - start, 0)
        # heavy index 0 is the empty heavy formula; paired with zero hydrogen it
        # is the empty formula, which is excluded by definition.
        empty_hit = (start[:, 0] == 0) & (length[:, 0] > 0)
        start[empty_hit, 0] += 1
        length[empty_hit, 0] -= 1
        counts += length.sum(axis=1)
        blocks.append((start, length))

    over = int(counts.max())
    if over > peak_cap:
        worst = int(np.argmax(counts))
        raise EnumerationCapExceeded(
            f"{table.formula}: peak {worst} (m/z {mzs[worst]:.4f}) matches {over:,} candidates, "
            f"over the per-peak cap of {peak_cap:,}"
        )
    return counts, blocks


def _as_inputs(mzs: np.ndarray, decimals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mzs = np.asarray(mzs, dtype=np.float64)
    decimals = np.asarray(decimals, dtype=np.int64)
    if mzs.shape != decimals.shape:
        raise ValueError(f"mzs {mzs.shape} and decimals {decimals.shape} disagree")
    return mzs, decimals


def match_edges(
    table: SubformulaTable,
    mzs: np.ndarray,
    decimals: np.ndarray,
    channels: tuple[IonChannel, ...],
    *,
    ppm: float = DEFAULT_PPM,
    peak_cap: int = PEAK_CANDIDATE_CAP,
) -> SpectrumEdges:
    """Every peak-to-candidate edge, keeping the peak association.

    A candidate matching several peaks yields several edges; it is not
    duplicated by the caller.

    Raises:
        EnumerationCapExceeded: if any peak exceeds ``peak_cap``.
    """
    mzs, decimals = _as_inputs(mzs, decimals)
    counts, blocks = _match_blocks(table, mzs, decimals, channels, ppm, peak_cap)
    empty = np.empty(0, dtype=np.int64)
    if not blocks:
        return SpectrumEdges(counts, empty, empty)

    n_peaks = int(mzs.shape[0])
    hydrogen_stride, channel_stride = _key_strides(table)
    n_hydrogen = table.max_hydrogen + 1
    # Both match the C-order ravel of the (n_peaks, n_hydrogen) cell grid.
    cell_hydrogen = np.tile(np.arange(n_hydrogen, dtype=np.int64), n_peaks)
    cell_peak = np.repeat(np.arange(n_peaks, dtype=np.int64), n_hydrogen)

    peak_chunks: list[np.ndarray] = []
    key_chunks: list[np.ndarray] = []
    for channel_index, (start, length) in enumerate(blocks):
        flat_length = length.ravel()
        if not flat_length.any():
            continue
        heavy_indices = _expand_ranges(start.ravel(), flat_length)
        hydrogens = np.repeat(cell_hydrogen, flat_length)
        peak_chunks.append(np.repeat(cell_peak, flat_length))
        key_chunks.append(
            heavy_indices * hydrogen_stride + hydrogens + channel_index * channel_stride
        )

    if not key_chunks:
        return SpectrumEdges(counts, empty, empty)
    return SpectrumEdges(counts, np.concatenate(peak_chunks), np.concatenate(key_chunks))


def key_components(table: SubformulaTable, keys: np.ndarray) -> tuple[np.ndarray, ...]:
    """Split candidate keys into ``(channel index, heavy index, hydrogen count)``."""
    hydrogen_stride, channel_stride = _key_strides(table)
    keys = np.asarray(keys, dtype=np.int64)
    channel_index, rest = np.divmod(keys, channel_stride)
    heavy_index, hydrogen = np.divmod(rest, hydrogen_stride)
    return channel_index, heavy_index, hydrogen


def theoretical_mz(
    table: SubformulaTable, channels: tuple[IonChannel, ...], keys: np.ndarray
) -> np.ndarray:
    """Theoretical m/z of each candidate key, vectorised."""
    channel_index, heavy_index, hydrogen = key_components(table, keys)
    offsets = np.asarray([channel.mz_offset for channel in channels], dtype=np.float64)
    neutral = table.heavy_masses[heavy_index] + hydrogen * _HYDROGEN_MASS
    return neutral + offsets[channel_index]


def match_spectrum(
    table: SubformulaTable,
    mzs: np.ndarray,
    decimals: np.ndarray,
    channels: tuple[IonChannel, ...],
    *,
    ppm: float = DEFAULT_PPM,
    peak_cap: int = PEAK_CANDIDATE_CAP,
    collect_keys: bool = True,
) -> SpectrumMatch:
    """Count candidates per peak and list the spectrum's distinct candidates."""
    if not collect_keys:
        mzs, decimals = _as_inputs(mzs, decimals)
        counts, _ = _match_blocks(table, mzs, decimals, channels, ppm, peak_cap)
        return SpectrumMatch(counts, np.empty(0, dtype=np.int64))

    edges = match_edges(table, mzs, decimals, channels, ppm=ppm, peak_cap=peak_cap)
    return SpectrumMatch(edges.peak_candidate_counts, np.unique(edges.edge_key))


def generate_candidates(
    table: SubformulaTable,
    mz: float,
    decimals: int,
    channels: tuple[IonChannel, ...],
    *,
    ppm: float = DEFAULT_PPM,
    peak_cap: int = PEAK_CANDIDATE_CAP,
) -> list[Candidate]:
    """Every candidate for a single peak, with theoretical m/z and mass errors.

    Reference implementation over the same matching core the audit uses, so the
    two cannot drift apart.
    """
    match = match_spectrum(
        table,
        np.asarray([mz], dtype=np.float64),
        np.asarray([decimals], dtype=np.int64),
        channels,
        ppm=ppm,
        peak_cap=peak_cap,
    )
    candidates: list[Candidate] = []
    for key in match.candidate_keys.tolist():
        channel, counts, neutral_mass = decode_key(table, channels, key)
        theoretical = neutral_mass + channel.mz_offset
        error_da = float(mz) - theoretical
        candidates.append(
            Candidate(
                neutral_formula=formula_to_string(counts),
                ion_state=channel.name,
                theoretical_mz=theoretical,
                error_da=error_da,
                error_ppm=error_da / theoretical * 1e6 if theoretical else float("nan"),
            )
        )
    candidates.sort(key=lambda c: (c.ion_state, c.theoretical_mz))
    return candidates
