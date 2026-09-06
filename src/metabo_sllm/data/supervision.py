"""Which peaks carry fragment-identity supervision, and in what order.

A peak is supervised only when the candidate generator found at least one
formula for it *and* its m/z was printed with enough decimals for the mass
window to mean anything.  Peaks that fail either test stay in the raw spectrum
-- they are still valid full-spectrum targets -- they just cannot teach a
model which formula produced them.

The model has a fixed number of fragment slots, so training will match against
at most :data:`FIXED_SLOT_COUNT` supervised peaks.  That selection is defined
here rather than at training time so the stored artifact pins it down: highest
intensity first, ties broken by lower m/z and then by original peak order.
Nothing is deleted -- the ordering is recorded, and the caller decides how many
slots to consume.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "FIXED_SLOT_COUNT",
    "MIN_SUPERVISION_DECIMALS",
    "supervision_mask",
    "supervision_rank",
    "top_slot_mask",
]

FIXED_SLOT_COUNT = 64
MIN_SUPERVISION_DECIMALS = 2


def supervision_mask(candidate_counts: np.ndarray, mz_decimal_places: np.ndarray) -> np.ndarray:
    """Peaks eligible for fragment-identity supervision.

    Requires at least one formula candidate and at least
    :data:`MIN_SUPERVISION_DECIMALS` decimal places in the reported m/z.
    """
    counts = np.asarray(candidate_counts)
    decimals = np.asarray(mz_decimal_places)
    if counts.shape != decimals.shape:
        raise ValueError(f"counts {counts.shape} and decimals {decimals.shape} disagree")
    return (counts > 0) & (decimals >= MIN_SUPERVISION_DECIMALS)


def supervision_rank(
    mask: np.ndarray, mzs: np.ndarray, intensities: np.ndarray
) -> np.ndarray:
    """Rank of each supervised peak; ``-1`` for peaks that are not supervised.

    Ordering is intensity descending, then m/z ascending, then original peak
    index ascending -- fully determined by the stored spectrum, so a training
    loader reproduces it exactly.
    """
    mask = np.asarray(mask, dtype=bool)
    mzs = np.asarray(mzs, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)
    if not (mask.shape == mzs.shape == intensities.shape):
        raise ValueError("mask, mzs and intensities must have the same shape")

    rank = np.full(mask.shape[0], -1, dtype=np.int32)
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        return rank
    # np.lexsort takes the primary key last.
    order = np.lexsort((selected, mzs[selected], -intensities[selected]))
    rank[selected[order]] = np.arange(selected.size, dtype=np.int32)
    return rank


def top_slot_mask(rank: np.ndarray, slots: int = FIXED_SLOT_COUNT) -> np.ndarray:
    """Peaks that would occupy a fragment slot: supervised and ranked below ``slots``."""
    rank = np.asarray(rank)
    return (rank >= 0) & (rank < slots)
