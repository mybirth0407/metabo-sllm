"""Did the predicted fragments name the right formulas?

This is a diagnostic, not the validation metric: it reads the candidate bags,
which the model never sees, and it runs only after predictions exist.  It
answers a different question than cosine does -- cosine asks whether the
spectrum looks right, this asks whether the model got there for the right
reasons.

One prediction may sit in several targets' bags, and one target's bag may
contain several predictions.  Counting every compatible pair would inflate the
hit rate, so predictions and targets are matched one-to-one by maximum
bipartite matching first.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ["FragmentDiagnostic", "match_predictions_to_targets", "score_fragments", "summarise_fragments"]


@dataclass
class FragmentDiagnostic:
    spectrum_uid: str
    targets: int
    predictions: int
    matched: int
    unique_targets: int
    unique_matched: int
    ambiguous_targets: int
    ambiguous_matched: int
    target_intensity: float
    matched_intensity: float
    active_slots: int
    duplicate_identity_slots: int


def match_predictions_to_targets(
    predicted: Sequence[tuple[str, str]], bags: Sequence[set[tuple[str, str]]]
) -> list[tuple[int, int]]:
    """One-to-one matching of predictions to targets they can explain.

    Maximises the number of matched targets; a prediction is used at most once.
    """
    if not predicted or not bags:
        return []
    cost = np.ones((len(predicted), len(bags)), dtype=np.float64)
    for row, identity in enumerate(predicted):
        for column, bag in enumerate(bags):
            if identity in bag:
                cost[row, column] = 0.0
    if not (cost == 0.0).any():
        return []
    rows, columns = linear_sum_assignment(cost)
    return [
        (int(row), int(column))
        for row, column in zip(rows, columns, strict=True)
        if cost[row, column] == 0.0
    ]


def score_fragments(
    spectrum_uid: str,
    predicted: Sequence[tuple[str, str]],
    bags: Sequence[set[tuple[str, str]]],
    target_intensities: Sequence[float],
    *,
    active_slots: int = 0,
    duplicate_identity_slots: int = 0,
) -> FragmentDiagnostic:
    """Hit rates for one spectrum, after one-to-one matching."""
    matches = match_predictions_to_targets(predicted, bags)
    matched_targets = {column for _, column in matches}
    sizes = [len(bag) for bag in bags]
    intensities = np.asarray(target_intensities, dtype=np.float64)

    return FragmentDiagnostic(
        spectrum_uid=spectrum_uid,
        targets=len(bags),
        predictions=len(predicted),
        matched=len(matches),
        unique_targets=sum(1 for size in sizes if size == 1),
        unique_matched=sum(1 for column in matched_targets if sizes[column] == 1),
        ambiguous_targets=sum(1 for size in sizes if size >= 2),
        ambiguous_matched=sum(1 for column in matched_targets if sizes[column] >= 2),
        target_intensity=float(intensities.sum()) if intensities.size else 0.0,
        matched_intensity=float(intensities[list(matched_targets)].sum())
        if matched_targets and intensities.size
        else 0.0,
        active_slots=active_slots,
        duplicate_identity_slots=duplicate_identity_slots,
    )


def summarise_fragments(diagnostics: list[FragmentDiagnostic]) -> dict:
    """Fold-level rates, pooled as sums rather than as an average of ratios."""
    if not diagnostics:
        return {"spectra": 0}

    def total(name: str) -> float:
        return float(sum(getattr(d, name) for d in diagnostics))

    targets = total("targets")
    predictions = total("predictions")
    matched = total("matched")
    unique = total("unique_targets")
    ambiguous = total("ambiguous_targets")
    intensity = total("target_intensity")
    slots = np.asarray([d.active_slots for d in diagnostics], dtype=np.float64)
    duplicates = total("duplicate_identity_slots")

    return {
        "spectra": len(diagnostics),
        "targets": int(targets),
        "predictions": int(predictions),
        "fragment_bag_hit": matched / targets if targets else None,
        "unique_target_hit": total("unique_matched") / unique if unique else None,
        "ambiguous_bag_hit": total("ambiguous_matched") / ambiguous if ambiguous else None,
        "intensity_weighted_bag_hit": (
            total("matched_intensity") / intensity if intensity else None
        ),
        "active_slot_precision": matched / predictions if predictions else None,
        "active_slots_mean": float(slots.mean()),
        "duplicate_identity_slots": int(duplicates),
        "duplicate_active_formula_rate": (
            duplicates / (duplicates + predictions) if (duplicates + predictions) else 0.0
        ),
    }
