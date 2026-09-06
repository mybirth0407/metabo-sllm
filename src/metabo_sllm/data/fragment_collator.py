"""Turn stored spectra into the tensors the 64-slot model consumes.

The artifact keeps everything; this is where the fixed slot budget bites.  At
most :data:`FIXED_SLOT_COUNT` supervised peaks become identity targets, chosen
by the ordering recorded in the artifact (intensity desc, m/z asc, peak index
asc).  Only the candidates attached to those targets are tensorised -- a
spectrum with thirty thousand candidates contributes only the bags of its
selected peaks -- but no bag is ever cut: a target keeps every candidate it
has.

Peaks that are not identity targets (no candidate, or a low-precision m/z) are
still carried in full as the full-spectrum target.

Intensities are transformed once, here: ``t = sqrt(y)`` normalised by the
2-norm over the *whole* raw spectrum, not over the selected peaks, so the
target vector keeps unit norm and the slots have to account for the peaks they
did not get.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from metabo_sllm.chem.candidates import (
    ION_STATE_TO_ID,
    ION_STATE_VOCABULARY,
    admissible_ion_state_ids,
)
from metabo_sllm.chem.formula import atomic_number, parse_formula
from metabo_sllm.data.supervision import FIXED_SLOT_COUNT
from metabo_sllm.model.input_formatter import format_row

__all__ = ["FragmentCollator", "INTENSITY_EPS", "select_targets", "transform_intensities"]

INTENSITY_EPS = 1e-12


def select_targets(supervision_rank: np.ndarray, slots: int = FIXED_SLOT_COUNT) -> np.ndarray:
    """Peak indices of the identity targets, ordered by supervision rank."""
    rank = np.asarray(supervision_rank)
    selected = np.flatnonzero((rank >= 0) & (rank < slots))
    return selected[np.argsort(rank[selected], kind="stable")]


def transform_intensities(intensities: np.ndarray) -> np.ndarray:
    """``sqrt(y)`` normalised by the 2-norm taken over every raw peak."""
    values = np.asarray(intensities, dtype=np.float64)
    root = np.sqrt(np.clip(values, 0.0, None))
    norm = np.sqrt(np.square(root).sum())
    return root / (norm + INTENSITY_EPS)


@dataclass(frozen=True)
class _RowTensors:
    text: str
    element_numbers: np.ndarray
    element_counts: np.ndarray
    target_peaks: np.ndarray
    target_intensities: np.ndarray
    candidate_counts: np.ndarray  # [C, R]
    candidate_ions: np.ndarray  # [C]
    candidate_targets: np.ndarray  # [C]
    admissible_ions: np.ndarray  # [n_ion] bool
    full_mzs: np.ndarray
    full_target: np.ndarray
    full_raw: np.ndarray


class FragmentCollator:
    """Collate artifact rows into a padded batch of tensors."""

    def __init__(
        self,
        tokenizer,
        *,
        max_text_length: int = 512,
        slots: int = FIXED_SLOT_COUNT,
        max_count: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.slots = slots
        self.max_count = max_count
        self.num_ion_states = len(ION_STATE_VOCABULARY)

    # ------------------------------------------------------------------ row

    def _prepare(self, row: Mapping[str, object]) -> _RowTensors:
        precursor = parse_formula(str(row["formula"]))
        symbols = sorted(precursor, key=atomic_number)
        element_numbers = np.asarray([atomic_number(s) for s in symbols], dtype=np.int64)
        element_counts = np.asarray([precursor[s] for s in symbols], dtype=np.int64)
        if element_counts.size and int(element_counts.max()) > self.max_count:
            raise ValueError(
                f"{row['spectrum_uid']}: precursor formula {row['formula']!r} has an element "
                f"count of {int(element_counts.max())}, above the count vocabulary "
                f"limit of {self.max_count}"
            )
        position = {symbol: index for index, symbol in enumerate(symbols)}

        intensities = np.asarray(row["intensities"], dtype=np.float64)
        full_target = transform_intensities(intensities)
        target_peaks = select_targets(np.asarray(row["supervision_rank"]), self.slots)

        peak_to_target = np.full(intensities.shape[0], -1, dtype=np.int64)
        peak_to_target[target_peaks] = np.arange(target_peaks.size, dtype=np.int64)

        edge_peak = np.asarray(row["edge_peak_index"], dtype=np.int64)
        edge_candidate = np.asarray(row["edge_candidate_index"], dtype=np.int64)
        keep = peak_to_target[edge_peak] >= 0 if edge_peak.size else np.zeros(0, dtype=bool)
        kept_candidates = edge_candidate[keep]
        kept_targets = peak_to_target[edge_peak[keep]]

        formulas: Sequence[str] = row["candidate_neutral_formula"]
        ion_states: Sequence[str] = row["candidate_ion_state"]
        cache: dict[int, np.ndarray] = {}
        candidate_counts = np.zeros((kept_candidates.size, element_numbers.size), dtype=np.int64)
        candidate_ions = np.zeros(kept_candidates.size, dtype=np.int64)
        for slot, candidate in enumerate(kept_candidates.tolist()):
            counts = cache.get(candidate)
            if counts is None:
                counts = np.zeros(element_numbers.size, dtype=np.int64)
                for symbol, value in parse_formula(formulas[candidate]).items():
                    counts[position[symbol]] = value
                cache[candidate] = counts
            candidate_counts[slot] = counts
            candidate_ions[slot] = ION_STATE_TO_ID[ion_states[candidate]]

        admissible = np.zeros(self.num_ion_states, dtype=bool)
        admissible[list(admissible_ion_state_ids(str(row["adduct"])))] = True

        return _RowTensors(
            text=format_row(row),
            element_numbers=element_numbers,
            element_counts=element_counts,
            target_peaks=target_peaks,
            target_intensities=full_target[target_peaks],
            candidate_counts=candidate_counts,
            candidate_ions=candidate_ions,
            candidate_targets=kept_targets,
            admissible_ions=admissible,
            full_mzs=np.asarray(row["mzs"], dtype=np.float64),
            full_target=full_target,
            full_raw=intensities,
        )

    # ---------------------------------------------------------------- batch

    def __call__(self, rows: Sequence[Mapping[str, object]]) -> dict:
        prepared = [self._prepare(row) for row in rows]
        batch = len(prepared)
        n_elements = max((p.element_numbers.size for p in prepared), default=1) or 1
        n_candidates = max((p.candidate_ions.size for p in prepared), default=1) or 1
        n_peaks = max((p.full_mzs.size for p in prepared), default=1) or 1

        encoded = self.tokenizer(
            [p.text for p in prepared],
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )

        element_ids = torch.zeros(batch, n_elements, dtype=torch.long)
        element_counts = torch.zeros(batch, n_elements, dtype=torch.long)
        element_mask = torch.zeros(batch, n_elements, dtype=torch.bool)
        target_peak_indices = torch.zeros(batch, self.slots, dtype=torch.long)
        target_peak_mask = torch.zeros(batch, self.slots, dtype=torch.bool)
        target_intensities = torch.zeros(batch, self.slots, dtype=torch.float32)
        candidate_formula_counts = torch.zeros(batch, n_candidates, n_elements, dtype=torch.long)
        candidate_ion_states = torch.zeros(batch, n_candidates, dtype=torch.long)
        candidate_to_peak = torch.zeros(batch, n_candidates, dtype=torch.long)
        candidate_mask = torch.zeros(batch, n_candidates, dtype=torch.bool)
        admissible_ion_mask = torch.zeros(batch, self.num_ion_states, dtype=torch.bool)
        full_peak_mzs = torch.zeros(batch, n_peaks, dtype=torch.float32)
        full_peak_intensities = torch.zeros(batch, n_peaks, dtype=torch.float32)
        full_peak_intensities_raw = torch.zeros(batch, n_peaks, dtype=torch.float32)
        full_peak_mask = torch.zeros(batch, n_peaks, dtype=torch.bool)

        for index, item in enumerate(prepared):
            n_e = item.element_numbers.size
            element_ids[index, :n_e] = torch.from_numpy(item.element_numbers)
            element_counts[index, :n_e] = torch.from_numpy(item.element_counts)
            element_mask[index, :n_e] = True

            n_t = item.target_peaks.size
            target_peak_indices[index, :n_t] = torch.from_numpy(item.target_peaks)
            target_peak_mask[index, :n_t] = True
            target_intensities[index, :n_t] = torch.from_numpy(
                item.target_intensities.astype(np.float32)
            )

            n_c = item.candidate_ions.size
            if n_c:
                candidate_formula_counts[index, :n_c, :n_e] = torch.from_numpy(
                    item.candidate_counts
                )
                candidate_ion_states[index, :n_c] = torch.from_numpy(item.candidate_ions)
                candidate_to_peak[index, :n_c] = torch.from_numpy(item.candidate_targets)
                candidate_mask[index, :n_c] = True

            admissible_ion_mask[index] = torch.from_numpy(item.admissible_ions)

            n_p = item.full_mzs.size
            full_peak_mzs[index, :n_p] = torch.from_numpy(item.full_mzs.astype(np.float32))
            full_peak_intensities[index, :n_p] = torch.from_numpy(
                item.full_target.astype(np.float32)
            )
            full_peak_intensities_raw[index, :n_p] = torch.from_numpy(
                item.full_raw.astype(np.float32)
            )
            full_peak_mask[index, :n_p] = True

        # What each row costs to score, so a later batch sampler can balance on
        # the quantity that actually drives memory: candidate element steps,
        # not spectrum count.
        token_counts = encoded["attention_mask"].sum(dim=1).tolist()
        cost_profile = [
            {
                "spectrum_uid": str(rows[index]["spectrum_uid"]),
                "num_tokens": int(token_counts[index]),
                "num_targets": int(prepared[index].target_peaks.size),
                "num_candidates_linked_to_top64": int(prepared[index].candidate_ions.size),
                "num_candidate_element_steps": int(
                    prepared[index].candidate_ions.size * prepared[index].element_numbers.size
                ),
                "num_peaks": int(prepared[index].full_mzs.size),
            }
            for index in range(batch)
        ]

        return {
            "spectrum_uid": [str(row["spectrum_uid"]) for row in rows],
            "cost_profile": cost_profile,
            "adduct": [str(row["adduct"]) for row in rows],
            "text": [p.text for p in prepared],
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "precursor_element_ids": element_ids,
            "precursor_element_counts": element_counts,
            "precursor_element_mask": element_mask,
            "target_peak_indices": target_peak_indices,
            "target_peak_mask": target_peak_mask,
            "target_intensities": target_intensities,
            "candidate_formula_counts": candidate_formula_counts,
            "candidate_ion_states": candidate_ion_states,
            "candidate_to_peak": candidate_to_peak,
            "candidate_mask": candidate_mask,
            "admissible_ion_mask": admissible_ion_mask,
            "full_peak_mzs": full_peak_mzs,
            "full_peak_intensities": full_peak_intensities,
            "full_peak_intensities_raw": full_peak_intensities_raw,
            "full_peak_mask": full_peak_mask,
        }
