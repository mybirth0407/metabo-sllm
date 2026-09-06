"""Candidate-free spectrum prediction.

At validation the model must stand on its own: given a molecule and the
acquisition settings, predict a spectrum.  Everything derived from the observed
spectrum -- the peaks, their intensities, the candidate formulas that were
enumerated around them, the supervision ranks -- is withheld, because a model
that saw any of it would be scored on information it will not have at test
time.

That contract is enforced here rather than trusted: :func:`build_model_inputs`
constructs only the permitted tensors, and :func:`predict_spectrum` refuses a
batch that carries a forbidden key.  Metrics are computed afterwards, by a
separate evaluator that reads the experimental spectrum.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from metabo_sllm.chem.candidates import (
    ION_STATE_VOCABULARY,
    admissible_ion_state_ids,
    channels_for_adduct,
)
from metabo_sllm.chem.formula import atomic_number, element_mass, element_symbol, formula_to_string, parse_formula
from metabo_sllm.model.input_formatter import format_row

__all__ = [
    "FORBIDDEN_INPUT_FIELDS",
    "FragmentPrediction",
    "MODEL_INPUT_FIELDS",
    "SpectrumPrediction",
    "build_model_inputs",
    "predict_spectrum",
]

# Exactly what the model may see.
MODEL_INPUT_FIELDS = frozenset(
    {
        "spectrum_uid",
        "adduct",
        "text",
        "input_ids",
        "attention_mask",
        "precursor_element_ids",
        "precursor_element_counts",
        "precursor_element_mask",
        "admissible_ion_mask",
    }
)

# Anything derived from the observed spectrum or from candidate enumeration.
FORBIDDEN_INPUT_FIELDS = frozenset(
    {
        "mzs",
        "intensities",
        "mz_decimal_places",
        "full_peak_mzs",
        "full_peak_intensities",
        "full_peak_intensities_raw",
        "full_peak_mask",
        "candidate_formula_counts",
        "candidate_ion_states",
        "candidate_to_peak",
        "candidate_mask",
        "candidate_neutral_formula",
        "candidate_ion_state",
        "candidate_theoretical_mz",
        "edge_peak_index",
        "edge_candidate_index",
        "edge_error_ppm",
        "supervision_mask",
        "supervision_rank",
        "target_peak_indices",
        "target_peak_mask",
        "target_intensities",
    }
)

# Fields of a stored row that are legitimately part of the model's conditioning.
CONDITIONING_KEYS = (
    "smiles",
    "formula",
    "adduct",
    "collision_energy",
    "instrument",
    "precursor_mz",
)


@dataclass(frozen=True)
class FragmentPrediction:
    formula: str
    ion_state: str
    mz: float
    presence: float
    raw_intensity: float
    weighted_intensity: float
    slot_index: int
    merged_slots: int = 1


@dataclass
class SpectrumPrediction:
    spectrum_uid: str
    fragments: list[FragmentPrediction] = field(default_factory=list)
    active_slots: int = 0
    empty_formula_slots: int = 0
    duplicate_identity_slots: int = 0

    @property
    def mz(self) -> np.ndarray:
        return np.asarray([f.mz for f in self.fragments], dtype=np.float64)

    @property
    def intensity(self) -> np.ndarray:
        return np.asarray([f.weighted_intensity for f in self.fragments], dtype=np.float64)


def build_model_inputs(
    rows: Sequence[Mapping[str, object]],
    tokenizer,
    *,
    max_text_length: int = 512,
) -> dict:
    """Tensors for candidate-free prediction, and nothing else.

    Built from the conditioning fields alone, so a row's peaks and candidates
    cannot leak in even by accident.
    """
    texts = []
    element_lists = []
    for row in rows:
        texts.append(format_row({key: row.get(key) for key in CONDITIONING_KEYS}))
        precursor = parse_formula(str(row["formula"]))
        symbols = sorted(precursor, key=atomic_number)
        element_lists.append(
            (
                np.asarray([atomic_number(s) for s in symbols], dtype=np.int64),
                np.asarray([precursor[s] for s in symbols], dtype=np.int64),
            )
        )

    encoded = tokenizer(
        texts, padding=True, truncation=True, max_length=max_text_length, return_tensors="pt"
    )
    batch = len(rows)
    width = max(numbers.size for numbers, _ in element_lists)
    element_ids = torch.zeros(batch, width, dtype=torch.long)
    element_counts = torch.zeros(batch, width, dtype=torch.long)
    element_mask = torch.zeros(batch, width, dtype=torch.bool)
    admissible = torch.zeros(batch, len(ION_STATE_VOCABULARY), dtype=torch.bool)

    for index, (numbers, counts) in enumerate(element_lists):
        size = numbers.size
        element_ids[index, :size] = torch.from_numpy(numbers)
        element_counts[index, :size] = torch.from_numpy(counts)
        element_mask[index, :size] = True
        admissible[index, list(admissible_ion_state_ids(str(rows[index]["adduct"])))] = True

    return {
        "spectrum_uid": [str(row["spectrum_uid"]) for row in rows],
        "adduct": [str(row["adduct"]) for row in rows],
        "text": texts,
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "precursor_element_ids": element_ids,
        "precursor_element_counts": element_counts,
        "precursor_element_mask": element_mask,
        "admissible_ion_mask": admissible,
    }


def assert_no_leakage(model_inputs: Mapping[str, object]) -> None:
    """Refuse a batch carrying anything the model must not see."""
    present = set(model_inputs) & FORBIDDEN_INPUT_FIELDS
    if present:
        raise ValueError(
            f"model inputs carry label-derived fields: {sorted(present)}. "
            "Validation prediction must not see the observed spectrum or its candidates."
        )
    unknown = set(model_inputs) - MODEL_INPUT_FIELDS
    if unknown:
        raise ValueError(f"unexpected model input fields: {sorted(unknown)}")


def _counts_to_formula(element_ids: np.ndarray, counts: np.ndarray) -> tuple[str, float]:
    mapping: dict[str, int] = {}
    mass = 0.0
    for number, count in zip(element_ids.tolist(), counts.tolist(), strict=True):
        if number == 0 or count <= 0:
            continue
        symbol = element_symbol(number)
        mapping[symbol] = int(count)
        mass += element_mass(symbol) * int(count)
    return formula_to_string(mapping), mass


@torch.no_grad()
def predict_spectrum(
    model,
    model_inputs: Mapping[str, object],
    *,
    presence_threshold: float = 0.5,
    beam_size: int = 1,
) -> list[SpectrumPrediction]:
    """Hard-decode every slot into a fragment, then merge duplicates.

    Slots below ``presence_threshold`` contribute nothing.  Slots that agree on
    ``(formula, ion_state)`` are summed rather than emitted twice.
    """
    if beam_size != 1:
        raise NotImplementedError("only greedy decoding is implemented")
    assert_no_leakage(model_inputs)

    was_training = model.training
    model.eval()
    try:
        outputs = model(dict(model_inputs))
        slots = outputs.slots
        batch, num_slots, slot_dim = slots.shape
        elements = model_inputs["precursor_element_ids"].shape[1]
        shape = (batch, num_slots, elements)

        counts = (
            model.formula_decoder.greedy_decode(
                slots.reshape(batch * num_slots, slot_dim),
                model_inputs["precursor_element_ids"].unsqueeze(1).expand(shape).reshape(-1, elements),
                model_inputs["precursor_element_counts"].unsqueeze(1).expand(shape).reshape(-1, elements),
                model_inputs["precursor_element_mask"].unsqueeze(1).expand(shape).reshape(-1, elements),
            )
            .view(batch, num_slots, elements)
            .cpu()
            .numpy()
        )
        ion_choice = outputs.ion_log_prob.argmax(dim=-1).cpu().numpy()
        presence = outputs.presence.cpu().numpy()
        raw_intensity = outputs.intensity.cpu().numpy()
        contribution = outputs.contribution.cpu().numpy()
    finally:
        model.train(was_training)

    element_ids = model_inputs["precursor_element_ids"].cpu().numpy()
    predictions: list[SpectrumPrediction] = []
    for index in range(batch):
        channels = {c.name: c for c in channels_for_adduct(model_inputs["adduct"][index])}
        merged: dict[tuple[str, str], dict] = {}
        active = 0
        empty = 0
        for slot in range(num_slots):
            if presence[index, slot] < presence_threshold:
                continue
            active += 1
            formula, mass = _counts_to_formula(element_ids[index], counts[index, slot])
            if not formula:
                empty += 1
                continue
            name = ION_STATE_VOCABULARY[int(ion_choice[index, slot])]
            channel = channels.get(name)
            if channel is None:
                continue
            key = (formula, name)
            entry = merged.get(key)
            if entry is None:
                merged[key] = {
                    "mz": mass + channel.mz_offset,
                    "presence": float(presence[index, slot]),
                    "raw": float(raw_intensity[index, slot]),
                    "weighted": float(contribution[index, slot]),
                    "slot": slot,
                    "best": float(contribution[index, slot]),
                    "merged": 1,
                }
            else:
                entry["weighted"] += float(contribution[index, slot])
                entry["raw"] += float(raw_intensity[index, slot])
                entry["merged"] += 1
                if float(contribution[index, slot]) > entry["best"]:
                    entry["best"] = float(contribution[index, slot])
                    entry["slot"] = slot
                    entry["presence"] = float(presence[index, slot])

        fragments = [
            FragmentPrediction(
                formula=formula,
                ion_state=ion,
                mz=entry["mz"],
                presence=entry["presence"],
                raw_intensity=entry["raw"],
                weighted_intensity=entry["weighted"],
                slot_index=entry["slot"],
                merged_slots=entry["merged"],
            )
            for (formula, ion), entry in sorted(merged.items())
        ]
        predictions.append(
            SpectrumPrediction(
                spectrum_uid=str(model_inputs["spectrum_uid"][index]),
                fragments=fragments,
                active_slots=active,
                empty_formula_slots=empty,
                duplicate_identity_slots=sum(f.merged_slots - 1 for f in fragments),
            )
        )
    return predictions
