"""Validation predictions must not depend on anything derived from the answer."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from metabo_sllm.chem.candidates import ION_STATE_TO_ID, ION_STATE_VOCABULARY
from metabo_sllm.evaluation.inference import (
    FORBIDDEN_INPUT_FIELDS,
    MODEL_INPUT_FIELDS,
    assert_no_leakage,
    build_model_inputs,
    predict_spectrum,
)
from metabo_sllm.model.formula_decoder import StructuredFormulaDecoder
from metabo_sllm.model.fragment_latent_model import ModelOutput
from metabo_sllm.model.heads import IntensityHead, IonStateHead, PresenceHead
from metabo_sllm.model.slot_decoder import SlotDecoder

SLOTS = 8
SLOT_DIM = 24
MEMORY_DIM = 16


class DummyTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(self, texts, padding=True, truncation=True, max_length=64, return_tensors="pt"):
        encoded = [[ord(c) % 97 + 1 for c in text[:max_length]] for text in texts]
        width = max(len(item) for item in encoded)
        ids = torch.zeros(len(encoded), width, dtype=torch.long)
        mask = torch.zeros(len(encoded), width, dtype=torch.long)
        for index, item in enumerate(encoded):
            ids[index, : len(item)] = torch.tensor(item, dtype=torch.long)
            mask[index, : len(item)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class StubModel(nn.Module):
    """The real decoding path, with a deterministic stand-in for the backbone."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.embedding = nn.Embedding(200, MEMORY_DIM)
        self.slot_decoder = SlotDecoder(
            MEMORY_DIM, num_slots=SLOTS, hidden_dim=SLOT_DIM, num_layers=1, num_heads=4, dropout=0.0
        )
        self.formula_decoder = StructuredFormulaDecoder(
            SLOT_DIM, hidden_dim=32, num_layers=1, num_heads=4, max_count=16, dropout=0.0
        )
        self.ion_head = IonStateHead(SLOT_DIM, hidden_dim=16)
        self.presence_head = PresenceHead(SLOT_DIM, hidden_dim=16)
        self.intensity_head = IntensityHead(SLOT_DIM, hidden_dim=16)
        with torch.no_grad():
            self.presence_head.net[-1].bias.fill_(2.0)  # keep slots active

    def forward(self, batch):
        memory = self.embedding(batch["input_ids"])
        slots = self.slot_decoder(memory, batch["attention_mask"])
        presence = torch.sigmoid(self.presence_head(slots))
        intensity = self.intensity_head(slots)
        return ModelOutput(
            slots=slots,
            presence_logits=self.presence_head(slots),
            presence=presence,
            intensity=intensity,
            contribution=presence * intensity,
            ion_log_prob=self.ion_head(slots, batch["admissible_ion_mask"]),
        )


def make_row(uid="nist_1:00", adduct="[M+H]+", intensities=None, candidates=None):
    intensities = intensities if intensities is not None else np.array([5.0, 3.0, 1.0])
    candidates = candidates or ["C2H6O", "CH4"]
    return {
        "spectrum_uid": uid,
        "smiles": "CCO",
        "formula": "C2H6O",
        "adduct": adduct,
        "collision_energy": 20.0,
        "instrument": "Orbitrap",
        "precursor_mz": 47.0491,
        # everything below must never reach the model
        "mzs": np.array([31.0, 45.0, 47.0]),
        "intensities": intensities,
        "mz_decimal_places": np.array([2, 2, 2]),
        "supervision_mask": np.array([True, True, True]),
        "supervision_rank": np.array([0, 1, 2]),
        "candidate_neutral_formula": candidates,
        "candidate_ion_state": ["protonated"] * len(candidates),
        "candidate_theoretical_mz": np.zeros(len(candidates)),
        "edge_peak_index": np.array([0, 1]),
        "edge_candidate_index": np.array([0, 1]),
        "edge_error_ppm": np.zeros(2),
    }


@pytest.fixture
def parts():
    return StubModel().eval(), DummyTokenizer()


# ------------------------------------------------------------------ contract


def test_model_inputs_contain_only_permitted_fields(parts):
    _, tokenizer = parts
    inputs = build_model_inputs([make_row()], tokenizer)

    assert set(inputs) <= MODEL_INPUT_FIELDS
    assert not set(inputs) & FORBIDDEN_INPUT_FIELDS


def test_forbidden_fields_are_refused(parts):
    model, tokenizer = parts
    inputs = build_model_inputs([make_row()], tokenizer)
    inputs["target_intensities"] = torch.zeros(1, 64)

    with pytest.raises(ValueError, match="label-derived"):
        assert_no_leakage(inputs)
    with pytest.raises(ValueError, match="label-derived"):
        predict_spectrum(model, inputs)


@pytest.mark.parametrize("field", ["candidate_mask", "mzs", "supervision_rank", "edge_peak_index"])
def test_each_forbidden_field_is_caught(parts, field):
    _, tokenizer = parts
    inputs = build_model_inputs([make_row()], tokenizer)
    inputs[field] = torch.zeros(1)

    with pytest.raises(ValueError):
        assert_no_leakage(inputs)


# ---------------------------------------------------------------- invariance


def _identities(predictions):
    return [
        [(f.formula, f.ion_state, round(f.mz, 6), round(f.weighted_intensity, 6))
         for f in prediction.fragments]
        for prediction in predictions
    ]


def test_prediction_ignores_experimental_intensity(parts):
    model, tokenizer = parts
    base = predict_spectrum(model, build_model_inputs([make_row()], tokenizer))
    altered = predict_spectrum(
        model,
        build_model_inputs([make_row(intensities=np.array([1.0, 999.0, 0.5]))], tokenizer),
    )

    assert _identities(base) == _identities(altered)


def test_prediction_ignores_candidate_content_and_order(parts):
    model, tokenizer = parts
    base = predict_spectrum(model, build_model_inputs([make_row()], tokenizer))
    reordered = predict_spectrum(
        model, build_model_inputs([make_row(candidates=["CH4", "C2H6O"])], tokenizer)
    )
    removed_row = make_row()
    for key in ("candidate_neutral_formula", "candidate_ion_state", "edge_peak_index",
                "edge_candidate_index", "supervision_rank", "mzs", "intensities"):
        removed_row.pop(key)
    removed = predict_spectrum(model, build_model_inputs([removed_row], tokenizer))

    assert _identities(base) == _identities(reordered) == _identities(removed)


def test_prediction_is_deterministic(parts):
    model, tokenizer = parts
    inputs = build_model_inputs([make_row()], tokenizer)

    assert _identities(predict_spectrum(model, inputs)) == _identities(
        predict_spectrum(model, inputs)
    )


def test_prediction_is_deterministic_in_train_mode(parts):
    model, tokenizer = parts
    model.train()
    inputs = build_model_inputs([make_row()], tokenizer)
    first = _identities(predict_spectrum(model, inputs))

    assert model.training, "predict_spectrum must restore the training flag"
    assert first == _identities(predict_spectrum(model, inputs))


# ------------------------------------------------------------------ decoding


def test_no_empty_formula_is_ever_emitted(parts):
    model, tokenizer = parts
    predictions = predict_spectrum(model, build_model_inputs([make_row()], tokenizer))

    for fragment in predictions[0].fragments:
        assert fragment.formula
        assert fragment.weighted_intensity >= 0.0


def test_duplicate_identities_are_merged_not_repeated(parts):
    model, tokenizer = parts
    prediction = predict_spectrum(model, build_model_inputs([make_row()], tokenizer))[0]
    identities = [(f.formula, f.ion_state) for f in prediction.fragments]

    assert len(identities) == len(set(identities))
    assert sum(f.merged_slots for f in prediction.fragments) <= SLOTS


def test_only_admissible_ion_states_are_predicted(parts):
    model, tokenizer = parts
    prediction = predict_spectrum(
        model, build_model_inputs([make_row(adduct="[M-H]-")], tokenizer)
    )[0]

    for fragment in prediction.fragments:
        assert fragment.ion_state in {"deprotonated"}


def test_sodium_adduct_may_use_the_sodiated_channel(parts):
    model, tokenizer = parts
    inputs = build_model_inputs([make_row(adduct="[M+Na]+")], tokenizer)

    assert bool(inputs["admissible_ion_mask"][0, ION_STATE_TO_ID["sodiated"]])
    prediction = predict_spectrum(model, inputs)[0]
    for fragment in prediction.fragments:
        assert fragment.ion_state in {"protonated", "sodiated"}


def test_presence_threshold_silences_slots(parts):
    model, tokenizer = parts
    inputs = build_model_inputs([make_row()], tokenizer)

    assert predict_spectrum(model, inputs, presence_threshold=1.01)[0].fragments == []
    assert predict_spectrum(model, inputs, presence_threshold=0.0)[0].active_slots == SLOTS


def test_ion_state_vocabulary_is_pinned():
    assert ION_STATE_VOCABULARY == (
        "protonated", "deprotonated", "sodiated", "potassiated", "chloride_retained"
    )
