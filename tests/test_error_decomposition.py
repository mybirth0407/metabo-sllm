"""The decomposition may read labels; the predictions it decomposes may not.

These tests pin two things: an oracle condition can only ever help (it is a
ceiling, not a result), and the prediction that every condition is measured
against is produced before any label is touched and does not change when the
labels do.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from metabo_sllm.chem.candidates import ION_STATE_TO_ID, ION_STATE_VOCABULARY
from metabo_sllm.evaluation.error_decomposition import CONDITIONS, beam_decode_formulas
from metabo_sllm.evaluation.inference import (
    FORBIDDEN_INPUT_FIELDS,
    MODEL_INPUT_FIELDS,
    assert_no_leakage,
    predict_spectrum,
)
from metabo_sllm.evaluation.spectrum_metrics import (
    PRIMARY_SPACE,
    BinningConfig,
    score_prediction,
)
from metabo_sllm.model.formula_decoder import StructuredFormulaDecoder
from metabo_sllm.model.fragment_latent_model import ModelOutput
from metabo_sllm.model.heads import IntensityHead, IonStateHead, PresenceHead
from metabo_sllm.model.slot_decoder import SlotDecoder

SLOTS = 8
SLOT_DIM = 24
MEMORY_DIM = 16
ELEMENTS = torch.tensor([[1, 6, 8]])
PRECURSOR = torch.tensor([[6, 2, 1]])
MASK = torch.ones(1, 3, dtype=torch.bool)


class StubModel(nn.Module):
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
            self.presence_head.net[-1].bias.fill_(2.0)

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


def model_inputs(seed: int = 0) -> dict:
    generator = torch.Generator().manual_seed(seed)
    admissible = torch.zeros(1, len(ION_STATE_VOCABULARY), dtype=torch.bool)
    admissible[0, ION_STATE_TO_ID["protonated"]] = True
    return {
        "spectrum_uid": ["uid"],
        "adduct": ["[M+H]+"],
        "text": ["SMILES=CCO"],
        "input_ids": torch.randint(1, 100, (1, 12), generator=generator),
        "attention_mask": torch.ones(1, 12, dtype=torch.long),
        "precursor_element_ids": ELEMENTS,
        "precursor_element_counts": PRECURSOR,
        "precursor_element_mask": MASK,
        "admissible_ion_mask": admissible,
    }


# ---------------------------------------------------------------- no leakage


def test_prediction_inputs_exclude_every_label_field():
    inputs = model_inputs()

    assert set(inputs) <= MODEL_INPUT_FIELDS
    assert not set(inputs) & FORBIDDEN_INPUT_FIELDS


def test_predictions_do_not_move_when_labels_change():
    """The decomposition changes labels constantly; predictions must not follow."""
    model = StubModel().eval()
    inputs = model_inputs()
    before = predict_spectrum(model, inputs)

    polluted = dict(inputs)
    polluted["candidate_mask"] = torch.ones(1, 4, dtype=torch.bool)
    with pytest.raises(ValueError):
        predict_spectrum(model, polluted)

    after = predict_spectrum(model, inputs)
    assert [(f.formula, f.ion_state, f.weighted_intensity) for f in before[0].fragments] == [
        (f.formula, f.ion_state, f.weighted_intensity) for f in after[0].fragments
    ]


def test_condition_names_mark_every_oracle():
    oracles = {name for name in CONDITIONS if name != "all_predicted"}

    assert "all_predicted" in CONDITIONS
    assert all("oracle" in name or name == "presence_forced_on_matched" for name in oracles)


# ------------------------------------------------------------------ ceilings


def _cosine(mz, intensity, true_mz, true_intensity, *, space=PRIMARY_SPACE):
    return score_prediction(
        "uid",
        np.asarray(mz, dtype=np.float64),
        np.asarray(intensity, dtype=np.float64),
        np.asarray(true_mz, dtype=np.float64),
        np.asarray(true_intensity, dtype=np.float64),
        BinningConfig(),
    ).cosine[space][100]


def test_oracle_identity_with_oracle_weights_is_a_ceiling():
    """Oracle weights are the raw observation, so they are perfect in legacy_raw."""
    true_mz = np.array([100.0, 200.0, 300.0])
    true_intensity = np.array([1.0, 0.5, 0.25])

    perfect = _cosine(true_mz, true_intensity, true_mz, true_intensity, space="legacy_raw")
    wrong_identity = _cosine(
        [111.0, 222.0, 333.0], true_intensity, true_mz, true_intensity, space="legacy_raw"
    )

    assert perfect == pytest.approx(1.0)
    assert wrong_identity < perfect


def test_each_weighting_is_perfect_in_exactly_one_space():
    """Which weights count as "correct" is a property of the space, not the model.

    Raw weights are perfect against a raw observation and wrong against a
    square-root one; square-root weights -- what the intensity head actually
    emits -- are the mirror image.  Reporting one number without naming its
    space is what let a square-root prediction be scored against raw
    intensities for the whole pilot.
    """
    true_mz = np.array([100.0, 200.0, 300.0])
    raw = np.array([100.0, 4.0, 1.0])
    root = np.sqrt(raw)

    assert _cosine(true_mz, raw, true_mz, raw, space="legacy_raw") == pytest.approx(1.0)
    assert _cosine(true_mz, raw, true_mz, raw, space=PRIMARY_SPACE) < 1.0

    assert _cosine(true_mz, root, true_mz, raw, space=PRIMARY_SPACE) == pytest.approx(1.0)
    assert _cosine(true_mz, root, true_mz, raw, space="legacy_raw") < 1.0


# ---------------------------------------------------------------------- beam


def test_beam_of_one_reproduces_greedy():
    model = StubModel().eval()
    slots = torch.randn(4, SLOT_DIM)
    elements = ELEMENTS.expand(4, -1)
    precursor = PRECURSOR.expand(4, -1)
    mask = MASK.expand(4, -1)

    greedy = model.formula_decoder.greedy_decode(slots, elements, precursor, mask)
    beam = beam_decode_formulas(
        model.formula_decoder, slots, elements, precursor, mask, beam=1
    )

    assert torch.equal(greedy, beam)


@pytest.mark.parametrize("width", [1, 2, 4])
def test_beam_respects_the_precursor_and_never_returns_empty(width):
    model = StubModel().eval()
    slots = torch.randn(6, SLOT_DIM)
    elements = ELEMENTS.expand(6, -1)
    precursor = PRECURSOR.expand(6, -1)
    mask = MASK.expand(6, -1)

    counts = beam_decode_formulas(
        model.formula_decoder, slots, elements, precursor, mask, beam=width
    )

    assert (counts <= precursor).all()
    assert (counts >= 0).all()
    assert (counts.sum(dim=1) > 0).all()


def test_wider_beam_never_lowers_the_model_log_probability():
    """A beam search maximises the decoder's own score, by construction."""
    model = StubModel().eval()
    slots = torch.randn(8, SLOT_DIM)
    elements = ELEMENTS.expand(8, -1)
    precursor = PRECURSOR.expand(8, -1)
    mask = MASK.expand(8, -1)

    narrow = beam_decode_formulas(model.formula_decoder, slots, elements, precursor, mask, beam=1)
    wide = beam_decode_formulas(model.formula_decoder, slots, elements, precursor, mask, beam=8)

    narrow_score = model.formula_decoder.log_prob(slots, elements, precursor, narrow, mask)
    wide_score = model.formula_decoder.log_prob(slots, elements, precursor, wide, mask)

    assert torch.all(wide_score >= narrow_score - 1e-5)


def test_beam_is_deterministic():
    model = StubModel().eval()
    slots = torch.randn(4, SLOT_DIM)
    args = (ELEMENTS.expand(4, -1), PRECURSOR.expand(4, -1), MASK.expand(4, -1))

    first = beam_decode_formulas(model.formula_decoder, slots, *args, beam=4)
    second = beam_decode_formulas(model.formula_decoder, slots, *args, beam=4)

    assert torch.equal(first, second)
