"""The evaluation contract: which space a cosine lives in, what each oracle bounds.

The reported ``cos@100`` compared a square-root-space prediction against raw
observed intensities.  A cosine between two different spaces is not a property
of the model, so the four spaces are named and pinned apart here rather than
left to whoever reads the number next.

Three contracts are fixed below.  The cosine has to agree with ms-pred's own
definition, re-derived independently in ``ms_pred_cosine`` rather than imported.
An oracle has to be a *perfect prediction*, which means it lives in the space
predictions live in -- otherwise ``peak_copy``, which copies the observation
outright, stops scoring 1.0.  And a spectrum's model input has to be a function
of that spectrum alone, not of whoever else shared its batch.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from metabo_sllm.chem.candidates import channels_for_adduct
from metabo_sllm.chem.formula import element_mass, parse_formula
from metabo_sllm.evaluation.contract_audit import (
    INTENSITY_SPACES,
    ORACLES,
    build_oracles,
    canonical_model_inputs,
    cosine_in_space,
)
from metabo_sllm.evaluation.inference import (
    MODEL_INPUT_FIELDS,
    assert_no_leakage,
    build_model_inputs,
)
from metabo_sllm.evaluation.spectrum_metrics import BinningConfig

CONFIG = BinningConfig()

# One dominant peak and a plateau of weak ones: the shape that makes the two
# spaces disagree.  A spectrum carried by a single base peak scores nearly the
# same either way, which would hide the mismatch instead of exposing it.
OBSERVED_MZ = np.array([81.0699, 105.0699, 133.0648, 161.0597, 179.0703])
OBSERVED_INTENSITY = np.array([100.0, 9.0, 9.0, 9.0, 9.0])


def ms_pred_cosine(pred_mz, pred_intensity, true_mz, true_intensity, *, k):
    """ms-pred's evaluation written out from its source, not imported from ours.

    ``common.bin_spectra(..., pool_fn="max")`` for the observation and summed
    bins for the prediction, then ``analysis/spec_pred_eval.py``: drop bins
    below ``min_inten``, keep the ``k`` largest, plain cosine with no further
    normalisation ("Don't renorm; already procesed prior!").
    """
    scale = (CONFIG.num_bins - 1) / CONFIG.upper_limit
    prediction = np.zeros(CONFIG.num_bins)
    observation = np.zeros(CONFIG.num_bins)
    for mz, value in zip(pred_mz, pred_intensity, strict=True):
        prediction[int(np.floor(mz * scale)) + 1] += value
    for mz, value in zip(true_mz, true_intensity, strict=True):
        index = int(np.floor(mz * scale)) + 1
        observation[index] = max(observation[index], value)

    if prediction.max() > 0:
        prediction = prediction / prediction.max()
        prediction[prediction < CONFIG.min_pred_intensity] = 0.0
    keep = sorted(np.flatnonzero(prediction), key=lambda i: -prediction[i])[:k]
    trimmed = np.zeros_like(prediction)
    trimmed[keep] = prediction[keep]

    denominator = np.linalg.norm(trimmed) * np.linalg.norm(observation)
    return float(trimmed @ observation / denominator) if denominator else 0.0


def sqrt_space_prediction(intensity=OBSERVED_INTENSITY):
    """What the intensity head is trained to emit: ``sqrt(y)`` up to a scale."""
    root = np.sqrt(intensity)
    return root / np.linalg.norm(root)


def score(prediction, space, *, k=100, mz=OBSERVED_MZ):
    return cosine_in_space(
        mz, prediction, OBSERVED_MZ, OBSERVED_INTENSITY, space=space, k=k, config=CONFIG
    )


# ------------------------------------------------------------- intensity spaces


def test_every_space_is_named_once():
    assert len(set(INTENSITY_SPACES)) == len(INTENSITY_SPACES) == 4
    assert len(set(ORACLES)) == len(ORACLES) == 4


def test_an_unnamed_space_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="unknown intensity space"):
        score(sqrt_space_prediction(), "sqrt")


def test_the_cosine_is_ms_preds_cosine():
    prediction = sqrt_space_prediction()
    for space, transform in (("legacy_raw", lambda y: y), ("canonical_sqrt", np.sqrt)):
        expected = ms_pred_cosine(
            OBSERVED_MZ, prediction, OBSERVED_MZ, transform(OBSERVED_INTENSITY), k=100
        )
        assert score(prediction, space) == pytest.approx(expected)


def test_a_perfect_prediction_scores_one_in_the_canonical_space():
    assert score(sqrt_space_prediction(), "canonical_sqrt") == pytest.approx(1.0)


def test_squaring_a_perfect_prediction_scores_one_against_raw_intensities():
    """The other self-consistent reading: bring the prediction down to raw space."""
    assert score(sqrt_space_prediction(), "raw_via_squared_prediction") == pytest.approx(1.0)


def test_the_same_perfect_prediction_is_penalised_in_the_legacy_space():
    """This is the whole defect: a perfect model scored well below 1.0."""
    legacy = score(sqrt_space_prediction(), "legacy_raw")

    assert legacy < 0.95


def test_taking_the_root_twice_is_not_the_canonical_space():
    assert score(sqrt_space_prediction(), "double_sqrt") < 1.0


def test_the_two_self_consistent_spaces_agree_and_the_mismatched_ones_do_not():
    prediction = sqrt_space_prediction()
    consistent = {score(prediction, s) for s in ("canonical_sqrt", "raw_via_squared_prediction")}
    mismatched = [score(prediction, s) for s in ("legacy_raw", "double_sqrt")]

    assert max(consistent) - min(consistent) < 1e-9
    assert all(value < max(consistent) - 1e-3 for value in mismatched)


def test_ranking_can_change_with_the_space():
    """Why the space has to be named: it is not a monotone relabelling.

    A prediction that leans on the base peak beats a flatter one in raw space
    and loses to it in square-root space, so "which model is better" is not a
    question the number answers until the space is stated.
    """
    peaked = np.array([1.0, 0.02, 0.02, 0.02, 0.02])
    flat = sqrt_space_prediction()

    assert score(peaked, "legacy_raw") > score(flat, "legacy_raw")
    assert score(peaked, "canonical_sqrt") < score(flat, "canonical_sqrt")


# -------------------------------------------------------------------- oracles


ADDUCT = "[M+H]+"
CANDIDATE_FORMULAS = ["C3H4O", "C4H6O2", "C6H6O", "C7H6O2"]


def candidate_mz(formula):
    proton = {c.name: c for c in channels_for_adduct(ADDUCT)}["protonated"].mz_offset
    counts = parse_formula(formula)
    return sum(element_mass(symbol) * n for symbol, n in counts.items()) + proton


def oracle_row():
    """Five peaks: four carry a candidate, three of those reach a slot.

    The observed m/z sit a fraction of a milli-Dalton off the theoretical ones,
    the way measured peaks do, so ``formula_rendered`` genuinely re-derives its
    positions instead of copying them.
    """
    rendered = np.asarray([candidate_mz(f) for f in CANDIDATE_FORMULAS])
    observed = np.concatenate([rendered + 1e-4, [423.1234]])  # last peak: no candidate
    intensity = np.array([100.0, 9.0, 9.0, 9.0, 9.0])
    return {
        "adduct": ADDUCT,
        "mzs": observed,
        "intensities": intensity,
        "edge_peak_index": np.arange(4),
        "edge_candidate_index": np.arange(4),
        # peaks 0-2 win a slot; peak 3 is representable but unsupervised
        "supervision_rank": np.array([0, 1, 2, -1, -1]),
        "candidate_neutral_formula": CANDIDATE_FORMULAS,
        "candidate_ion_state": ["protonated"] * 4,
    }


def oracle_score(oracle, name, space, *, k=100):
    mz, intensity = getattr(oracle, name)
    return cosine_in_space(mz, intensity, *oracle.observed, space=space, k=k, config=CONFIG)


def test_oracles_count_the_peaks_they_are_built_from():
    oracle = build_oracles(oracle_row())

    assert oracle.total_peaks == 5
    assert oracle.representable_peaks == 4
    assert oracle.supervised_peaks == 3


def test_the_observed_side_keeps_the_raw_intensities():
    """Everything is scored against the observation, so it must not be transformed."""
    oracle = build_oracles(oracle_row())

    assert oracle.observed[1] == pytest.approx(oracle_row()["intensities"])


def test_oracles_are_predictions_and_so_live_in_square_root_space():
    oracle = build_oracles(oracle_row())

    assert oracle.peak_copy[1] == pytest.approx(np.sqrt(oracle_row()["intensities"]))


def test_peak_copy_scores_one_in_both_self_consistent_spaces():
    """The invariant that catches a space mismatch inside the oracle itself."""
    oracle = build_oracles(oracle_row())

    for space in ("canonical_sqrt", "raw_via_squared_prediction"):
        assert oracle_score(oracle, "peak_copy", space) == pytest.approx(1.0)


def test_peak_copy_falls_short_in_the_mismatched_spaces():
    """A copy of the observation stops scoring 1.0 the moment the spaces differ."""
    oracle = build_oracles(oracle_row())

    for space in ("legacy_raw", "double_sqrt"):
        assert oracle_score(oracle, "peak_copy", space) < 1.0 - 1e-3


def test_rendering_from_formulas_lands_in_the_observed_bins():
    """``formula_rendered`` recomputes m/z and must still hit the same bins.

    If it did not, the oracle would be measuring the mass tolerance rather than
    what the formula supervision can express.
    """
    oracle = build_oracles(oracle_row())

    assert oracle.rendered_offset_da < 1e-3
    assert oracle_score(oracle, "formula_rendered", "canonical_sqrt") == pytest.approx(
        oracle_score(oracle, "support_mask", "canonical_sqrt")
    )


def test_the_ceilings_are_ordered_by_how_much_they_drop():
    """Each oracle keeps a subset of the peaks of the one before it."""
    oracle = build_oracles(oracle_row())
    copy = oracle_score(oracle, "peak_copy", "canonical_sqrt")
    support = oracle_score(oracle, "support_mask", "canonical_sqrt")
    slots = oracle_score(oracle, "slot_capacity", "canonical_sqrt")

    assert copy == pytest.approx(1.0)
    assert support < copy
    assert slots < support


def test_peak_copy_says_nothing_about_the_model():
    """It reaches 1.0 whatever the spectrum looks like, so it bounds nothing."""
    row = oracle_row()
    row["intensities"] = np.array([3.0, 91.0, 0.4, 12.0, 55.0])
    oracle = build_oracles(row)

    assert oracle_score(oracle, "peak_copy", "canonical_sqrt") == pytest.approx(1.0)


# --------------------------------------------------------- canonical batching


class StubTokenizer:
    """Character-level stand-in carrying the padding contract we depend on."""

    def __call__(
        self, texts, *, padding=None, truncation=False, max_length=None, return_tensors=None
    ):
        if isinstance(texts, str):
            texts = [texts]
        ids = [[(ord(character) % 97) + 1 for character in text] for text in texts]
        if truncation and max_length is not None:
            ids = [row[:max_length] for row in ids]
        width = max_length if padding == "max_length" else max(len(row) for row in ids)
        input_ids = torch.zeros(len(ids), width, dtype=torch.long)
        attention = torch.zeros(len(ids), width, dtype=torch.long)
        for index, row in enumerate(ids):
            input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention[index, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention}


def make_row(uid, smiles, formula="C7H6O2"):
    return {
        "spectrum_uid": uid,
        "smiles": smiles,
        "formula": formula,
        "adduct": ADDUCT,
        "collision_energy": 20.0,
        "instrument": "HCD",
        "precursor_mz": 123.0441,
    }


SHORT = make_row("short", "CCO")
LONG = make_row("long", "CC(=O)Oc1ccccc1C(=O)O" * 4)
OTHER = make_row("other", "c1ccccc1O")


def canonical(rows):
    return canonical_model_inputs(rows, StubTokenizer(), sequence_length=256)


def test_canonical_inputs_do_not_move_with_their_neighbours():
    """The property that makes a prediction a function of its own spectrum."""
    alone = canonical([SHORT])
    with_long = canonical([SHORT, LONG])
    reversed_order = canonical([OTHER, LONG, SHORT])

    for key in ("input_ids", "attention_mask"):
        assert torch.equal(alone[key][0], with_long[key][0])
        assert torch.equal(alone[key][0], reversed_order[key][2])


def test_ragged_batching_does_move_with_its_neighbours():
    """The defect canonical batching exists to remove, stated as a test.

    Left ragged, the encoder sees a different amount of padding depending on
    who shares the batch, and a slot near the greedy argmax boundary flips.
    """
    tokenizer = StubTokenizer()
    alone = build_model_inputs([SHORT], tokenizer, max_text_length=512)
    with_long = build_model_inputs([SHORT, LONG], tokenizer, max_text_length=512)

    assert alone["input_ids"].shape[1] != with_long["input_ids"].shape[1]


def test_canonical_element_tensors_have_a_fixed_width():
    """Element padding is fixed too, for the same reason the token width is."""
    for rows in ([SHORT], [SHORT, LONG], [OTHER, LONG, SHORT]):
        inputs = canonical(rows)

        assert inputs["precursor_element_ids"].shape[1] == 16
        assert inputs["precursor_element_counts"].shape[1] == 16
        assert inputs["precursor_element_mask"].shape[1] == 16


def test_canonical_inputs_carry_nothing_the_model_may_not_see():
    inputs = canonical([SHORT, OTHER])

    assert_no_leakage(inputs)
    assert set(inputs) <= MODEL_INPUT_FIELDS


def test_canonical_inputs_are_bitwise_repeatable():
    first, second = canonical([SHORT, LONG]), canonical([SHORT, LONG])

    for key, value in first.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, second[key])
        else:
            assert value == second[key]
