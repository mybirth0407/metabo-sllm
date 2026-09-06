"""The collator decides what reaches the GPU; these pin that decision down."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from metabo_sllm.chem.candidates import ION_STATE_TO_ID
from metabo_sllm.data.fragment_collator import (
    FragmentCollator,
    select_targets,
    transform_intensities,
)
from metabo_sllm.data.supervision import FIXED_SLOT_COUNT, supervision_rank


class DummyTokenizer:
    """Deterministic stand-in so collator tests do not load a 0.6B backbone."""

    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(self, texts, padding=True, truncation=True, max_length=64, return_tensors="pt"):
        encoded = [[ord(c) % 97 + 1 for c in text[:max_length]] for text in texts]
        width = max(len(e) for e in encoded)
        ids = torch.zeros(len(encoded), width, dtype=torch.long)
        mask = torch.zeros(len(encoded), width, dtype=torch.long)
        for index, item in enumerate(encoded):
            ids[index, : len(item)] = torch.tensor(item, dtype=torch.long)
            mask[index, : len(item)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def make_row(
    *,
    uid="nist_1:00",
    formula="C2H6O",
    adduct="[M+H]+",
    n_peaks=8,
    supervised=None,
    decimals=None,
    candidates=None,
    edges=None,
    intensities=None,
):
    mzs = np.arange(1, n_peaks + 1, dtype=np.float64) * 10.0
    if intensities is None:
        intensities = np.arange(n_peaks, 0, -1, dtype=np.float64)
    decimals = np.full(n_peaks, 4, dtype=np.int64) if decimals is None else np.asarray(decimals)
    if candidates is None:
        candidates = [("C2H6O", "protonated")]
    if edges is None:
        edges = [(index, 0) for index in range(n_peaks)]
    edge_peak = np.asarray([e[0] for e in edges], dtype=np.int64)
    edge_candidate = np.asarray([e[1] for e in edges], dtype=np.int64)

    counts = np.zeros(n_peaks, dtype=np.int64)
    for peak in edge_peak:
        counts[peak] += 1
    if supervised is None:
        supervised = (counts > 0) & (decimals >= 2)
    rank = supervision_rank(supervised, mzs, intensities)

    return {
        "spectrum_uid": uid,
        "parent_spec": uid.split(":")[0],
        "fold": "train",
        "collision_energy": 20.0,
        "smiles": "CCO",
        "formula": formula,
        "inchikey": "AAA-BBB-C",
        "adduct": adduct,
        "instrument": "Orbitrap",
        "precursor_mz": 47.0491,
        "mzs": mzs,
        "intensities": intensities,
        "mz_decimal_places": decimals,
        "supervision_mask": supervised,
        "supervision_rank": rank,
        "candidate_neutral_formula": [c[0] for c in candidates],
        "candidate_ion_state": [c[1] for c in candidates],
        "candidate_theoretical_mz": np.zeros(len(candidates)),
        "edge_peak_index": edge_peak,
        "edge_candidate_index": edge_candidate,
        "edge_error_ppm": np.zeros(len(edges)),
    }


@pytest.fixture
def collator():
    return FragmentCollator(DummyTokenizer(), max_text_length=64)


# ------------------------------------------------------------------ selection


def test_intensity_transform_normalises_over_every_peak():
    intensities = np.array([9.0, 16.0, 0.0, 25.0])
    transformed = transform_intensities(intensities)

    np.testing.assert_allclose(np.linalg.norm(transformed), 1.0, atol=1e-9)
    np.testing.assert_allclose(transformed, np.sqrt(intensities) / np.sqrt(intensities.sum()))


def test_select_targets_takes_at_most_the_slot_count_in_rank_order():
    count = FIXED_SLOT_COUNT + 30
    mask = np.ones(count, dtype=bool)
    mzs = np.arange(count, dtype=np.float64)
    intensities = np.arange(count, dtype=np.float64)[::-1].copy()
    rank = supervision_rank(mask, mzs, intensities)

    selected = select_targets(rank)

    assert selected.size == FIXED_SLOT_COUNT
    assert selected.tolist() == list(range(FIXED_SLOT_COUNT))
    assert rank[selected].tolist() == list(range(FIXED_SLOT_COUNT))


def test_more_supervised_peaks_than_slots_are_capped(collator):
    count = FIXED_SLOT_COUNT + 20
    row = make_row(n_peaks=count, edges=[(i, 0) for i in range(count)])
    batch = collator([row])

    assert int(batch["target_peak_mask"].sum()) == FIXED_SLOT_COUNT
    assert batch["full_peak_mask"].sum() == count  # the raw spectrum keeps every peak


def test_low_precision_and_unsupported_peaks_stay_in_the_full_spectrum(collator):
    decimals = np.array([4, 1, 4, 0, 4, 4, 4, 4])
    row = make_row(decimals=decimals, edges=[(i, 0) for i in range(6)])
    batch = collator([row])

    targets = batch["target_peak_indices"][0][batch["target_peak_mask"][0]].tolist()
    assert 1 not in targets and 3 not in targets  # low precision
    assert 6 not in targets and 7 not in targets  # no candidate
    assert int(batch["full_peak_mask"][0].sum()) == 8
    np.testing.assert_allclose(
        batch["full_peak_intensities_raw"][0, :8].numpy(),
        row["intensities"].astype(np.float32),
    )


# ----------------------------------------------------------------- candidates


def test_only_candidates_of_selected_targets_are_tensorised(collator):
    # peak 0 is supervised, peak 1 is low precision so its candidate is dropped
    row = make_row(
        n_peaks=2,
        decimals=np.array([4, 1]),
        candidates=[("C2H6O", "protonated"), ("CH4", "protonated")],
        edges=[(0, 0), (1, 1)],
    )
    batch = collator([row])

    assert int(batch["candidate_mask"].sum()) == 1
    assert int(batch["candidate_to_peak"][0, 0]) == 0


def test_ambiguous_bags_are_kept_whole(collator):
    bag = [("C2H6O", "protonated"), ("CH4", "protonated"), ("C2H4", "protonated")]
    row = make_row(n_peaks=1, candidates=bag, edges=[(0, 0), (0, 1), (0, 2)])
    batch = collator([row])

    assert int(batch["candidate_mask"].sum()) == 3
    assert batch["candidate_to_peak"][0, :3].tolist() == [0, 0, 0]


def test_candidate_counts_align_with_precursor_elements(collator):
    row = make_row(n_peaks=1, formula="C2H6O", candidates=[("CH4", "protonated")], edges=[(0, 0)])
    batch = collator([row])

    # elements ordered by atomic number: H(1), C(6), O(8)
    assert batch["precursor_element_ids"][0].tolist() == [1, 6, 8]
    assert batch["precursor_element_counts"][0].tolist() == [6, 2, 1]
    assert batch["candidate_formula_counts"][0, 0].tolist() == [4, 1, 0]


def test_ion_state_ids_and_admissible_mask(collator):
    row = make_row(adduct="[M+Na]+", candidates=[("C2H6O", "sodiated")], edges=[(0, 0)])
    batch = collator([row])

    assert int(batch["candidate_ion_states"][0, 0]) == ION_STATE_TO_ID["sodiated"]
    admissible = batch["admissible_ion_mask"][0]
    assert bool(admissible[ION_STATE_TO_ID["protonated"]])
    assert bool(admissible[ION_STATE_TO_ID["sodiated"]])
    assert not bool(admissible[ION_STATE_TO_ID["deprotonated"]])


def test_target_intensities_come_from_the_full_spectrum_normalisation(collator):
    row = make_row(n_peaks=4, intensities=np.array([9.0, 16.0, 0.0, 25.0]))
    batch = collator([row])

    expected = transform_intensities(row["intensities"])
    selected = batch["target_peak_indices"][0][batch["target_peak_mask"][0]].numpy()
    np.testing.assert_allclose(
        batch["target_intensities"][0][batch["target_peak_mask"][0]].numpy(),
        expected[selected].astype(np.float32),
        atol=1e-6,
    )


def test_batches_pad_to_the_widest_row(collator):
    small = make_row(uid="a:00", n_peaks=2)
    large = make_row(uid="b:00", n_peaks=6, formula="C6H12O6")
    batch = collator([small, large])

    assert batch["full_peak_mask"].shape[0] == 2
    assert batch["full_peak_mask"][0].sum() == 2
    assert batch["full_peak_mask"][1].sum() == 6
    assert batch["precursor_element_mask"].shape[1] == 3
    assert batch["target_peak_mask"].shape[1] == FIXED_SLOT_COUNT


def test_precursor_count_above_the_vocabulary_fails_loudly():
    collator = FragmentCollator(DummyTokenizer(), max_count=8)
    row = make_row(formula="C2H600O")

    with pytest.raises(ValueError, match="count vocabulary"):
        collator([row])
