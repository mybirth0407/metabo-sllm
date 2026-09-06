"""Evaluation Contract Audit: intensity spaces, oracle definitions, reproducibility.

Three things had been left implicit and are pinned down here.

**Intensity space.** The intensity head is trained against ``sqrt(y)`` normalised
over the whole spectrum, but the reported cosine compared that output against
the *raw* observed intensities. Comparing two different spaces is not a
property of the model, so every metric below is computed in each space and the
spaces are named rather than assumed.

**Oracles.** "Oracle" had covered two very different ceilings: copying the
observed peaks outright, and rendering peaks from candidate formulas. The first
is 1.0 by construction and says nothing; only the second measures what the
formula supervision can express. They are separate functions here, with names
that say which is which.

**Reproducibility.** A spectrum's prediction depended on how much padding its
batch happened to carry. Canonical batching pads every sequence to a fixed
length so a spectrum's result depends on the spectrum alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from metabo_sllm.chem.candidates import ION_STATE_VOCABULARY, channels_for_adduct
from metabo_sllm.chem.formula import element_mass, parse_formula
from metabo_sllm.data.supervision import FIXED_SLOT_COUNT
from metabo_sllm.evaluation.spectrum_metrics import BinningConfig, bin_index

__all__ = [
    "INTENSITY_SPACES",
    "ORACLES",
    "OracleSpectra",
    "build_oracles",
    "canonical_model_inputs",
    "cosine_in_space",
]

# How a prediction and an observation are brought into a common space before
# the cosine. The model's intensity head is trained against
# ``sqrt(y)/||sqrt(y)||``, so its output is already a square-root-space
# quantity; that is what makes ``legacy_raw`` a category error rather than a
# stylistic choice.
#
# ``canonical_sqrt`` is ms-pred's own contract. Its preprocessing
# (``common/misc_utils.py``: ``spec[:,1] /= spec[:,1].max()`` then
# ``np.sqrt(...)``) stores spectra already square-rooted, ``bin_from_str``
# notes "Don't renorm; already processed prior!", and
# ``analysis/spec_pred_eval.py`` bins those values with max pooling and takes a
# plain cosine. ``norm_spectrum`` -- the only other sqrt in the codebase -- is
# dead code: every call site is commented out.
INTENSITY_SPACES = (
    "legacy_raw",  # prediction as-is vs raw observation: mismatched, what was reported
    "canonical_sqrt",  # prediction as-is vs sqrt(observation): both square-root space
    "raw_via_squared_prediction",  # prediction squared vs raw observation: both raw space
    "double_sqrt",  # sqrt of both: shown only to rule out a naive reading
)

ORACLES = (
    "peak_copy_oracle",
    "support_mask_oracle",
    "formula_rendered_oracle",
    "slot_capacity_oracle",
)


def _bin(mz: np.ndarray, intensity: np.ndarray, config: BinningConfig, reduce: str) -> np.ndarray:
    output = np.zeros(config.num_bins, dtype=np.float64)
    mz = np.asarray(mz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if mz.size == 0:
        return output
    index = bin_index(mz, config)
    keep = (index >= 0) & (index < config.num_bins)
    if not keep.any():
        return output
    if reduce == "max":
        np.maximum.at(output, index[keep], intensity[keep])
    else:
        np.add.at(output, index[keep], intensity[keep])
    return output


def cosine_in_space(
    predicted_mz: np.ndarray,
    predicted_intensity: np.ndarray,
    observed_mz: np.ndarray,
    observed_intensity: np.ndarray,
    *,
    space: str,
    k: int,
    config: BinningConfig,
) -> float:
    """``cos@k`` after stating which space each side is in.

    The prediction always arrives in square-root space, because that is what the
    intensity head was trained to produce.  What varies is what it is compared
    against; see :data:`INTENSITY_SPACES`.
    """
    predicted_intensity = np.clip(np.asarray(predicted_intensity, dtype=np.float64), 0.0, None)
    observed_intensity = np.clip(np.asarray(observed_intensity, dtype=np.float64), 0.0, None)

    if space == "legacy_raw":
        pass
    elif space == "canonical_sqrt":
        observed_intensity = np.sqrt(observed_intensity)
    elif space == "raw_via_squared_prediction":
        predicted_intensity = np.square(predicted_intensity)
    elif space == "double_sqrt":
        predicted_intensity = np.sqrt(predicted_intensity)
        observed_intensity = np.sqrt(observed_intensity)
    else:
        raise ValueError(f"unknown intensity space {space!r}")

    prediction = _bin(predicted_mz, predicted_intensity, config, "sum")
    peak = prediction.max()
    if peak > 0:
        prediction = prediction / peak
        prediction[prediction < config.min_pred_intensity] = 0.0
    observation = _bin(observed_mz, observed_intensity, config, "max")

    nonzero = int(np.count_nonzero(prediction))
    if nonzero == 0:
        return 0.0
    if nonzero > k:
        cut = np.argpartition(prediction, -k)[-k:]
        trimmed = np.zeros_like(prediction)
        trimmed[cut] = prediction[cut]
        prediction = trimmed
    denominator = np.linalg.norm(prediction) * np.linalg.norm(observation)
    return float(prediction @ observation / denominator) if denominator else 0.0


@dataclass
class OracleSpectra:
    """One spectrum's oracles, each as a peak list a perfect model could emit.

    ``observed`` is the thing every oracle is scored against and carries the
    raw intensities.  The oracle peak lists carry ``sqrt`` of those intensities
    instead, because that is the space the intensity head is trained to produce
    -- an oracle is a *perfect prediction*, so it has to live where predictions
    live.  Handing back raw intensities here would score the oracle in one
    space against an observation in another, and ``peak_copy`` would stop
    reaching 1.0 in the very space the contract says is canonical.
    """

    observed: tuple[np.ndarray, np.ndarray]
    peak_copy: tuple[np.ndarray, np.ndarray]
    support_mask: tuple[np.ndarray, np.ndarray]
    formula_rendered: tuple[np.ndarray, np.ndarray]
    slot_capacity: tuple[np.ndarray, np.ndarray]
    representable_peaks: int
    supervised_peaks: int
    total_peaks: int
    rendered_offset_da: float


def build_oracles(row) -> OracleSpectra:
    """Four ceilings, separated so none of them can be mistaken for the others.

    ``peak_copy`` reproduces the observation exactly and scores 1.0 in every
    self-consistent space; it is a sanity check, not a ceiling on anything the
    model could do.  ``formula_rendered`` is the one that says what the formula
    supervision can express, because it places peaks at masses computed from
    candidate formulas rather than copying the observed m/z.
    """
    observed_mz = np.asarray(row["mzs"], dtype=np.float64)
    observed_intensity = np.asarray(row["intensities"], dtype=np.float64)
    # what a perfect intensity head would emit for these peaks
    oracle_intensity = np.sqrt(np.clip(observed_intensity, 0.0, None))
    edge_peak = np.asarray(row["edge_peak_index"], dtype=np.int64)
    edge_candidate = np.asarray(row["edge_candidate_index"], dtype=np.int64)
    rank = np.asarray(row["supervision_rank"], dtype=np.int64)
    formulas = row["candidate_neutral_formula"]
    ions = row["candidate_ion_state"]
    channels = {c.name: c for c in channels_for_adduct(str(row["adduct"]))}

    # one candidate per representable peak; bag members agree to within the
    # mass tolerance, which is far below the bin width
    first_candidate: dict[int, int] = {}
    for peak, candidate in zip(edge_peak.tolist(), edge_candidate.tolist(), strict=True):
        first_candidate.setdefault(int(peak), int(candidate))

    representable = np.asarray(sorted(first_candidate), dtype=np.int64)
    supervised = np.flatnonzero((rank >= 0) & (rank < FIXED_SLOT_COUNT))

    rendered_mz = np.zeros(representable.size, dtype=np.float64)
    offsets = []
    cache: dict[int, float] = {}
    for position, peak in enumerate(representable.tolist()):
        candidate = first_candidate[peak]
        mass = cache.get(candidate)
        if mass is None:
            counts = parse_formula(formulas[candidate])
            mass = sum(element_mass(symbol) * n for symbol, n in counts.items())
            cache[candidate] = mass
        channel = channels.get(ions[candidate])
        value = mass + (channel.mz_offset if channel else 0.0)
        rendered_mz[position] = value
        offsets.append(abs(value - observed_mz[peak]))

    rendered_lookup = dict(zip(representable.tolist(), rendered_mz.tolist(), strict=True))
    slot_peaks = np.asarray([p for p in supervised.tolist() if p in rendered_lookup], dtype=np.int64)
    slot_mz = np.asarray([rendered_lookup[int(p)] for p in slot_peaks], dtype=np.float64)

    return OracleSpectra(
        observed=(observed_mz, observed_intensity),
        peak_copy=(observed_mz, oracle_intensity),
        support_mask=(observed_mz[representable], oracle_intensity[representable]),
        formula_rendered=(rendered_mz, oracle_intensity[representable]),
        slot_capacity=(slot_mz, oracle_intensity[slot_peaks]),
        representable_peaks=int(representable.size),
        supervised_peaks=int(supervised.size),
        total_peaks=int(observed_mz.size),
        rendered_offset_da=float(np.max(offsets)) if offsets else 0.0,
    )


def canonical_model_inputs(
    rows, tokenizer, *, sequence_length: int = 256
) -> dict:
    """Model inputs padded to a fixed width, so batching cannot change a result.

    With ragged padding the encoder's output moves at float level with the
    length of whoever else is in the batch, and a slot sitting near the greedy
    argmax boundary flips.  Padding every sequence to the same width removes the
    dependence entirely rather than hiding it behind a tolerance.
    """
    from metabo_sllm.evaluation.inference import CONDITIONING_KEYS
    from metabo_sllm.chem.formula import atomic_number
    from metabo_sllm.chem.candidates import admissible_ion_state_ids
    from metabo_sllm.model.input_formatter import format_row

    texts = [format_row({key: row.get(key) for key in CONDITIONING_KEYS}) for row in rows]
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=sequence_length,
        return_tensors="pt",
    )

    batch = len(rows)
    element_lists = []
    for row in rows:
        precursor = parse_formula(str(row["formula"]))
        symbols = sorted(precursor, key=atomic_number)
        element_lists.append(
            (
                np.asarray([atomic_number(s) for s in symbols], dtype=np.int64),
                np.asarray([precursor[s] for s in symbols], dtype=np.int64),
            )
        )
    # a fixed element width too, for the same reason
    width = max(16, max(numbers.size for numbers, _ in element_lists))
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
