"""Greedy slot decoding and spectrum rendering.

Each slot is decoded to one fragment: a formula, an admissible ion state, the
theoretical m/z that pair implies, and an amplitude.  Slots that agree on
``(formula, ion_state)`` are summed rather than double counted, and the result
is accumulated into mass bins.

Only greedy decoding is implemented.  Beam search and marginal top-K rendering
are deliberately absent; the interfaces take the arguments they would need so
adding them later does not change call sites.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch

from metabo_sllm.chem.candidates import ION_STATE_VOCABULARY, IonChannel, channels_for_adduct
from metabo_sllm.chem.formula import element_mass, element_symbol, formula_to_string

__all__ = [
    "BinningConfig",
    "RenderedFragment",
    "bin_spectrum",
    "deterministic",
    "greedy_identity_metrics",
    "render_greedy",
    "slot_diversity",
]


@dataclass(frozen=True)
class BinningConfig:
    """Mass binning for evaluation, matching ms-pred's ``bin_spectra``.

    This is the canonical ms-pred setting, not a local choice: ``num_bins:
    15000`` appears in
    ``metabo_data/results/glacier_nist23/scaffold_1_rnd1/args.yaml`` and in the
    massformer baseline's ``args.yaml``; ``ppm_tol: 20`` and ``loss_fn: cosine``
    appear in the glacier, iceberg and marason configs.  ``upper_limit`` is
    ms-pred's 1500 Da default, which is what those 15000 bins span.

    Binning is only for inference and evaluation.  Training's spectrum loss
    works on peak indices, not bins -- see :func:`bin_index` versus
    ``losses.fragment_losses.spectrum_cosine_loss``.
    """

    num_bins: int = 15000
    upper_limit: float = 1500.0
    ppm_tolerance: float = 20.0
    source: str = "metabo_data/results/{glacier,massformer}_nist23/scaffold_1_rnd1/args.yaml"


def bin_index(mz: np.ndarray, config: BinningConfig | None = None) -> np.ndarray:
    """ms-pred bin index: ``floor(mz * (num_bins - 1) / upper_limit) + 1``.

    Valid indices are ``0 <= b < num_bins``; anything outside is dropped by
    :func:`bin_spectrum` rather than clamped onto the edge bin.
    """
    config = config or BinningConfig()
    scaled = np.asarray(mz, dtype=np.float64) * ((config.num_bins - 1) / config.upper_limit)
    return np.floor(scaled).astype(np.int64) + 1


@dataclass
class RenderedFragment:
    formula: str
    ion_state: str
    mz: float
    intensity: float


@dataclass
class RenderedSpectrum:
    spectrum_uid: str
    fragments: list[RenderedFragment] = field(default_factory=list)


@contextmanager
def deterministic(module: torch.nn.Module):
    """Run ``module`` without dropout, then restore its training flag.

    Decoding is a measurement, not a training step: with the model in train
    mode the formula decoder's dropout would randomise the argmax and make the
    reported identity accuracy look far worse than the model actually is.
    """
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)


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


def _channel_by_name(adduct: str) -> dict[str, IonChannel]:
    return {channel.name: channel for channel in channels_for_adduct(adduct)}


@torch.no_grad()
def render_greedy(
    model,
    batch: dict,
    outputs,
    *,
    presence_threshold: float = 0.5,
    beam_size: int = 1,
) -> list[RenderedSpectrum]:
    """Decode every slot greedily and merge duplicate identities.

    ``beam_size`` must be 1; the parameter exists so a future beam search can
    be enabled without changing callers.
    """
    if beam_size != 1:
        raise NotImplementedError("only greedy decoding is implemented in this version")

    slots = outputs.slots
    batch_size, num_slots, slot_dim = slots.shape
    elements = batch["precursor_element_ids"].shape[1]
    shape = (batch_size, num_slots, elements)

    with deterministic(model.formula_decoder):
        counts = model.formula_decoder.greedy_decode(
            slots.reshape(batch_size * num_slots, slot_dim),
            batch["precursor_element_ids"].unsqueeze(1).expand(shape).reshape(-1, elements),
            batch["precursor_element_counts"].unsqueeze(1).expand(shape).reshape(-1, elements),
            batch["precursor_element_mask"].unsqueeze(1).expand(shape).reshape(-1, elements),
        ).view(batch_size, num_slots, elements)

    ion_choice = outputs.ion_log_prob.argmax(dim=-1)

    element_ids = batch["precursor_element_ids"].cpu().numpy()
    counts_np = counts.cpu().numpy()
    ion_np = ion_choice.cpu().numpy()
    presence_np = outputs.presence.detach().cpu().numpy()
    # the rendered amplitude is the gated contribution, matching training
    intensity_np = outputs.contribution.detach().cpu().numpy()

    rendered: list[RenderedSpectrum] = []
    for index in range(batch_size):
        channels = _channel_by_name(batch["adduct"][index])
        merged: dict[tuple[str, str], list[float]] = {}
        for slot in range(num_slots):
            if presence_np[index, slot] < presence_threshold:
                continue
            formula, mass = _counts_to_formula(element_ids[index], counts_np[index, slot])
            if not formula:
                continue
            name = ION_STATE_VOCABULARY[int(ion_np[index, slot])]
            channel = channels.get(name)
            if channel is None:
                continue
            key = (formula, name)
            entry = merged.setdefault(key, [mass + channel.mz_offset, 0.0])
            entry[1] += float(intensity_np[index, slot])
        rendered.append(
            RenderedSpectrum(
                spectrum_uid=batch["spectrum_uid"][index],
                fragments=[
                    RenderedFragment(formula=f, ion_state=s, mz=mz, intensity=value)
                    for (f, s), (mz, value) in sorted(merged.items())
                ],
            )
        )
    return rendered


def bin_spectrum(
    mz: np.ndarray, intensity: np.ndarray, config: BinningConfig | None = None
) -> np.ndarray:
    """Accumulate peaks into ms-pred mass bins.

    Peaks whose bin index falls outside ``[0, num_bins)`` are dropped, not
    clamped: folding an out-of-range mass onto the last bin would invent a peak
    the model never predicted.
    """
    config = config or BinningConfig()
    output = np.zeros(config.num_bins, dtype=np.float64)
    mz = np.asarray(mz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if mz.size == 0:
        return output
    index = bin_index(mz, config)
    keep = (index >= 0) & (index < config.num_bins)
    if not keep.any():
        return output
    np.add.at(output, index[keep], intensity[keep])
    return output


@torch.no_grad()
def slot_diversity(model, batch: dict, outputs) -> dict:
    """Are the slots actually distinct, or has the set collapsed to one answer?

    Counting how many slots clear a presence threshold does not catch this:
    sixty-four identical slots can all report "present".  What matters is
    whether they point in different directions and decode to different
    fragments.
    """
    slots = outputs.slots
    batch_size, num_slots, slot_dim = slots.shape
    normalised = torch.nn.functional.normalize(slots, dim=-1)
    cosine = torch.bmm(normalised, normalised.transpose(1, 2))
    diagonal = torch.eye(num_slots, dtype=torch.bool, device=cosine.device).unsqueeze(0)
    mean_cosine = float(torch.nanmean(cosine.masked_fill(diagonal, float("nan"))))

    elements = batch["precursor_element_ids"].shape[1]
    shape = (batch_size, num_slots, elements)
    with deterministic(model.formula_decoder):
        counts = (
            model.formula_decoder.greedy_decode(
                slots.reshape(batch_size * num_slots, slot_dim),
                batch["precursor_element_ids"].unsqueeze(1).expand(shape).reshape(-1, elements),
                batch["precursor_element_counts"].unsqueeze(1).expand(shape).reshape(-1, elements),
                batch["precursor_element_mask"].unsqueeze(1).expand(shape).reshape(-1, elements),
            )
            .view(batch_size, num_slots, elements)
            .cpu()
            .numpy()
        )
    distinct = [
        len({tuple(counts[index, slot]) for slot in range(num_slots)})
        for index in range(batch_size)
    ]
    return {
        "slot_cosine_mean": mean_cosine,
        "distinct_greedy_formulas_mean": float(np.mean(distinct)),
        "distinct_greedy_formulas_min": int(np.min(distinct)),
        "num_slots": num_slots,
        "intensity_std_across_slots": float(outputs.intensity.std(dim=1).mean()),
        "presence_std_across_slots": float(outputs.presence.std(dim=1).mean()),
    }


@torch.no_grad()
def greedy_identity_metrics(model, batch: dict, outputs, assignment) -> dict:
    """How often the greedy fragment of a matched slot lands inside its bag."""
    if len(assignment) == 0:
        return {
            "argmax_in_candidate_bag": None,
            "unique_candidate_accuracy": None,
            "ambiguous_bag_hit": None,
            "matched_pairs": 0,
        }

    slots = outputs.slots
    batch_size, num_slots, slot_dim = slots.shape
    elements = batch["precursor_element_ids"].shape[1]
    shape = (batch_size, num_slots, elements)
    with deterministic(model.formula_decoder):
        counts = (
            model.formula_decoder.greedy_decode(
                slots.reshape(batch_size * num_slots, slot_dim),
                batch["precursor_element_ids"].unsqueeze(1).expand(shape).reshape(-1, elements),
                batch["precursor_element_counts"].unsqueeze(1).expand(shape).reshape(-1, elements),
                batch["precursor_element_mask"].unsqueeze(1).expand(shape).reshape(-1, elements),
            )
            .view(batch_size, num_slots, elements)
            .cpu()
            .numpy()
        )
    ion_choice = outputs.ion_log_prob.argmax(dim=-1).cpu().numpy()

    candidate_counts = batch["candidate_formula_counts"].cpu().numpy()
    candidate_ions = batch["candidate_ion_states"].cpu().numpy()
    candidate_peak = batch["candidate_to_peak"].cpu().numpy()
    candidate_mask = batch["candidate_mask"].cpu().numpy()

    batches = assignment.batch_index.cpu().numpy()
    slot_ids = assignment.slot_index.cpu().numpy()
    targets = assignment.target_index.cpu().numpy()

    hits = []
    unique_hits = []
    ambiguous_hits = []
    for b, slot, target in zip(batches, slot_ids, targets, strict=True):
        members = np.flatnonzero(candidate_mask[b] & (candidate_peak[b] == target))
        if members.size == 0:
            continue
        same_formula = (candidate_counts[b, members] == counts[b, slot]).all(axis=1)
        same_ion = candidate_ions[b, members] == ion_choice[b, slot]
        hit = bool((same_formula & same_ion).any())
        hits.append(hit)
        (unique_hits if members.size == 1 else ambiguous_hits).append(hit)

    def mean(values: list[bool]) -> float | None:
        return float(np.mean(values)) if values else None

    return {
        "argmax_in_candidate_bag": mean(hits),
        "unique_candidate_accuracy": mean(unique_hits),
        "ambiguous_bag_hit": mean(ambiguous_hits),
        "matched_pairs": len(hits),
    }
