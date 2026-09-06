"""Where does the missing cosine go?

Everything here is **diagnostic only**.  It reads candidate bags and observed
intensities, which the model must never see, so none of it may be used to
produce a prediction.  The contract is enforced by construction: a batch's
predictions are made first, from the model-input subset alone, and are then
frozen before any label is touched.

The decomposition swaps one predicted quantity at a time for its oracle value
and re-scores, so the cosine lost to formula identity, ion state, presence and
intensity can be read off separately rather than argued about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from metabo_sllm.chem.candidates import ION_STATE_VOCABULARY, channels_for_adduct
from metabo_sllm.chem.formula import element_mass, element_symbol, formula_to_string
from metabo_sllm.losses.candidate_scoring import candidate_pairs_for_assignment
from metabo_sllm.losses.matching import NEG, hungarian_assign

__all__ = [
    "CONDITIONS",
    "SpectrumRecord",
    "analyse_batch",
    "beam_decode_formulas",
]

# Every condition below is a counterfactual: "predicted" keeps the model's
# value, "oracle" substitutes the supervision's. Only ``all_predicted`` is a
# real model result.
CONDITIONS = (
    "all_predicted",
    "oracle_identity_predicted_weights",
    "predicted_identity_oracle_weights",
    "oracle_formula_predicted_ion",
    "predicted_formula_oracle_ion",
    "oracle_identity_oracle_weights",
    "oracle_identity_sqrt_weights",
    "presence_forced_on_matched",
)


@dataclass
class SpectrumRecord:
    """One spectrum's predictions, oracles and per-condition peak lists."""

    spectrum_uid: str
    conditions: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    identity: dict = field(default_factory=dict)


# --------------------------------------------------------------------- beams


@torch.no_grad()
def beam_decode_formulas(
    formula_decoder,
    slots: torch.Tensor,
    element_ids: torch.Tensor,
    precursor_counts: torch.Tensor,
    element_mask: torch.Tensor,
    *,
    beam: int = 1,
) -> torch.Tensor:
    """Best formula per slot under a beam of ``beam``, candidate-free.

    Scores come only from the decoder's own masked distribution -- the same one
    greedy uses -- so a wider beam cannot smuggle in knowledge of the answer.
    ``beam=1`` reproduces greedy exactly.
    """
    rows, length = element_ids.shape
    device = slots.device
    counts = torch.zeros(rows, beam, length, dtype=torch.long, device=device)
    previous = torch.full((rows, beam, length), formula_decoder.bos_count, dtype=torch.long, device=device)
    scores = torch.full((rows, beam), NEG, device=device, dtype=torch.float32)
    scores[:, 0] = 0.0  # only the first beam is alive before the first step

    flat = rows * beam
    slot_flat = slots.unsqueeze(1).expand(rows, beam, slots.shape[-1]).reshape(flat, -1)
    element_flat = element_ids.unsqueeze(1).expand(rows, beam, length).reshape(flat, length)
    counts_flat_precursor = precursor_counts.unsqueeze(1).expand(rows, beam, length).reshape(flat, length)
    mask_flat = element_mask.unsqueeze(1).expand(rows, beam, length).reshape(flat, length)

    for step in range(length):
        logits = formula_decoder.logits(
            slot_flat,
            element_flat,
            counts_flat_precursor,
            previous.reshape(flat, length),
            mask_flat,
        )[:, step, :].view(rows, beam, -1)
        log_probs = torch.log_softmax(logits, dim=-1)
        log_probs = torch.nan_to_num(log_probs, neginf=NEG)

        alive = element_mask[:, step]
        candidate_scores = scores.unsqueeze(-1) + log_probs
        vocabulary = candidate_scores.shape[-1]
        flat_scores = candidate_scores.view(rows, beam * vocabulary)
        width = min(beam, flat_scores.shape[1])
        best, index = torch.topk(flat_scores, width, dim=1)
        origin = index // vocabulary
        value = index % vocabulary

        new_counts = torch.gather(
            counts, 1, origin.unsqueeze(-1).expand(rows, width, length)
        )
        new_previous = torch.gather(
            previous, 1, origin.unsqueeze(-1).expand(rows, width, length)
        )
        chosen = torch.where(alive.unsqueeze(1), value, torch.zeros_like(value))
        new_counts[:, :, step] = chosen
        if step + 1 < length:
            new_previous[:, :, step + 1] = chosen
        counts, previous = new_counts, new_previous
        scores = torch.where(alive.unsqueeze(1), best, scores[:, :width])
        # rows whose element is padding keep their beams unchanged

    return counts[:, 0, :] * element_mask.to(counts.dtype)


# ------------------------------------------------------------------ analysis


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


def _peaks(entries: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    if not entries:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    mz = np.asarray([m for m, _ in entries], dtype=np.float64)
    intensity = np.asarray([i for _, i in entries], dtype=np.float64)
    return mz, intensity


@torch.no_grad()
def analyse_batch(
    model,
    batch: dict,
    rows: list,
    predictions: list,
    *,
    presence_threshold: float = 0.5,
    beams: tuple[int, ...] = (),
) -> list[SpectrumRecord]:
    """Build every counterfactual peak list for one batch.

    ``predictions`` must already have been produced from the model-input subset
    alone; this function only reads them.
    """
    outputs = model(batch)
    slots = outputs.slots
    batch_size, num_slots, slot_dim = slots.shape
    elements = batch["precursor_element_ids"].shape[1]
    shape = (batch_size, num_slots, elements)

    greedy = (
        model.formula_decoder.greedy_decode(
            slots.reshape(batch_size * num_slots, slot_dim),
            batch["precursor_element_ids"].unsqueeze(1).expand(shape).reshape(-1, elements),
            batch["precursor_element_counts"].unsqueeze(1).expand(shape).reshape(-1, elements),
            batch["precursor_element_mask"].unsqueeze(1).expand(shape).reshape(-1, elements),
        )
        .view(batch_size, num_slots, elements)
    )
    greedy_log_prob = model.formula_decoder.log_prob(
        slots.reshape(batch_size * num_slots, slot_dim),
        batch["precursor_element_ids"].unsqueeze(1).expand(shape).reshape(-1, elements),
        batch["precursor_element_counts"].unsqueeze(1).expand(shape).reshape(-1, elements),
        greedy.reshape(-1, elements),
        batch["precursor_element_mask"].unsqueeze(1).expand(shape).reshape(-1, elements),
    ).view(batch_size, num_slots)

    beam_counts = {}
    for width in beams:
        beam_counts[width] = (
            beam_decode_formulas(
                model.formula_decoder,
                slots.reshape(batch_size * num_slots, slot_dim),
                batch["precursor_element_ids"].unsqueeze(1).expand(shape).reshape(-1, elements),
                batch["precursor_element_counts"].unsqueeze(1).expand(shape).reshape(-1, elements),
                batch["precursor_element_mask"].unsqueeze(1).expand(shape).reshape(-1, elements),
                beam=width,
            )
            .view(batch_size, num_slots, elements)
            .cpu()
            .numpy()
        )

    # ---- labels enter only from here on, and only for diagnostics ----
    cost = model.compute_matching_cost_no_grad(
        slots, outputs.presence, outputs.intensity, batch
    )
    assignment = hungarian_assign(cost, batch["target_peak_mask"])
    ion_log_prob = model.ion_head(slots, batch["admissible_ion_mask"])

    pair_batch, pair_candidate, pair_slot, pair_index = candidate_pairs_for_assignment(
        batch, assignment
    )
    if pair_batch.numel():
        pair_scores = (
            model.formula_decoder.log_prob(
                slots[pair_batch, pair_slot],
                batch["precursor_element_ids"][pair_batch],
                batch["precursor_element_counts"][pair_batch],
                batch["candidate_formula_counts"][pair_batch, pair_candidate],
                batch["precursor_element_mask"][pair_batch],
            )
            + ion_log_prob[
                pair_batch, pair_slot, batch["candidate_ion_states"][pair_batch, pair_candidate]
            ]
        )
    else:
        pair_scores = torch.zeros(0, device=slots.device)

    element_ids = batch["precursor_element_ids"].cpu().numpy()
    candidate_counts = batch["candidate_formula_counts"].cpu().numpy()
    candidate_ions = batch["candidate_ion_states"].cpu().numpy()
    presence = outputs.presence.cpu().numpy()
    contribution = outputs.contribution.cpu().numpy()
    ion_choice = outputs.ion_log_prob.argmax(dim=-1).cpu().numpy()
    greedy_np = greedy.cpu().numpy()
    greedy_lp = greedy_log_prob.cpu().numpy()
    ion_lp = ion_log_prob.cpu().numpy()
    a_batch = assignment.batch_index.cpu().numpy()
    a_slot = assignment.slot_index.cpu().numpy()
    a_target = assignment.target_index.cpu().numpy()
    peak_indices = batch["target_peak_indices"].cpu().numpy()
    pair_batch_np = pair_batch.cpu().numpy()
    pair_candidate_np = pair_candidate.cpu().numpy()
    pair_index_np = pair_index.cpu().numpy()
    pair_scores_np = pair_scores.cpu().numpy()

    records: list[SpectrumRecord] = []
    for index, (row, prediction) in enumerate(zip(rows, predictions, strict=True)):
        channels = {c.name: c for c in channels_for_adduct(str(row["adduct"]))}
        raw = np.asarray(row["intensities"], dtype=np.float64)
        observed_mz = np.asarray(row["mzs"], dtype=np.float64)
        root = np.sqrt(np.clip(raw, 0.0, None))
        transformed = root / (np.sqrt(np.square(root).sum()) + 1e-12)

        keep = a_batch == index
        slot_of_target: dict[int, int] = {}
        for slot, target in zip(a_slot[keep], a_target[keep], strict=True):
            slot_of_target[int(target)] = int(slot)
        target_of_slot = {slot: target for target, slot in slot_of_target.items()}

        # best bag member per matched pair, by the slot's own likelihood
        oracle_for_slot: dict[int, tuple[str, str, float]] = {}
        pair_rows = np.flatnonzero(pair_batch_np == index)
        if pair_rows.size:
            for pair in np.unique(pair_index_np[pair_rows]):
                members = pair_rows[pair_index_np[pair_rows] == pair]
                best = members[int(np.argmax(pair_scores_np[members]))]
                candidate = int(pair_candidate_np[best])
                formula, mass = _counts_to_formula(
                    element_ids[index], candidate_counts[index, candidate]
                )
                ion = ION_STATE_VOCABULARY[int(candidate_ions[index, candidate])]
                channel = channels.get(ion)
                mz = mass + channel.mz_offset if channel else float("nan")
                # pair_index is global across the batch, so index the assignment
                # arrays directly rather than a per-spectrum slice.
                oracle_for_slot[int(a_slot[int(pair)])] = (formula, ion, mz)

        active = [s for s in range(num_slots) if presence[index, s] >= presence_threshold]
        predicted_identity: dict[int, tuple[str, str, float]] = {}
        for slot in range(num_slots):
            formula, mass = _counts_to_formula(element_ids[index], greedy_np[index, slot])
            ion = ION_STATE_VOCABULARY[int(ion_choice[index, slot])]
            channel = channels.get(ion)
            if formula and channel is not None:
                predicted_identity[slot] = (formula, ion, mass + channel.mz_offset)

        def emit(slot_set, identity_source, weight_source) -> list[tuple[float, float]]:
            peaks = []
            for slot in slot_set:
                identity = identity_source(slot)
                if identity is None:
                    continue
                weight = weight_source(slot)
                if weight is None or weight <= 0:
                    continue
                peaks.append((identity[2], weight))
            return peaks

        def predicted_id(slot):
            return predicted_identity.get(slot)

        def oracle_id(slot):
            return oracle_for_slot.get(slot, predicted_identity.get(slot))

        def mixed_formula_oracle(slot):
            """Oracle formula, predicted ion state."""
            oracle = oracle_for_slot.get(slot)
            if oracle is None:
                return predicted_identity.get(slot)
            predicted = predicted_identity.get(slot)
            ion = predicted[1] if predicted else oracle[1]
            channel = channels.get(ion)
            if channel is None:
                return oracle
            mass = oracle[2] - channels[oracle[1]].mz_offset
            return (oracle[0], ion, mass + channel.mz_offset)

        def mixed_ion_oracle(slot):
            """Predicted formula, oracle ion state."""
            predicted = predicted_identity.get(slot)
            oracle = oracle_for_slot.get(slot)
            if predicted is None or oracle is None:
                return predicted
            channel = channels.get(oracle[1])
            if channel is None:
                return predicted
            mass = predicted[2] - channels[predicted[1]].mz_offset
            return (predicted[0], oracle[1], mass + channel.mz_offset)

        def predicted_weight(slot):
            return float(contribution[index, slot])

        def oracle_weight(slot):
            target = target_of_slot.get(slot)
            if target is None:
                return None
            peak = int(peak_indices[index, target])
            return float(raw[peak]) if peak < raw.size else None

        def oracle_weight_sqrt(slot):
            target = target_of_slot.get(slot)
            if target is None:
                return None
            peak = int(peak_indices[index, target])
            return float(transformed[peak]) if peak < transformed.size else None

        matched_slots = sorted(target_of_slot)
        conditions = {
            "all_predicted": _peaks(emit(active, predicted_id, predicted_weight)),
            "oracle_identity_predicted_weights": _peaks(
                emit(active, oracle_id, predicted_weight)
            ),
            "predicted_identity_oracle_weights": _peaks(
                emit(
                    [s for s in active if s in target_of_slot],
                    predicted_id,
                    lambda s: oracle_weight(s)
                    if predicted_identity.get(s)
                    and oracle_for_slot.get(s)
                    and predicted_identity[s][:2] == oracle_for_slot[s][:2]
                    else None,
                )
            ),
            "oracle_formula_predicted_ion": _peaks(
                emit(active, mixed_formula_oracle, predicted_weight)
            ),
            "predicted_formula_oracle_ion": _peaks(
                emit(active, mixed_ion_oracle, predicted_weight)
            ),
            "oracle_identity_oracle_weights": _peaks(
                emit(matched_slots, oracle_id, oracle_weight)
            ),
            "oracle_identity_sqrt_weights": _peaks(
                emit(matched_slots, oracle_id, oracle_weight_sqrt)
            ),
            "presence_forced_on_matched": _peaks(
                emit(sorted(set(active) | set(matched_slots)), predicted_id, predicted_weight)
            ),
        }
        for width, decoded in beam_counts.items():
            entries = []
            for slot in active:
                formula, mass = _counts_to_formula(element_ids[index], decoded[index, slot])
                ion = ION_STATE_VOCABULARY[int(ion_choice[index, slot])]
                channel = channels.get(ion)
                if formula and channel is not None:
                    entries.append((mass + channel.mz_offset, float(contribution[index, slot])))
            conditions[f"beam{width}"] = _peaks(entries)

        # identity-level diagnostics for the matched pairs
        bags = {}
        for pair_row in pair_rows:
            pair = int(pair_index_np[pair_row])
            candidate = int(pair_candidate_np[pair_row])
            formula, _ = _counts_to_formula(element_ids[index], candidate_counts[index, candidate])
            ion = ION_STATE_VOCABULARY[int(candidate_ions[index, candidate])]
            bags.setdefault(pair, []).append(((formula, ion), float(pair_scores_np[pair_row])))

        identity = {
            "matched_pairs": int(keep.sum()),
            "greedy_in_bag": 0,
            "greedy_formula_in_bag": 0,
            "greedy_exact_oracle": 0,
            "unique_pairs": 0,
            "unique_in_bag": 0,
            "ambiguous_pairs": 0,
            "ambiguous_in_bag": 0,
            "bag_log_prob": [],
            "best_in_bag": [],
            "greedy_log_prob": [],
            "element_correct": 0,
            "element_total": 0,
            "element_abs_error": 0.0,
            "position_correct": [],
            "position_total": [],
        }
        for pair, members in bags.items():
            slot = int(a_slot[int(pair)])
            names = {name for name, _ in members}
            formulas = {name[0] for name, _ in members}
            predicted = predicted_identity.get(slot)
            if predicted is not None:
                if predicted[:2] in names:
                    identity["greedy_in_bag"] += 1
                if predicted[0] in formulas:
                    identity["greedy_formula_in_bag"] += 1
            oracle = oracle_for_slot.get(slot)
            if oracle is not None and predicted is not None and predicted[:2] == oracle[:2]:
                identity["greedy_exact_oracle"] += 1
            bucket = "unique" if len(names) == 1 else "ambiguous"
            identity[f"{bucket}_pairs"] += 1
            if predicted is not None and predicted[:2] in names:
                identity[f"{bucket}_in_bag"] += 1
            scores_here = [score for _, score in members]
            identity["best_in_bag"].append(max(scores_here))
            identity["bag_log_prob"].append(
                float(torch.logsumexp(torch.tensor(scores_here), dim=0))
            )
            identity["greedy_log_prob"].append(
                float(greedy_lp[index, slot] + ion_lp[index, slot, int(ion_choice[index, slot])])
            )
            # per-element and per-position agreement with the oracle formula
            oracle_candidate = max(members, key=lambda item: item[1])[0]
            oracle_counts = None
            for pair_row in pair_rows:
                if int(pair_index_np[pair_row]) != pair:
                    continue
                candidate = int(pair_candidate_np[pair_row])
                formula, _ = _counts_to_formula(
                    element_ids[index], candidate_counts[index, candidate]
                )
                ion = ION_STATE_VOCABULARY[int(candidate_ions[index, candidate])]
                if (formula, ion) == oracle_candidate:
                    oracle_counts = candidate_counts[index, candidate]
                    break
            if oracle_counts is None:
                continue
            mask = element_ids[index] > 0
            predicted_counts = greedy_np[index, slot]
            agree = (predicted_counts == oracle_counts) & mask
            identity["element_correct"] += int(agree.sum())
            identity["element_total"] += int(mask.sum())
            identity["element_abs_error"] += float(
                np.abs(predicted_counts[mask] - oracle_counts[mask]).sum()
            )
            for position in range(int(mask.sum())):
                while len(identity["position_correct"]) <= position:
                    identity["position_correct"].append(0)
                    identity["position_total"].append(0)
                identity["position_correct"][position] += int(
                    predicted_counts[position] == oracle_counts[position]
                )
                identity["position_total"][position] += 1

        records.append(
            SpectrumRecord(
                spectrum_uid=str(row["spectrum_uid"]),
                conditions=conditions,
                identity=identity,
                meta={
                    "observed_mz": observed_mz,
                    "observed_intensity": raw,
                    "targets": int(batch["target_peak_mask"][index].sum()),
                    "active_slots": len(active),
                    "adduct": str(row["adduct"]),
                    "instrument": str(row["instrument"]),
                    "collision_energy": float(row["collision_energy"]),
                    "precursor_mz": float(row["precursor_mz"]),
                    "peaks": int(observed_mz.size),
                    "unique_pairs": identity["unique_pairs"],
                    "ambiguous_pairs": identity["ambiguous_pairs"],
                    "predicted_fragments": len(prediction.fragments),
                },
            )
        )
    return records
