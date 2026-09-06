#!/usr/bin/env python3
"""Correctness checks that have to pass before any real training run.

Three questions, each answered by measurement rather than argument:

``analyse``  where the spectrum cosine actually goes, and whether the number of
             active slots matches the number of supervision targets.
``stress``   whether a batch of genuinely demanding spectra -- including one at
             the 64-target cap and one with the largest candidate bag -- runs
             forward and backward.

Everything reads the train fold only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
_SCRIPTS = Path(__file__).resolve().parent
for entry in (str(_SRC), str(_SCRIPTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from metabo_sllm.data.fragment_collator import (  # noqa: E402
    INTENSITY_EPS,
    select_targets,
)
from metabo_sllm.data.fragment_dataset import FragmentSupervisionDataset  # noqa: E402
from metabo_sllm.data.supervision import FIXED_SLOT_COUNT  # noqa: E402
from metabo_sllm.losses.fragment_losses import spectrum_cosine_loss  # noqa: E402
from metabo_sllm.model.fragment_latent_model import training_step  # noqa: E402
from metabo_sllm.rendering.spectrum import deterministic  # noqa: E402
from smoke_fragment_model import build, seed_everything, to_device  # noqa: E402

TRAIN_FOLD = "train"
COSINE_EPS = 1e-8


def summarise(values: list[float] | np.ndarray) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        "mean": float(array.mean()),
        "min": float(array.min()),
        "p50": float(np.percentile(array, 50)),
        "max": float(array.max()),
    }


def cosine(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = np.linalg.norm(prediction) * np.linalg.norm(target) + COSINE_EPS
    return float(prediction @ target / denominator)


def train_reference(config, args, device):
    """Reproduce the tiny-overfit checkpoint: same seed, spectra, steps, lr."""
    seed_everything(config.runtime.seed)
    dataset = FragmentSupervisionDataset(
        config.data.root,
        TRAIN_FOLD,
        shards=[0],
        limit=args.num_spectra,
        exclude_zero_target=config.data.exclude_zero_target_train,
    )
    model, collator, weights = build(config)
    model.to(device).train()
    rows = [dataset[index] for index in range(args.num_spectra)]
    groups = [
        rows[start : start + args.batch_size]
        for start in range(0, len(rows), args.batch_size)
    ]
    batches = [to_device(collator(group), device) for group in groups]

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(
        trainable, lr=config.optim.lr, weight_decay=config.optim.weight_decay
    )
    for _ in range(args.steps):
        for batch in batches:
            optimiser.zero_grad(set_to_none=True)
            _, losses = training_step(model, batch, weights)
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, config.optim.grad_clip)
            optimiser.step()
    return model, weights, groups, batches


# --------------------------------------------------------------- spectrum loss


def decompose_spectrum(model, weights, groups, batches) -> dict:
    """Split the spectrum cosine into oracle, forced-presence and gated parts."""
    records: list[dict] = []
    implementation_means: list[tuple[float, int]] = []

    with torch.no_grad():
        for group, batch in zip(groups, batches, strict=True):
            outputs, _ = training_step(model, batch, weights)
            assignment = outputs.extras["assignment"]
            implementation = float(
                spectrum_cosine_loss(
                    outputs.contribution,
                    assignment,
                    batch["target_peak_indices"],
                    batch["full_peak_intensities"],
                    batch["full_peak_mask"],
                )
            )
            implementation_means.append((implementation, len(group)))

            batch_index = assignment.batch_index.cpu().numpy()
            slot_index = assignment.slot_index.cpu().numpy()
            target_index = assignment.target_index.cpu().numpy()
            intensity = outputs.intensity.cpu().numpy()
            contribution = outputs.contribution.cpu().numpy()
            presence = outputs.presence.cpu().numpy()
            peak_indices = batch["target_peak_indices"].cpu().numpy()
            collator_target = batch["full_peak_intensities"].cpu().numpy()
            peak_mask = batch["full_peak_mask"].cpu().numpy()

            for local, row in enumerate(group):
                raw = np.asarray(row["intensities"], dtype=np.float64)
                # independent recomputation of the target transform
                root = np.sqrt(np.clip(raw, 0.0, None))
                target = root / (np.sqrt(np.square(root).sum()) + INTENSITY_EPS)
                selected = select_targets(np.asarray(row["supervision_rank"]))

                oracle_closed_form = float(np.sqrt(np.square(target[selected]).sum()))
                oracle_vector = np.zeros_like(target)
                oracle_vector[selected] = target[selected]
                oracle_from_vector = cosine(oracle_vector, target)

                keep = batch_index == local
                slots = slot_index[keep]
                targets = target_index[keep]
                peaks = peak_indices[local, targets]

                forced = np.zeros_like(target)
                gated = np.zeros_like(target)
                forced[peaks] = intensity[local, slots]
                gated[peaks] = contribution[local, slots]

                stored = collator_target[local][peak_mask[local]]
                records.append(
                    {
                        "spectrum_uid": str(row["spectrum_uid"]),
                        "peaks": int(target.size),
                        "targets": int(selected.size),
                        "target_transform_max_abs_diff": float(
                            np.abs(stored[: target.size] - target).max()
                        ),
                        "oracle_cosine": oracle_closed_form,
                        "oracle_cosine_from_vector": oracle_from_vector,
                        "oracle_formula_vs_vector_gap": abs(
                            oracle_closed_form - oracle_from_vector
                        ),
                        "forced_presence_cosine": cosine(forced, target),
                        "predicted_contribution_cosine": cosine(gated, target),
                        "matched": int(slots.size),
                        "mean_presence_matched": float(presence[local, slots].mean())
                        if slots.size
                        else float("nan"),
                    }
                )

    for record in records:
        record["current_spectrum_cosine"] = record["predicted_contribution_cosine"]
        record["oracle_minus_forced_gap"] = (
            record["oracle_cosine"] - record["forced_presence_cosine"]
        )
        record["forced_minus_current_gap"] = (
            record["forced_presence_cosine"] - record["current_spectrum_cosine"]
        )
        record["delta_spec"] = record["oracle_cosine"] - record["current_spectrum_cosine"]

    total = sum(count for _, count in implementation_means)
    implementation_loss = sum(v * c for v, c in implementation_means) / total
    forced_mean = float(np.mean([r["forced_presence_cosine"] for r in records]))
    gated_mean = float(np.mean([r["predicted_contribution_cosine"] for r in records]))
    oracle_mean = float(np.mean([r["oracle_cosine"] for r in records]))

    return {
        "per_spectrum": records,
        "means": {
            "oracle_cosine": oracle_mean,
            "forced_presence_cosine": forced_mean,
            "predicted_contribution_cosine": gated_mean,
            "current_spectrum_cosine": gated_mean,
            "oracle_minus_forced_gap": oracle_mean - forced_mean,
            "forced_minus_current_gap": forced_mean - gated_mean,
            "delta_spec": oracle_mean - gated_mean,
        },
        "implementation": {
            "spectrum_loss_returned": implementation_loss,
            "one_minus_forced_presence_cosine": 1.0 - forced_mean,
            "one_minus_gated_cosine": 1.0 - gated_mean,
            "gap_vs_forced": abs(implementation_loss - (1.0 - forced_mean)),
            "gap_vs_gated": abs(implementation_loss - (1.0 - gated_mean)),
            "predicted_contribution_is": "sigmoid(presence_logit) * softplus(intensity_logit)",
        },
        "checks": {
            "oracle_two_ways_agree": max(
                r["oracle_formula_vs_vector_gap"] for r in records
            )
            <= 1e-6,
            "target_transform_matches_collator": max(
                r["target_transform_max_abs_diff"] for r in records
            )
            <= 1e-6,
            "implementation_matches_gated": abs(implementation_loss - (1.0 - gated_mean))
            <= 1e-6,
            "delta_spec_within_threshold": (oracle_mean - gated_mean) <= 0.03,
        },
        "max_oracle_two_way_gap": max(r["oracle_formula_vs_vector_gap"] for r in records),
        "max_target_transform_gap": max(
            r["target_transform_max_abs_diff"] for r in records
        ),
    }


# ------------------------------------------------------------------ slot audit


def audit_slots(model, weights, groups, batches) -> dict:
    """Compare active slots against the number of supervision targets."""
    per_spectrum: list[dict] = []
    with torch.no_grad():
        for group, batch in zip(groups, batches, strict=True):
            outputs, _ = training_step(model, batch, weights)
            assignment = outputs.extras["assignment"]
            slots = outputs.slots
            batch_size, num_slots, slot_dim = slots.shape
            elements = batch["precursor_element_ids"].shape[1]
            shape = (batch_size, num_slots, elements)
            with deterministic(model.formula_decoder):
                counts = (
                    model.formula_decoder.greedy_decode(
                        slots.reshape(batch_size * num_slots, slot_dim),
                        batch["precursor_element_ids"]
                        .unsqueeze(1)
                        .expand(shape)
                        .reshape(-1, elements),
                        batch["precursor_element_counts"]
                        .unsqueeze(1)
                        .expand(shape)
                        .reshape(-1, elements),
                        batch["precursor_element_mask"]
                        .unsqueeze(1)
                        .expand(shape)
                        .reshape(-1, elements),
                    )
                    .view(batch_size, num_slots, elements)
                    .cpu()
                    .numpy()
                )
            ion_choice = outputs.ion_log_prob.argmax(dim=-1).cpu().numpy()
            presence = outputs.presence.cpu().numpy()
            candidate_counts = batch["candidate_formula_counts"].cpu().numpy()
            candidate_ions = batch["candidate_ion_states"].cpu().numpy()
            candidate_peak = batch["candidate_to_peak"].cpu().numpy()
            candidate_mask = batch["candidate_mask"].cpu().numpy()
            target_mask = batch["target_peak_mask"].cpu().numpy()

            batch_index = assignment.batch_index.cpu().numpy()
            slot_index = assignment.slot_index.cpu().numpy()
            target_index = assignment.target_index.cpu().numpy()

            for local, row in enumerate(group):
                targets = int(target_mask[local].sum())
                active = presence[local] > 0.5
                keep = batch_index == local
                matched_slots = slot_index[keep]
                matched_targets = target_index[keep]

                matched_active = int(active[matched_slots].sum()) if matched_slots.size else 0
                unmatched = np.setdiff1d(np.arange(num_slots), matched_slots)
                false_positive = (
                    float(active[unmatched].mean()) if unmatched.size else float("nan")
                )

                active_ids = np.flatnonzero(active)
                active_formulas = [tuple(counts[local, s]) for s in active_ids]
                distinct_active = len(set(active_formulas))
                empty_active = sum(1 for f in active_formulas if sum(f) == 0)

                in_bag = []
                for slot, target in zip(matched_slots, matched_targets, strict=True):
                    members = np.flatnonzero(
                        candidate_mask[local] & (candidate_peak[local] == target)
                    )
                    if members.size == 0:
                        continue
                    same = (candidate_counts[local, members] == counts[local, slot]).all(axis=1)
                    in_bag.append(
                        bool((same & (candidate_ions[local, members] == ion_choice[local, slot])).any())
                    )

                matched_formulas = [tuple(counts[local, s]) for s in matched_slots]
                per_spectrum.append(
                    {
                        "spectrum_uid": str(row["spectrum_uid"]),
                        "targets": targets,
                        "active_slots": int(active.sum()),
                        "matched_slots": int(matched_slots.size),
                        "matched_presence_recall": matched_active / targets if targets else None,
                        "unmatched_false_positive_rate": false_positive,
                        "distinct_active_formulas": distinct_active,
                        "active_formula_duplicate_ratio": (
                            1.0 - distinct_active / len(active_formulas)
                            if active_formulas
                            else None
                        ),
                        "distinct_matched_formulas": len(set(matched_formulas)),
                        "matched_formula_duplicate_ratio": (
                            1.0 - len(set(matched_formulas)) / len(matched_formulas)
                            if matched_formulas
                            else None
                        ),
                        "matched_identity_in_bag": float(np.mean(in_bag)) if in_bag else None,
                        "active_empty_formulas": empty_active,
                    }
                )

    def column(name):
        return [r[name] for r in per_spectrum if r[name] is not None]

    return {
        "per_spectrum": per_spectrum,
        "targets_per_spectrum": summarise(column("targets")),
        "active_slots_per_spectrum": summarise(column("active_slots")),
        "matched_presence_recall": summarise(column("matched_presence_recall")),
        "unmatched_false_positive_rate": summarise(column("unmatched_false_positive_rate")),
        "distinct_active_formulas": summarise(column("distinct_active_formulas")),
        "active_formula_duplicate_ratio": summarise(column("active_formula_duplicate_ratio")),
        "matched_formula_duplicate_ratio": summarise(column("matched_formula_duplicate_ratio")),
        "matched_identity_in_bag": summarise(column("matched_identity_in_bag")),
        "active_empty_formulas_total": int(sum(column("active_empty_formulas"))),
        "active_vs_targets_mean_ratio": (
            float(np.mean(column("active_slots")) / np.mean(column("targets")))
            if column("targets")
            else None
        ),
    }


def run_analyse(args, config) -> dict:
    device = torch.device(args.device)
    model, weights, groups, batches = train_reference(config, args, device)
    model.eval()
    report = {
        "fold": TRAIN_FOLD,
        "test_fold_read": False,
        "num_spectra": args.num_spectra,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "seed": config.runtime.seed,
        "learning_rate": config.optim.lr,
        "spectrum_decomposition": decompose_spectrum(model, weights, groups, batches),
        "slot_audit": audit_slots(model, weights, groups, batches),
    }
    return report


# ----------------------------------------------------------------- stress test


def pick_stress_rows(config, args) -> tuple[list[dict], list[dict]]:
    """Choose spectra that actually exercise the 64-slot budget."""
    dataset = FragmentSupervisionDataset(
        config.data.root, TRAIN_FOLD, shards=list(range(args.scan_shards))
    )
    profiles = []
    for index in range(len(dataset)):
        row = dataset[index]
        selected = select_targets(np.asarray(row["supervision_rank"]))
        if selected.size == 0:
            continue
        peak_to_target = np.full(np.asarray(row["mzs"]).size, -1, dtype=np.int64)
        peak_to_target[selected] = np.arange(selected.size)
        edge_peak = np.asarray(row["edge_peak_index"])
        keep = peak_to_target[edge_peak] >= 0 if edge_peak.size else np.zeros(0, bool)
        linked_targets = peak_to_target[edge_peak[keep]] if keep.any() else np.zeros(0, int)
        per_target = np.bincount(linked_targets, minlength=selected.size) if keep.any() else np.zeros(selected.size, int)
        profiles.append(
            {
                "index": index,
                "spectrum_uid": str(row["spectrum_uid"]),
                "targets": int(selected.size),
                "linked_candidates": int(keep.sum()),
                "ambiguous_targets": int((per_target >= 2).sum()),
                "total_candidates_in_row": len(row["candidate_ion_state"]),
                "peaks": int(np.asarray(row["mzs"]).size),
            }
        )

    chosen: dict[str, dict] = {}

    def take(label, candidates, key, reverse=False):
        if not candidates:
            return
        best = sorted(candidates, key=key, reverse=reverse)[0]
        chosen.setdefault(best["spectrum_uid"], {**best, "why": label})

    take("targets_1_to_8", [p for p in profiles if 1 <= p["targets"] <= 8], lambda p: -p["targets"])
    take("targets_near_32", profiles, lambda p: abs(p["targets"] - 32))
    take("targets_at_cap", [p for p in profiles if p["targets"] >= FIXED_SLOT_COUNT], lambda p: -p["linked_candidates"])
    take("most_ambiguous", profiles, lambda p: p["ambiguous_targets"], reverse=True)
    take("most_linked_candidates", profiles, lambda p: p["linked_candidates"], reverse=True)

    selection = list(chosen.values())
    rows = [dataset[entry["index"]] for entry in selection]
    return rows, selection


def run_stress(args, config) -> dict:
    device = torch.device(args.device)
    seed_everything(config.runtime.seed)
    rows, selection = pick_stress_rows(config, args)
    model, collator, weights = build(config)
    model.to(device).train()
    batch = to_device(collator(rows), device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    outputs, losses = training_step(model, batch, weights)
    forward_seconds = time.perf_counter() - started
    started = time.perf_counter()
    losses.total.backward()
    backward_seconds = time.perf_counter() - started

    tensorised = int(batch["candidate_mask"].sum())
    per_row_tensorised = batch["candidate_mask"].sum(dim=1).cpu().tolist()
    stored_total = sum(entry["total_candidates_in_row"] for entry in selection)

    base_grads = sum(1 for _, p in model.encoder.base_parameters() if p.grad is not None)
    lora_grads = sum(1 for _, p in model.encoder.lora_parameters() if p.grad is not None)
    new_grads = sum(
        1
        for module in (
            model.slot_decoder,
            model.formula_decoder,
            model.ion_head,
            model.presence_head,
            model.intensity_head,
        )
        for p in module.parameters()
        if p.grad is not None
    )

    values = {
        "total": float(losses.total.item()),
        "identity_bag_nll": float(losses.identity.item()),
        "presence": float(losses.presence.item()),
        "intensity": float(losses.intensity.item()),
        "spectrum": float(losses.spectrum.item()),
    }
    report = {
        "fold": TRAIN_FOLD,
        "test_fold_read": False,
        "selection": selection,
        "batch_size": len(rows),
        "shapes": {
            "input_ids": list(batch["input_ids"].shape),
            "slots": list(outputs.slots.shape),
            "presence": list(outputs.presence.shape),
            "intensity": list(outputs.intensity.shape),
            "matched_bag_nll": list(outputs.extras["matched_bag_nll"].shape),
            "matching_cost": list(outputs.extras["matching_cost"].shape),
            "full_peak_intensities": list(batch["full_peak_intensities"].shape),
        },
        "full_candidate_scores_in_output": "candidate_log_prob" in outputs.extras,
        "cost_profile": batch["cost_profile"],
        "gradient_candidate_slot_pairs": int(
            sum(int(v) for v in batch["candidate_mask"].sum(dim=1).cpu().tolist())
        ),
        "targets_per_row": batch["target_peak_mask"].sum(dim=1).cpu().tolist(),
        "candidates_tensorised_per_row": per_row_tensorised,
        "candidates_tensorised_total": tensorised,
        "candidates_stored_in_rows_total": stored_total,
        "candidates_left_on_disk": stored_total - tensorised,
        "losses": values,
        "all_losses_finite": all(np.isfinite(v) for v in values.values()),
        "no_nan_or_inf_in_outputs": bool(
            torch.isfinite(outputs.slots).all()
            and torch.isfinite(outputs.presence).all()
            and torch.isfinite(outputs.intensity).all()
            and torch.isfinite(outputs.extras["matched_bag_nll"]).all()
            and torch.isfinite(outputs.extras["matching_cost"]).all()
        ),
        "matched_pairs": losses.extras["matched_pairs"],
        "gradients": {
            "qwen_base_params_with_grad": base_grads,
            "lora_params_with_grad": lora_grads,
            "new_module_params_with_grad": new_grads,
        },
        "timing_seconds": {
            "forward": round(forward_seconds, 4),
            "backward": round(backward_seconds, 4),
        },
    }
    if device.type == "cuda":
        report["peak_gpu_memory_gib"] = round(
            torch.cuda.max_memory_allocated(device) / 2**30, 3
        )
    return report


def run_zero_targets(args, config) -> dict:
    """How much of the train fold has no supervision target at all."""
    dataset = FragmentSupervisionDataset(config.data.root, TRAIN_FOLD)
    stats = dict(dataset.zero_target_stats)
    kept = FragmentSupervisionDataset(config.data.root, TRAIN_FOLD, exclude_zero_target=True)
    stats["rows_after_exclusion"] = len(kept)
    stats["fold"] = TRAIN_FOLD
    stats["test_fold_read"] = False
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["analyse", "stress", "zero-targets"])
    parser.add_argument("--config", default="configs/model/qwen_formula_slots_v0.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-spectra", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--scan-shards", type=int, default=4)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    config = OmegaConf.load(args.config)
    runners = {"analyse": run_analyse, "stress": run_stress, "zero-targets": run_zero_targets}
    report = runners[args.mode](args, config)
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
