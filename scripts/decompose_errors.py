#!/usr/bin/env python3
"""Error decomposition for a trained checkpoint. Diagnostic only; nothing is trained.

Predictions are generated once, from the model-input subset alone, and frozen.
Only afterwards are candidate bags and observed intensities opened, to build
counterfactuals that replace one predicted quantity at a time with its oracle
value. Oracle numbers are ceilings, not model performance, and are labelled as
such everywhere.

The test fold is never opened.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from metabo_sllm.data.fragment_collator import FragmentCollator  # noqa: E402
from metabo_sllm.data.fragment_dataset import FragmentSupervisionDataset  # noqa: E402
from metabo_sllm.evaluation.error_decomposition import (  # noqa: E402
    CONDITIONS,
    analyse_batch,
)
from metabo_sllm.evaluation.inference import (  # noqa: E402
    MODEL_INPUT_FIELDS,
    assert_no_leakage,
    predict_spectrum,
)
from metabo_sllm.evaluation.spectrum_metrics import (  # noqa: E402
    BinningConfig,
    score_prediction,
)
from metabo_sllm.model.qwen_encoder import load_tokenizer  # noqa: E402
from metabo_sllm.training.checkpoint import load_checkpoint  # noqa: E402
from metabo_sllm.training.trainer import configure_attention_backends  # noqa: E402
from smoke_fragment_model import build as build_model_parts  # noqa: E402
from smoke_fragment_model import seed_everything  # noqa: E402

FORBIDDEN_FOLD = "test"


def distribution(values: np.ndarray) -> dict:
    if values.size == 0:
        return {}
    return {
        "mean": float(values.mean()),
        "median": float(np.percentile(values, 50)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def bucket(value: float, edges: list[float]) -> str:
    for low, high in zip([-np.inf, *edges], [*edges, np.inf], strict=True):
        if low <= value < high:
            low_text = "-inf" if low == -np.inf else f"{low:g}"
            high_text = "inf" if high == np.inf else f"{high:g}"
            return f"[{low_text},{high_text})"
    return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/train/qwen_formula_slots_v0_pilot.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", default="valid")
    parser.add_argument("--out", required=True, help="diagnostics directory")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--beams", default="1,4,16")
    args = parser.parse_args(argv)
    if args.fold == FORBIDDEN_FOLD:
        raise SystemExit("error: the test fold is not available to this tool")

    beams = tuple(int(b) for b in args.beams.split(",") if b)
    config = OmegaConf.load(args.config)
    config.model = OmegaConf.load(_ROOT / config.model_config)
    seed_everything(config.training.seed)
    configure_attention_backends()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _, _ = build_model_parts(config.model)
    model.to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.eval()
    tokenizer = load_tokenizer(config.model.encoder.model_name_or_path)
    collator = FragmentCollator(tokenizer, max_text_length=config.data.max_text_length)
    dataset = FragmentSupervisionDataset(config.data.root, args.fold, limit=args.limit)
    binning = BinningConfig(
        upper_limit=config.evaluation.upper_limit,
        num_bins=config.evaluation.num_bins,
        min_pred_intensity=config.evaluation.min_pred_intensity,
        top_k=tuple(config.evaluation.top_k),
    )
    threshold = config.evaluation.presence_threshold

    names = list(CONDITIONS) + [f"beam{b}" for b in beams]
    cosines: dict[str, list[float]] = {name: [] for name in names}
    cosines20: dict[str, list[float]] = {name: [] for name in names}
    zero_counts: dict[str, int] = {name: 0 for name in names}
    peak_counts: dict[str, list[int]] = {name: [] for name in names}
    per_row: list[dict] = []
    identity_totals: dict[str, float] = defaultdict(float)
    position_correct: list[int] = []
    position_total: list[int] = []
    bag_log_prob: list[float] = []
    best_in_bag: list[float] = []
    greedy_log_prob: list[float] = []

    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(dataset), args.batch_size):
            rows = [dataset[i] for i in range(start, min(start + args.batch_size, len(dataset)))]
            batch = collator(rows)
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            # Predictions are made from the permitted subset only, then frozen.
            model_inputs = {k: v for k, v in batch.items() if k in MODEL_INPUT_FIELDS}
            assert_no_leakage(model_inputs)
            predictions = predict_spectrum(model, model_inputs, presence_threshold=threshold)

            records = analyse_batch(
                model, batch, rows, predictions, presence_threshold=threshold, beams=beams
            )
            for row, record in zip(rows, records, strict=True):
                observed_mz = record.meta["observed_mz"]
                observed_intensity = record.meta["observed_intensity"]
                entry = {"spectrum_uid": record.spectrum_uid}
                for name in names:
                    mz, intensity = record.conditions[name]
                    score = score_prediction(
                        record.spectrum_uid, mz, intensity, observed_mz, observed_intensity, binning
                    )
                    cosines[name].append(score.cosine[100])
                    cosines20[name].append(score.cosine[20])
                    zero_counts[name] += int(score.zero_prediction)
                    peak_counts[name].append(score.predicted_peaks)
                    entry[f"cos100_{name}"] = score.cosine[100]
                    entry[f"cos20_{name}"] = score.cosine[20]
                identity = record.identity
                for key in (
                    "matched_pairs", "greedy_in_bag", "greedy_formula_in_bag",
                    "greedy_exact_oracle", "unique_pairs", "unique_in_bag",
                    "ambiguous_pairs", "ambiguous_in_bag", "element_correct",
                    "element_total", "element_abs_error",
                ):
                    identity_totals[key] += identity[key]
                bag_log_prob.extend(identity["bag_log_prob"])
                best_in_bag.extend(identity["best_in_bag"])
                greedy_log_prob.extend(identity["greedy_log_prob"])
                for position, (correct, total) in enumerate(
                    zip(identity["position_correct"], identity["position_total"], strict=True)
                ):
                    while len(position_correct) <= position:
                        position_correct.append(0)
                        position_total.append(0)
                    position_correct[position] += correct
                    position_total[position] += total

                entry.update(
                    {
                        "targets": record.meta["targets"],
                        "active_slots": record.meta["active_slots"],
                        "peaks": record.meta["peaks"],
                        "adduct": record.meta["adduct"],
                        "instrument": record.meta["instrument"],
                        "collision_energy": record.meta["collision_energy"],
                        "precursor_mz": record.meta["precursor_mz"],
                        "unique_pairs": record.meta["unique_pairs"],
                        "ambiguous_pairs": record.meta["ambiguous_pairs"],
                        "predicted_fragments": record.meta["predicted_fragments"],
                        "matched_pairs": identity["matched_pairs"],
                        "greedy_in_bag": identity["greedy_in_bag"],
                    }
                )
                per_row.append(entry)
    elapsed = time.perf_counter() - started

    # ------------------------------------------------------------- summaries
    conditions_summary = {}
    for name in names:
        values = np.asarray(cosines[name])
        values20 = np.asarray(cosines20[name])
        conditions_summary[name] = {
            "oracle": name != "all_predicted" and not name.startswith("beam"),
            "cos@100": distribution(values),
            "cos@20": distribution(values20),
            "zero_prediction_spectra": zero_counts[name],
            "mean_predicted_peaks": float(np.mean(peak_counts[name])),
        }

    total_pairs = identity_totals["matched_pairs"] or 1
    identity_summary = {
        "matched_pairs": int(identity_totals["matched_pairs"]),
        "free_running_identity_in_bag": identity_totals["greedy_in_bag"] / total_pairs,
        "free_running_formula_in_bag": identity_totals["greedy_formula_in_bag"] / total_pairs,
        "free_running_exact_oracle": identity_totals["greedy_exact_oracle"] / total_pairs,
        "unique_pairs": int(identity_totals["unique_pairs"]),
        "unique_in_bag": (
            identity_totals["unique_in_bag"] / identity_totals["unique_pairs"]
            if identity_totals["unique_pairs"] else None
        ),
        "ambiguous_pairs": int(identity_totals["ambiguous_pairs"]),
        "ambiguous_in_bag": (
            identity_totals["ambiguous_in_bag"] / identity_totals["ambiguous_pairs"]
            if identity_totals["ambiguous_pairs"] else None
        ),
        "element_count_exact_accuracy": (
            identity_totals["element_correct"] / identity_totals["element_total"]
            if identity_totals["element_total"] else None
        ),
        "element_count_mae": (
            identity_totals["element_abs_error"] / identity_totals["element_total"]
            if identity_totals["element_total"] else None
        ),
        "position_accuracy": [
            (c / t if t else None) for c, t in zip(position_correct, position_total, strict=True)
        ],
        "position_support": position_total,
    }
    bag = np.asarray(bag_log_prob)
    best = np.asarray(best_in_bag)
    free = np.asarray(greedy_log_prob)
    teacher_forced = {
        "bag_nll_mean": float(-bag.mean()) if bag.size else None,
        "bag_probability_mass_mean": float(np.exp(bag).mean()) if bag.size else None,
        "bag_probability_mass_median": float(np.median(np.exp(bag))) if bag.size else None,
        "best_in_bag_log_prob_mean": float(best.mean()) if best.size else None,
        "free_running_log_prob_mean": float(free.mean()) if free.size else None,
        "margin_best_in_bag_minus_free_running_mean": (
            float((best - free).mean()) if best.size else None
        ),
        "fraction_free_running_beats_best_in_bag": (
            float((free > best).mean()) if best.size else None
        ),
    }

    # ------------------------------------------------------------ stratified
    frame = {key: [row.get(key) for row in per_row] for key in per_row[0]}
    base = np.asarray(frame["cos100_all_predicted"], dtype=np.float64)

    def stratify(labels) -> dict:
        groups: dict[str, list[float]] = defaultdict(list)
        for label, value in zip(labels, base, strict=True):
            groups[str(label)].append(float(value))
        return {
            key: {
                "n": len(values),
                "cos@100_mean": float(np.mean(values)),
                "cos@100_median": float(np.median(values)),
            }
            for key, values in sorted(groups.items())
        }

    targets = np.asarray(frame["targets"])
    unique_share = np.divide(
        np.asarray(frame["unique_pairs"], dtype=np.float64),
        np.maximum(np.asarray(frame["matched_pairs"], dtype=np.float64), 1),
    )
    stratified = {
        "zero_prediction": stratify(
            ["zero" if p == 0 else "nonzero" for p in frame["predicted_fragments"]]
        ),
        "unique_target_share": stratify([bucket(v, [0.25, 0.5, 0.75, 0.95]) for v in unique_share]),
        "target_peaks": stratify([bucket(v, [4, 8, 16, 32, 64]) for v in targets]),
        "precursor_mz": stratify(
            [bucket(v, [200, 300, 400, 600, 900]) for v in frame["precursor_mz"]]
        ),
        "instrument": stratify(frame["instrument"]),
        "adduct": stratify(frame["adduct"]),
        "collision_energy": stratify(
            [bucket(v, [10, 20, 30, 40, 60]) for v in frame["collision_energy"]]
        ),
        "active_slots": stratify([bucket(v, [1, 8, 16, 32, 48]) for v in frame["active_slots"]]),
    }

    report = {
        "diagnostic_only": True,
        "note": "conditions with oracle=true are ceilings, not model performance",
        "fold": args.fold,
        "test_fold_accessed": False,
        "checkpoint": str(args.checkpoint),
        "spectra": len(per_row),
        "elapsed_seconds": round(elapsed, 1),
        "beams": list(beams),
        "conditions": conditions_summary,
        "teacher_forced_vs_free_running": teacher_forced,
        "identity": identity_summary,
        "stratified_all_predicted": stratified,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "best_error_decomposition.json").write_text(json.dumps(report, indent=2) + "\n")
    pq.write_table(pa.table(frame), out / "per_spectrum_decomposition.parquet", compression="zstd")
    print(json.dumps({k: v for k, v in report.items() if k != "stratified_all_predicted"}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
