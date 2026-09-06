#!/usr/bin/env python3
"""Candidate-free validation of the fragment-latent model.

Predictions are made from the molecule and acquisition settings alone; the
observed spectrum is opened only afterwards, to score what was predicted. The
test fold is refused.

    PYTHONPATH=src python3 scripts/evaluate_fragment_model.py \
        --config configs/train/qwen_formula_slots_v0_pilot.yaml \
        --checkpoint runs/.../checkpoints/best --out valid.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from metabo_sllm.data.fragment_dataset import FragmentSupervisionDataset  # noqa: E402
from metabo_sllm.data.supervision import FIXED_SLOT_COUNT  # noqa: E402
from metabo_sllm.evaluation.fragment_metrics import (  # noqa: E402
    score_fragments,
    summarise_fragments,
)
from metabo_sllm.evaluation.inference import (  # noqa: E402
    build_model_inputs,
    predict_spectrum,
)
from metabo_sllm.evaluation.spectrum_metrics import (  # noqa: E402
    BinningConfig,
    score_prediction,
    summarise_scores,
)
from metabo_sllm.training.distributed import DistributedContext, gather_objects  # noqa: E402

FORBIDDEN_FOLD = "test"


def target_bags(row) -> tuple[list[set[tuple[str, str]]], list[float]]:
    """Candidate bags of the top-64 supervised peaks, for the diagnostic only."""
    rank = np.asarray(row["supervision_rank"])
    selected = np.flatnonzero((rank >= 0) & (rank < FIXED_SLOT_COUNT))
    selected = selected[np.argsort(rank[selected], kind="stable")]
    if selected.size == 0:
        return [], []

    position = {int(peak): index for index, peak in enumerate(selected)}
    bags: list[set[tuple[str, str]]] = [set() for _ in selected]
    formulas = row["candidate_neutral_formula"]
    ions = row["candidate_ion_state"]
    for peak, candidate in zip(
        np.asarray(row["edge_peak_index"]).tolist(),
        np.asarray(row["edge_candidate_index"]).tolist(),
        strict=True,
    ):
        index = position.get(int(peak))
        if index is not None:
            bags[index].add((formulas[candidate], ions[candidate]))
    intensities = np.asarray(row["intensities"], dtype=np.float64)[selected]
    return bags, intensities.tolist()


def evaluate_fold(
    model,
    dataset: FragmentSupervisionDataset,
    tokenizer,
    context: DistributedContext,
    *,
    fold: str,
    binning: BinningConfig,
    batch_size: int = 32,
    max_text_length: int = 512,
    presence_threshold: float = 0.5,
    with_fragment_diagnostic: bool = True,
) -> tuple[dict, list, list]:
    """Predict, then score. Returns (summary, predictions, spectrum scores)."""
    if fold == FORBIDDEN_FOLD:
        raise ValueError("evaluation on the test fold is not permitted")

    indices = list(range(context.rank, len(dataset), context.world_size))
    predictions = []
    scores = []
    diagnostics = []
    started = time.perf_counter()

    for start in range(0, len(indices), batch_size):
        rows = [dataset[index] for index in indices[start : start + batch_size]]
        model_inputs = build_model_inputs(rows, tokenizer, max_text_length=max_text_length)
        on_device = {
            key: value.to(context.device) if isinstance(value, torch.Tensor) else value
            for key, value in model_inputs.items()
        }
        batch_predictions = predict_spectrum(
            model, on_device, presence_threshold=presence_threshold
        )
        # Only now is the observed spectrum opened.
        for row, prediction in zip(rows, batch_predictions, strict=True):
            scores.append(
                score_prediction(
                    prediction.spectrum_uid,
                    prediction.mz,
                    prediction.intensity,
                    np.asarray(row["mzs"], dtype=np.float64),
                    np.asarray(row["intensities"], dtype=np.float64),
                    binning,
                )
            )
            if with_fragment_diagnostic:
                bags, intensities = target_bags(row)
                diagnostics.append(
                    score_fragments(
                        prediction.spectrum_uid,
                        [(f.formula, f.ion_state) for f in prediction.fragments],
                        bags,
                        intensities,
                        active_slots=prediction.active_slots,
                        duplicate_identity_slots=prediction.duplicate_identity_slots,
                    )
                )
        predictions.extend(batch_predictions)

    elapsed = time.perf_counter() - started
    all_scores = [s for bucket in gather_objects(scores, context) for s in bucket]
    all_diagnostics = [d for bucket in gather_objects(diagnostics, context) for d in bucket]

    summary = summarise_scores(all_scores, binning)
    summary["fold"] = fold
    summary["seconds"] = round(elapsed, 2)
    summary["spectra_per_second"] = (
        round(len(all_scores) / elapsed, 1) if elapsed > 0 else None
    )
    if with_fragment_diagnostic:
        summary["fragment_diagnostic"] = summarise_fragments(all_diagnostics)
    return summary, predictions, scores


def main(argv: list[str] | None = None) -> int:
    from omegaconf import OmegaConf

    from metabo_sllm.evaluation.prediction_writer import write_predictions
    from metabo_sllm.model.qwen_encoder import load_tokenizer
    from metabo_sllm.training.checkpoint import load_checkpoint
    from smoke_fragment_model import build as build_model_parts

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", default="valid")
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    if args.fold == FORBIDDEN_FOLD:
        raise SystemExit("error: evaluation on the test fold is not permitted")

    config = OmegaConf.load(args.config)
    model_config = OmegaConf.load(_ROOT / config.model_config)
    context = DistributedContext.from_environment()
    model, _, _ = build_model_parts(model_config)
    model.to(context.device)
    load_checkpoint(args.checkpoint, model=model, map_location=context.device)

    tokenizer = load_tokenizer(model_config.encoder.model_name_or_path)
    dataset = FragmentSupervisionDataset(config.data.root, args.fold, limit=args.limit)
    binning = BinningConfig(
        upper_limit=config.evaluation.upper_limit,
        num_bins=config.evaluation.num_bins,
        min_pred_intensity=config.evaluation.min_pred_intensity,
        top_k=tuple(config.evaluation.top_k),
    )
    summary, predictions, scores = evaluate_fold(
        model, dataset, tokenizer, context, fold=args.fold, binning=binning,
        batch_size=args.batch_size, max_text_length=config.data.max_text_length,
    )
    print(json.dumps(summary, indent=2))
    if args.out and context.is_main:
        write_predictions(args.out, predictions, scores, fold=args.fold)
        print(f"wrote {args.out}")
    context.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
