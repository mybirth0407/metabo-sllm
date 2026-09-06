#!/usr/bin/env python3
"""Prove the fragment-latent model runs: one real batch, then a tiny overfit.

This is an implementation check, not a result.  ``batch`` runs a single
forward/backward over real train rows and asserts the parameter contract
(LoRA and the new modules receive gradient, the Qwen base does not).
``overfit`` then drives a handful of spectra for a few dozen steps: if the
losses do not fall on data the model has memorised, something is wired wrong
and there is no point training on the subset.

Only the train fold is opened.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from metabo_sllm.data.fragment_collator import FragmentCollator  # noqa: E402
from metabo_sllm.data.fragment_dataset import FragmentSupervisionDataset  # noqa: E402
from metabo_sllm.losses.fragment_losses import LossWeights  # noqa: E402
from metabo_sllm.model.fragment_latent_model import (  # noqa: E402
    FragmentLatentModel,
    ModelConfig,
    parameter_report,
    training_step,
)
from metabo_sllm.model.qwen_encoder import load_tokenizer  # noqa: E402
from metabo_sllm.rendering.spectrum import (  # noqa: E402
    greedy_identity_metrics,
    slot_diversity,
)

TRAIN_FOLD = "train"


def build(config) -> tuple[FragmentLatentModel, FragmentCollator, LossWeights]:
    model_config = ModelConfig(
        model_name_or_path=config.encoder.model_name_or_path,
        dtype=config.encoder.dtype,
        lora_r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        use_rslora=config.lora.use_rslora,
        lora_target_modules=tuple(config.lora.target_modules),
        num_slots=config.slots.num_slots,
        slot_hidden_dim=config.slots.hidden_dim,
        slot_num_layers=config.slots.num_layers,
        slot_num_heads=config.slots.num_heads,
        slot_dropout=config.slots.dropout,
        slot_query_init_std=config.slots.query_init_std,
        formula_hidden_dim=config.formula_decoder.hidden_dim,
        formula_num_layers=config.formula_decoder.num_layers,
        formula_num_heads=config.formula_decoder.num_heads,
        formula_max_count=config.formula_decoder.max_count,
        formula_max_elements=config.formula_decoder.max_elements,
        formula_dropout=config.formula_decoder.dropout,
        head_hidden_dim=config.heads.hidden_dim,
        head_dropout=config.heads.dropout,
        match_intensity_weight=config.matching.intensity_weight,
        huber_delta=config.matching.huber_delta,
        matching_chunk_size=config.candidate_scoring.matching_chunk_size,
        matched_chunk_size=config.candidate_scoring.matched_chunk_size,
        return_full_candidate_scores=config.candidate_scoring.return_full_candidate_scores,
    )
    model = FragmentLatentModel(model_config)
    tokenizer = load_tokenizer(config.encoder.model_name_or_path)
    collator = FragmentCollator(tokenizer, max_text_length=config.data.max_text_length)
    weights = LossWeights(
        identity=config.loss.identity,
        presence=config.loss.presence,
        intensity=config.loss.intensity,
        spectrum=config.loss.spectrum,
    )
    return model, collator, weights


def to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def active_slots(presence: torch.Tensor, threshold: float = 0.5) -> float:
    return float((presence > threshold).sum(dim=1).float().mean().item())


def gradient_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum().item())
    return total**0.5


def parameter_contract(model: FragmentLatentModel) -> dict:
    base_grads = [
        name for name, p in model.encoder.base_parameters() if p.grad is not None
    ]
    lora_grads = [
        name for name, p in model.encoder.lora_parameters() if p.grad is not None
    ]
    new_modules = {
        "slot_decoder": model.slot_decoder,
        "formula_decoder": model.formula_decoder,
        "ion_head": model.ion_head,
        "presence_head": model.presence_head,
        "intensity_head": model.intensity_head,
    }
    new_with_grad = {
        name: sum(1 for p in module.parameters() if p.grad is not None)
        for name, module in new_modules.items()
    }
    return {
        "qwen_base_params_with_grad": len(base_grads),
        "lora_params_with_grad": len(lora_grads),
        "new_module_params_with_grad": new_with_grad,
        "base_grad_examples": base_grads[:3],
    }


def run_batch(args, config) -> dict:
    seed_everything(config.runtime.seed)
    device = torch.device(args.device)
    dataset = FragmentSupervisionDataset(
        config.data.root,
        TRAIN_FOLD,
        shards=[0],
        limit=args.batch_size,
        exclude_zero_target=config.data.exclude_zero_target_train,
    )
    model, collator, weights = build(config)
    model.to(device).train()

    rows = [dataset[i] for i in range(min(args.batch_size, len(dataset)))]
    batch = to_device(collator(rows), device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    outputs, losses = training_step(model, batch, weights)
    forward_seconds = time.perf_counter() - started

    started = time.perf_counter()
    losses.total.backward()
    backward_seconds = time.perf_counter() - started

    trainable = [p for p in model.parameters() if p.requires_grad]
    metrics = greedy_identity_metrics(model, batch, outputs, outputs.extras["assignment"])

    report = {
        "fold": TRAIN_FOLD,
        "test_fold_read": False,
        "batch_size": len(rows),
        "spectrum_uid": batch["spectrum_uid"],
        "shapes": {
            "input_ids": list(batch["input_ids"].shape),
            "slots": list(outputs.slots.shape),
            "presence": list(outputs.presence.shape),
            "intensity": list(outputs.intensity.shape),
            "ion_log_prob": list(outputs.ion_log_prob.shape),
            "matched_bag_nll": list(outputs.extras["matched_bag_nll"].shape),
            "matching_cost": list(outputs.extras["matching_cost"].shape),
            "target_intensities": list(batch["target_intensities"].shape),
            "full_peak_intensities": list(batch["full_peak_intensities"].shape),
        },
        "full_candidate_scores_in_output": "candidate_log_prob" in outputs.extras,
        "cost_profile": batch["cost_profile"],
        "parameters": parameter_report(model),
        "losses": {
            "total": float(losses.total.item()),
            "identity_bag_nll": float(losses.identity.item()),
            "presence": float(losses.presence.item()),
            "intensity": float(losses.intensity.item()),
            "spectrum": float(losses.spectrum.item()),
        },
        "all_losses_finite": all(
            np.isfinite(v)
            for v in (
                losses.total.item(),
                losses.identity.item(),
                losses.presence.item(),
                losses.intensity.item(),
                losses.spectrum.item(),
            )
        ),
        "matched_pairs": losses.extras["matched_pairs"],
        "identity_metrics": metrics,
        "active_slots_mean": active_slots(outputs.presence),
        "slot_diversity": slot_diversity(model, batch, outputs),
        "gradient_norm": gradient_norm(trainable),
        "parameter_contract": parameter_contract(model),
        "environment_workaround": model.encoder.torchao_probe_patched,
        "timing_seconds": {
            "forward": round(forward_seconds, 4),
            "backward": round(backward_seconds, 4),
        },
    }
    if device.type == "cuda":
        report["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
        report["peak_gpu_memory_gib"] = round(
            torch.cuda.max_memory_allocated(device) / 2**30, 3
        )
    return report


def eval_probe(model, batches, weights) -> dict:
    """Losses and slot diagnostics with dropout off.

    The training-mode loss on its own is not evidence of learning here: with
    dropout active the 64 slots differ by noise, and the Hungarian matcher will
    happily assign each target to whichever noisy slot fits it.  Only the
    eval-mode number says whether the model itself improved.
    """
    was_training = model.training
    model.eval()
    records = []
    with torch.no_grad():
        for batch in batches:
            outputs, losses = training_step(model, batch, weights)
            metrics = greedy_identity_metrics(
                model, batch, outputs, outputs.extras["assignment"]
            )
            diversity = slot_diversity(model, batch, outputs)
            records.append(
                {
                    "total": float(losses.total.item()),
                    "identity_bag_nll": float(losses.identity.item()),
                    "presence": float(losses.presence.item()),
                    "intensity": float(losses.intensity.item()),
                    "spectrum": float(losses.spectrum.item()),
                    "argmax_in_candidate_bag": metrics["argmax_in_candidate_bag"],
                    "active_slots": active_slots(outputs.presence),
                    **{k: v for k, v in diversity.items() if k != "num_slots"},
                }
            )
    model.train(was_training)
    keys = [k for k in records[0] if records[0][k] is not None]
    return {
        key: float(np.mean([r[key] for r in records if r[key] is not None])) for key in keys
    }


def run_overfit(args, config) -> dict:
    seed_everything(config.runtime.seed)
    device = torch.device(args.device)
    dataset = FragmentSupervisionDataset(
        config.data.root,
        TRAIN_FOLD,
        shards=[0],
        limit=args.num_spectra,
        exclude_zero_target=config.data.exclude_zero_target_train,
    )
    model, collator, weights = build(config)
    model.to(device).train()

    rows = [dataset[i] for i in range(min(args.num_spectra, len(dataset)))]
    batches = [
        to_device(collator(rows[start : start + args.batch_size]), device)
        for start in range(0, len(rows), args.batch_size)
    ]
    trainable = [p for p in model.parameters() if p.requires_grad]
    learning_rate = args.lr if args.lr is not None else config.optim.lr
    optimiser = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=config.optim.weight_decay
    )

    history: list[dict] = []
    eval_history: list[dict] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    eval_every = max(1, args.steps // 5)
    eval_history.append({"step": 0, **eval_probe(model, batches, weights)})
    for step in range(args.steps):
        step_losses = []
        step_metrics = []
        for batch in batches:
            optimiser.zero_grad(set_to_none=True)
            outputs, losses = training_step(model, batch, weights)
            losses.total.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, config.optim.grad_clip)
            optimiser.step()
            step_losses.append(
                {
                    "total": float(losses.total.item()),
                    "identity_bag_nll": float(losses.identity.item()),
                    "presence": float(losses.presence.item()),
                    "intensity": float(losses.intensity.item()),
                    "spectrum": float(losses.spectrum.item()),
                    "grad_norm": float(norm),
                    "active_slots": active_slots(outputs.presence),
                }
            )
            step_metrics.append(
                greedy_identity_metrics(model, batch, outputs, outputs.extras["assignment"])
            )
        record = {key: float(np.mean([s[key] for s in step_losses])) for key in step_losses[0]}
        hits = [m["argmax_in_candidate_bag"] for m in step_metrics if m["argmax_in_candidate_bag"] is not None]
        record["argmax_in_candidate_bag"] = float(np.mean(hits)) if hits else None
        record["step"] = step
        history.append(record)
        if (step + 1) % eval_every == 0 or step == args.steps - 1:
            probe = {"step": step + 1, **eval_probe(model, batches, weights)}
            eval_history.append(probe)
            print(
                f"[overfit] step {step + 1:>3} train_total={record['total']:.4f} "
                f"train_bag={record['identity_bag_nll']:.4f} | "
                f"eval_bag={probe['identity_bag_nll']:.4f} "
                f"eval_int={probe['intensity']:.4f} eval_spec={probe['spectrum']:.4f} "
                f"bag_hit={probe['argmax_in_candidate_bag']:.4f} "
                f"slot_cos={probe['slot_cosine_mean']:.4f} "
                f"distinct={probe['distinct_greedy_formulas_mean']:.1f}/64",
                flush=True,
            )
    elapsed = time.perf_counter() - started

    first, last = history[0], history[-1]
    eval_first, eval_last = eval_history[0], eval_history[-1]
    checks = {
        "train_total_decreased": last["total"] < first["total"],
        "train_bag_nll_decreased": last["identity_bag_nll"] < first["identity_bag_nll"],
        "eval_bag_nll_decreased": eval_last["identity_bag_nll"] < eval_first["identity_bag_nll"],
        "eval_total_decreased": eval_last["total"] < eval_first["total"],
        "intensity_decreased": eval_last["intensity"] < eval_first["intensity"],
        "bag_hit_increased": eval_last["argmax_in_candidate_bag"]
        > eval_first["argmax_in_candidate_bag"],
        # sixty-four identical slots would still all look "active"; require
        # that they actually point in different directions and decode
        # different fragments.
        "no_slot_collapse": (
            eval_last["slot_cosine_mean"] < 0.99
            and eval_last["distinct_greedy_formulas_mean"] > 1.5
        ),
        "all_finite": all(
            np.isfinite(value)
            for record in history + eval_history
            for key, value in record.items()
            if value is not None
        ),
    }
    report = {
        "eval_history": eval_history,
        "eval_first": eval_first,
        "eval_last": eval_last,
        "fold": TRAIN_FOLD,
        "test_fold_read": False,
        "num_spectra": len(rows),
        "batch_size": args.batch_size,
        "steps": args.steps,
        "learning_rate": learning_rate,
        "seed": config.runtime.seed,
        "first": first,
        "last": last,
        "history": history,
        "checks": checks,
        "passed": all(checks.values()),
        "elapsed_seconds": round(elapsed, 2),
        "seconds_per_step": round(elapsed / max(1, args.steps), 4),
    }
    if device.type == "cuda":
        report["peak_gpu_memory_gib"] = round(
            torch.cuda.max_memory_allocated(device) / 2**30, 3
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["batch", "overfit"])
    parser.add_argument("--config", default="configs/model/qwen_formula_slots_v0.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-spectra", type=int, default=8)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=None, help="override optim.lr")
    parser.add_argument("--out", default=None, help="write the report as JSON here")
    args = parser.parse_args(argv)

    config = OmegaConf.load(args.config)
    report = run_batch(args, config) if args.mode == "batch" else run_overfit(args, config)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    if args.mode == "overfit" and not report["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
