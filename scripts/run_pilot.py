#!/usr/bin/env python3
"""Ten-epoch pilot: train on scaffold_sub_10/train, validate candidate-free.

Validation never sees the observed spectrum until after the model has
predicted; the test fold is not opened, named, or written anywhere.

    PYTHONPATH=src python3 -m torch.distributed.run --standalone \
        --nproc_per_node=4 scripts/run_pilot.py \
        --config configs/train/qwen_formula_slots_v0_pilot.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from evaluate_fragment_model import evaluate_fold  # noqa: E402
from metabo_sllm.data.fragment_collator import FragmentCollator  # noqa: E402
from metabo_sllm.data.fragment_dataset import FragmentSupervisionDataset  # noqa: E402
from metabo_sllm.evaluation.prediction_writer import write_predictions  # noqa: E402
from metabo_sllm.evaluation.spectrum_metrics import BinningConfig, PRIMARY_SPACE  # noqa: E402
from metabo_sllm.losses.fragment_losses import LossWeights  # noqa: E402
from metabo_sllm.model.qwen_encoder import load_tokenizer  # noqa: E402
from metabo_sllm.training.checkpoint import supervision_manifest_hash  # noqa: E402
from metabo_sllm.training.distributed import DistributedContext, gather_objects  # noqa: E402
from metabo_sllm.training.dynamic_batch_sampler import (  # noqa: E402
    BatchBudgets,
    DynamicBatchSampler,
)
from metabo_sllm.training.trainer import (  # noqa: E402
    Trainer,
    TrainingConfig,
    build_optimizer,
    build_scheduler,
    epoch_summary,
)
from smoke_fragment_model import build as build_model_parts  # noqa: E402
from smoke_fragment_model import seed_everything  # noqa: E402

TRAIN_FOLD = "train"


def environment() -> dict:
    import peft
    import transformers

    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }


def git_state() -> dict:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(_ROOT), *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def file_hash(path: Path) -> str | None:
    import hashlib

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def append_jsonl(path: Path, record: dict, context: DistributedContext) -> None:
    if not context.is_main:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/train/qwen_formula_slots_v0_pilot.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="debug only")
    parser.add_argument("--valid-limit", type=int, default=None, help="debug only")
    args = parser.parse_args(argv)

    config = OmegaConf.load(args.config)
    config.model = OmegaConf.load(_ROOT / config.model_config)
    epochs = args.epochs or config.training.epochs

    context = DistributedContext.from_environment()
    seed_everything(config.training.seed + context.rank)
    output = Path(config.training.output_dir)
    if context.is_main:
        output.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, output / "config.yaml")

    train_set = FragmentSupervisionDataset(
        config.data.root,
        TRAIN_FOLD,
        exclude_zero_target=config.data.exclude_zero_target_train,
    )
    valid_set = FragmentSupervisionDataset(
        config.data.root, config.data.valid_fold, limit=args.valid_limit
    )

    model, _, _ = build_model_parts(config.model)
    tokenizer = load_tokenizer(config.model.encoder.model_name_or_path)
    collator = FragmentCollator(tokenizer, max_text_length=config.data.max_text_length)
    model.to(context.device)

    costs = train_set.cost_table(
        tokenizer=tokenizer, max_text_length=config.data.max_text_length
    )
    sampler = DynamicBatchSampler(
        costs,
        budgets=BatchBudgets(
            max_spectra=config.batching.max_spectra,
            max_qwen_tokens=config.batching.max_qwen_tokens,
            max_linked_candidates=config.batching.max_linked_candidates,
        ),
        world_size=context.world_size,
        rank=context.rank,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        seed=config.training.seed,
        bucket_multiplier=config.batching.bucket_multiplier,
    )
    steps_per_epoch = args.steps_per_epoch or sampler.optimizer_steps_per_epoch()
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, int(round(total_steps * config.scheduler.warmup_ratio)))

    optimizer = build_optimizer(
        model,
        learning_rate=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        betas=config.optimizer.betas,
        eps=config.optimizer.eps,
        fused=config.optimizer.fused,
    )
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=config.scheduler.warmup_ratio,
        min_lr_ratio=config.scheduler.min_lr_ratio,
    )
    trainer = Trainer(
        model=model,
        dataset=train_set,
        collator=collator,
        sampler=sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        context=context,
        config=TrainingConfig(
            max_steps=0,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            max_grad_norm=config.training.max_grad_norm,
            precision=config.training.precision,
            find_unused_parameters=config.training.find_unused_parameters,
            ddp_no_sync=config.training.ddp_no_sync,
            log_every=config.training.log_every,
            save_every=0,
            output_dir=str(output),
            seed=config.training.seed,
            loss=LossWeights(
                identity=config.loss.identity,
                presence=config.loss.presence,
                intensity=config.loss.intensity,
                spectrum=config.loss.spectrum,
                prefix=float(config.loss.get("prefix", 0.0)),
            ),
            prefix_warmup_steps=int(config.training.get("prefix_warmup_steps", 0)),
            prefix_decay_rate=float(config.training.get("prefix_decay_rate", 1.0)),
            prefix_decay_period=int(config.training.get("prefix_decay_period", 1)),
        ),
        run_config=OmegaConf.to_container(config, resolve=True),
        supervision_root=config.data.root,
        repo_root=str(_ROOT),
    )

    binning = BinningConfig(
        upper_limit=config.evaluation.upper_limit,
        num_bins=config.evaluation.num_bins,
        min_pred_intensity=config.evaluation.min_pred_intensity,
        top_k=tuple(config.evaluation.top_k),
    )
    manifest = {
        "run": config.name,
        "git": git_state(),
        "seed": config.training.seed,
        "world_size": context.world_size,
        "environment": environment(),
        "supervision_root": str(config.data.root),
        "supervision_manifest_sha256": supervision_manifest_hash(config.data.root),
        "qwen_model_path": str(config.model.encoder.model_name_or_path),
        "qwen_weights_sha256": file_hash(
            Path(config.model.encoder.model_name_or_path) / "model.safetensors"
        ),
        "train_spectra": len(train_set),
        "valid_spectra": len(valid_set),
        "zero_target_stats": train_set.zero_target_stats,
        "epochs": epochs,
        "optimizer_steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": total_steps,
        "warmup_steps": warmup_steps,
        "sampler": sampler.diagnostics().__dict__,
        "test_fold_accessed": False,
    }
    if context.is_main:
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        print(f"[pilot] train={len(train_set):,} valid={len(valid_set):,} "
              f"steps/epoch={steps_per_epoch} total={total_steps} warmup={warmup_steps}",
              flush=True)

    def validate(tag: str, epoch: int) -> dict:
        summary, predictions, scores = evaluate_fold(
            model, valid_set, tokenizer, context,
            fold=config.data.valid_fold, binning=binning,
            batch_size=config.evaluation.batch_size,
            max_text_length=config.data.max_text_length,
            presence_threshold=config.evaluation.presence_threshold,
        )
        summary["epoch"] = epoch
        summary["global_step"] = trainer.global_step
        summary["tag"] = tag
        append_jsonl(output / "valid_metrics.jsonl", summary, context)
        if context.is_main:
            print(
                f"[valid] {tag:<8} "
                f"cos@100={summary[PRIMARY_SPACE]['cos@100']['mean']:.4f} "
                f"cos@20={summary[PRIMARY_SPACE]['cos@20']['mean']:.4f} "
                f"(legacy_raw cos@100={summary['legacy_raw']['cos@100']['mean']:.4f}) "
                f"bag_hit={summary['fragment_diagnostic']['fragment_bag_hit']} "
                f"zero={summary['zero_prediction_spectra']} "
                f"peaks={summary['predicted_peaks']['mean']:.1f} "
                f"({summary['seconds']:.0f}s)",
                flush=True,
            )
        return summary, predictions, scores

    started = time.perf_counter()
    baseline, _, _ = validate("step-0", 0)
    # Selection runs on the primary space; the legacy number is recorded
    # beside it so the two can be compared, never so it can pick the winner.
    best = {"space": PRIMARY_SPACE, "cos@100": -1.0, "cos@20": -1.0, "epoch": None}

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        trainer.epoch = epoch
        outcome = trainer.train(max_steps=steps_per_epoch)
        summary = epoch_summary(outcome["history"])
        summary["epoch"] = epoch
        summary["global_step"] = trainer.global_step
        append_jsonl(output / "train_metrics.jsonl", summary, context)
        if context.is_main:
            print(
                f"[epoch] {epoch:>2} loss={summary['total_loss']:.4f} "
                f"bag={summary['bag_nll']:.4f} pres={summary['presence_loss']:.4f} "
                f"int={summary['intensity_loss']:.4f} spec={summary['spectrum_loss']:.4f} "
                f"pre={summary['prefix_loss']:.4f} "
                f"bag_hit={summary['argmax_in_candidate_bag']:.4f} "
                f"recall={summary['matched_presence_recall']:.4f} "
                f"active={summary['active_slots']:.1f} lr={summary['learning_rate']:.3e} "
                f"gnorm={summary['gradient_norm']:.3f} "
                f"{summary['spectra_per_second']:.0f} spectra/s "
                f"mem={summary['peak_gpu_memory_gib']:.1f}GiB",
                flush=True,
            )

        valid_summary, predictions, scores = validate(f"epoch{epoch}", epoch)
        trainer.save(output / "checkpoints" / "last")

        primary = valid_summary[PRIMARY_SPACE]
        current = (primary["cos@100"]["mean"], primary["cos@20"]["mean"])
        incumbent = (best["cos@100"], best["cos@20"])
        if current > incumbent:
            best = {
                "space": PRIMARY_SPACE,
                "cos@100": current[0],
                "cos@20": current[1],
                "cos@100_legacy_raw": valid_summary["legacy_raw"]["cos@100"]["mean"],
                "epoch": epoch,
                "global_step": trainer.global_step,
            }
            trainer.save(output / "checkpoints" / "best")
            gathered = [p for bucket in gather_objects(predictions, context) for p in bucket]
            gathered_scores = [s for bucket in gather_objects(scores, context) for s in bucket]
            if context.is_main:
                write_predictions(
                    output / "predictions" / "valid_best.parquet",
                    gathered,
                    gathered_scores,
                    fold=config.data.valid_fold,
                    metadata={"epoch": epoch, "run": config.name},
                )

    elapsed = time.perf_counter() - started
    final = {
        **manifest,
        "baseline_cos@100": baseline[PRIMARY_SPACE]["cos@100"]["mean"],
        "baseline_cos@100_legacy_raw": baseline["legacy_raw"]["cos@100"]["mean"],
        "best": best,
        "elapsed_seconds": round(elapsed, 1),
        "peak_gpu_memory_gib_per_rank": [
            round(float(v), 3)
            for v in gather_objects(
                torch.cuda.max_memory_allocated(context.device) / 2**30
                if context.device.type == "cuda"
                else 0.0,
                context,
            )
        ],
    }
    if context.is_main:
        (output / "run_manifest.json").write_text(json.dumps(final, indent=2, default=str) + "\n")
        print(f"[pilot] done in {elapsed / 60:.1f} min; best {json.dumps(best)}", flush=True)
        if (output / "checkpoints" / "best").is_dir():
            shutil.copy(
                output / "config.yaml", output / "checkpoints" / "best" / "config.yaml"
            )
    context.barrier()
    context.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
