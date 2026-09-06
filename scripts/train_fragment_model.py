#!/usr/bin/env python3
"""Train the fragment-latent model on fragment_supervision_v1/train.

Single process::

    PYTHONPATH=src python3 scripts/train_fragment_model.py \
        --config configs/train/qwen_formula_slots_v0_smoke.yaml --max-steps 10

Four GPUs::

    PYTHONPATH=src torchrun --nproc_per_node=4 scripts/train_fragment_model.py \
        --config configs/train/qwen_formula_slots_v0_smoke.yaml

The valid and test folds are never opened.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from metabo_sllm.data.fragment_collator import FragmentCollator  # noqa: E402
from metabo_sllm.data.fragment_dataset import FragmentSupervisionDataset  # noqa: E402
from metabo_sllm.losses.fragment_losses import LossWeights  # noqa: E402
from metabo_sllm.model.qwen_encoder import load_tokenizer  # noqa: E402
from metabo_sllm.training.checkpoint import load_checkpoint  # noqa: E402
from metabo_sllm.training.distributed import DistributedContext  # noqa: E402
from metabo_sllm.training.dynamic_batch_sampler import (  # noqa: E402
    BatchBudgets,
    DynamicBatchSampler,
)
from metabo_sllm.training.trainer import (  # noqa: E402
    Trainer,
    TrainingConfig,
    build_optimizer,
    build_scheduler,
)
from smoke_fragment_model import build as build_model_parts  # noqa: E402
from smoke_fragment_model import seed_everything  # noqa: E402

TRAIN_FOLD = "train"


def load_config(path: str):
    config = OmegaConf.load(path)
    model_config = OmegaConf.load(_ROOT / config.model_config)
    config.model = model_config
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/train/qwen_formula_slots_v0_smoke.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--resume", default=None, help="checkpoint directory to resume from")
    parser.add_argument("--limit-spectra", type=int, default=None, help="use a fixed subset")
    parser.add_argument("--shards", type=int, default=None, help="use only the first N shards")
    parser.add_argument("--report", default=None, help="write a JSON run report here")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.output_dir is not None:
        config.training.output_dir = args.output_dir
    if args.save_every is not None:
        config.training.save_every = args.save_every

    context = DistributedContext.from_environment()
    seed_everything(config.training.seed + context.rank)

    dataset = FragmentSupervisionDataset(
        config.data.root,
        TRAIN_FOLD,
        shards=list(range(args.shards)) if args.shards else None,
        limit=args.limit_spectra,
        exclude_zero_target=config.data.exclude_zero_target_train,
    )
    model, collator, _ = build_model_parts(config.model)
    tokenizer = load_tokenizer(config.model.encoder.model_name_or_path)
    collator = FragmentCollator(tokenizer, max_text_length=config.data.max_text_length)
    model.to(context.device)

    costs = dataset.cost_table(
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

    optimizer = build_optimizer(
        model,
        learning_rate=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        betas=config.optimizer.betas,
        eps=config.optimizer.eps,
        fused=config.optimizer.fused,
    )
    # The cosine shape depends on the horizon it was built with, so a resumed
    # run has to rebuild the same curve -- taken from the checkpoint, not from
    # however many steps this continuation happens to ask for.
    schedule_total = max(1, config.training.max_steps)
    if args.resume:
        original = json.loads((Path(args.resume) / "manifest.json").read_text())
        schedule_total = max(1, int(original["config"]["training"]["max_steps"]))
    scheduler = build_scheduler(
        optimizer,
        total_steps=schedule_total,
        warmup_ratio=config.scheduler.warmup_ratio,
        min_lr_ratio=config.scheduler.min_lr_ratio,
    )

    resumed_state = None
    if args.resume:
        resumed_state = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=context.device,
        )
        sampler.load_state_dict(resumed_state["sampler"])

    trainer = Trainer(
        model=model,
        dataset=dataset,
        collator=collator,
        sampler=sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        context=context,
        config=TrainingConfig(
            max_steps=config.training.max_steps,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            max_grad_norm=config.training.max_grad_norm,
            precision=config.training.precision,
            find_unused_parameters=config.training.find_unused_parameters,
            ddp_no_sync=config.training.ddp_no_sync,
            log_every=config.training.log_every,
            save_every=config.training.save_every,
            output_dir=config.training.output_dir,
            seed=config.training.seed,
            loss=LossWeights(
                identity=config.loss.identity,
                presence=config.loss.presence,
                intensity=config.loss.intensity,
                spectrum=config.loss.spectrum,
            ),
        ),
        run_config=OmegaConf.to_container(config, resolve=True),
        supervision_root=config.data.root,
        repo_root=str(_ROOT),
    )
    if resumed_state is not None:
        trainer.global_step = int(resumed_state["global_step"])
        trainer.epoch = int(resumed_state["epoch"])

    diagnostics = sampler.diagnostics()
    if context.is_main:
        print(f"[train] dataset spectra: {len(dataset)}", flush=True)
        print(f"[train] zero-target stats: {dataset.zero_target_stats}", flush=True)
        print(f"[train] sampler: {json.dumps(diagnostics.__dict__, default=str)}", flush=True)

    outcome = trainer.train()
    overlap = trainer.spectrum_overlap()
    steps_per_rank = [
        int(value) for value in _gather_int(outcome["steps"], context)
    ]
    peak_per_rank = _gather_peak_memory(context)

    final_checkpoint = None
    if config.training.save_every == 0:
        final_checkpoint = trainer.save(
            Path(config.training.output_dir) / f"checkpoint-{trainer.global_step:08d}"
        )

    report = {
        "fold": TRAIN_FOLD,
        "test_fold_read": False,
        "world_size": context.world_size,
        "dataset_spectra": len(dataset),
        "zero_target_stats": dataset.zero_target_stats,
        "sampler": diagnostics.__dict__,
        "steps_completed": outcome["steps"],
        "steps_per_rank": steps_per_rank,
        "steps_equal_across_ranks": len(set(steps_per_rank)) == 1,
        "peak_gpu_memory_gib_per_rank": peak_per_rank,
        "global_step": trainer.global_step,
        "spectrum_overlap": overlap,
        "consumed_spectra": outcome["consumed_spectra"],
        "history": outcome["history"],
        "scheduler_total_steps": schedule_total,
        "resumed_from": args.resume,
        "final_checkpoint": str(final_checkpoint) if final_checkpoint else None,
    }
    if context.is_main and args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
        print(f"[train] wrote {args.report}", flush=True)

    context.barrier()
    context.shutdown()
    return 0


def _gather_int(value: int, context: DistributedContext) -> list[int]:
    from metabo_sllm.training.distributed import gather_objects

    return gather_objects(int(value), context)


def _gather_peak_memory(context: DistributedContext) -> list[float]:
    """Peak allocation on each GPU, not just the one rank 0 happens to hold."""
    from metabo_sllm.training.distributed import gather_objects

    peak = (
        torch.cuda.max_memory_allocated(context.device) / 2**30
        if context.device.type == "cuda"
        else 0.0
    )
    return [round(float(value), 3) for value in gather_objects(peak, context)]


if __name__ == "__main__":
    raise SystemExit(main())
