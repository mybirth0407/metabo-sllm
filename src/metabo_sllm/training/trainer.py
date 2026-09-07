"""Training loop: dynamic batches, gradient accumulation, DDP, BF16.

Microbatches inside one optimizer step vary in size, so each one's loss is
weighted by its share of the step's spectra rather than by a flat
``1/accumulation`` -- otherwise a batch of two spectra would pull as hard as a
batch of sixteen.  The shares are known before the step runs, because the
sampler hands over the whole plan up front.

All but the last microbatch run under ``no_sync()``, so gradients are
all-reduced once per optimizer step instead of once per microbatch.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel

from metabo_sllm.losses.fragment_losses import LossWeights
from metabo_sllm.model.fragment_latent_model import training_step
from metabo_sllm.training.checkpoint import save_checkpoint
from metabo_sllm.training.distributed import (
    DistributedContext,
    all_reduce_sum,
    gather_objects,
)
from metabo_sllm.training.metrics import batch_workload, gradient_norm, slot_metrics

__all__ = ["NonFiniteLossError", "TrainingConfig", "Trainer", "build_optimizer", "build_scheduler"]


class NonFiniteLossError(RuntimeError):
    """Raised with the offending spectra when a loss stops being finite."""


@dataclass
class TrainingConfig:
    max_steps: int = 30
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    precision: str = "bf16"
    find_unused_parameters: bool = False
    ddp_no_sync: bool = True
    log_every: int = 1
    save_every: int = 0
    output_dir: str = "runs/smoke"
    seed: int = 0
    loss: LossWeights = field(default_factory=LossWeights)
    # Schedule for the prefix term, after GLACIER: full weight for
    # ``prefix_warmup_steps``, then multiplied by ``prefix_decay_rate`` per
    # ``prefix_decay_period`` steps. A rate of 1.0 means no decay.
    prefix_warmup_steps: int = 0
    prefix_decay_rate: float = 1.0
    prefix_decay_period: int = 1


_AUTOCAST = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}


def configure_attention_backends() -> None:
    """Turn off the cuDNN scaled-dot-product backend.

    Under bf16 autocast the slot and formula decoders' ``nn.MultiheadAttention``
    calls hit cuDNN's fused attention graph, which fails to build for these
    shapes ("mha_graph->execute ... to be true, but got false"). The flash,
    memory-efficient and math backends all handle them, so only cuDNN is
    disabled -- the model and its numerics are untouched.
    """
    if not torch.cuda.is_available():
        return
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)


def build_optimizer(model, *, learning_rate: float, weight_decay: float, betas, eps, fused: bool):
    """AdamW over the trainable parameters only; the frozen backbone stays out."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("no trainable parameters")
    kwargs = dict(lr=learning_rate, weight_decay=weight_decay, betas=tuple(betas), eps=eps)
    if fused and torch.cuda.is_available():
        try:
            return torch.optim.AdamW(parameters, fused=True, **kwargs)
        except (RuntimeError, ValueError):
            pass  # fall through to the portable implementation
    return torch.optim.AdamW(parameters, **kwargs)


def build_scheduler(optimizer, *, total_steps: int, warmup_ratio: float, min_lr_ratio: float):
    """Linear warmup into a cosine decay that flattens at ``min_lr_ratio``."""
    warmup = max(1, int(round(total_steps * warmup_ratio)))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        if total_steps <= warmup:
            return min_lr_ratio
        progress = (step - warmup) / max(1, total_steps - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class Trainer:
    def __init__(
        self,
        *,
        model,
        dataset,
        collator,
        sampler,
        optimizer,
        scheduler,
        context: DistributedContext,
        config: TrainingConfig,
        run_config: dict,
        supervision_root: str,
        repo_root: str,
    ) -> None:
        configure_attention_backends()
        self.raw_model = model
        self.dataset = dataset
        self.collator = collator
        self.sampler = sampler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.context = context
        self.config = config
        self.run_config = run_config
        self.supervision_root = supervision_root
        self.repo_root = repo_root

        self.model = model
        if context.distributed:
            self.model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank] if context.device.type == "cuda" else None,
                find_unused_parameters=config.find_unused_parameters,
            )
        self.trainable = [p for p in model.parameters() if p.requires_grad]
        self.global_step = 0
        self.epoch = 0
        self.output_dir = Path(config.output_dir)
        self.log_path = self.output_dir / "metrics.jsonl"
        self.consumed_spectra: list[str] = []
        if context.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- utilities

    def _autocast(self):
        dtype = _AUTOCAST.get(self.config.precision)
        if dtype is None or self.context.device.type != "cuda":
            return torch.autocast(device_type="cpu", enabled=False)
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _collate(self, indices: list[int]) -> dict:
        rows = [self.dataset[index] for index in indices]
        batch = self.collator(rows)
        return {
            key: value.to(self.context.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _log(self, record: dict) -> None:
        if not self.context.is_main:
            return
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(
            f"[train] step {record['step']:>5} lr={record['learning_rate']:.3e} "
            f"loss={record['total_loss']:.4f} bag={record['bag_nll']:.4f} "
            f"pres={record['presence_loss']:.4f} int={record['intensity_loss']:.4f} "
            f"spec={record['spectrum_loss']:.4f} pre={record['prefix_loss']:.4f} "
            f"bag_hit={_fmt(record['argmax_in_candidate_bag'])} "
            f"recall={_fmt(record['matched_presence_recall'])} "
            f"active={_fmt(record['active_slots'])} "
            f"spectra={record['batch_spectra']} cand={record['batch_candidates']} "
            f"gnorm={record['gradient_norm']:.3f} "
            f"mem={record['peak_gpu_memory_gib']:.2f}GiB "
            f"step_time={record['step_time']:.3f}s",
            flush=True,
        )

    # ------------------------------------------------------------------ loop

    def train(self, max_steps: int | None = None) -> dict:
        """Run up to ``max_steps`` optimizer steps of the current epoch.

        ``None`` (or 0) means run the epoch out.
        """
        self.model.train()
        self.consumed_spectra = []
        plan = self.sampler.remaining_for_rank()
        accumulation = self.config.gradient_accumulation_steps
        steps_available = len(plan) // accumulation
        limit = max_steps if max_steps is not None else self.config.max_steps
        steps = steps_available if not limit else min(limit, steps_available)
        if steps == 0:
            raise RuntimeError(
                f"rank {self.context.rank} has {len(plan)} batches, too few for one "
                f"optimizer step of {accumulation} microbatches"
            )

        if self.context.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.context.device)

        history: list[dict] = []
        cursor = 0
        for _ in range(steps):
            micro = plan[cursor : cursor + accumulation]
            cursor += accumulation
            # Advance the sampler before the step, not after the loop: a
            # checkpoint written mid-run has to record the batches already
            # consumed, or resuming replays them.
            self.sampler.consumed_batches += len(micro)
            history.append(self._optimizer_step(micro))

        return {
            "steps": len(history),
            "history": history,
            "consumed_spectra": self.consumed_spectra,
            "global_step": self.global_step,
        }

    def _loss_weights(self) -> LossWeights:
        """Loss weights at this step: the prefix term warms up, then decays."""
        weights = self.config.loss
        if weights.prefix == 0.0 or self.config.prefix_decay_rate == 1.0:
            return weights
        after = self.global_step - self.config.prefix_warmup_steps
        if after <= 0:
            return weights
        ratio = self.config.prefix_decay_rate ** (after / max(1, self.config.prefix_decay_period))
        return replace(weights, prefix=weights.prefix * ratio)

    def _optimizer_step(self, micro_batches: list[list[int]]) -> dict:
        started = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)

        data_started = time.perf_counter()
        collated = [self._collate(indices) for indices in micro_batches]
        data_time = time.perf_counter() - data_started

        # Weight each microbatch by its share of the step's spectra: the losses
        # are per-spectrum means, so a flat 1/accumulation would over-weight
        # small batches.
        sizes = [float(batch["target_peak_mask"].shape[0]) for batch in collated]
        total_size = sum(sizes) or 1.0

        totals = {
            "total_loss": 0.0,
            "identity_loss": 0.0,
            "presence_loss": 0.0,
            "intensity_loss": 0.0,
            "spectrum_loss": 0.0,
            "prefix_loss": 0.0,
        }
        workload = {
            "batch_spectra": 0,
            "batch_targets": 0,
            "batch_candidates": 0,
            "batch_tokens": 0,
            "batch_peaks": 0,
        }
        counters = {
            "bag_hits": 0.0,
            "bag_pairs": 0.0,
            "presence_recalled": 0.0,
            "presence_targets": 0.0,
            "active_slots": 0.0,
            "active_spectra": 0.0,
        }
        forward_time = backward_time = 0.0

        for position, (indices, batch) in enumerate(zip(micro_batches, collated, strict=True)):
            last = position == len(collated) - 1
            weight = sizes[position] / total_size
            sync = self._sync_context(last)
            with sync:
                mark = time.perf_counter()
                with self._autocast():
                    outputs, losses = training_step(
                        self.raw_model, batch, self._loss_weights()
                    )
                forward_time += time.perf_counter() - mark

                self._guard_finite(losses, batch, indices)
                mark = time.perf_counter()
                (losses.total * weight).backward()
                backward_time += time.perf_counter() - mark

            for key, value in (
                ("total_loss", losses.total),
                ("identity_loss", losses.identity),
                ("presence_loss", losses.presence),
                ("intensity_loss", losses.intensity),
                ("spectrum_loss", losses.spectrum),
                ("prefix_loss", losses.prefix),
            ):
                totals[key] += float(value.item()) * weight
            for key, value in batch_workload(batch).items():
                workload[key] += value
            self.consumed_spectra.extend(batch["spectrum_uid"])

            if last:
                measured = slot_metrics(
                    self.raw_model, batch, outputs, outputs.extras["assignment"]
                )
                for key in counters:
                    counters[key] += float(measured[key])

        norm = torch.nn.utils.clip_grad_norm_(self.trainable, self.config.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

        record = self._reduce_record(
            totals, workload, counters, float(norm), started, data_time, forward_time, backward_time
        )
        if self.global_step % max(1, self.config.log_every) == 0:
            self._log(record)
        if self.config.save_every and self.global_step % self.config.save_every == 0:
            self.save(self.output_dir / f"checkpoint-{self.global_step:08d}")
        return record

    def _sync_context(self, last: bool):
        if last or not self.context.distributed or not self.config.ddp_no_sync:
            return _NullContext()
        return self.model.no_sync()

    def _guard_finite(self, losses, batch: dict, indices: list[int]) -> None:
        value = float(losses.total.item())
        if math.isfinite(value):
            return
        raise NonFiniteLossError(
            json.dumps(
                {
                    "rank": self.context.rank,
                    "global_step": self.global_step,
                    "total_loss": value,
                    "spectrum_uid": list(batch["spectrum_uid"]),
                    "dataset_indices": list(indices),
                    "workload": batch_workload(batch),
                    "cost_profile": batch["cost_profile"],
                },
                indent=2,
            )
        )

    def _reduce_record(
        self, totals, workload, counters, norm, started, data_time, forward_time, backward_time
    ) -> dict:
        context = self.context
        reduced_losses = {
            key: all_reduce_sum(value, context) / context.world_size
            for key, value in totals.items()
        }
        reduced_workload = {
            key: int(all_reduce_sum(float(value), context)) for key, value in workload.items()
        }
        reduced_counters = {
            key: all_reduce_sum(value, context) for key, value in counters.items()
        }
        peak = (
            torch.cuda.max_memory_allocated(context.device) / 2**30
            if context.device.type == "cuda"
            else 0.0
        )
        return {
            "step": self.global_step,
            "epoch": self.epoch,
            "learning_rate": float(self.scheduler.get_last_lr()[0]),
            "total_loss": reduced_losses["total_loss"],
            "identity_loss": reduced_losses["identity_loss"],
            "presence_loss": reduced_losses["presence_loss"],
            "intensity_loss": reduced_losses["intensity_loss"],
            "spectrum_loss": reduced_losses["spectrum_loss"],
            "prefix_loss": reduced_losses["prefix_loss"],
            "prefix_weight": float(self._loss_weights().prefix),
            "bag_nll": reduced_losses["identity_loss"],
            "argmax_in_candidate_bag": _ratio(
                reduced_counters["bag_hits"], reduced_counters["bag_pairs"]
            ),
            "matched_presence_recall": _ratio(
                reduced_counters["presence_recalled"], reduced_counters["presence_targets"]
            ),
            "active_slots": _ratio(
                reduced_counters["active_slots"], reduced_counters["active_spectra"]
            ),
            **reduced_workload,
            "gradient_norm": all_reduce_sum(norm, context) / context.world_size,
            "peak_gpu_memory_gib": round(peak, 3),
            "data_time": round(data_time, 4),
            "forward_time": round(forward_time, 4),
            "backward_time": round(backward_time, 4),
            "step_time": round(time.perf_counter() - started, 4),
        }

    # ------------------------------------------------------------ checkpoint

    def save(self, directory: Path) -> Path | None:
        self.context.barrier()
        if not self.context.is_main:
            self.context.barrier()
            return None
        save_checkpoint(
            directory,
            model=self.raw_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            sampler_state=self.sampler.state_dict(),
            epoch=self.epoch,
            global_step=self.global_step,
            config=self.run_config,
            supervision_root=self.supervision_root,
            repo_root=self.repo_root,
        )
        self.context.barrier()
        return directory

    def spectrum_overlap(self) -> dict:
        """Whether any two ranks consumed the same spectrum."""
        per_rank = gather_objects(sorted(set(self.consumed_spectra)), self.context)
        seen: dict[str, int] = {}
        duplicates = 0
        for rank, uids in enumerate(per_rank):
            for uid in uids:
                if uid in seen:
                    duplicates += 1
                else:
                    seen[uid] = rank
        return {
            "unique_spectra": len(seen),
            "duplicate_spectra_across_ranks": duplicates,
            "per_rank_spectra": [len(uids) for uids in per_rank],
        }


def epoch_summary(history: list[dict]) -> dict:
    """Averages over an epoch's optimizer steps, for the per-epoch log."""
    if not history:
        return {}
    numeric = [
        "total_loss",
        "identity_loss",
        "presence_loss",
        "intensity_loss",
        "spectrum_loss",
        "prefix_loss",
        "bag_nll",
        "gradient_norm",
        "learning_rate",
    ]
    summary = {
        key: float(sum(record[key] for record in history) / len(history)) for key in numeric
    }
    for key in ("argmax_in_candidate_bag", "matched_presence_recall", "active_slots"):
        values = [record[key] for record in history if record[key] is not None]
        summary[key] = float(sum(values) / len(values)) if values else None
    for key in ("batch_spectra", "batch_targets", "batch_candidates", "batch_tokens"):
        summary[key] = int(sum(record[key] for record in history))
    total_time = sum(record["step_time"] for record in history)
    summary["steps"] = len(history)
    summary["seconds"] = round(total_time, 2)
    summary["spectra_per_second"] = (
        round(summary["batch_spectra"] / total_time, 1) if total_time > 0 else None
    )
    summary["peak_gpu_memory_gib"] = max(record["peak_gpu_memory_gib"] for record in history)
    return summary


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"
