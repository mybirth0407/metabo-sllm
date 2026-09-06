"""Batches sized by scoring work, not by spectrum count.

Spectra differ enormously in what they cost to score: in the stress set one
spectrum links 1,902 candidates while another links 12.  Fixed-size batches
would have to be sized for the worst case and would then waste the GPU on
every ordinary batch.

So a batch fills until any one budget -- spectra, Qwen tokens, or linked
candidates -- would be exceeded, and closes just before.  A spectrum that
alone exceeds a budget becomes its own batch rather than being dropped or
trimmed: no candidate bag is ever cut to make something fit.

Spectra of similar cost are kept near each other so padding stays small, but
the order inside each bucket, and the order of the batches themselves, is
reshuffled every epoch from ``seed`` so the model does not see the same
grouping twice.  Everything is a pure function of ``(seed, epoch)``, which is
what makes resume exact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = ["BatchBudgets", "DynamicBatchSampler", "SamplerDiagnostics"]


@dataclass(frozen=True)
class BatchBudgets:
    """Per-GPU limits. A batch closes before any of them is exceeded."""

    max_spectra: int = 16
    max_qwen_tokens: int = 4096
    max_linked_candidates: int = 2048

    def exceeded(self, spectra: int, tokens: int, candidates: int) -> bool:
        return (
            spectra > self.max_spectra
            or tokens > self.max_qwen_tokens
            or candidates > self.max_linked_candidates
        )


@dataclass
class SamplerDiagnostics:
    """What an epoch's plan looks like, and what it had to leave behind."""

    epoch: int
    total_spectra: int
    total_batches: int
    used_batches: int
    dropped_batches: int
    dropped_spectra: int
    singleton_batches: int
    batches_per_rank: int
    optimizer_steps: int
    workload: dict = field(default_factory=dict)


class DynamicBatchSampler:
    """Deterministic, budget-aware batches split across DDP ranks.

    Rank ``r`` receives batches ``r, r + world_size, r + 2 * world_size, ...``
    of the epoch's plan, so no spectrum is consumed by two ranks, and the plan
    is truncated to a whole number of optimizer steps per rank so every rank
    steps the optimizer the same number of times.
    """

    def __init__(
        self,
        costs: Mapping[str, np.ndarray],
        *,
        budgets: BatchBudgets | None = None,
        world_size: int = 1,
        rank: int = 0,
        gradient_accumulation_steps: int = 1,
        seed: int = 0,
        bucket_multiplier: int = 50,
    ) -> None:
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"invalid rank {rank} for world size {world_size}")
        if gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")

        self.tokens = np.asarray(costs["num_tokens"], dtype=np.int64)
        self.candidates = np.asarray(
            costs["num_candidates_linked_to_top64"], dtype=np.int64
        )
        self.targets = np.asarray(costs["num_targets"], dtype=np.int64)
        self.element_steps = np.asarray(
            costs["num_candidate_element_steps"], dtype=np.int64
        )
        self.size = int(self.tokens.shape[0])

        self.budgets = budgets or BatchBudgets()
        self.world_size = world_size
        self.rank = rank
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.seed = seed
        self.bucket_multiplier = bucket_multiplier

        self.epoch = 0
        self.consumed_batches = 0
        self._plan: list[list[int]] | None = None
        self._diagnostics: SamplerDiagnostics | None = None

    # ------------------------------------------------------------------ plan

    def set_epoch(self, epoch: int) -> None:
        if epoch != self.epoch:
            self.consumed_batches = 0
        self.epoch = epoch
        self._plan = None
        self._diagnostics = None

    def _ordering(self) -> np.ndarray:
        """Cost-sorted indices, shuffled inside buckets of similar cost."""
        order = np.lexsort((self.tokens, self.element_steps))
        bucket = max(1, self.budgets.max_spectra * self.bucket_multiplier)
        generator = np.random.default_rng((self.seed, self.epoch, 0xB0CC))
        for start in range(0, order.size, bucket):
            window = order[start : start + bucket]
            generator.shuffle(window)
            order[start : start + bucket] = window
        return order

    def _pack(self, order: np.ndarray) -> tuple[list[list[int]], int]:
        batches: list[list[int]] = []
        singletons = 0
        current: list[int] = []
        tokens = candidates = 0
        for index in order.tolist():
            item_tokens = int(self.tokens[index])
            item_candidates = int(self.candidates[index])
            alone = self.budgets.exceeded(1, item_tokens, item_candidates)
            if alone:
                # Too big for any batch, but it is not the sampler's job to
                # shrink a spectrum: give it a batch of its own.
                if current:
                    batches.append(current)
                    current, tokens, candidates = [], 0, 0
                batches.append([index])
                singletons += 1
                continue
            if current and self.budgets.exceeded(
                len(current) + 1, tokens + item_tokens, candidates + item_candidates
            ):
                batches.append(current)
                current, tokens, candidates = [], 0, 0
            current.append(index)
            tokens += item_tokens
            candidates += item_candidates
        if current:
            batches.append(current)
        return batches, singletons

    def plan(self) -> list[list[int]]:
        """Every batch of this epoch, before the split across ranks."""
        if self._plan is not None:
            return self._plan

        batches, singletons = self._pack(self._ordering())
        generator = np.random.default_rng((self.seed, self.epoch, 0xBA7C))
        permutation = generator.permutation(len(batches))
        batches = [batches[position] for position in permutation]

        # Every rank must run the same number of optimizer steps, and each step
        # eats gradient_accumulation_steps batches, so the tail that cannot fill
        # one whole round is dropped -- and counted.
        stride = self.world_size * self.gradient_accumulation_steps
        used = (len(batches) // stride) * stride
        dropped = batches[used:]
        batches = batches[:used]

        self._plan = batches
        self._diagnostics = SamplerDiagnostics(
            epoch=self.epoch,
            total_spectra=self.size,
            total_batches=used + len(dropped),
            used_batches=used,
            dropped_batches=len(dropped),
            dropped_spectra=sum(len(item) for item in dropped),
            singleton_batches=singletons,
            batches_per_rank=used // self.world_size if self.world_size else 0,
            optimizer_steps=used // stride if stride else 0,
            workload=self._workload(batches),
        )
        return self._plan

    def _workload(self, batches: Sequence[Sequence[int]]) -> dict:
        if not batches:
            return {}

        def stats(values: list[int]) -> dict:
            array = np.asarray(values, dtype=np.float64)
            return {
                "mean": float(array.mean()),
                "min": int(array.min()),
                "p50": float(np.percentile(array, 50)),
                "p90": float(np.percentile(array, 90)),
                "max": int(array.max()),
            }

        return {
            "spectra": stats([len(item) for item in batches]),
            "tokens": stats([int(self.tokens[item].sum()) for item in batches]),
            "targets": stats([int(self.targets[item].sum()) for item in batches]),
            "linked_candidates": stats(
                [int(self.candidates[item].sum()) for item in batches]
            ),
            "candidate_element_steps": stats(
                [int(self.element_steps[item].sum()) for item in batches]
            ),
        }

    def diagnostics(self) -> SamplerDiagnostics:
        self.plan()
        assert self._diagnostics is not None
        return self._diagnostics

    # ------------------------------------------------------------------ rank

    def batches_for_rank(self, rank: int | None = None) -> list[list[int]]:
        """This rank's slice of the plan, in order."""
        rank = self.rank if rank is None else rank
        return self.plan()[rank :: self.world_size]

    def remaining_for_rank(self) -> list[list[int]]:
        """Batches this rank has not consumed yet, honouring a resumed state."""
        return self.batches_for_rank()[self.consumed_batches :]

    def optimizer_steps_per_epoch(self) -> int:
        return self.diagnostics().optimizer_steps

    def __len__(self) -> int:
        return len(self.batches_for_rank())

    def __iter__(self):
        for batch in self.remaining_for_rank():
            self.consumed_batches += 1
            yield batch

    # ----------------------------------------------------------------- state

    def state_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "consumed_batches": self.consumed_batches,
            "seed": self.seed,
            "world_size": self.world_size,
            "rank": self.rank,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "budgets": {
                "max_spectra": self.budgets.max_spectra,
                "max_qwen_tokens": self.budgets.max_qwen_tokens,
                "max_linked_candidates": self.budgets.max_linked_candidates,
            },
            "dataset_size": self.size,
        }

    def load_state_dict(self, state: Mapping) -> None:
        """Restore position. The plan is rebuilt, not stored: it is a pure
        function of ``(seed, epoch)``, so replaying it is exact."""
        if int(state["dataset_size"]) != self.size:
            raise ValueError(
                f"checkpoint was written for {state['dataset_size']} spectra, "
                f"this dataset has {self.size}"
            )
        if int(state["seed"]) != self.seed:
            raise ValueError("checkpoint seed does not match the configured seed")
        if int(state["world_size"]) != self.world_size:
            raise ValueError(
                f"checkpoint used world size {state['world_size']}, this run uses "
                f"{self.world_size}; the batch split would differ"
            )
        self.epoch = int(state["epoch"])
        self._plan = None
        self._diagnostics = None
        self.consumed_batches = int(state["consumed_batches"])
