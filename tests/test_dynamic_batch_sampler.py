"""The sampler decides what each GPU sees, so its guarantees are load-bearing."""

from __future__ import annotations

import numpy as np
import pytest

from metabo_sllm.training.dynamic_batch_sampler import BatchBudgets, DynamicBatchSampler

BUDGETS = BatchBudgets(max_spectra=4, max_qwen_tokens=100, max_linked_candidates=200)


def make_costs(size: int = 64, seed: int = 0, oversized: int = 0) -> dict:
    generator = np.random.default_rng(seed)
    tokens = generator.integers(10, 30, size).astype(np.int64)
    candidates = generator.integers(1, 60, size).astype(np.int64)
    targets = generator.integers(1, 64, size).astype(np.int64)
    elements = generator.integers(3, 9, size).astype(np.int64)
    for index in range(oversized):
        candidates[index] = BUDGETS.max_linked_candidates * 3
    return {
        "num_tokens": tokens,
        "num_candidates_linked_to_top64": candidates,
        "num_targets": targets,
        "num_candidate_element_steps": candidates * elements,
    }


def build(**kwargs) -> DynamicBatchSampler:
    settings = {
        "budgets": BUDGETS,
        "world_size": 1,
        "rank": 0,
        "gradient_accumulation_steps": 1,
        "seed": 0,
    }
    costs = kwargs.pop("costs", None) or make_costs()
    settings.update(kwargs)
    return DynamicBatchSampler(costs, **settings)


# ------------------------------------------------------------------- budgets


def test_batches_respect_every_budget():
    sampler = build()
    costs = sampler

    for batch in sampler.plan():
        if len(batch) == 1:
            continue  # a singleton is allowed to exceed a budget
        assert len(batch) <= BUDGETS.max_spectra
        assert int(costs.tokens[batch].sum()) <= BUDGETS.max_qwen_tokens
        assert int(costs.candidates[batch].sum()) <= BUDGETS.max_linked_candidates


def test_oversized_spectrum_becomes_a_singleton_and_is_not_dropped():
    costs = make_costs(oversized=3)
    sampler = build(costs=costs)
    plan = sampler.plan()

    huge = {
        index
        for index in range(costs["num_tokens"].size)
        if costs["num_candidates_linked_to_top64"][index] > BUDGETS.max_linked_candidates
    }
    placed = {index for batch in plan for index in batch}
    for index in huge & placed:
        owner = next(batch for batch in plan if index in batch)
        assert owner == [index], "an oversized spectrum must get a batch of its own"
    assert sampler.diagnostics().singleton_batches >= 1


def test_no_spectrum_is_duplicated_or_split():
    sampler = build()
    seen = [index for batch in sampler.plan() for index in batch]

    assert len(seen) == len(set(seen))


def test_dropped_tail_is_counted_not_hidden():
    sampler = build(world_size=3, gradient_accumulation_steps=2)
    diagnostics = sampler.diagnostics()

    assert diagnostics.used_batches % (3 * 2) == 0
    assert diagnostics.dropped_batches == diagnostics.total_batches - diagnostics.used_batches
    if diagnostics.dropped_batches:
        assert diagnostics.dropped_spectra > 0


def test_workload_diagnostics_are_reported():
    workload = build().diagnostics().workload

    for key in ("spectra", "tokens", "targets", "linked_candidates", "candidate_element_steps"):
        assert set(workload[key]) == {"mean", "min", "p50", "p90", "max"}


# ------------------------------------------------------------- determinism


def test_plan_is_a_function_of_seed_and_epoch():
    first = build()
    second = build()

    assert first.plan() == second.plan()


def test_epoch_changes_the_order():
    sampler = build()
    epoch_zero = sampler.plan()
    sampler.set_epoch(1)
    epoch_one = sampler.plan()

    assert epoch_zero != epoch_one
    assert sorted(i for b in epoch_zero for i in b) != [] and len(epoch_one) > 0


def test_seed_changes_the_order():
    assert build(seed=0).plan() != build(seed=1).plan()


# -------------------------------------------------------------------- ranks


def test_ranks_never_share_a_spectrum():
    world = 4
    samplers = [build(world_size=world, rank=r, gradient_accumulation_steps=2) for r in range(world)]
    per_rank = [
        {index for batch in sampler.batches_for_rank() for index in batch}
        for sampler in samplers
    ]

    for left in range(world):
        for right in range(left + 1, world):
            assert not (per_rank[left] & per_rank[right])


def test_every_rank_runs_the_same_number_of_optimizer_steps():
    world, accumulation = 4, 2
    samplers = [
        build(world_size=world, rank=r, gradient_accumulation_steps=accumulation)
        for r in range(world)
    ]
    counts = [len(sampler.batches_for_rank()) for sampler in samplers]

    assert len(set(counts)) == 1
    assert counts[0] % accumulation == 0
    assert samplers[0].optimizer_steps_per_epoch() == counts[0] // accumulation


def test_rank_slices_cover_the_whole_plan():
    world = 4
    sampler = build(world_size=world, gradient_accumulation_steps=1)
    covered = [
        batch for rank in range(world) for batch in sampler.batches_for_rank(rank)
    ]

    assert sorted(map(tuple, covered)) == sorted(map(tuple, sampler.plan()))


# -------------------------------------------------------------------- state


def test_resume_skips_the_batches_already_consumed():
    sampler = build()
    everything = sampler.batches_for_rank()
    consumed = []
    for position, batch in enumerate(sampler):
        consumed.append(batch)
        if position == 4:
            break
    state = sampler.state_dict()

    resumed = build()
    resumed.load_state_dict(state)
    remaining = resumed.remaining_for_rank()

    assert len(consumed) == 5
    assert remaining == everything[5:]
    assert not any(batch in consumed for batch in remaining)


def test_state_round_trips_through_a_new_sampler():
    sampler = build()
    sampler.set_epoch(2)
    next(iter(sampler))
    restored = build()
    restored.load_state_dict(sampler.state_dict())

    assert restored.epoch == 2
    assert restored.consumed_batches == 1
    assert restored.plan() == sampler.plan()


def test_resume_rejects_a_different_dataset():
    state = build().state_dict()
    other = DynamicBatchSampler(make_costs(size=32, seed=5), budgets=BUDGETS)

    with pytest.raises(ValueError, match="spectra"):
        other.load_state_dict(state)


def test_resume_rejects_a_different_world_size():
    state = build(world_size=2, rank=0).state_dict()

    with pytest.raises(ValueError, match="world size"):
        build(world_size=4, rank=0).load_state_dict(state)


def test_invalid_rank_is_rejected():
    with pytest.raises(ValueError):
        build(world_size=2, rank=2)
