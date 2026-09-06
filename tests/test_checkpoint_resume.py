"""A checkpoint must carry the training state and none of the frozen backbone."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from metabo_sllm.training.checkpoint import (
    CHECKPOINT_FILES,
    HEAD_MODULES,
    CheckpointPaths,
    head_state_dict,
    load_checkpoint,
    save_checkpoint,
    supervision_manifest_hash,
)
from metabo_sllm.training.dynamic_batch_sampler import BatchBudgets, DynamicBatchSampler
from metabo_sllm.training.trainer import build_optimizer, build_scheduler


class FakeBackbone(nn.Module):
    """Stands in for the peft-wrapped Qwen: huge, frozen, and never saved."""

    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(64, 64)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.randn(4, 64))

    def save_pretrained(self, directory: str) -> None:
        from pathlib import Path

        from safetensors.torch import save_file

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        save_file({"lora_A": self.lora_A.detach().clone()}, str(path / "adapter_model.safetensors"))
        (path / "adapter_config.json").write_text("{}")

    def load_state_dict(self, state, strict=True):  # noqa: D401 - torch signature
        return super().load_state_dict(state, strict=strict)


class FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = FakeBackbone()


class FakeModel(nn.Module):
    """Same attribute names the real model exposes to the checkpointer."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = FakeEncoder()
        self.slot_decoder = nn.Linear(8, 8)
        self.formula_decoder = nn.Linear(8, 8)
        self.ion_head = nn.Linear(8, 3)
        self.presence_head = nn.Linear(8, 1)
        self.intensity_head = nn.Linear(8, 1)


def make_sampler() -> DynamicBatchSampler:
    generator = np.random.default_rng(0)
    size = 40
    costs = {
        "num_tokens": generator.integers(10, 20, size).astype(np.int64),
        "num_candidates_linked_to_top64": generator.integers(1, 20, size).astype(np.int64),
        "num_targets": generator.integers(1, 10, size).astype(np.int64),
        "num_candidate_element_steps": generator.integers(10, 100, size).astype(np.int64),
    }
    return DynamicBatchSampler(
        costs, budgets=BatchBudgets(max_spectra=4, max_qwen_tokens=80, max_linked_candidates=60)
    )


def write(tmp_path, step: int = 10):
    model = FakeModel()
    optimizer = build_optimizer(
        model, learning_rate=1e-4, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8, fused=False
    )
    scheduler = build_scheduler(optimizer, total_steps=30, warmup_ratio=0.1, min_lr_ratio=0.1)
    sampler = make_sampler()
    for _ in range(step):
        for parameter in optimizer.param_groups[0]["params"]:
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
        next(iter(sampler))
    supervision = tmp_path / "supervision"
    supervision.mkdir()
    (supervision / "manifest.json").write_text('{"schema_version": "fragment_supervision_v1"}')

    directory = tmp_path / f"checkpoint-{step:08d}"
    save_checkpoint(
        directory,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler_state=sampler.state_dict(),
        epoch=0,
        global_step=step,
        config={"training": {"max_steps": 30}},
        supervision_root=supervision,
        repo_root=tmp_path,
    )
    return model, optimizer, scheduler, sampler, directory, supervision


# ------------------------------------------------------------------ contents


def test_checkpoint_writes_the_expected_files(tmp_path):
    *_, directory, _ = write(tmp_path)
    paths = CheckpointPaths(directory)

    for name in CHECKPOINT_FILES:
        assert (directory / name).exists(), name
    assert paths.adapter.is_dir()
    assert (paths.adapter / "adapter_model.safetensors").is_file()


def test_no_backbone_weights_are_copied(tmp_path):
    *_, directory, _ = write(tmp_path)

    heads = torch.load(directory / "model_heads.pt", weights_only=False)
    assert set(heads) == set(HEAD_MODULES)
    flattened = [f"{module}.{key}" for module, state in heads.items() for key in state]
    assert not any("encoder" in name or "backbone" in name for name in flattened)

    from safetensors.torch import load_file

    adapter = load_file(str(directory / "adapter" / "adapter_model.safetensors"))
    assert set(adapter) == {"lora_A"}
    assert "base.weight" not in adapter

    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["contains_backbone_weights"] is False


def test_manifest_records_provenance(tmp_path):
    *_, directory, supervision = write(tmp_path)
    manifest = json.loads((directory / "manifest.json").read_text())

    assert manifest["supervision_manifest_sha256"] == supervision_manifest_hash(supervision)
    assert manifest["config"]["training"]["max_steps"] == 30
    assert set(manifest["git"]) == {"commit", "branch", "dirty"}


def test_head_state_dict_covers_queries_and_projection():
    model = FakeModel()
    state = head_state_dict(model)

    assert set(state) == set(HEAD_MODULES)
    # the learned queries and the Qwen-to-slot projection live in slot_decoder
    assert "slot_decoder" in state


# -------------------------------------------------------------------- resume


def test_resume_restores_step_scheduler_and_optimizer(tmp_path):
    _, optimizer, scheduler, sampler, directory, _ = write(tmp_path, step=10)
    expected_lr = scheduler.get_last_lr()[0]
    expected_state = optimizer.state_dict()["state"]

    fresh_model = FakeModel()
    fresh_optimizer = build_optimizer(
        fresh_model, learning_rate=1e-4, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8, fused=False
    )
    fresh_scheduler = build_scheduler(
        fresh_optimizer, total_steps=30, warmup_ratio=0.1, min_lr_ratio=0.1
    )
    state = load_checkpoint(
        directory, model=fresh_model, optimizer=fresh_optimizer, scheduler=fresh_scheduler
    )

    assert state["global_step"] == 10
    assert fresh_scheduler.get_last_lr()[0] == pytest.approx(expected_lr)
    assert set(fresh_optimizer.state_dict()["state"]) == set(expected_state)
    for key, value in expected_state.items():
        torch.testing.assert_close(
            fresh_optimizer.state_dict()["state"][key]["exp_avg"], value["exp_avg"]
        )


def test_resume_restores_head_weights(tmp_path):
    model, *_, directory, _ = write(tmp_path)
    fresh = FakeModel()

    assert not torch.allclose(fresh.slot_decoder.weight, model.slot_decoder.weight)
    load_checkpoint(directory, model=fresh)
    torch.testing.assert_close(fresh.slot_decoder.weight, model.slot_decoder.weight)
    torch.testing.assert_close(fresh.presence_head.bias, model.presence_head.bias)


def test_resume_continues_the_sampler_without_repeating(tmp_path):
    *_, sampler, directory, _ = write(tmp_path, step=10)
    consumed = sampler.batches_for_rank()[:10]

    state = json.loads((directory / "trainer_state.json").read_text())
    resumed = make_sampler()
    resumed.load_state_dict(state["sampler"])
    remaining = resumed.remaining_for_rank()

    assert resumed.consumed_batches == 10
    assert remaining == sampler.batches_for_rank()[10:]
    assert not any(batch in consumed for batch in remaining)


def test_loading_a_non_checkpoint_directory_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path, model=FakeModel())
