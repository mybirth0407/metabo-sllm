"""Adapter-only checkpoints.

The Qwen backbone is frozen, so writing its 596M parameters into every
checkpoint would copy 2.4 GB of weights that are already on disk and identical
in every checkpoint.  Only what training actually changes is saved: the LoRA
adapter, the slot/formula/head modules, and enough state to resume the exact
batch order.

Layout::

    checkpoint-00000020/
    ├── adapter/            LoRA weights (peft format)
    ├── model_heads.pt      slot decoder, formula decoder, heads
    ├── optimizer.pt
    ├── scheduler.pt
    ├── trainer_state.json  epoch, global step, sampler position, RNG
    └── manifest.json       config, git commit, supervision manifest hash
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

__all__ = [
    "CHECKPOINT_FILES",
    "CheckpointPaths",
    "head_state_dict",
    "load_checkpoint",
    "save_checkpoint",
    "supervision_manifest_hash",
]

CHECKPOINT_FILES = (
    "adapter",
    "model_heads.pt",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "manifest.json",
)

# Everything trainable that is not the LoRA adapter.
HEAD_MODULES = (
    "slot_decoder",
    "formula_decoder",
    "ion_head",
    "presence_head",
    "intensity_head",
)


@dataclass(frozen=True)
class CheckpointPaths:
    root: Path

    @property
    def adapter(self) -> Path:
        return self.root / "adapter"

    @property
    def heads(self) -> Path:
        return self.root / "model_heads.pt"

    @property
    def optimizer(self) -> Path:
        return self.root / "optimizer.pt"

    @property
    def scheduler(self) -> Path:
        return self.root / "scheduler.pt"

    @property
    def trainer_state(self) -> Path:
        return self.root / "trainer_state.json"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"


def head_state_dict(model) -> dict:
    """Trainable non-adapter weights, including the queries and projection.

    Both of those live inside ``slot_decoder``, so naming the module is enough.
    """
    return {
        name: getattr(model, name).state_dict() for name in HEAD_MODULES
    }


def git_commit(repo: Path) -> dict:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def supervision_manifest_hash(root: str | Path) -> str | None:
    path = Path(root) / "manifest.json"
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def random_states() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_states(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu") else state["torch"])
    if "cuda" not in state or not torch.cuda.is_available():
        return
    # A four-GPU checkpoint may be reloaded on one GPU (and the reverse), so
    # restore per device over the overlap rather than demanding the counts match.
    saved = [tensor.cpu() for tensor in state["cuda"]]
    for index in range(min(len(saved), torch.cuda.device_count())):
        torch.cuda.set_rng_state(saved[index], index)


def save_checkpoint(
    directory: str | Path,
    *,
    model,
    optimizer,
    scheduler,
    sampler_state: dict,
    epoch: int,
    global_step: int,
    config: dict,
    supervision_root: str | Path,
    repo_root: str | Path,
) -> CheckpointPaths:
    """Write a checkpoint; the frozen backbone is deliberately not included."""
    paths = CheckpointPaths(Path(directory))
    paths.root.mkdir(parents=True, exist_ok=True)

    model.encoder.backbone.save_pretrained(str(paths.adapter))
    torch.save(head_state_dict(model), paths.heads)
    torch.save(optimizer.state_dict(), paths.optimizer)
    torch.save(scheduler.state_dict(), paths.scheduler)
    torch.save(random_states(), paths.root / "random_states.pt")

    paths.trainer_state.write_text(
        json.dumps(
            {
                "epoch": epoch,
                "global_step": global_step,
                "sampler": sampler_state,
            },
            indent=2,
        )
        + "\n"
    )
    paths.manifest.write_text(
        json.dumps(
            {
                "config": config,
                "git": git_commit(Path(repo_root)),
                "supervision_root": str(supervision_root),
                "supervision_manifest_sha256": supervision_manifest_hash(supervision_root),
                "contains_backbone_weights": False,
                "head_modules": list(HEAD_MODULES),
            },
            indent=2,
        )
        + "\n"
    )
    return paths


def load_checkpoint(
    directory: str | Path,
    *,
    model,
    optimizer=None,
    scheduler=None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict:
    """Restore weights and training position from a checkpoint directory."""
    paths = CheckpointPaths(Path(directory))
    if not paths.trainer_state.is_file():
        raise FileNotFoundError(f"not a checkpoint directory: {paths.root}")

    heads = torch.load(paths.heads, map_location=map_location, weights_only=False)
    for name in HEAD_MODULES:
        getattr(model, name).load_state_dict(heads[name])

    _load_adapter(model.encoder.backbone, paths.adapter, map_location)

    if optimizer is not None and paths.optimizer.is_file():
        optimizer.load_state_dict(
            torch.load(paths.optimizer, map_location=map_location, weights_only=False)
        )
    if scheduler is not None and paths.scheduler.is_file():
        scheduler.load_state_dict(
            torch.load(paths.scheduler, map_location=map_location, weights_only=False)
        )

    rng_path = paths.root / "random_states.pt"
    if restore_rng and rng_path.is_file():
        restore_random_states(torch.load(rng_path, map_location="cpu", weights_only=False))

    return json.loads(paths.trainer_state.read_text())


def _adapter_tensors(directory: Path, map_location) -> dict:
    """Raw tensors as ``save_pretrained`` wrote them, with no key surgery."""
    from safetensors.torch import load_file

    safetensors = directory / "adapter_model.safetensors"
    binary = directory / "adapter_model.bin"
    if safetensors.is_file():
        return load_file(str(safetensors))
    if binary.is_file():
        return torch.load(binary, map_location=map_location, weights_only=False)
    return {}


def _load_adapter(backbone, directory: Path, map_location) -> None:
    """Put the adapter weights back.

    peft renames parameters when it wraps a model, so its own loader is the
    only thing that reliably maps a saved adapter back onto the live modules;
    hand-written key rewriting drifts the moment peft changes its layout.
    """
    try:
        from peft import PeftModel, set_peft_model_state_dict
    except ImportError:
        PeftModel = None

    if PeftModel is not None and isinstance(backbone, PeftModel):
        from peft import load_peft_weights

        weights = load_peft_weights(str(directory), device=str(map_location))
        set_peft_model_state_dict(backbone, weights)
        return

    tensors = _adapter_tensors(directory, map_location)
    if not tensors:
        return
    result = backbone.load_state_dict(tensors, strict=False)
    unexpected = list(getattr(result, "unexpected_keys", []))
    if unexpected:
        raise RuntimeError(f"adapter checkpoint has unexpected keys: {unexpected[:5]}")
