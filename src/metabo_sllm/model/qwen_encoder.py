"""Frozen Qwen backbone with LoRA adapters, used as a molecular text encoder.

The whole last hidden state is the molecular memory the slot decoder attends
over -- pooling to a single vector would throw away exactly the token-level
structure (which atoms, which ring, which adduct) the fragment slots need.

Weights are loaded from a local directory only.  If they are not there this
raises instead of reaching for the network or quietly substituting a different
backbone: a different encoder is a different experiment.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

__all__ = [
    "LORA_TARGET_MODULES",
    "MissingBackboneError",
    "QwenEncoder",
    "load_tokenizer",
    "neutralise_torchao_probe",
]

LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


class MissingBackboneError(RuntimeError):
    """Raised when the configured backbone weights are not on disk."""


def neutralise_torchao_probe() -> str | None:
    """Stop peft's torchao probe from raising on an older torchao build.

    While building LoRA layers peft walks a chain of dispatchers, one of which
    asks whether torchao is available.  On this machine torchao is installed but
    predates the version peft wants, and peft's probe *raises* instead of
    reporting absence -- which aborts adapter construction even though no
    torchao quantisation is involved here.  Reporting absence is the honest
    answer for this configuration.

    Returns the original error message when the probe was patched, else None.
    """
    try:
        from peft import import_utils
        from peft.tuners.lora import torchao as lora_torchao
    except ImportError:
        return None
    try:
        import_utils.is_torchao_available()
    except ImportError as exc:
        import_utils.is_torchao_available = lambda: False
        lora_torchao.is_torchao_available = lambda: False
        return str(exc)
    return None


def load_tokenizer(model_name_or_path: str | Path):
    """Tokenizer for the backbone, from local files only."""
    from transformers import AutoTokenizer

    path = Path(model_name_or_path)
    if not path.is_dir():
        raise MissingBackboneError(f"backbone directory not found: {path}")
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class QwenEncoder(nn.Module):
    """Qwen3 backbone, frozen, with LoRA/RSLoRA adapters on the projections."""

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        use_rslora: bool = True,
        target_modules: tuple[str, ...] = LORA_TARGET_MODULES,
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel

        path = Path(model_name_or_path)
        if not path.is_dir():
            raise MissingBackboneError(
                f"backbone directory not found: {path}. Provide local weights; this encoder "
                "never downloads and never substitutes another model."
            )
        if not any(path.glob("*.safetensors")) and not any(path.glob("*.bin")):
            raise MissingBackboneError(f"no model weights under {path}")
        if dtype not in _DTYPES:
            raise ValueError(f"unknown dtype {dtype!r}, expected one of {sorted(_DTYPES)}")

        self.torchao_probe_patched = neutralise_torchao_probe()
        base = AutoModel.from_pretrained(
            str(path), local_files_only=True, dtype=_DTYPES[dtype]
        )
        for parameter in base.parameters():
            parameter.requires_grad_(False)

        lora = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_rslora=use_rslora,
            target_modules=list(target_modules),
            bias="none",
        )
        self.backbone = get_peft_model(base, lora)
        self.hidden_size = int(base.config.hidden_size)
        self.model_name_or_path = str(path)

    def base_parameters(self):
        """Backbone parameters that must never receive gradient."""
        for name, parameter in self.backbone.named_parameters():
            if "lora_" not in name:
                yield name, parameter

    def lora_parameters(self):
        for name, parameter in self.backbone.named_parameters():
            if "lora_" in name:
                yield name, parameter

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Last hidden state of the backbone, ``[B, L, d_qwen]``."""
        output = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False
        )
        return output.last_hidden_state
