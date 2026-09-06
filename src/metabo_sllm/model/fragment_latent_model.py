"""Graph-free fragment-latent model: Qwen memory -> 64 slots -> fragment heads."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from metabo_sllm.chem.candidates import ION_STATE_VOCABULARY
from metabo_sllm.data.supervision import FIXED_SLOT_COUNT
from metabo_sllm.losses.candidate_scoring import (
    matched_bag_nll,
    matching_cost_no_grad,
    reference_candidate_log_prob,
)
from metabo_sllm.losses.fragment_losses import LossWeights, compute_losses
from metabo_sllm.losses.matching import hungarian_assign
from metabo_sllm.model.formula_decoder import StructuredFormulaDecoder
from metabo_sllm.model.heads import IntensityHead, IonStateHead, PresenceHead
from metabo_sllm.model.qwen_encoder import LORA_TARGET_MODULES, QwenEncoder
from metabo_sllm.model.slot_decoder import SlotDecoder

__all__ = ["FragmentLatentModel", "ModelConfig", "parameter_report", "training_step"]


@dataclass
class ModelConfig:
    model_name_or_path: str
    dtype: str = "float32"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    use_rslora: bool = True
    lora_target_modules: tuple[str, ...] = LORA_TARGET_MODULES
    num_slots: int = FIXED_SLOT_COUNT
    slot_hidden_dim: int = 512
    slot_num_layers: int = 4
    slot_num_heads: int = 8
    slot_dropout: float = 0.1
    slot_query_init_std: float = 1.0
    formula_hidden_dim: int = 256
    formula_num_layers: int = 2
    formula_num_heads: int = 4
    formula_max_count: int = 512
    formula_max_elements: int = 32
    formula_dropout: float = 0.0
    head_hidden_dim: int = 256
    head_dropout: float = 0.0
    match_intensity_weight: float = 0.25
    huber_delta: float = 1.0
    matching_chunk_size: int = 64
    matched_chunk_size: int = 256
    return_full_candidate_scores: bool = False


@dataclass
class ModelOutput:
    """Per-slot predictions.

    Deliberately without a ``[B, 64, C]`` candidate score tensor: holding one
    is what made a 1,902-candidate spectrum cost 106 GiB.  Ask for it through
    ``return_full_candidate_scores`` when debugging and it arrives in
    ``extras``.
    """

    slots: torch.Tensor
    presence_logits: torch.Tensor
    presence: torch.Tensor
    intensity: torch.Tensor
    # what the slot actually contributes to the spectrum: presence * amplitude
    contribution: torch.Tensor
    ion_log_prob: torch.Tensor
    extras: dict = field(default_factory=dict)


class FragmentLatentModel(nn.Module):
    """SMILES + metadata -> 64 fragment slots -> formula, ion, presence, intensity."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = QwenEncoder(
            config.model_name_or_path,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            use_rslora=config.use_rslora,
            target_modules=tuple(config.lora_target_modules),
            dtype=config.dtype,
        )
        self.slot_decoder = SlotDecoder(
            self.encoder.hidden_size,
            num_slots=config.num_slots,
            hidden_dim=config.slot_hidden_dim,
            num_layers=config.slot_num_layers,
            num_heads=config.slot_num_heads,
            dropout=config.slot_dropout,
            query_init_std=config.slot_query_init_std,
        )
        self.formula_decoder = StructuredFormulaDecoder(
            config.slot_hidden_dim,
            hidden_dim=config.formula_hidden_dim,
            num_layers=config.formula_num_layers,
            num_heads=config.formula_num_heads,
            max_count=config.formula_max_count,
            max_elements=config.formula_max_elements,
            dropout=config.formula_dropout,
        )
        self.ion_head = IonStateHead(
            config.slot_hidden_dim,
            hidden_dim=config.head_hidden_dim,
            dropout=config.head_dropout,
        )
        self.presence_head = PresenceHead(
            config.slot_hidden_dim,
            hidden_dim=config.head_hidden_dim,
            dropout=config.head_dropout,
        )
        self.intensity_head = IntensityHead(
            config.slot_hidden_dim,
            hidden_dim=config.head_hidden_dim,
            dropout=config.head_dropout,
        )
        self.ion_state_vocabulary = ION_STATE_VOCABULARY

    # ----------------------------------------------------------------- parts

    def encode(self, batch: dict) -> torch.Tensor:
        return self.encoder(batch["input_ids"], batch["attention_mask"])

    def decode_slots(self, batch: dict) -> torch.Tensor:
        memory = self.encode(batch)
        return self.slot_decoder(memory, batch["attention_mask"])

    # ------------------------------------------------------- candidate scoring

    def compute_matching_cost_no_grad(
        self,
        slots: torch.Tensor,
        presence: torch.Tensor,
        intensity: torch.Tensor,
        batch: dict,
        candidate_chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Pass 1: the whole ``[B, 64, T]`` cost, with no gradient graph."""
        return matching_cost_no_grad(
            self.formula_decoder,
            self.ion_head,
            slots,
            presence * intensity,
            batch,
            candidate_chunk_size=candidate_chunk_size or self.config.matching_chunk_size,
            match_intensity_weight=self.config.match_intensity_weight,
            huber_delta=self.config.huber_delta,
        )

    def compute_matched_bag_nll(
        self,
        slots: torch.Tensor,
        batch: dict,
        assignment,
        candidate_chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Pass 2: bag NLL for the matched pairs only, with gradients on."""
        return matched_bag_nll(
            self.formula_decoder,
            self.ion_head,
            slots,
            batch,
            assignment,
            candidate_chunk_size=candidate_chunk_size or self.config.matched_chunk_size,
        )

    # --------------------------------------------------------------- forward

    def forward(self, batch: dict) -> ModelOutput:
        slots = self.decode_slots(batch)
        presence_logits = self.presence_head(slots)
        presence = torch.sigmoid(presence_logits)
        intensity = self.intensity_head(slots)
        # A slot's contribution is gated by its presence. Scoring the raw
        # amplitude instead would let a slot the model believes is absent still
        # account for a peak.
        contribution = presence * intensity
        ion_log_prob = self.ion_head(slots, batch["admissible_ion_mask"])
        return ModelOutput(
            slots=slots,
            presence_logits=presence_logits,
            presence=presence,
            intensity=intensity,
            contribution=contribution,
            ion_log_prob=ion_log_prob,
        )


def training_step(
    model: FragmentLatentModel,
    batch: dict,
    weights: LossWeights | None = None,
    *,
    return_full_candidate_scores: bool = False,
) -> tuple[ModelOutput, object]:
    """Forward, match on a detached cost, then score only the matched pairs.

    ``return_full_candidate_scores`` is a debugging escape hatch: it rebuilds
    the ``[B, 64, C]`` grid the two-pass split exists to avoid, so it defaults
    to off and never runs during training.
    """
    outputs = model(batch)
    cost = model.compute_matching_cost_no_grad(
        outputs.slots, outputs.presence, outputs.intensity, batch
    )
    assignment = hungarian_assign(cost, batch["target_peak_mask"])
    matched = model.compute_matched_bag_nll(outputs.slots, batch, assignment)

    losses = compute_losses(
        matched_bag_nll=matched,
        presence_logits=outputs.presence_logits,
        predicted_intensity=outputs.contribution,
        target_intensities=batch["target_intensities"],
        target_peak_indices=batch["target_peak_indices"],
        target_peak_mask=batch["target_peak_mask"],
        full_peak_intensities=batch["full_peak_intensities"],
        full_peak_mask=batch["full_peak_mask"],
        assignment=assignment,
        weights=weights or LossWeights(),
        huber_delta=model.config.huber_delta,
    )
    outputs.extras["assignment"] = assignment
    outputs.extras["matched_bag_nll"] = matched.detach()
    outputs.extras["matching_cost"] = cost
    if return_full_candidate_scores or model.config.return_full_candidate_scores:
        outputs.extras["candidate_log_prob"] = reference_candidate_log_prob(
            model.formula_decoder, model.ion_head, outputs.slots, batch
        )
    return outputs, losses


def parameter_report(model: FragmentLatentModel) -> dict:
    """Total / trainable / frozen parameter counts, broken down by component."""

    def count(parameters) -> int:
        return sum(p.numel() for p in parameters)

    lora = [p for name, p in model.encoder.lora_parameters()]
    base = [p for name, p in model.encoder.base_parameters()]
    decoder = list(model.slot_decoder.parameters())
    formula = list(model.formula_decoder.parameters())
    heads = (
        list(model.ion_head.parameters())
        + list(model.presence_head.parameters())
        + list(model.intensity_head.parameters())
    )
    total = list(model.parameters())
    trainable = [p for p in total if p.requires_grad]
    return {
        "total": count(total),
        "trainable": count(trainable),
        "frozen": count(total) - count(trainable),
        "qwen_base_frozen": count(base),
        "lora": count(lora),
        "slot_decoder": count(decoder),
        "formula_decoder": count(formula),
        "heads": count(heads),
        "base_requires_grad": any(p.requires_grad for p in base),
    }
