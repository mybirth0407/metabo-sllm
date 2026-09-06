"""Per-slot output heads: ion state, presence, intensity."""

from __future__ import annotations

import torch
from torch import nn

from metabo_sllm.chem.candidates import ION_STATE_VOCABULARY

__all__ = ["IntensityHead", "IonStateHead", "PresenceHead"]


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class IonStateHead(nn.Module):
    """Distribution over ion states, restricted to those the adduct allows."""

    def __init__(self, slot_dim: int, *, hidden_dim: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_states = len(ION_STATE_VOCABULARY)
        self.net = _mlp(slot_dim, hidden_dim, self.num_states, dropout)

    def forward(self, slots: torch.Tensor, admissible_mask: torch.Tensor) -> torch.Tensor:
        """Log-probabilities over ion states, ``[B, S, n_states]``.

        Args:
            slots: ``[B, S, d]``.
            admissible_mask: ``[B, n_states]``, True where the adduct permits it.
        """
        logits = self.net(slots)
        allowed = admissible_mask.unsqueeze(1).expand_as(logits)
        logits = logits.masked_fill(~allowed, float("-inf"))
        return torch.log_softmax(logits, dim=-1)


class PresenceHead(nn.Module):
    """Whether a slot carries a fragment at all."""

    def __init__(self, slot_dim: int, *, hidden_dim: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = _mlp(slot_dim, hidden_dim, 1, dropout)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """Presence logits, ``[B, S]``."""
        return self.net(slots).squeeze(-1)


class IntensityHead(nn.Module):
    """Non-negative intensity contributed by a slot."""

    def __init__(self, slot_dim: int, *, hidden_dim: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = _mlp(slot_dim, hidden_dim, 1, dropout)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """Intensity amplitudes, ``[B, S]``, strictly non-negative."""
        return torch.nn.functional.softplus(self.net(slots).squeeze(-1))
