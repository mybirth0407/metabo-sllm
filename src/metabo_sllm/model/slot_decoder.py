"""Fixed set of learned fragment queries decoded against the Qwen memory.

Each layer lets the slots talk to each other (so they can divide the spectrum
between them), then read the molecule (cross-attention over the backbone's
token states), then transform.

No positional encoding is added to the slots: they are a *set*, and the learned
queries already start out different from one another.  Adding positions would
impose an order the targets do not have.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["SlotDecoder"]


class SlotDecoder(nn.Module):
    def __init__(
        self,
        memory_dim: int,
        *,
        num_slots: int = 64,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        feedforward_multiplier: int = 4,
        query_init_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim

        # Unit-scale, like DETR's query embedding. A small init (0.02) is the
        # usual choice for weights but is wrong for queries: cross-attention
        # adds a common read of the same memory to every slot, and with tiny
        # queries that shared component dominates the residual stream, leaving
        # all 64 slots pointing the same way (cosine ~0.999) before training
        # even starts. The queries have to enter the residual at the scale
        # LayerNorm works in for the slots to stay distinguishable.
        self.queries = nn.Parameter(torch.empty(num_slots, hidden_dim))
        nn.init.normal_(self.queries, std=query_init_std)

        self.memory_proj = nn.Linear(memory_dim, hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)

        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * feedforward_multiplier,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, memory: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Decode slots from a token memory.

        Args:
            memory: backbone hidden states, ``[B, L, d_memory]``.
            attention_mask: ``1`` for real tokens, ``[B, L]``.

        Returns:
            Slot representations, ``[B, num_slots, hidden_dim]``.
        """
        batch = memory.shape[0]
        projected = self.memory_norm(self.memory_proj(memory.to(self.memory_proj.weight.dtype)))
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        padding = ~attention_mask.to(torch.bool)
        slots = self.decoder(queries, projected, memory_key_padding_mask=padding)
        return self.out_norm(slots)
