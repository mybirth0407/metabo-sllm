"""Autoregressive decoder over per-element atom counts.

A fragment formula is not a class label.  Treating it as one would need a
vocabulary the size of the subformula space and would let the model emit
formulas the precursor cannot supply.  Instead the decoder walks the
precursor's own elements in atomic-number order and predicts each count, with
the logits above ``precursor[e]`` masked to ``-inf`` -- so an impossible count
has probability exactly zero rather than a small one the model must learn to
avoid.

The empty formula is excluded the same way, and at every stage: if every
element so far took count zero, the *last* element cannot take zero either.
Because that constraint is a function of the counts already emitted, it holds
identically under teacher forcing and under greedy decoding, and the decoder
represents ``p(F | F != empty, z, P)`` -- a proper distribution whose mass over
non-empty formulas sums to one.  Forbidding the empty formula only at inference
time, as an earlier version did, would have left training probability mass on a
fragment that cannot exist.

Only elements the precursor actually contains are decoded; padded element
slots contribute nothing to the log-probability.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["StructuredFormulaDecoder"]


class StructuredFormulaDecoder(nn.Module):
    def __init__(
        self,
        slot_dim: int,
        *,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        max_count: int = 512,
        max_atomic_number: int = 118,
        max_elements: int = 32,
        # Zero by default: candidates are scored twice per step -- once to
        # build the matching cost, once for the matched pairs -- and dropout
        # would make those two passes disagree, so the assignment would be
        # chosen against a different distribution than the loss is taken from.
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_count = max_count
        self.max_elements = max_elements
        self.hidden_dim = hidden_dim
        # counts 0..max_count, plus one "start of sequence" symbol
        self.bos_count = max_count + 1

        self.element_embedding = nn.Embedding(max_atomic_number + 1, hidden_dim, padding_idx=0)
        self.precursor_count_embedding = nn.Embedding(max_count + 1, hidden_dim)
        self.previous_count_embedding = nn.Embedding(max_count + 2, hidden_dim)
        self.position_embedding = nn.Embedding(max_elements, hidden_dim)
        self.slot_proj = nn.Linear(slot_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.body = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.count_head = nn.Linear(hidden_dim, max_count + 1)

    # ------------------------------------------------------------------ core

    def _validate(self, precursor_counts: torch.Tensor, element_mask: torch.Tensor) -> None:
        if element_mask.shape[1] > self.max_elements:
            raise ValueError(
                f"formula has {element_mask.shape[1]} elements, above max_elements="
                f"{self.max_elements}"
            )
        if not precursor_counts.numel():
            return
        largest = int(precursor_counts.masked_fill(~element_mask, 0).max().item())
        if largest > self.max_count:
            raise ValueError(
                f"precursor element count {largest} exceeds the count vocabulary "
                f"(0..{self.max_count}); refusing to clip"
            )
        if not element_mask.any(dim=1).all():
            raise ValueError("a precursor formula has no elements")
        # Every present element must supply at least one atom, otherwise
        # "non-empty" is unreachable and the masked distribution is empty.
        if int(precursor_counts.masked_fill(~element_mask, 1).min().item()) < 1:
            raise ValueError(
                "a precursor element has count 0; no non-empty subformula exists"
            )

    def _last_element(self, element_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(element_mask.shape[1], device=element_mask.device)
        return (element_mask.to(torch.int64) * positions).max(dim=1).values

    def _hidden(
        self,
        slot_repr: torch.Tensor,
        element_ids: torch.Tensor,
        precursor_counts: torch.Tensor,
        previous_counts: torch.Tensor,
        element_mask: torch.Tensor,
    ) -> torch.Tensor:
        n, length = element_ids.shape
        positions = torch.arange(length, device=element_ids.device).unsqueeze(0).expand(n, -1)
        features = (
            self.element_embedding(element_ids)
            + self.precursor_count_embedding(precursor_counts.clamp(0, self.max_count))
            + self.previous_count_embedding(previous_counts.clamp(0, self.bos_count))
            + self.position_embedding(positions)
            + self.slot_proj(slot_repr).unsqueeze(1)
        )
        features = self.input_norm(features)
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=element_ids.device), diagonal=1
        )
        # A fully padded row would make every key invalid; keep such rows alive
        # and drop them later through element_mask.
        padding = ~element_mask
        padding = torch.where(padding.all(dim=1, keepdim=True), torch.zeros_like(padding), padding)
        hidden = self.body(features, mask=causal, src_key_padding_mask=padding)
        return self.out_norm(hidden)

    def logits(
        self,
        slot_repr: torch.Tensor,
        element_ids: torch.Tensor,
        precursor_counts: torch.Tensor,
        previous_counts: torch.Tensor,
        element_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Count logits per element position, ``[N, R, max_count + 1]``.

        Two masks, both to ``-inf`` so the probabilities are exactly zero:
        counts above the precursor's supply, and -- on the last element of a
        formula whose earlier counts were all zero -- the count zero that would
        complete the empty formula.
        """
        hidden = self._hidden(
            slot_repr, element_ids, precursor_counts, previous_counts, element_mask
        )
        logits = self.count_head(hidden)
        grid = torch.arange(self.max_count + 1, device=logits.device)
        allowed = grid.view(1, 1, -1) <= precursor_counts.unsqueeze(-1)
        logits = logits.masked_fill(~allowed, float("-inf"))

        # previous_counts[t] holds f_{t-1} (and a start symbol at t = 0), so a
        # cumulative sum gives the atoms committed before position t.
        emitted = previous_counts.clone()
        emitted[:, 0] = 0
        emitted = emitted.masked_fill(~element_mask, 0)
        committed = torch.cumsum(emitted, dim=1)

        positions = torch.arange(element_ids.shape[1], device=logits.device).unsqueeze(0)
        is_last = element_mask & (positions == self._last_element(element_mask).unsqueeze(1))
        forbid_zero = is_last & (committed == 0)
        zero_logits = logits[:, :, 0].masked_fill(forbid_zero, float("-inf"))
        return torch.cat([zero_logits.unsqueeze(-1), logits[:, :, 1:]], dim=-1)

    # -------------------------------------------------------------- training

    def log_prob(
        self,
        slot_repr: torch.Tensor,
        element_ids: torch.Tensor,
        precursor_counts: torch.Tensor,
        target_counts: torch.Tensor,
        element_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced ``log p(F | slot, precursor)``, ``[N]``."""
        self._validate(precursor_counts, element_mask)
        previous = torch.roll(target_counts, shifts=1, dims=1)
        previous[:, 0] = self.bos_count
        logits = self.logits(
            slot_repr, element_ids, precursor_counts, previous, element_mask
        )
        log_probs = torch.log_softmax(logits, dim=-1)
        gathered = log_probs.gather(2, target_counts.clamp(0, self.max_count).unsqueeze(-1))
        gathered = gathered.squeeze(-1)
        return (gathered * element_mask.to(gathered.dtype)).sum(dim=1)

    # ------------------------------------------------------------- inference

    @torch.no_grad()
    def greedy_decode(
        self,
        slot_repr: torch.Tensor,
        element_ids: torch.Tensor,
        precursor_counts: torch.Tensor,
        element_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Greedy counts per element, ``[N, R]``.

        Uses exactly the masked distribution :meth:`log_prob` scores, so the
        empty formula is unreachable here for the same reason it has zero
        probability during training.
        """
        self._validate(precursor_counts, element_mask)
        _, length = element_ids.shape
        counts = torch.zeros_like(element_ids)
        previous = torch.full_like(element_ids, self.bos_count)

        for step in range(length):
            logits = self.logits(
                slot_repr, element_ids, precursor_counts, previous, element_mask
            )[:, step, :]
            choice = logits.argmax(dim=-1)
            counts[:, step] = torch.where(
                element_mask[:, step], choice, torch.zeros_like(choice)
            )
            if step + 1 < length:
                previous[:, step + 1] = counts[:, step]
        return counts * element_mask.to(counts.dtype)
