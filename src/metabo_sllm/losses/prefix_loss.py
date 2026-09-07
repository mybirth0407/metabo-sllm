"""Set-level identity supervision over the candidate prefix tree.

Per-slot identity is learned only through the pairs Hungarian matching picks,
so a slot that is not matched -- or is matched to the wrong peak -- gets no
useful signal about which formulas exist in this spectrum.  This term
supervises the formula decoder directly, conditioned on the molecule rather
than on a slot: at every distinct prefix of the candidate formulas, the counts
that some candidate actually takes next should carry the probability mass.

It is SCARF's prefix-tree objective in the marginal form our categorical head
already speaks, ``-log sum_{k in children} p(k | prefix, molecule)``, with one
term per distinct (spectrum, depth, prefix) node and the mean taken over
nodes.  A spectrum with a single candidate reduces exactly to that candidate's
teacher-forced negative log-probability spread over its element positions.

The decoder is the same module the slots use, so what it learns here about a
molecule's fragments is what the slots decode with.
"""

from __future__ import annotations

import torch

__all__ = ["prefix_marginal_nll"]


def prefix_marginal_nll(decoder, molecule_repr: torch.Tensor, batch: dict) -> torch.Tensor:
    """Mean over prefix nodes of ``-log`` mass on the children candidates take."""
    counts = batch["candidate_formula_counts"]
    candidate_mask = batch["candidate_mask"]
    element_ids = batch["precursor_element_ids"]
    precursor = batch["precursor_element_counts"]
    element_mask = batch["precursor_element_mask"]

    spectrum, candidate = candidate_mask.nonzero(as_tuple=True)
    if spectrum.numel() == 0:
        return molecule_repr.sum() * 0.0

    target = counts[spectrum, candidate]
    present = element_mask[spectrum]
    previous = torch.roll(target, shifts=1, dims=1)
    previous[:, 0] = decoder.bos_count
    logits = decoder.logits(
        molecule_repr[spectrum], element_ids[spectrum], precursor[spectrum], previous, present
    )
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    vocabulary = log_probs.shape[-1]

    per_node = []
    for depth in range(log_probs.shape[1]):
        rows = present[:, depth].nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        # Two candidates share a node when they belong to the same spectrum
        # and agree on every count before this depth.
        key = torch.cat([spectrum[rows].unsqueeze(1), target[rows, :depth]], dim=1)
        nodes, inverse = torch.unique(key, dim=0, return_inverse=True)
        children = torch.zeros(nodes.shape[0], vocabulary, dtype=torch.bool, device=rows.device)
        children[inverse, target[rows, depth].clamp(0, vocabulary - 1)] = True
        # Every member of a node sees identical logits -- same prefix, same
        # conditioning, no dropout -- so score each node once, via its first
        # member.
        first = torch.full((nodes.shape[0],), rows.numel(), dtype=torch.long, device=rows.device)
        first = first.scatter_reduce(
            0, inverse, torch.arange(rows.numel(), device=rows.device), reduce="amin"
        )
        node_log_probs = log_probs[rows[first], depth]
        per_node.append(torch.logsumexp(node_log_probs.masked_fill(~children, float("-inf")), dim=-1))

    return -torch.cat(per_node).mean()
