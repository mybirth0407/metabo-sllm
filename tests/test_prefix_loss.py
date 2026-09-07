"""The prefix-tree marginal: what it reduces to, and how it merges candidates.

With one candidate every prefix node has exactly one child, so the term is that
candidate's teacher-forced NLL spread over its element positions.  With two
candidates that share a prefix, the shared nodes are scored once and the node
where they part gets both children, so the term can only be as small or smaller
than either candidate alone.
"""

from __future__ import annotations

import pytest
import torch

from metabo_sllm.losses.prefix_loss import prefix_marginal_nll
from metabo_sllm.model.formula_decoder import StructuredFormulaDecoder

ELEMENTS = torch.tensor([[6, 1, 8]])  # C, H, O
PRECURSOR = torch.tensor([[6, 12, 6]])
MASK = torch.ones(1, 3, dtype=torch.bool)


def decoder():
    torch.manual_seed(0)
    return StructuredFormulaDecoder(4, hidden_dim=16, num_layers=1, num_heads=2, max_count=16).eval()


def batch(candidates: list[list[int]]) -> dict:
    counts = torch.tensor(candidates).unsqueeze(0)  # [1, C, 3]
    return {
        "candidate_formula_counts": counts,
        "candidate_mask": torch.ones(1, counts.shape[1], dtype=torch.bool),
        "precursor_element_ids": ELEMENTS,
        "precursor_element_counts": PRECURSOR,
        "precursor_element_mask": MASK,
    }


def test_single_candidate_is_its_teacher_forced_nll_per_position():
    model = decoder()
    molecule = torch.randn(1, 4)
    single = batch([[3, 6, 3]])

    loss = prefix_marginal_nll(model, molecule, single)
    log_prob = model.log_prob(molecule, ELEMENTS, PRECURSOR, torch.tensor([[3, 6, 3]]), MASK)

    assert loss.item() == pytest.approx(-log_prob.item() / 3, rel=1e-5)


def test_shared_prefix_merges_children_and_can_only_help():
    model = decoder()
    molecule = torch.randn(1, 4)
    a, b = [3, 6, 3], [3, 6, 2]  # part only at the last element

    pair = prefix_marginal_nll(model, molecule, batch([a, b]))
    alone_a = prefix_marginal_nll(model, molecule, batch([a]))
    alone_b = prefix_marginal_nll(model, molecule, batch([b]))

    assert pair.item() <= min(alone_a.item(), alone_b.item()) + 1e-6


def test_duplicate_candidates_do_not_change_the_term():
    """A node is a node; listing the same formula twice must not double-count it."""
    model = decoder()
    molecule = torch.randn(1, 4)

    once = prefix_marginal_nll(model, molecule, batch([[3, 6, 3]]))
    twice = prefix_marginal_nll(model, molecule, batch([[3, 6, 3], [3, 6, 3]]))

    assert once.item() == pytest.approx(twice.item(), rel=1e-6)


def test_no_candidates_is_zero_with_a_gradient_path():
    model = decoder()
    molecule = torch.randn(1, 4, requires_grad=True)
    empty = batch([[3, 6, 3]])
    empty["candidate_mask"] = torch.zeros(1, 1, dtype=torch.bool)

    loss = prefix_marginal_nll(model, molecule, empty)

    assert loss.item() == 0.0
    assert loss.requires_grad


def test_the_term_reaches_the_decoder_parameters():
    model = decoder().train()
    molecule = torch.randn(1, 4)

    prefix_marginal_nll(model, molecule, batch([[3, 6, 3], [2, 4, 1]])).backward()

    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
