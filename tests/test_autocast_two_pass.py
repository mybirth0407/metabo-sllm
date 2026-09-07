"""Gradients must survive a no-grad pass that runs first under autocast.

The candidate scorer evaluates the formula decoder twice per step inside one
mixed-precision context: without gradients to build the matching cost, then
with gradients on the matched pairs.  Autocast caches the bf16 copy of each
fp32 weight for the length of the context; a copy first made under ``no_grad``
carries no autograd history, so the gradient pass reused it and the decoder's
linear weights -- but not its LayerNorms or embeddings, which stay in fp32 --
received nothing.  Every run trained that way left the decoder body and the
count head at their initialisation.

This pins the trainer's autocast context to the behaviour that keeps the
gradient.  It needs a CUDA device, which is where bf16 autocast exists.
"""

from __future__ import annotations

import pytest
import torch

from metabo_sllm.model.formula_decoder import StructuredFormulaDecoder
from metabo_sllm.training.trainer import autocast_context

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 autocast needs CUDA")


def _two_pass_gradients(context) -> tuple[bool, bool]:
    torch.manual_seed(0)
    decoder = StructuredFormulaDecoder(8, hidden_dim=32, num_layers=1, num_heads=4, max_count=16)
    decoder = decoder.cuda()
    slot = torch.randn(4, 8, device="cuda")
    elements = torch.tensor([[6, 1, 8]] * 4, device="cuda")
    precursor = torch.tensor([[6, 12, 6]] * 4, device="cuda")
    mask = torch.ones(4, 3, dtype=torch.bool, device="cuda")
    target = torch.tensor([[3, 6, 3]] * 4, device="cuda")

    with context:
        with torch.no_grad():
            previous = torch.roll(target, shifts=1, dims=1)
            previous[:, 0] = decoder.bos_count
            decoder.logits(slot, elements, precursor, previous, mask)
        loss = -decoder.log_prob(slot, elements, precursor, target, mask).sum()
    loss.backward()

    head = decoder.count_head.weight.grad
    norm = decoder.input_norm.weight.grad
    return (head is not None and bool(head.abs().sum() > 0)), (norm is not None)


def test_the_trainer_context_keeps_the_linear_gradients():
    head_has_grad, norm_has_grad = _two_pass_gradients(autocast_context("bf16"))

    assert norm_has_grad
    assert head_has_grad


def test_the_default_cache_is_what_lost_them():
    """The failure mode, kept so the fix cannot be undone without noticing."""
    default = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    head_has_grad, norm_has_grad = _two_pass_gradients(default)

    assert norm_has_grad
    assert not head_has_grad
