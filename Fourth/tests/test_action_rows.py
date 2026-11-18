"""Exercise row-limited training with the ordinary full-vocabulary ASR loss."""

from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_asr.action_rows import ActionRows


def test_action_rows_match_dense_forward_and_gradients():
    torch.manual_seed(4)
    ids = [6, 2]
    embedding_weight = torch.randn(8, 5, dtype=torch.float64)
    output_weight = torch.randn_like(embedding_weight)
    inputs = torch.tensor([[0, 6, 6], [1, 2, 0]])
    # Include ordinary transcription targets and both actions.
    labels = torch.tensor([1, 6, 2, 0, 3, 7])
    inputs_rows = ActionRows(embedding_weight, ids)
    output_rows = ActionRows(output_weight, ids)
    with torch.no_grad():
        inputs_rows.delta.normal_()
        output_rows.delta.normal_()
    transform = torch.randn(5, 5, dtype=torch.float64, requires_grad=True)

    hidden = inputs_rows.embedding(inputs, F.embedding(inputs, embedding_weight)) @ transform
    logits = output_rows.logits(hidden, F.linear(hidden, output_weight))
    F.cross_entropy(logits.flatten(0, 1), labels).backward()

    dense_embedding = embedding_weight.index_copy(0, inputs_rows.token_ids, inputs_rows.effective_rows(embedding_weight))
    dense_head = output_weight.index_copy(0, output_rows.token_ids, output_rows.effective_rows(output_weight))
    dense_embedding = dense_embedding.detach().requires_grad_()
    dense_head = dense_head.detach().requires_grad_()
    dense_transform = transform.detach().requires_grad_()
    expected = F.linear(F.embedding(inputs, dense_embedding) @ dense_transform, dense_head)
    F.cross_entropy(expected.flatten(0, 1), labels).backward()
    torch.testing.assert_close(logits, expected)
    torch.testing.assert_close(inputs_rows.delta.grad, dense_embedding.grad[ids])
    torch.testing.assert_close(output_rows.delta.grad, dense_head.grad[ids])
    torch.testing.assert_close(transform.grad, dense_transform.grad)


def test_adamw_only_owns_action_rows_and_keeps_ordinary_rows_bitwise_intact():
    torch.manual_seed(7)
    embedding = nn.Embedding(9, 4).requires_grad_(False)
    head = nn.Linear(4, 9, bias=False).requires_grad_(False)
    ids = [2, 8]
    in_rows = ActionRows(embedding.weight, ids)
    out_rows = ActionRows(head.weight, ids)
    in_before = embedding.weight.detach().clone()
    out_before = head.weight.detach().clone()
    optimizer = torch.optim.AdamW([in_rows.delta, out_rows.delta], lr=.05, weight_decay=.1)
    inputs = torch.tensor([2, 8, 3])
    hidden = in_rows.embedding(inputs, embedding(inputs))
    logits = out_rows.logits(hidden, head(hidden))
    torch.testing.assert_close(logits, head(embedding(inputs)), rtol=0, atol=0)
    # Even with only ordinary words as targets, loss reaches the trainable rows.
    F.cross_entropy(logits, torch.tensor([1, 3, 0])).backward()
    optimizer.step()
    for base, original, updates in ((embedding.weight, in_before, in_rows), (head.weight, out_before, out_rows)):
        assert base.grad is None
        assert torch.equal(base, original)
        assert optimizer.state[updates.delta]["exp_avg"].shape == (2, 4)
        effective = base.index_copy(0, updates.token_ids, updates.effective_rows(base))
        changed_rows = (effective != original).any(dim=1).nonzero().flatten().tolist()
        assert changed_rows == ids


@pytest.mark.parametrize("bad_ids", [[], [2, 2], [-1, 2], [1, 6], [True]])
def test_invalid_row_ids_fail_before_training(bad_ids):
    with pytest.raises(ValueError, match="ID"):
        ActionRows(torch.randn(6, 3), bad_ids)
