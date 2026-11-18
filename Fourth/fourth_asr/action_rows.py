"""Small, independently trainable corrections to selected vocabulary rows."""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class ActionRows(nn.Module):
    """Learn a [number of action tokens, hidden size] delta on a frozen table."""

    def __init__(self, weight: torch.Tensor, token_ids: Sequence[int]) -> None:
        super().__init__()
        ids = tuple(token_ids)
        if weight.ndim != 2 or not weight.is_floating_point():
            raise ValueError("expected a floating-point vocabulary matrix")
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("action token IDs must be nonempty and unique")
        if any(type(i) is not int or not 0 <= i < weight.size(0) for i in ids):
            raise ValueError("action token ID outside the vocabulary")
        self.register_buffer(
            "token_ids", torch.tensor(ids, dtype=torch.long, device=weight.device),
            persistent=False,
        )
        self.delta = nn.Parameter(weight.new_zeros(len(ids), weight.size(1)))

    def embedding(self, input_ids: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        """Apply the input-table correction at action positions, including repeated IDs."""
        if base_output.shape[:-1] != input_ids.shape:
            raise ValueError("input IDs and embedding positions must have matching shapes")
        matches = input_ids.unsqueeze(-1) == self.token_ids
        correction = matches.to(self.delta.dtype) @ self.delta
        return base_output + correction.to(base_output.dtype)

    def logits(self, hidden_states: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        """Apply the output-table correction to selected columns of the full vocabulary logits."""
        correction = F.linear(hidden_states, self.delta.to(hidden_states.dtype))
        return base_logits.index_add(-1, self.token_ids, correction.to(base_logits.dtype))

    def effective_rows(self, weight: torch.Tensor) -> torch.Tensor:
        """Materialize only the rows needed to write ordinary dense checkpoint weights."""
        return weight.index_select(0, self.token_ids) + self.delta.to(weight.dtype)
