"""Compare the packed text kernel with independent, per-token causal GQA."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from omnicommit_packed_attention import packed_causal_attention
from omnicommit_packed_adapter import _wrap_packed_forward


def per_token_reference(q, k, v, lengths, spans, scale):
    heads, dim = q.shape[1:]
    kv_indices = torch.arange(heads) // (heads // k.size(1))
    scale = dim ** -0.5 if scale is None else scale
    output = []
    start = 0
    for length, span in zip(lengths, spans):
        for t in range(start, start + length):
            keys = k[start:t + 1, kv_indices]
            scores = torch.einsum("hd,thd->ht", q[t], keys) * scale
            values = v[start:t + 1, kv_indices]
            output.append(torch.einsum("ht,thd->hd", scores.softmax(-1), values))
        output.extend(q.new_zeros(heads, dim) for _ in range(span - length))
        start += span
    return torch.stack(output).flatten(1)


@pytest.mark.parametrize("kv_heads,spans,scale", [(2, (2, 5, 2), 0.31), (4, (1, 3, 2), None)])
def test_output_and_gradients_match_per_token_reference(kv_heads, spans, scale):
    generator = torch.Generator().manual_seed(7)
    lengths = (1, 3, 2)
    def offsets(counts):
        return torch.tensor((0, *counts), dtype=torch.int32).cumsum(0, dtype=torch.int32)
    # Strided views cover noncontiguous projections, including grouped heads.
    q, k, v = [torch.randn(sum(spans), heads * 2, 6, dtype=torch.float64, generator=generator)
               [:, ::2, :].requires_grad_() for heads in (4, kv_heads, kv_heads)]
    got = packed_causal_attention(q, k, v, offsets(lengths),
                                 cu_seqlens_padded=offsets(spans) if spans != lengths else None,
                                 scale=scale)
    expected = per_token_reference(q, k, v, lengths, spans, scale)
    torch.testing.assert_close(got, expected)
    upstream = torch.randn(got.shape, dtype=got.dtype, generator=generator)
    for actual, target in zip(torch.autograd.grad(got, (q, k, v), upstream),
                              torch.autograd.grad(expected, (q, k, v), upstream)):
        torch.testing.assert_close(actual, target)


def test_output_of_later_example_has_no_gradient_into_previous_example_or_future_token():
    q, k, v = [torch.randn(7, h, 3, requires_grad=True) for h in (4, 2, 2)]
    output = packed_causal_attention(q, k, v, torch.tensor([0, 4, 7]))
    # Second token of the second sequence only has access to tokens 4 and 5.
    for gradient in torch.autograd.grad(output[5].sum(), (q, k, v)):
        assert gradient[:4].count_nonzero() == 0 and gradient[6:].count_nonzero() == 0
        assert gradient[4:6].count_nonzero() > 0


@pytest.mark.parametrize("logical,physical", [
    ([1, 3, 8], None), ([0, 4, 4, 8], None), ([0, 7], None),
    ([0, 2, 7], [0, 8]), ([0, 4, 6], [0, 3, 8]),
])
def test_invalid_layout_rejected(logical, physical):
    q = torch.zeros(8, 2, 8)
    with pytest.raises(ValueError):
        packed_causal_attention(q, q, q, torch.tensor(logical),
                                cu_seqlens_padded=torch.tensor(physical) if physical else None)


def original_forward(self, query, key, value, attention_mask, attn_mask_type,
                     attention_bias=None, packed_seq_params=None):
    return query, key, value, attention_mask, attn_mask_type, attention_bias, packed_seq_params


def layer_and_inputs():
    config = SimpleNamespace(context_parallel_size=1, attention_dropout=0.,
                             apply_query_key_layer_scaling=False, softmax_scale=0.71)
    layer = SimpleNamespace(config=config, attention_type="self", training=True)
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    padded = torch.tensor([0, 3, 6], dtype=torch.int32)
    params = SimpleNamespace(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu.clone(),
                             cu_seqlens_q_padded=padded, cu_seqlens_kv_padded=padded.clone())
    tensors = [torch.randn(6, h, 3) for h in (4, 2, 2)]
    return layer, params, tensors


def test_adapter_uses_packed_offsets_scale_and_allowed_sdpa_backends(monkeypatch):
    layer, params, (q, k, v) = layer_and_inputs()
    sdpa = torch.nn.functional.scaled_dot_product_attention
    def assert_backends(*args, **kwargs):
        assert not torch.backends.cuda.flash_sdp_enabled()
        assert not torch.backends.cuda.cudnn_sdp_enabled()
        return sdpa(*args, **kwargs)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", assert_backends)
    actual = _wrap_packed_forward(original_forward)(
        layer, q, k, v, None, SimpleNamespace(name="padding_causal"), packed_seq_params=params)
    expected = per_token_reference(q, k, v, (2, 3), (3, 3), layer.config.softmax_scale)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("layout", [None, "sbhd"])
def test_adapter_forwards_dense_calls_exactly(layout):
    layer, _, (q, k, v) = layer_and_inputs()
    params = None if layout is None else SimpleNamespace(qkv_format=layout)
    mask, kind, bias = object(), object(), object()
    expected = (q, k, v, mask, kind, bias, params)
    actual = _wrap_packed_forward(original_forward)(layer, *expected)
    assert len(actual) == len(expected) and all(a is e for a, e in zip(actual, expected))


@pytest.mark.parametrize("unsupported", ["mask", "bias", "noncausal", "cross", "cp",
                                          "logical_mismatch", "padded_mismatch"])
def test_adapter_rejects_unsupported_packed_calls(unsupported):
    layer, params, tensors = layer_and_inputs()
    mask = object() if unsupported == "mask" else None
    bias = object() if unsupported == "bias" else None
    mask_type = SimpleNamespace(name="no_mask" if unsupported == "noncausal" else "causal")
    if unsupported == "cross":
        layer.attention_type = "cross"
    if unsupported == "cp":
        layer.config.context_parallel_size = 2
    if unsupported == "logical_mismatch":
        params.cu_seqlens_kv = torch.tensor([0, 1, 5])
    if unsupported == "padded_mismatch":
        params.cu_seqlens_kv_padded = None
    with pytest.raises(ValueError):
        _wrap_packed_forward(original_forward)(layer, *tensors, mask, mask_type,
                                               attention_bias=bias, packed_seq_params=params)
