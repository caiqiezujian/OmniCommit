"""Causal self-attention on packed Thinker sequences without a FlashAttention kernel."""

from contextlib import contextmanager

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


@contextmanager
def non_flash_sdpa(*, efficient):
    backends = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH] if efficient else [SDPBackend.MATH]
    with sdpa_kernel(backends):
        # PyTorch 2.3's public context manager toggles flash, efficient and math
        # but leaves a previously enabled cuDNN SDPA backend untouched.
        restore_cudnn = torch.backends.cuda.cudnn_sdp_enabled()
        if restore_cudnn:
            torch.backends.cuda.enable_cudnn_sdp(False)
        try:
            yield
        finally:
            if restore_cudnn:
                torch.backends.cuda.enable_cudnn_sdp(True)


def _boundaries(cumulative, name):
    if (not isinstance(cumulative, torch.Tensor) or cumulative.ndim != 1
            or cumulative.numel() < 2 or cumulative.dtype not in (torch.int32, torch.int64)):
        raise ValueError(f"{name} must be a one-dimensional integer tensor with at least two entries")
    offsets = cumulative.detach().tolist()
    if offsets[0] != 0 or any(end <= start for start, end in zip(offsets, offsets[1:])):
        raise ValueError(f"{name} must strictly increase from zero")
    return offsets


def packed_causal_attention(query, key, value, cu_seqlens, *, cu_seqlens_padded=None,
                            scale=None, dropout_p=0.0):
    """Compute independent square causal attention for each example in a THD tensor.

    cu_seqlens counts real tokens. If examples were padded before concatenation,
    cu_seqlens_padded gives their starts in physical storage, with trailing
    padding local to each example. Padded outputs and input gradients are zero.
    Output is flattened to [total physical tokens, query_heads * head_dim].
    Q/K/V have already been projected and positions applied by Megatron.
    """
    if (query.ndim != 3 or key.ndim != 3 or key.shape != value.shape
            or query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2]):
        raise ValueError("Packed self-attention requires THD Q/K/V with matching tokens and head dimensions")
    if (not query.is_floating_point() or query.dtype != key.dtype or query.dtype != value.dtype
            or query.device != key.device or query.device != value.device):
        raise ValueError("Q/K/V must have the same floating-point dtype and device")
    total, heads, dim = query.shape
    kv_heads = key.shape[1]
    if min(total, heads, kv_heads, dim) < 1 or heads % kv_heads:
        raise ValueError("Expected positive dimensions and a whole number of query heads per KV head")
    logical = _boundaries(cu_seqlens, "cu_seqlens")
    physical = logical
    if cu_seqlens_padded is not None and cu_seqlens_padded is not cu_seqlens:
        physical = _boundaries(cu_seqlens_padded, "cu_seqlens_padded")
    if len(physical) != len(logical) or physical[-1] != total:
        raise ValueError("Packed boundaries must describe the same examples and cover all physical tokens")

    groups = heads // kv_heads
    outputs = []
    with non_flash_sdpa(efficient=query.is_cuda):
        for i, (start, stop) in enumerate(zip(physical, physical[1:])):
            length = logical[i + 1] - logical[i]
            end = start + length
            if end > stop:
                raise ValueError("Logical sequence length exceeds its physical packed span")
            q = query[start:end]
            k = key[start:end]
            v = value[start:end]
            # Torch's efficient backend expects equal head counts. Only expand
            # the current example, keeping Megatron's contiguous group order.
            if groups > 1:
                k = k.repeat_interleave(groups, dim=1)
                v = v.repeat_interleave(groups, dim=1)
            context = F.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0), dropout_p=dropout_p, is_causal=True, scale=scale,
            )
            outputs.append(context.squeeze(0).transpose(0, 1))
            if end < stop:
                outputs.append(query.new_zeros((stop - end, heads, dim)))
    return torch.cat(outputs, dim=0).reshape(total, heads * dim)
