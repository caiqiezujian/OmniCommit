"""Megatron adapter for Qwen3-Omni packed training without FlashAttention."""
from functools import wraps
import inspect

import torch

from omnicommit_packed_attention import packed_causal_attention


def _self_attention_offsets(params):
    # SWIFT builds one PackedSeqParams per microbatch. Reuse its host offsets
    # across the decoder layers and checkpoint recomputation.
    cached = getattr(params, "_omnicommit_host_offsets", None)
    if cached is not None:
        return cached
    q, k = params.cu_seqlens_q, params.cu_seqlens_kv
    qp = getattr(params, "cu_seqlens_q_padded", None)
    kp = getattr(params, "cu_seqlens_kv_padded", None)
    if q is None or k is None or (q is not k and not torch.equal(q, k)):
        raise ValueError("Packed self-attention requires identical query and key sequence boundaries")
    if qp is not kp and (qp is None or kp is None or not torch.equal(qp, kp)):
        raise ValueError("Packed self-attention requires identical padded query and key boundaries")
    cached = q.detach().cpu(), qp.detach().cpu() if qp is not None else None
    params._omnicommit_host_offsets = cached
    return cached


def _wrap_packed_forward(original):
    expected = ("self", "query", "key", "value", "attention_mask", "attn_mask_type",
                "attention_bias", "packed_seq_params")
    if tuple(inspect.signature(original).parameters) != expected:
        raise RuntimeError("The installed Megatron TEDotProductAttention.forward signature differs from the supported one")

    @wraps(original)
    def forward(self, query, key, value, attention_mask, attn_mask_type, attention_bias=None, packed_seq_params=None):
        if packed_seq_params is None or packed_seq_params.qkv_format != "thd":
            return original(self, query, key, value, attention_mask, attn_mask_type,
                            attention_bias=attention_bias, packed_seq_params=packed_seq_params)
        config = self.config
        if (attn_mask_type.name not in ("causal", "padding_causal") or attention_mask is not None
                or attention_bias is not None or getattr(self, "attention_type", "self") != "self"):
            raise ValueError("--packed-no-flash requires causal self-attention with sequence boundaries and no extra mask or bias")
        if config.context_parallel_size != 1:
            raise ValueError("--packed-no-flash requires context_parallel_size=1")
        if (config.apply_query_key_layer_scaling or getattr(config, "window_size", None) is not None
                or getattr(config, "softmax_type", "vanilla") != "vanilla" or getattr(config, "fp8", None)):
            raise ValueError("--packed-no-flash supports the Qwen Thinker's full-context vanilla attention in BF16/FP16/FP32")
        if self.training and config.attention_dropout != 0:
            raise ValueError("--packed-no-flash requires attention_dropout=0 for the Qwen Thinker recipe")
        offsets, padded = _self_attention_offsets(packed_seq_params)
        return packed_causal_attention(query, key, value, offsets, cu_seqlens_padded=padded,
                                       scale=getattr(config, "softmax_scale", None))

    forward._omnicommit_packed_no_flash = True
    return forward


def install_packed_no_flash():
    from megatron.core.extensions.transformer_engine import TEDotProductAttention

    torch.backends.cuda.enable_flash_sdp(False)
    if not getattr(TEDotProductAttention.forward, "_omnicommit_packed_no_flash", False):
        TEDotProductAttention.forward = _wrap_packed_forward(TEDotProductAttention.forward)
        print("[OmniCommit] Packed Thinker attention: per-example PyTorch SDPA, FlashAttention disabled.", flush=True)
