#!/usr/bin/env python3
"""Probe the Thinker's packed BF16 attention kernel on the first visible GPU.

Run in the training container with the same CUDA_VISIBLE_DEVICES as the launcher.
Tests one local TE FlashAttention forward/backward with two independent examples.
The full model, audio collator, distributed training and checkpointing require
a separate real SFT smoke run.
"""

import math
import os


def reference_sequence(torch, q, k, v):
    """Independent FP32 causal GQA reference for one short sequence (THD)."""
    groups = q.size(1) // k.size(1)
    q = q.transpose(0, 1).float()
    k = k.transpose(0, 1).float().repeat_interleave(groups, dim=0)
    v = v.transpose(0, 1).float().repeat_interleave(groups, dim=0)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.size(-1))
    future_tokens = torch.ones(scores.shape[-2:], dtype=torch.bool, device=q.device).triu(1)
    weights = scores.masked_fill(future_tokens, float("-inf")).softmax(dim=-1)
    return (weights @ v).transpose(0, 1)


def check_kernel(torch, attention, device):
    lengths = (37, 79)
    total = sum(lengths)
    head_size = 128
    cu = torch.tensor([0, lengths[0], total], dtype=torch.int32, device=device)
    q, k, v = [
        torch.randn(total, heads, head_size, device=device, dtype=torch.bfloat16, requires_grad=True)
        for heads in (32, 4, 4)
    ]
    output = attention(q, k, v, cu_seqlens_q=cu, cu_seqlens_kv=cu,
                       max_seqlen_q=max(lengths), max_seqlen_kv=max(lengths))
    # TE releases return either flattened hidden size or explicit heads.
    output = output.reshape_as(q)
    with torch.no_grad():
        expected = torch.cat([
            reference_sequence(torch, q_part, k_part, v_part)
            for q_part, k_part, v_part in zip(q.split(lengths), k.split(lengths), v.split(lengths))
        ])
        torch.testing.assert_close(output.float(), expected, rtol=0.025, atol=0.02)

    output.backward(torch.randn_like(output))
    for name, tensor in (("Q", q), ("K", k), ("V", v)):
        gradient = tensor.grad
        if gradient is None or not gradient.isfinite().all() or not gradient.count_nonzero():
            raise RuntimeError(f"Packed attention produced missing, nonfinite or all-zero {name} gradients.")


def main():
    # Set before importing TE; ensure the test cannot silently use the cuDNN or unfused backend.
    os.environ.update(NVTE_FLASH_ATTN="1", NVTE_FUSED_ATTN="0", NVTE_UNFUSED_ATTN="0")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the packed FlashAttention probe.")
    from transformer_engine.pytorch.attention import DotProductAttention

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(23)
    print(f"[packed-attention] {torch.cuda.get_device_name(device)}, CUDA {torch.version.cuda}, "
          "TE FlashAttention only, BF16 THD, 32Q/4KV x 128, causal", flush=True)
    attention = DotProductAttention(num_attention_heads=32, num_gqa_groups=4, kv_channels=128,
                                    attention_dropout=0.0, qkv_format="thd", attn_mask_type="padding_causal")
    check_kernel(torch, attention, device)
    torch.cuda.synchronize(device)
    print("[packed-attention] PASS: isolated outputs and finite Q/K/V gradients.", flush=True)


if __name__ == "__main__":
    main()
