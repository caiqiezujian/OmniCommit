#!/usr/bin/env python3
"""Tiny BF16 forward/backward through the installed patched Megatron text layer."""
import os
from unittest.mock import patch

from check_packed_attention import check_kernel


def main():
    os.environ.update(NVTE_FLASH_ATTN="0", NVTE_FUSED_ATTN="0", NVTE_UNFUSED_ATTN="1")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Run this packed attention check in the training container on a visible GPU.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(23)

    # Match the trainer's Megatron source selection and TE compatibility patches.
    import swift.megatron
    from megatron.core.extensions.transformer_engine import TEDotProductAttention
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.transformer_config import TransformerConfig
    from omnicommit_packed_adapter import install_packed_no_flash

    install_packed_no_flash()
    config = TransformerConfig(num_layers=1, hidden_size=4096, num_attention_heads=32,
                               num_query_groups=4, kv_channels=128, attention_dropout=0.0,
                               bf16=True, params_dtype=torch.bfloat16)
    layer = TEDotProductAttention(config, layer_number=1, attn_mask_type=AttnMaskType.causal,
                                  attention_type="self").to(device)

    def packed_forward(q, k, v, **lengths):
        params = PackedSeqParams(qkv_format="thd", **lengths)
        return layer(q, k, v, attention_mask=None, attn_mask_type=AttnMaskType.causal,
                     packed_seq_params=params)

    sdpa = torch.nn.functional.scaled_dot_product_attention
    def checked_sdpa(*args, **kwargs):
        if torch.backends.cuda.flash_sdp_enabled() or torch.backends.cuda.cudnn_sdp_enabled():
            raise AssertionError("The packed text path left Flash or cuDNN SDPA enabled.")
        return sdpa(*args, **kwargs)

    print(f"[no-flash-packing] {torch.cuda.get_device_name(device)}; Megatron BF16 packed "
          "causal self-attention, 32 query heads / 4 KV heads, head size 128", flush=True)
    # Fail if Megatron ever delegates this packed call to TE's attention kernels.
    with patch.object(TEDotProductAttention.__base__, "forward",
                      side_effect=AssertionError("Packed call unexpectedly reached TE attention")), \
            patch.object(torch.nn.functional, "scaled_dot_product_attention", side_effect=checked_sdpa) as calls:
        check_kernel(torch, packed_forward, device)
        if not calls.called:
            raise AssertionError("Packed forward did not enter the segmented PyTorch SDPA path.")
    torch.cuda.synchronize(device)
    print("[no-flash-packing] PASS: packed outputs match two isolated reference sequences; "
          "Q/K/V gradients are finite and nonzero.", flush=True)


if __name__ == "__main__":
    main()
