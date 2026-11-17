"""Configure the Third Web service and instantiate its ASR backend."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .streaming import ReplayBackend, StreamOptions


def add_runtime_arguments(parser):
    parser.add_argument("--backend", choices=("hf", "vllm", "replay"), default="hf")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--replies", type=Path, help="Replay only: JSON array of scripted replies; never real ASR")
    parser.add_argument("--number-format", choices=("verbatim", "normalized"), default="verbatim")
    parser.add_argument("--context-policy", choices=("window", "verified", "rolling"), default="window")
    parser.add_argument("--recent-seconds", type=float, default=12)
    parser.add_argument("--max-context-seconds", type=float, default=30)
    parser.add_argument("--max-pending-seconds", type=float, default=30)
    parser.set_defaults(max_speakers=1, max_reference_seconds=.5, max_total_reference_seconds=.5)
    parser.add_argument("--rolling-tail-seconds", type=float, default=2)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--attention", choices=("sdpa", "flash_attention_2", "eager"), default="sdpa")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM only")
    parser.add_argument("--gpu-memory-utilization", type=float, default=.85, help="vLLM only")
    parser.add_argument("--prefix-caching", "--enable-prefix-caching", dest="enable_prefix_caching",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Reuse KV blocks for shared prefixes across vLLM requests (default: enabled)")
    parser.add_argument("--mm-processor-cache-gb", type=float, default=1, help="vLLM only")
    parser.add_argument("--vllm-image-slots", type=int, choices=(0,1), default=1,
                        help="vLLM Omni initialization compatibility; requests stay audio-only")


def runtime_options(args, options_class=StreamOptions):
    names = options_class.__dataclass_fields__
    return options_class(**{name: getattr(args, name) for name in names})


def build_backend(args, *, manifest_validator=None, max_audio_item_seconds=0.0):
    if args.backend == "replay":
        replies = json.loads(args.replies.read_text(encoding="utf-8")) if args.replies else None
        if replies is not None and not isinstance(replies, list):
            raise ValueError("Replay replies must be a JSON array")
        return ReplayBackend(replies)
    if not args.model:
        raise ValueError("--model is required for real inference")
    checks = {} if manifest_validator is None else {"manifest_validator": manifest_validator}
    if args.backend == "vllm":
        from .vllm_backend import VLLMBackend
        if args.context_policy == "window":
            # The engine fixes the number of audio slots at startup. Reserve room
            # for comparing longer windows in the UI without reloading weights.
            max_audio_turns = math.ceil(args.max_window_seconds / args.chunk_seconds)
        else:
            max_audio_turns = math.ceil(args.max_context_seconds / args.chunk_seconds) + args.max_speakers + 2
        return VLLMBackend(
            args.model, max_context_tokens=args.max_context_tokens, max_new_tokens=args.max_new_tokens,
            tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization,
            max_audio_turns=max_audio_turns,
            max_audio_seconds=max(.5, args.chunk_seconds, args.max_reference_seconds, max_audio_item_seconds),
            enable_prefix_caching=args.enable_prefix_caching, mm_processor_cache_gb=args.mm_processor_cache_gb,
            image_slots=args.vllm_image_slots, dtype=args.dtype, **checks,
        )
    from .inference_backend import HFBackend
    return HFBackend(args.model, max_context_tokens=args.max_context_tokens,
                     max_new_tokens=args.max_new_tokens, device_map=args.device_map,
                     dtype=args.dtype, attn_implementation=args.attention, **checks)
