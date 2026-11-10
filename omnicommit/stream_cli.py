"""CLI and trusted control events shared by file and Web inference."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

from .streaming import ReplayBackend, StreamOptions, StreamSession


def add_runtime_arguments(parser):
    parser.add_argument("--backend", choices=("hf", "vllm", "replay"), default="hf")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--replies", type=Path, help="Replay only: JSON array of scripted replies; never real ASR")
    parser.add_argument("--chunk-seconds", type=float, choices=(.5, .8, 1.0), default=.5)
    parser.add_argument("--number-format", choices=("verbatim", "normalized"), default="verbatim")
    parser.add_argument("--context-policy", choices=("verified", "rolling"), default="verified")
    parser.add_argument("--recent-seconds", type=float, default=12)
    parser.add_argument("--max-context-seconds", type=float, default=30)
    parser.add_argument("--max-pending-seconds", type=float, default=30)
    parser.add_argument("--max-speakers", type=int, default=16)
    parser.add_argument("--max-reference-seconds", type=float, default=5)
    parser.add_argument("--max-total-reference-seconds", type=float, default=30)
    parser.add_argument("--rolling-tail-seconds", type=float, default=2)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--attention", choices=("sdpa", "flash_attention_2", "eager"), default="sdpa")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM only")
    parser.add_argument("--gpu-memory-utilization", type=float, default=.85, help="vLLM only")
    parser.add_argument("--mm-processor-cache-gb", type=float, default=1, help="vLLM only")
    parser.add_argument("--vllm-image-slots", type=int, choices=(0,1), default=1,
                        help="vLLM Omni initialization compatibility; requests stay audio-only")


def runtime_options(args):
    names = StreamOptions.__dataclass_fields__
    return StreamOptions(**{name: getattr(args, name) for name in names})


def build_backend(args):
    if args.backend == "replay":
        replies = json.loads(args.replies.read_text(encoding="utf-8")) if args.replies else None
        if replies is not None and not isinstance(replies, list):
            raise ValueError("Replay replies must be a JSON array")
        return ReplayBackend(replies)
    if not args.model:
        raise ValueError("--model is required for real inference")
    if args.backend == "vllm":
        from .vllm_backend import VLLMBackend
        return VLLMBackend(args.model,max_context_tokens=args.max_context_tokens,
            max_new_tokens=args.max_new_tokens,tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_audio_turns=math.ceil(args.max_context_seconds/args.chunk_seconds)+args.max_speakers+2,
            max_audio_seconds=max(args.chunk_seconds,args.max_reference_seconds),
            mm_processor_cache_gb=args.mm_processor_cache_gb,image_slots=args.vllm_image_slots,dtype=args.dtype)
    from .inference_backend import HFBackend
    return HFBackend(args.model, max_context_tokens=args.max_context_tokens,
                     max_new_tokens=args.max_new_tokens, device_map=args.device_map,
                     dtype=args.dtype, attn_implementation=args.attention)


def apply_control(session, event):
    if not isinstance(event, dict):
        raise ValueError("Control must be an object")
    if event.get("op") == "commit":
        session.verify_commit(event["through_seconds"], verified=event.get("verified", False))
    elif event.get("op") == "reference":
        session.register_reference(**{k: event[k] for k in
            ("speaker", "language", "text", "start", "end", "verified", "clean", "overlap")})
    else:
        raise ValueError("Unknown control op")


def iter_audio(path, chunk_seconds):
    """Bounded file reads; cut source first, then resample each new chunk."""
    import soundfile as sf
    from scipy.signal import resample_poly
    with sf.SoundFile(str(path)) as stream:
        if not len(stream):
            raise ValueError("Empty input audio")
        sr = stream.samplerate
        step = round(chunk_seconds * sr)
        while True:
            audio = stream.read(step, dtype="float32", always_2d=True)
            if not len(audio):
                break
            audio = audio.mean(axis=1)
            if sr != 16000:
                divisor = math.gcd(sr, 16000)
                audio = resample_poly(audio, 16000 // divisor, sr // divisor)
            yield audio.astype(np.float32)


def main(argv=None):
    parser = argparse.ArgumentParser(description="OmniCommit streaming WAV inference; no ground-truth text sent to model")
    add_runtime_arguments(parser)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=Path, help="Trusted causal control JSONL (manual/oracle or external verified aligner)")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output exists; choose a new path")
    options = runtime_options(args)
    if options.context_policy == "rolling":
        print("WARNING: rolling is a bounded-tail heuristic, not verified consumption; held speech may be lost.", file=sys.stderr)
    events = []
    if args.events:
        for line in args.events.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                event = json.loads(line)
                after = event.get("after_seconds")
                if isinstance(after, bool) or not isinstance(after, (int, float)) or not math.isfinite(after) or after < 0:
                    parser.error("Every control event needs finite nonnegative after_seconds")
                events.append(event)
        if events != sorted(events, key=lambda e: e["after_seconds"]):
            parser.error("Control events must be sorted causally by after_seconds")
    session = StreamSession(build_backend(args), options)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pointer = 0
    status = 0
    with args.output.open("x", encoding="utf-8") as output:
        try:
            for audio in iter_audio(args.audio, options.chunk_seconds):
                result = session.step(audio)
                result.update(backend=args.backend, source_audio=str(args.audio.resolve()))
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                while pointer < len(events) and events[pointer]["after_seconds"] <= session.heard_seconds + 1e-8:
                    apply_control(session, events[pointer])
                    pointer += 1
            if pointer < len(events):
                raise ValueError("Control events refer to future audio beyond EOF")
            result = session.step(end_of_audio=True)
            result.update(backend=args.backend, source_audio=str(args.audio.resolve()))
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
        except Exception as exc:
            output.write(json.dumps({"event": "error", "error": str(exc),
                "heard_seconds": session.heard_seconds, "backend": args.backend}, ensure_ascii=False) + "\n")
            print(str(exc), file=sys.stderr)
            status = 1
        finally:
            session.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
