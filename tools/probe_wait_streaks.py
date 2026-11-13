#!/usr/bin/env python3
"""Replay the onset of each sustained WAIT in an independent short Second session."""
import argparse
from collections import deque
from contextlib import closing
from itertools import islice
from pathlib import Path
import math
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omnicommit.diagnostic_media import iter_media_chunks
from omnicommit.diagnostic_runtime import add_diagnostic_arguments, resolve_runtime
from omnicommit.streaming import SAMPLE_RATE
from tools.compare_stream_traces import read_trace, records
from tools.diagnose_stream import ObservedBackend, open_trace, pcm_sha256
from tools.diagnose_switch_segment import SESSION_OPTIONS


def select_probes(frames, args):
    """Window ends at the first threshold crossing and never uses following audio."""
    waits = 0
    for turn, (event, (_, end), result) in enumerate(frames, 1):
        if event != "audio":
            break
        waits = waits + 1 if result["action"] == "WAIT" else 0
        if waits == args.wait_turns and args.start_seconds <= end <= args.end_seconds:
            yield max(1, turn - waits - args.lead_turns + 1), turn


def engine_from_trace(header):
    if header["protocol"] != "second" or header.get("automatic_reference") != "disabled":
        raise ValueError("Expected a Second file diagnostic trace with automatic_reference disabled")
    backends = {"HFBackend": "hf", "VLLMBackend": "vllm"}
    backend = backends.get(header.get("backend"))
    if not backend or not header.get("model"):
        raise ValueError("A paired experiment requires an HF/vLLM checkpoint trace")
    if backend == "vllm" and (type(header.get("enable_prefix_caching")) is not bool
                              or header.get("mm_processor_cache_gb") is None):
        raise ValueError("Missing vLLM cache settings in the original trace")
    parser = argparse.ArgumentParser(add_help=False)
    add_diagnostic_arguments(parser)
    cmd = ["--backend", backend, "--model", header["model"], "--dtype", header["requested_dtype"]]
    for name in SESSION_OPTIONS:
        cmd.extend(("--" + name.replace("_", "-"), str(header[name])))
    for name, value in header.get("engine_options", {}).items():
        if name in ("attention", "device_map", "tensor_parallel_size", "gpu_memory_utilization", "vllm_image_slots"):
            cmd.extend(("--" + name.replace("_", "-"), str(value)))
    if backend == "vllm":
        cmd.extend(("--prefix-caching" if header["enable_prefix_caching"] else "--no-prefix-caching",
                    "--mm-processor-cache-gb", str(header["mm_processor_cache_gb"])))
    return resolve_runtime(parser.parse_args(cmd))


def replay_window(packets, frames, first, last, backend, options, session_factory, hashes):
    observed = ObservedBackend(backend)
    session = session_factory(observed, options)
    results = []
    started = time.perf_counter()
    error = None
    try:
        for turn, pcm, final_chunk in packets:
            if turn < first:
                continue
            observed.reset()
            try:
                shadow = session.step(pcm, final_chunk=final_chunk)
            except Exception as exc:
                error = {"turn": turn, "message": str(exc), "error_type": type(exc).__name__,
                         "raw_model_text": observed.raw_text}
                break
            results.append({"turn": turn, "raw_model_text": observed.raw_text,
                            "action": shadow["action"], "generation_seconds": shadow["generation_seconds"],
                            "num_cached_tokens": shadow["num_cached_tokens"]})
    finally:
        session.close()
    return {"type": "shadow_probe", "first_turn": first, "last_turn": last,
            "all_packet_hashes_verified": all(turn in hashes for turn in range(first, last+1)),
            "start_seconds": frames[first-1][1][0], "end_seconds": frames[last-1][1][1],
            "original": [dict(turn=turn, **frames[turn-1][2]) for turn in range(first, last+1)],
            "candidate": results, "error": error, "elapsed_seconds": time.perf_counter()-started}


def run(args, write):
    header, frames = read_trace(args.trace)
    protocol, options, session_factory, backend_factory = engine_from_trace(header)
    windows = list(islice(select_probes(frames, args), args.max_probes))
    write({"type": "shadow_ready", "source_trace": str(args.trace.resolve()), "original_settings": header,
           "wait_turns": args.wait_turns, "lead_turns": args.lead_turns, "selected": len(windows),
           "speaker_ids": "local_to_each_short_session",
           "engine_settings_from_trace": bool(header.get("engine_options"))})
    failed = 0
    if windows:
        selected = {turn for first, last in windows for turn in range(first, last + 1)}
        hashes = {r["turn"]: r["pcm_sha256"] for r in records(args.trace)
                  if r.get("turn") in selected and "pcm_sha256" in r}
        backend = backend_factory()
        ring = deque(maxlen=args.wait_turns+args.lead_turns)
        with closing(iter_media_chunks(Path(header["input"]), options.chunk_seconds, protocol)) as chunks:
            current = 0
            for first, last in windows:
                while current < last:
                    packet = next(chunks, None)
                    if packet is None:
                        raise ValueError("Original media ended before the selected trace turn")
                    current += 1
                    pcm, final_chunk = packet
                    t0, t1 = frames[current-1][1]
                    if len(pcm) != round((t1-t0)*SAMPLE_RATE):
                        raise ValueError(f"Original packet length differs from the trace at turn {current}")
                    if current in hashes and hashes[current] != pcm_sha256(pcm):
                        raise ValueError(f"Original PCM content differs from the trace at turn {current}")
                    ring.append((current, pcm, final_chunk))
                probe = replay_window(ring, frames, first, last, backend, options, session_factory, hashes)
                write(probe)
                failed += probe["error"] is not None
    write({"type": "shadow_summary", "probed": len(windows), "failed": failed})
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Original diagnose_stream.py JSON/JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="New shadow JSON/JSONL; original trace is read-only")
    parser.add_argument("--wait-turns", type=int, default=3)
    parser.add_argument("--lead-turns", type=int, default=1, help="Extra observed chunks preceding the WAIT streak")
    parser.add_argument("--max-probes", type=int, default=12)
    parser.add_argument("--start-seconds", type=float, default=0, help="Minimum threshold-crossing time")
    parser.add_argument("--end-seconds", type=float, help="Maximum threshold-crossing time")
    args = parser.parse_args(argv)
    if min(args.wait_turns, args.max_probes) < 1 or args.lead_turns < 0:
        parser.error("Expected positive wait-turns/max-probes and nonnegative lead-turns")
    if not math.isfinite(args.start_seconds) or args.start_seconds < 0 or (args.end_seconds is not None
            and (not math.isfinite(args.end_seconds) or args.end_seconds < args.start_seconds)):
        parser.error("Expected a finite 0 <= start-seconds <= end-seconds interval")
    if args.end_seconds is None:
        args.end_seconds = math.inf
    if args.output.suffix.lower() not in (".json", ".jsonl"):
        parser.error("--output must end in .json or .jsonl")
    try:
        with open_trace(args.output) as write:
            try:
                return run(args, write)
            except (ValueError, OSError, RuntimeError, KeyError, ImportError) as exc:
                write({"type": "shadow_error", "error": str(exc), "error_type": type(exc).__name__})
                raise
    except (ValueError, OSError, RuntimeError, KeyError, ImportError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    sys.exit(main())
