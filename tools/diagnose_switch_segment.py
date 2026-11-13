#!/usr/bin/env python3
"""Replay a marked interval in three fresh sessions using the original diagnostic traces."""
import argparse
from contextlib import closing, redirect_stdout
from decimal import Decimal
import json
from pathlib import Path
import re
import subprocess
import sys
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omnicommit.diagnostic_media import iter_media_chunks
from omnicommit.streaming import SAMPLE_RATE
from tools import compare_switch_segment


RUNS = {"hf": ("HFBackend", None), "vllm-on": ("VLLMBackend", True),
        "vllm-off": ("VLLMBackend", False)}
SESSION_OPTIONS = ("protocol", "chunk_ms", "number_format", "context_policy", "recent_seconds",
                   "rolling_tail_seconds", "max_context_seconds", "max_pending_seconds", "max_speakers",
                   "max_reference_seconds", "max_total_reference_seconds", "max_context_tokens", "max_new_tokens")


def read_settings(paths):
    headers = {}
    for name, path in paths.items():
        with path.open(encoding="utf-8-sig") as stream:
            header = json.loads(stream.readline())
        if not isinstance(header, dict) or header.get("type") != "ready":
            raise ValueError(f"{path}: expected a diagnose_stream ready header")
        for key in (*SESSION_OPTIONS, "input", "model", "requested_dtype", "backend", "enable_prefix_caching",
                    "mm_processor_cache_gb", "automatic_reference"):
            if key not in header:
                raise ValueError(f"{path}: missing ready.{key}")
        backend, cache = RUNS[name]
        if header["backend"] != backend or header["enable_prefix_caching"] is not cache:
            raise ValueError(f"{path}: expected {backend} with prefix_caching={cache}")
        if header["automatic_reference"] != "disabled":
            raise ValueError(f"{path}: fresh file diagnostics require automatic_reference=disabled")
        headers[name] = header
    hf = headers["hf"]
    for name in ("vllm-on", "vllm-off"):
        for key in ("input", "model", "requested_dtype", *SESSION_OPTIONS):
            if headers[name][key] != hf[key]:
                raise ValueError(f"{paths[name]}: ready.{key} differs from {paths['hf']}")
    if headers["vllm-on"]["mm_processor_cache_gb"] != headers["vllm-off"]["mm_processor_cache_gb"]:
        raise ValueError("The two vLLM traces used different mm_processor_cache_gb settings")
    return headers


def extract_packets(source, output, start, end, chunk_ms, protocol):
    if (not start.is_finite() or not end.is_finite() or not 0 <= start < end
            or type(chunk_ms) is not int or chunk_ms <= 0):
        raise ValueError("Expected a finite interval 0 <= start < end and positive chunk_ms")
    step = Decimal(chunk_ms) / 1000
    if start % step or end % step:
        raise ValueError(f"Both boundaries must lie on the original {chunk_ms} ms grid to compare identical packets")
    first_turn = int(start / step)
    count = int((end - start) / step)
    written = 0
    # Reuse the original playback resampler and grid. The full run may have
    # decoded lossy media or resampled each packet separately from 44.1 kHz.
    file = output.open("xb")
    try:
        with file, wave.open(file, "wb") as clip, closing(iter_media_chunks(source, float(step), protocol)) as packets:
            clip.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
            for index, (pcm, _) in enumerate(packets):
                if index < first_turn:
                    continue
                if len(pcm) != SAMPLE_RATE * chunk_ms // 1000:
                    break
                clip.writeframesraw(np.rint(pcm * 32768).astype("<i2").tobytes())
                written += 1
                if written == count:
                    return
        raise ValueError(f"{source}: interval [{start}, {end}) s extends past the decoded audio")
    except BaseException:
        output.unlink(missing_ok=True)
        raise


def run_experiment(args):
    paths = {name: Path(f"{args.trace_prefix}.{name}.jsonl") for name in RUNS}
    headers = read_settings(paths)
    common = headers["hf"]
    source = Path(common["input"])
    model = Path(common["model"])
    if not source.is_file() or not model.is_dir():
        raise ValueError(f"Expected original media and checkpoint from the traces: {source} and {model}")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    clip = out / "clip.wav"
    report = out / "comparison.txt"
    fresh = {name: out / f"{name}.jsonl" for name in RUNS}
    for path in (clip, report, *fresh.values()):
        if path.exists():
            raise ValueError(f"{path} already exists. Choose another --output-dir")
    print(f"Source: {source}\nModel: {model}\nExtracting [{args.start_seconds}, {args.end_seconds}) s "
          f"on the original {common['chunk_ms']} ms grid", flush=True)
    extract_packets(source, clip, args.start_seconds, args.end_seconds, common["chunk_ms"], common["protocol"])

    base = [sys.executable, str(ROOT / "tools" / "diagnose_stream.py"), str(clip),
            "--model", str(model), "--dtype", common["requested_dtype"]]
    for key in SESSION_OPTIONS:
        base.extend(("--" + key.replace("_", "-"), str(common[key])))
    failed = []
    report_args = ["--start-seconds", str(args.start_seconds), "--end-seconds", str(args.end_seconds),
                   "--language", args.language]
    for name, (backend, cache) in RUNS.items():
        cli = [*base, "--output", str(fresh[name])]
        if backend == "HFBackend":
            cli.extend(("--backend", "hf"))
        else:
            cli.extend(("--backend", "vllm", "--mm-processor-cache-gb", str(headers[name]["mm_processor_cache_gb"]),
                        "--prefix-caching" if cache else "--no-prefix-caching"))
        print(f"Starting {name} in a fresh process: {fresh[name]}", flush=True)
        code = subprocess.run(cli).returncode
        if code in (-2, 130):
            raise KeyboardInterrupt
        if code:
            failed.append(name)
            print(f"{name} exited with status {code}; continuing the comparison.", file=sys.stderr)
        report_args.extend(("--" + name, str(fresh[name]), "--full-" + name, str(paths[name])))
    with report.open("x", encoding="utf-8") as stream, redirect_stdout(stream):
        report_status = compare_switch_segment.main(report_args)
    print(report.read_text(encoding="utf-8"), end="")
    print(f"Report: {report.resolve()}")
    return 1 if failed else report_status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_prefix", help="Path before .hf.jsonl / .vllm-on.jsonl / .vllm-off.jsonl")
    parser.add_argument("--start-seconds", type=Decimal, required=True, help="Inclusive time in the original media")
    parser.add_argument("--end-seconds", type=Decimal, required=True, help="Exclusive time in the original media")
    parser.add_argument("--language", default="de", help="Target language tag to locate in the outputs")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", args.language):
        parser.error("Expected --language to be a language code such as de")
    try:
        return run_experiment(args)
    except (OSError, ValueError, ArithmeticError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
