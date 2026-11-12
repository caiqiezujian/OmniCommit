#!/usr/bin/env python3
"""Find the first differing turn in each pair of diagnose_stream.py traces."""
import argparse
from contextlib import closing
import json
from pathlib import Path


def records(path):
    with path.open(encoding="utf-8-sig") as stream:
        if path.suffix.lower() == ".json":
            rows = json.load(stream)
            if not isinstance(rows, list):
                raise ValueError(f"{path}: expected an array of trace events")
            yield from rows
        else:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{line_number}: {exc}") from exc


def read_trace(path):
    # Keep only comparison fields in memory, so long JSONL traces never retain
    # cached-token statistics, timings or repeated speaker/segment metadata.
    frames = []
    previous_end = 0
    terminal = False
    with closing(records(path)) as rows:
        header = next(rows, None)
        if not isinstance(header, dict) or header.get("type") != "ready":
            message = header.get("error", "missing ready event") if isinstance(header, dict) else "missing ready event"
            raise ValueError(f"{path}: inference did not start: {message}")
        for field in ("protocol", "model", "input", "chunk_ms"):
            if field not in header:
                raise ValueError(f"{path}: missing ready.{field}")
        for turn, row in enumerate(rows, 1):
            if terminal or not isinstance(row, dict):
                raise ValueError(f"{path}: unexpected record at turn {turn}")
            if row.get("type") == "error":
                interval = (row.get("attempted_start_seconds", row.get("heard_seconds")), row.get("attempted_end_seconds"))
                output = {key: row.get(key) for key in ("stage", "error_type", "error", "backend_called", "raw_model_text")}
                frames.append(("error", interval, output))
                terminal = True
                continue
            event = row.get("event")
            if row.get("type") != "result" or row.get("turn") != turn or event not in ("audio", "end_of_audio"):
                raise ValueError(f"{path}: missing or unexpected result at turn {turn}")
            start, end = row.get("start_seconds"), row.get("end_seconds")
            if start != previous_end or not isinstance(end, (int, float)) or not (
                end > start if event == "audio" else end == start
            ):
                raise ValueError(f"{path}: invalid audio interval at turn {turn}")
            if row.get("action") not in ("WAIT", "DIRECT") or not isinstance(row.get("raw_model_text"), str):
                raise ValueError(f"{path}: invalid action or missing raw_model_text at turn {turn}")
            frames.append((event, (start, end), {"action": row["action"], "raw_model_text": row["raw_model_text"]}))
            previous_end = end
            terminal = event == "end_of_audio"
    if not terminal:
        raise ValueError(f"{path}: incomplete trace after {len(frames)} turns (no end_of_audio or error)")
    return header, frames


def compare(path_a, trace_a, path_b, trace_b):
    header_a, a = trace_a
    header_b, b = trace_b
    # These settings affect the input or dialogue. Engine/cache settings and
    # generation timings are intentionally allowed to differ between runs.
    fields = ("input", "model", "protocol", "chunk_ms", "context_policy", "recent_seconds",
              "rolling_tail_seconds", "max_pending_seconds", "max_context_seconds", "number_format",
              "max_context_tokens", "max_new_tokens", "automatic_reference")
    if "requested_dtype" in header_a and "requested_dtype" in header_b:
        fields += ("requested_dtype",)
    for key in fields:
        if header_a.get(key) != header_b.get(key):
            raise ValueError(f"{path_a} vs {path_b}: ready.{key} differs: {header_a.get(key)!r} / {header_b.get(key)!r}")
    print(f"{path_a} vs {path_b}")
    for turn, (left, right) in enumerate(zip(a, b), 1):
        if left == right:
            continue
        if "error" in (left[0], right[0]):
            reason = "execution outcome differs"
        elif left[:2] != right[:2]:
            reason = "audio grid differs"
        else:
            reason = "model output differs"
        print(f"  First difference at turn {turn}: {reason}")
        for path, (event, (start, end), output) in ((path_a, left), (path_b, right)):
            print(f"  {path}: {event} [{start}, {end}] s  {json.dumps(output, ensure_ascii=False)}")
        return
    # Both traces were validated to end with an EOS or error. Matching
    # terminal events therefore imply matching lengths.
    event, interval, output = a[-1]
    if event == "error":
        print(f"  Matched {len(a)-1} results; both stopped at turn {len(a)} {interval}: {output['error']}")
    else:
        print(f"  Matched all {len(a)} turns through end_of_audio ({interval[1]:g} s)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs=3, metavar="TRACE", help="HF, vLLM cache-on and vLLM cache-off log files")
    args = parser.parse_args(argv)
    try:
        runs = [(path, read_trace(path)) for path in args.traces]
        for path, (header, _) in runs:
            print(f"{path}: {header.get('backend')}, prefix_cache={header.get('enable_prefix_caching')}, "
                  f"mm_processor_cache_gb={header.get('mm_processor_cache_gb')}, requested_dtype={header.get('requested_dtype')}")
        hf, vllm_on, vllm_off = runs
        for left, right in ((vllm_on, vllm_off), (hf, vllm_on), (hf, vllm_off)):
            compare(*left, *right)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
