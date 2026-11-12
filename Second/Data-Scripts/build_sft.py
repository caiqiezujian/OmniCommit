"""Build independent streaming SFT JSONL files from one timed annotation JSONL."""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from second_asr.protocol import WAIT, sort_tokens
from second_asr.recast import build_sft_row, validate_chunk_ms
from second_asr.source import parse_record


def json_line(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"


def prepare_record(record, audio_root, steps, number_format):
    audios = record.get("audios") if isinstance(record, dict) else None
    if not (isinstance(audios, list) and len(audios) == 1 and
            isinstance(audios[0], str) and audios[0].strip()):
        raise ValueError("Expected one original audio path in audios")
    path = (audio_root / audios[0]).resolve(strict=True)
    try:
        info = sf.info(str(path))
    except (RuntimeError, OSError) as exc:
        raise ValueError(f"Cannot read audio metadata for {path}: {exc}") from exc
    if info.frames < 1 or info.samplerate < 1:
        raise ValueError(f"Invalid audio length or sample rate: {path}")
    duration_ms = info.frames * 1000 / info.samplerate
    units = parse_record(record, duration_ms)
    tokens = {WAIT}
    for unit in units:
        for lang, speaker, _ in unit.spans:
            tokens.update((f"<{lang}>", f"<{speaker}>"))
    result = {}
    for step in steps:
        row = build_sft_row(units, path, info.frames, info.samplerate, step, number_format)
        real_replies = row["messages"][2:-2:2]
        waits = sum(msg["content"] == WAIT for msg in real_replies)
        result[step] = (json_line(row).encode("utf-8"), len(real_replies), waits)
    # Track uncovered time separately. The text alone cannot establish silence.
    gaps = [(0, units[0].start_ms), (units[-1].end_ms, duration_ms)]
    gaps.extend((a.end_ms, b.start_ms) for a, b in zip(units, units[1:]))
    longest_gap = max(gaps, key=lambda gap: gap[1] - gap[0])
    return result, tokens, longest_gap


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source JSONL with per-unit timestamps in ms")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, help="Base for relative audio paths (default: input directory)")
    parser.add_argument("--chunk-ms", nargs="+", type=int, default=[500], metavar="MS")
    parser.add_argument("--number-format", choices=["normalized", "verbatim"], default="normalized")
    args = parser.parse_args(argv)
    steps = args.chunk_ms
    try:
        for step in steps:
            validate_chunk_ms(step)
    except ValueError as exc:
        parser.error(str(exc))
    if len(set(steps)) != len(steps):
        parser.error("--chunk-ms values must be unique")
    if not args.input.is_file():
        parser.error(f"Missing input file: {args.input}")
    audio_root = args.audio_root or args.input.parent
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    paths = {step: output / f"train.{step}ms.jsonl" for step in steps}
    tokens_files = [path.with_suffix(".jsonl.tokens.txt") for path in paths.values()]
    rejected, summary = output / "rejected.jsonl", output / "summary.json"
    for path in [*paths.values(), *tokens_files, rejected, summary]:
        if path.exists():
            parser.error(f"Output already exists: {path}. Select a fresh --output-dir.")

    kept = bad = 0
    longest_gap = None
    inventory = {WAIT}
    counts = {step: {"audio_turns": 0, "wait_audio_turns": 0} for step in steps}
    with args.input.open("rb") as source, ExitStack() as stack:
        writers = {step: stack.enter_context(path.open("xb")) for step, path in paths.items()}
        errors = stack.enter_context(rejected.open("x", encoding="utf-8", newline="\n"))
        for line_number, raw in enumerate(source, 1):
            try:
                line = raw.decode("utf-8-sig" if line_number == 1 else "utf-8")
                record = json.loads(line)
                rows, tokens, (gap_start, gap_end) = prepare_record(record, audio_root, steps, args.number_format)
            except (ValueError, KeyError, TypeError, OSError, OverflowError, RecursionError) as exc:
                # Serialize errors as ASCII so even malformed Unicode input can be reported.
                errors.write(json.dumps({"line": line_number, "error": str(exc)}, ensure_ascii=True) + "\n")
                errors.flush()
                bad += 1
            else:
                # Prepare every grid before the first write, keeping their samples in sync.
                # Storage failures propagate rather than turning into data rejections.
                for step, (payload, turns, waits) in rows.items():
                    writers[step].write(payload)
                    writers[step].flush()
                    counts[step]["audio_turns"] += turns
                    counts[step]["wait_audio_turns"] += waits
                kept += 1
                inventory.update(tokens)
                gap_ms = gap_end - gap_start
                if gap_ms > 0 and (longest_gap is None or gap_ms > longest_gap["duration_ms"]):
                    longest_gap = {"line": line_number, "start_ms": gap_start, "end_ms": gap_end,
                                   "duration_ms": gap_ms}
            if line_number == 1 or line_number % 1000 == 0:
                print(f"Rows {line_number} | kept {kept} | rejected {bad}", file=sys.stderr, flush=True)
        for path in tokens_files:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write("".join(t + "\n" for t in sort_tokens(inventory)))
        report = {"input": str(args.input.resolve()), "number_format": args.number_format,
                  "kept": kept, "rejected": bad, "chunk_ms": steps, "by_chunk_ms": counts,
                  "longest_unannotated_gap": longest_gap}
        with summary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(f"Finished | kept {kept} | rejected {bad} | {summary}", file=sys.stderr, flush=True)
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
