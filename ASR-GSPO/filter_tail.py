#!/usr/bin/env python3
"""Keep open endings; filter punctuated endings with <=1 s of remaining audio.

Reads audio headers, never waveforms. Does not segment audio, edit punctuation,
or build streaming turns. Kept JSONL lines are copied unchanged. Row errors are
recorded and skipped so a missing WAV/alignment cannot stop a long filter job.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from fractions import Fraction
from functools import lru_cache
import json
from pathlib import Path
import sys

import soundfile as sf

from asr_gspo.data import (_check_audio_identity, _native_aligned_row,
                           transcript_parts)
from asr_gspo.tail_source import TailSourceError, inspect_source_tail

# Server defaults: edit these paths, or override them on the command line.
INPUT_JSONL = "/data/s50062929/FLEX/SFT_data/full_10000h_augmented.jsonl"
OUTPUT_JSONL = "/data/s50062929/FLEX/SFT_data/full_10000h_augmented.gspo_tail_checked.jsonl"
MIN_TAIL_MS = 1000


RowRejected = TailSourceError


@lru_cache(maxsize=8192)
def audio_header(path):
    info = sf.info(str(path))
    if info.frames <= 0 or info.samplerate <= 0:
        raise RowRejected("empty_audio")
    return info.frames, info.samplerate


def inspect_tail(row, audio_root, min_tail_ms=MIN_TAIL_MS):
    if not isinstance(row, dict):
        raise RowRejected("not_json_object")
    if row.get("kind") == "aligned_asr":
        row = _native_aligned_row(row)
    paths = row.get("audios")
    if not isinstance(paths, list) or len(paths) != 1 or not isinstance(paths[0], str) or not paths[0].strip():
        raise RowRejected("expected_one_original_audio")
    path = (Path(audio_root) / paths[0]).resolve(strict=True)
    frames, rate = audio_header(path)
    duration = Fraction(frames * 1000, rate)
    _check_audio_identity(row, path, Path(audio_root), float(duration))
    text = "".join(part[2] for part in transcript_parts(row))
    details = dict(audio=str(path), audio_duration_ms=float(duration))
    try:
        details.update(inspect_source_tail(row, text, duration, min_tail_ms=min_tail_ms))
    except TailSourceError as exc:
        raise RowRejected(exc.reason, **details, **exc.details) from exc
    # Validate provenance only for the tail annotation actually consulted. An
    # unused old alignment must not reject a timestamp-free open ending, or a
    # row whose valid final end came from the separate timestamps array.
    origin = details.get("tail_timestamp_source")
    if origin is not None:
        if origin.startswith("alignment."):
            container = row["alignment"]
            metadata = [container]
            entries = container[origin.split(".", 1)[1]]
        else:
            metadata = []
            entries = row[origin]
        metadata.append(entries[details["tail_timestamp_index"]])
        for item in metadata:
            _check_audio_identity(item, path, Path(audio_root), float(duration))
            if row.get("id") is not None and item.get("id") is not None and str(row["id"]) != str(item["id"]):
                raise RowRejected("tail_timestamp_id_mismatch")
    return details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", default=Path(INPUT_JSONL))
    parser.add_argument("--output", type=Path, help="Filtered source JSONL; kept lines remain unchanged")
    parser.add_argument("--rejects", type=Path, help="Line numbers and rejection reasons; no audio or full-row copies")
    parser.add_argument("--audio-root", type=Path, help="Base for relative audio paths; defaults to input directory")
    parser.add_argument("--min-tail-ms", type=int, default=MIN_TAIL_MS,
                        help="Punctuated endings require gap > threshold; unpunctuated endings are kept")
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.min_tail_ms < 0 or args.progress_every <= 0:
        parser.error("min-tail-ms must be nonnegative and progress-every must be positive")
    output = args.output or (Path(OUTPUT_JSONL) if args.input == Path(INPUT_JSONL)
                             else args.input.with_name(args.input.stem + ".tail_filtered.jsonl"))
    rejects = args.rejects or output.with_suffix(".rejected.jsonl")
    report_path = output.with_suffix(".report.json")
    paths = [p.resolve() for p in (args.input, output, rejects, report_path)]
    if len(set(paths)) != len(paths):
        parser.error("Input/output/rejects/report paths must be different")
    for path in (output, rejects, report_path):
        if path.exists():
            parser.error(f"Output already exists: {path}; choose a new output path")
    root = args.audio_root or args.input.parent
    counts, reasons, kept_reasons = Counter(), Counter(), Counter()
    audio_header.cache_clear()
    report = dict(input=str(args.input), output=str(output), rejects=str(rejects),
                  audio_root=str(root.resolve()), min_tail_ms=args.min_tail_ms,
                  comparison="unpunctuated tail OR audio_duration_ms - last_text_end_ms > min_tail_ms", completed=False)
    interrupted = False
    with args.input.open("rb") as source, ExitStack() as stack:
        for path in (output, rejects, report_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        target = stack.enter_context(output.open("xb"))
        reject_log = stack.enter_context(rejects.open("x", encoding="utf-8"))
        try:
            for line, raw in enumerate(source, 1):
                counts["lines"] += 1
                try:
                    row = json.loads(raw.decode("utf-8-sig" if line == 1 else "utf-8"))
                    details = inspect_tail(row, root, args.min_tail_ms)
                except (ValueError, OSError, RuntimeError, OverflowError, TypeError, KeyError, RecursionError) as exc:
                    reason = exc.reason if isinstance(exc, RowRejected) else "invalid_record_or_audio"
                    rejected = dict(line=line, reason=reason)
                    rejected.update(exc.details if isinstance(exc, RowRejected) else {"error": str(exc)})
                    reject_log.write(json.dumps(rejected, ensure_ascii=True, allow_nan=False) + "\n")
                    counts["rejected"] += 1
                    reasons[reason] += 1
                else:
                    target.write(raw)
                    counts["kept"] += 1
                    kept_reasons[details["tail_decision"]] += 1
                if line % args.progress_every == 0:
                    target.flush()
                    reject_log.flush()
                    print(f"rows={line} kept={counts['kept']} rejected={counts['rejected']}", flush=True)
            report["completed"] = True
        except KeyboardInterrupt:
            interrupted = True
        finally:
            target.flush()
            reject_log.flush()
            report.update(counts=dict(counts), rejection_reasons=dict(reasons), kept_reasons=dict(kept_reasons))
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if interrupted:
        raise SystemExit(130)
    return report


if __name__ == "__main__":
    main()
