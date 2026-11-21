#!/usr/bin/env python3
"""Prepare FLEX-style online GSPO rows from OmniCommit full-recording JSONL."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from collections import Counter
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path

from asr_gspo.data import prepare_record, validate_chunk_ms


@contextmanager
def staged_output(destination, *, publish_empty=True):
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(mode="wt", encoding="utf-8", newline="\n", dir=destination.parent,
                                         prefix=destination.name + ".", suffix=".partial", delete=False)
    try:
        with handle:
            yield handle
        if publish_empty or os.stat(handle.name).st_size:
            os.replace(handle.name, destination)
    finally:
        Path(handle.name).unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, help="Base for relative audio paths (default: input directory)")
    parser.add_argument("--chunk-ms", default=1000, type=int)
    parser.add_argument("--max-turns", default=65, type=int,
                        help="Maximum number of rollout turns including EOF (default: %(default)s)")
    parser.add_argument("--tail-policy", choices=("keep", "short-open", "annotated-prefix"), default="keep",
                        help="Keep full source audio/text by default; open-tail policies require aligned timing mode")
    parser.add_argument("--timing-mode", choices=("tail-only", "aligned"), default="tail-only",
                        help="tail-only: full source transcript, timestamps used only by filter_tail.py; aligned: opt in to detailed timing rewards")
    parser.add_argument("--skip-invalid", action="store_true", help="Continue after invalid input rows")
    parser.add_argument("--rejects", type=Path, help="Write skipped row numbers and reasons here (implies --skip-invalid)")
    args = parser.parse_args(argv)
    try:
        validate_chunk_ms(args.chunk_ms)
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_turns < 2:
        parser.error("--max-turns must allow at least one audio turn and EOF")
    if args.timing_mode == "tail-only" and args.tail_policy != "keep":
        parser.error("--timing-mode tail-only requires --tail-policy keep")
    paths = [p.resolve() for p in (args.input, args.output, args.rejects) if p is not None]
    if len(set(paths)) != len(paths):
        parser.error("Input, output and rejects must be distinct paths")
    audio_root = args.audio_root or args.input.parent
    counts = Counter()
    buckets, timing = Counter(), Counter()
    with args.input.open("rb") as source, ExitStack() as stack:
        target = stack.enter_context(staged_output(args.output, publish_empty=False))
        errors = stack.enter_context(staged_output(args.rejects)) if args.rejects else None
        # Keep the million-row identity index on disk, independently of the
        # growing prepared output. Its lifetime is limited to this run.
        scratch = stack.enter_context(tempfile.TemporaryDirectory(dir=args.output.parent))
        ids = stack.enter_context(closing(sqlite3.connect(str(Path(scratch) / "ids.sqlite"))))
        ids.execute("CREATE TABLE seen(id TEXT PRIMARY KEY) WITHOUT ROWID")
        for line, raw in enumerate(source, 1):
            try:
                record = json.loads(raw.decode("utf-8-sig" if line == 1 else "utf-8"))
                prepared = prepare_record(record, audio_root, args.chunk_ms, args.max_turns,
                                          tail_policy=args.tail_policy, timing_mode=args.timing_mode)
                encoded = json.dumps(prepared, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                # Unpaired UTF-16 surrogates are syntactically valid JSON escapes,
                # but would make the prepared UTF-8 row unwritable.
                encoded.encode("utf-8")
                task = json.loads(prepared["asr_task"])
                identifier = task["id"]
                if ids.execute("INSERT OR IGNORE INTO seen VALUES (?)", (identifier,)).rowcount != 1:
                    raise ValueError(f"Duplicate task ID {identifier!r}")
            except (ValueError, OSError, RuntimeError, OverflowError, TypeError, RecursionError) as exc:
                if not (args.skip_invalid or args.rejects):
                    raise SystemExit(f"{args.input}:{line}: {exc}") from exc
                message = json.dumps({"line": line, "error": str(exc)}, ensure_ascii=True)
                print(message, file=errors if errors is not None else sys.stderr, flush=True)
                counts["rejected"] += 1
            else:
                target.write(encoded + "\n")
                counts["kept"] += 1
                buckets[task["bucket"]] += 1
                timing["mode_" + task["timing_mode"]] += 1
                if any(unit.get("end_ms") is not None for unit in task["alignment"]):
                    timing["has_reference_end"] += 1
                elif task.get("switches"):
                    timing["switch_onset_without_reference_end"] += 1
                else:
                    timing["no_reference_end_or_switch"] += 1
            if line % 1000 == 0:
                print(f"rows={line} kept={counts['kept']} rejected={counts['rejected']}", file=sys.stderr, flush=True)
    print(f"kept={counts['kept']} rejected={counts['rejected']} output={args.output}", file=sys.stderr, flush=True)
    print(json.dumps({"buckets": buckets, "timing_rows": timing}), file=sys.stderr)
    if not counts["kept"]:
        raise SystemExit("No valid GSPO examples were prepared")


if __name__ == "__main__":
    main()
