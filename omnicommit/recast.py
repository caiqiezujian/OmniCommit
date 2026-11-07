"""Remold semantic annotations onto fixed audio steps, with causal EOS flush."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from .alignment import finite, group_units, render_units, tagged_units, source_rows
from .io import Journal, fingerprint, row_id, write_row


def recast_row(row, chunk_seconds=0.5, number_format="verbatim"):
    if row.get("schema_version") != 2 or row.get("kind") != "aligned_asr" or row.get("status") != "ok":
        raise ValueError("Expected successful schema_version=2 aligned_asr row")
    if chunk_seconds not in (0.5, 1.0):
        raise ValueError("chunk_seconds must be 0.5 or 1.0")
    if number_format not in ("verbatim", "normalized"):
        raise ValueError("Unknown number format")
    duration = finite(row.get("duration_ms"), "duration_ms")
    if duration <= 0:
        raise ValueError("Audio must have positive duration")
    segments = row.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("No aligned segments")
    units, starts, ends, actions = [], [], [], []
    previous_start = previous_end = -1.0
    reconstructed = []
    for segment in segments:
        if not isinstance(segment, dict) or segment.get("action") not in ("WAIT", "DIRECT"):
            raise ValueError("Malformed aligned segment")
        start = finite(segment.get("start_ms"), "segment.start_ms")
        end = finite(segment.get("end_ms"), "segment.end_ms")
        if not 0 <= start < end <= duration + 1e-6:
            raise ValueError("Segment timestamps outside audio")
        if start < previous_start - 1e-6 or end < previous_end - 1e-6:
            raise ValueError("Segment timestamps must be chronological")
        attributed = segment.get("attributed")
        if not isinstance(attributed, list) or not attributed:
            raise ValueError("Missing source attribution")
        local = tagged_units("".join("<%s><%s>%s" % (v["language"], v["speaker"], v["text"])
                                     for v in attributed))
        units.append(local)
        reconstructed.extend(local)
        starts.append(start)
        ends.append(end)
        actions.append(segment["action"])
        previous_start, previous_end = start, end
    if render_units(reconstructed) != row.get("transcript"):
        raise ValueError("Aligned segments do not reconstruct the attributed transcript")
    # State changes are read from source attribution, never inferred from what
    # has already been emitted; otherwise WAIT can repeat forever at a switch.
    previous_state = None
    for action, local in zip(actions, units):
        for language, speaker, _ in local:
            if previous_state is not None and previous_state != (language, speaker) and action != "WAIT":
                raise ValueError("An attribution switch must carry a WAIT annotation")
            previous_state = language, speaker
    step = chunk_seconds * 1000
    endpoints = [step * i for i in range(1, math.ceil(duration / step))] + [duration]
    turns, consumed, emitted, previous_time = [], 0, 0, 0.0
    for end in endpoints:
        first_due = consumed
        while consumed < len(segments) and ends[consumed] <= end + 1e-6:
            consumed += 1
        # Consume EACH hard event once. Several source units can finish in one
        # physical round; one WAIT is sufficient to hold all of that round.
        gates = [i for i in range(first_due, consumed) if actions[i] == "WAIT"]
        release_start = emitted
        if not gates:
            emitted = consumed
        output_units = [unit for local in units[release_start:emitted] for unit in local]
        action = "DIRECT" if output_units else "WAIT"
        # A DIRECT response can still leave an incomplete next name in audio.
        # Such a boundary is NOT safe for independent training windows.
        in_progress = consumed < len(segments) and starts[consumed] < end - 1e-6
        turns.append({"event": "audio", "start_ms": previous_time, "end_ms": end,
                      "action": action, "segments": group_units(output_units) if output_units else [],
                      "safe_cut": emitted == consumed and not in_progress,
                      "provenance": {"newly_completed_source_indices": list(range(first_due, consumed)),
                                     "hard_wait_source_indices": gates,
                                     "emitted_source_indices": list(range(release_start, emitted)),
                                     "pending_source_indices": list(range(emitted, consumed)),
                                     "incomplete_source_present": in_progress}})
        previous_time = end
    if consumed != len(segments):
        raise ValueError("Some annotated text extends beyond source audio")
    tail = [unit for local in units[emitted:] for unit in local]
    # EOS is an actual protocol event, not silence/padding or an invented extra
    # future audio chunk. Include it even when no pending transcript remains.
    turns.append({"event": "end_of_audio", "start_ms": duration, "end_ms": duration,
                  "action": "DIRECT" if tail else "WAIT",
                  "segments": group_units(tail) if tail else [], "safe_cut": True,
                  "provenance": {"emitted_source_indices": list(range(emitted, len(segments))),
                                 "pending_source_indices": []}})
    emitted_units = [(part["language"], part["speaker"], char)
                     for turn in turns for part in turn["segments"] for char in part["text"]]
    if emitted_units != reconstructed:
        raise AssertionError("Remolding lost or duplicated source content")
    return {"schema_version": 2, "id": row.get("id"), "status": "ok", "audio": row["audio"],
            "duration_ms": duration, "turns": turns, "transcript": row["transcript"],
            "source_segments": segments,
            "policy": {"number_format": number_format, "chunk_seconds": chunk_seconds,
                       "hard_wait": "atomic-source-unit-completion-once-per-physical-round",
                       "eos_user_text": "<end_of_audio>",
                       "prefix_format": "full-language-speaker-per-direct-span"},
            "alignment": row.get("alignment"), "audio_info": row.get("audio_info"),
            "provenance": {"algorithm": "fixed-audio-step-atomic-completion-hard-wait-v2",
                           "annotation_flush": row.get("annotation_flush"),
                           "eos_flush_source_count": len(segments) - emitted,
                           "causal_targets": True}}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="Aligned schema2 JSONL")
    p.add_argument("--output", required=True)
    p.add_argument("--chunk-seconds", type=float, choices=[0.5, 1.0], default=0.5)
    p.add_argument("--number-format", choices=["verbatim", "normalized"], default="verbatim",
                   help="Must match existing annotation text; does not rewrite digits")
    p.add_argument("--resume", action="store_true")
    return p


def run(args):
    if Path(args.input).resolve() == Path(args.output).resolve():
        raise ValueError("Input and output must differ")
    config = {"pipeline": "recast-v2", "chunk_seconds": args.chunk_seconds,
              "number_format": args.number_format}
    journal = Journal(args.output, config, args.resume)
    counts = {"written": 0, "failed": 0, "resumed": 0}
    complete = False
    try:
        with open(str(args.output) + ".rejected.jsonl", "a", encoding="utf-8") as rejected:
            for line, row, error in source_rows(args.input):
                try:
                    identifier = row_id(row, args.input, line)
                except ValueError as exc:
                    identifier, error = row_id({}, args.input, line), str(exc)
                digest = fingerprint(row)
                if journal.previous(digest, identifier) is not None:
                    counts["resumed"] += 1
                    continue
                try:
                    if error:
                        raise ValueError(error)
                    result = recast_row(row, args.chunk_seconds, args.number_format)
                except Exception as exc:
                    counts["failed"] += 1
                    result = {"schema_version": 2, "id": identifier, "status": "error",
                              "output": "", "error": str(exc)}
                    write_row(rejected, {"line": line, "id": identifier, "error": str(exc)})
                    print(f"[ERROR] id={identifier}: {exc}", file=sys.stderr, flush=True)
                result["input_hash"] = digest
                journal.append(result)
                counts["written"] += 1
        complete = True
    finally:
        journal.close(check_excess=complete)
    print(json.dumps(counts), flush=True)
    return counts


def main(argv=None):
    try:
        run(parser().parse_args(argv))
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2
    return 0
