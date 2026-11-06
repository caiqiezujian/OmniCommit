"""Model-admissible, final-anchored LSP and incremental DIRECT/WAIT labels."""
from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
import re
import sys
import unicodedata

from .io import Journal, fingerprint, row_id, source_rows, write_row
from .protocol import canonical_segments, from_units, plain_text, to_units


def agreement_text(text):
    return "".join(c.casefold() for c in unicodedata.normalize("NFKC", text)
                   if not c.isspace() and not unicodedata.category(c).startswith("P"))


def reference_audit(trace, full):
    reference = trace.get("reference_segments")
    common = {"teacher_languages": sorted({s["language"] for s in full}),
              "teacher_speakers": sorted({s["speaker"] for s in full}),
              "note": "Lexical agreement only, not acoustic or speaker-identity verification; numerical formats may differ."}
    if reference is None or trace.get("reference_parse_error"):
        return {"status": "unavailable", "text_agreement": None,
                "reference_languages": None, "reference_speakers": None,
                "reference_parse_error": trace.get("reference_parse_error"), **common}
    reference = canonical_segments(reference)
    original = agreement_text(plain_text(reference))
    predicted = agreement_text(plain_text(full))
    score = SequenceMatcher(None, original, predicted, autojunk=False).ratio()
    return {"status": "available", "text_agreement": score,
            "reference_languages": sorted({s["language"] for s in reference}),
            "reference_speakers": sorted({s["speaker"] for s in reference}),
            "reference_parse_error": None, **common}


def make_trajectory(trace, holdback_chars=0, min_reference_agreement=0.0):
    if trace.get("status") != "ok":
        raise ValueError("Collection failed for this source record")
    if isinstance(holdback_chars, bool) or not isinstance(holdback_chars, int) or holdback_chars < 0:
        raise ValueError("holdback_chars must be a nonnegative integer")
    duration = trace.get("duration_ms")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid audio duration")
    full = canonical_segments(trace["full"]["segments"])
    final = to_units(full)
    number_format = trace.get("teacher", {}).get("number_format", "verbatim")
    if number_format not in ("verbatim", "normalized"):
        raise ValueError("Unknown teacher number_format")
    # Conservative offline guard for digit strings (including dates/decimals).
    # It is NOT a complete named-entity or numeric-expression detector.
    protected_numbers, offset = [], 0
    for segment in full:
        if number_format == "normalized":
            for match in re.finditer(r"[0-9]+(?:[.,:/-][0-9]+)*(?:[%‰])?", segment["text"]):
                protected_numbers.append((offset + match.start(), offset + match.end()))
        offset += len(segment["text"])
    if not final:
        raise ValueError("Full teacher transcript is empty; no speech trajectory to supervise")
    audit = reference_audit(trace, full)
    if audit["status"] == "unavailable" and min_reference_agreement > 0:
        raise ValueError("Reference audit unavailable; cannot satisfy configured agreement threshold")
    if audit["text_agreement"] is not None and audit["text_agreement"] < min_reference_agreement:
        raise ValueError("Full teacher/reference lexical agreement below configured threshold")
    prefixes = trace.get("prefixes")
    if not isinstance(prefixes, list) or not prefixes:
        raise ValueError("Trace has no prefix hypotheses")
    previous_end, stable_length, committed = 0.0, 0, 0
    turns = []
    for index, prefix in enumerate(prefixes):
        end = prefix.get("end_ms")
        if isinstance(end, bool) or not isinstance(end, (int, float)) or not math.isfinite(end):
            raise ValueError("Invalid prefix endpoint")
        if not previous_end < end <= duration:
            raise ValueError("Prefix endpoints must increase within audio duration")
        hypothesis = to_units(prefix["segments"])
        admissible = len(hypothesis) <= len(final) and hypothesis == final[:len(hypothesis)]
        # Whole earlier output criterion, NOT partial-credit longest common prefix.
        if admissible:
            stable_length = max(stable_length, len(hypothesis))
        is_final = index == len(prefixes) - 1
        if is_final and (abs(end - duration) > 1e-6 or hypothesis != final):
            raise ValueError("Last prefix must be the complete-audio teacher hypothesis")
        release = len(final) if is_final else max(committed, stable_length - holdback_chars)
        if not is_final:
            for start, stop in protected_numbers:
                if start < release < stop:
                    release = max(committed, start)
                    break
        # An English whitespace by itself is not a useful DIRECT. Retain it for
        # the next lexical increment so that the final text stays exactly equal.
        delta = final[committed:release]
        delta_segments = from_units(delta)
        if delta_segments and any(not s["text"].strip() for s in delta_segments):
            release = committed
            delta_segments = []
        turns.append({"start_ms": previous_end, "end_ms": end,
                      "action": "DIRECT" if delta_segments else "WAIT",
                      "segments": delta_segments,
                      "provenance": {"prefix_index": index, "whole_hypothesis_matches": admissible,
                                     "lsp_characters": stable_length, "committed_characters": release}})
        committed, previous_end = release, end
    if committed != len(final):
        raise ValueError("Final transcript was not completely emitted")
    return {"schema_version": 1, "id": trace["id"], "status": "ok", "audio": trace["audio"],
            "duration_ms": duration, "turns": turns, "reference": trace.get("reference"),
            "reference_parse_error": trace.get("reference_parse_error"),
            "policy": {"number_format": number_format, "protect_digit_spans": number_format == "normalized"},
            "audio_info": trace.get("audio_info"), "reference_audit": audit,
            "teacher": trace.get("teacher", {}),
            "provenance": {"algorithm": "whole-earlier-output-final-anchored-joint-character-lsp-v1",
                           "holdback_chars": holdback_chars,
                           "scope": "Text/language/speaker self-consistency is an offline proxy, not a guarantee of acoustic certainty."}}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="Collected teacher trace JSONL")
    p.add_argument("--output", required=True)
    p.add_argument("--holdback-chars", type=int, default=0)
    p.add_argument("--min-reference-agreement", type=float, default=0.0)
    p.add_argument("--resume", action="store_true")
    return p


def run(args):
    if Path(args.input).resolve() == Path(args.output).resolve():
        raise ValueError("Input and output must differ")
    if args.holdback_chars < 0 or not 0 <= args.min_reference_agreement <= 1:
        raise ValueError("Invalid holdback/quality threshold")
    config = {"version": 2, "holdback_chars": args.holdback_chars,
              "min_reference_agreement": args.min_reference_agreement}
    journal = Journal(args.output, config, args.resume)
    counts = {"written": 0, "failed": 0, "resumed": 0}
    completed = False
    try:
        with open(str(args.output) + ".rejected.jsonl", "a", encoding="utf-8", newline="\n") as rejected:
            for line, trace, error in source_rows(args.input):
                try:
                    identifier = row_id(trace, args.input, line)
                except ValueError as exc:
                    identifier = row_id({}, args.input, line)
                    error = str(exc)
                source_hash = fingerprint(trace)
                if journal.previous(source_hash, identifier) is not None:
                    counts["resumed"] += 1
                    continue
                try:
                    if error:
                        raise ValueError(error)
                    output = make_trajectory(trace, args.holdback_chars, args.min_reference_agreement)
                except Exception as exc:
                    output = {"schema_version": 1, "id": identifier, "status": "error", "output": "",
                              "audio": trace.get("audio"), "error": str(exc)}
                    counts["failed"] += 1
                    write_row(rejected, {"id": identifier, "source_line": line, "error": str(exc)})
                    print(f"[ERROR] id={identifier}: {exc}", file=sys.stderr, flush=True)
                output["input_hash"] = source_hash
                journal.append(output)
                counts["written"] += 1
        completed = True
    finally:
        journal.close(check_excess=completed)
    print(json.dumps(counts), flush=True)
    return counts


def main(argv=None):
    try:
        run(parser().parse_args(argv))
    except Exception as error:
        print(f"[FATAL] {error}", file=sys.stderr)
        return 2
    return 0
