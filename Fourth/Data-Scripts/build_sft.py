#!/usr/bin/env python3
"""Merge attributed fragments and build Fourth streaming SFT directly from timed JSONL."""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_asr.annotation_tail import project_open_tail
from fourth_asr.protocol import SPK_CHANGE, WAIT, get_protocol
from fourth_asr.recast import build_sft_row, end_frame, validate_chunk_ms
from fourth_asr.semantic_preparation import prepare_units
from fourth_asr.timestamped_source import HEADER_TIME, TAG, reference_spans


# Keep tabs and ordinary spaces outside line-break positions exactly as annotated.
LINE_BREAK = re.compile(r"[\r\n\v\f\x85\u2028\u2029]")
LAYOUT = r" \t\r\n\v\f\x85\u2028\u2029"
BLANKS = re.compile(rf"[{LAYOUT}]+")
MARKED_PART = re.compile(rf"(?P<tag><[^<>]+>)|(?P<gap>[{LAYOUT}]+)|(?P<body>[^<{LAYOUT}]+|<)")
TEXT_KEYS = ("source_text", "text", "text_plain")


def _nonlayout_text(marked):
    return BLANKS.sub("", TAG.sub("", marked))


def _timed_text_fields(segments, timestamps):
    """Locate duplicate text fields by non-layout character offset in the transcript."""
    if len(timestamps) != len(segments) or any(not isinstance(t, dict) for t in timestamps):
        raise ValueError("Expected one timestamp per source segment")
    if any("index" in t for t in timestamps):
        if any(type(t.get("index")) is not int for t in timestamps):
            raise ValueError("Invalid timestamp index")
        timestamps = sorted(timestamps, key=lambda t: t["index"])
    offset = 0
    for segment, stamp in zip(segments, timestamps):
        if not isinstance(segment, dict) or not isinstance(segment.get("source_text"), str):
            raise ValueError("Expected source_text in each output segment")
        yield segment, "source_text", offset
        for key in TEXT_KEYS:
            if isinstance(stamp.get(key), str):
                yield stamp, key, offset
        child_offset = offset
        children = stamp.get("spans")
        if isinstance(children, list):
            for child in children:
                if not isinstance(child, dict):
                    raise ValueError("Invalid timed span")
                for key in TEXT_KEYS:
                    if isinstance(child.get(key), str):
                        yield child, key, child_offset
                body = child.get("text_plain", child.get("text"))
                if not isinstance(body, str):
                    raise ValueError("Timed span requires text or text_plain")
                child_offset += len(_nonlayout_text(body))
        offset += len(_nonlayout_text(segment["source_text"]))


def _break_offsets(text, offset):
    plain = TAG.sub("", text)
    cursor = 0
    for gap in BLANKS.finditer(plain):
        offset += gap.start() - cursor
        if LINE_BREAK.search(gap[0]):
            yield offset
        cursor = gap.end()


def _english_space(letters, pos):
    if not 0 < pos < len(letters):
        return False
    left, right = letters[pos-1:pos+1]
    def word_character(ch):
        return ch.isdecimal() or (ch.isalpha() and "LATIN" in unicodedata.name(ch, ""))

    if not word_character(right):
        return False
    # Find the base letter of a decomposed accent at the end of an English word.
    base = pos - 1
    while base and unicodedata.category(letters[base]).startswith("M"):
        base -= 1
    if left in ".," and right.isdigit() and pos >= 2 and letters[pos-2].isdigit():
        return False  # A decimal or grouped number can span a printed line.
    return word_character(letters[base]) or left in ".,;:!?)]\"”"


def _render_linebreaks(text, offset, breaks, spaces):
    """Place a needed blank in the same attributed span as the word on its right."""
    result = []
    for part in MARKED_PART.finditer(text):
        value = part[0]
        if part.lastgroup == "body":
            for char in value:
                if offset in spaces:
                    result.append(" ")
                result.append(char)
                offset += 1
        elif part.lastgroup == "tag" or offset not in breaks:
            result.append(value)
    return "".join(result)


def normalize_source_record(record):
    """Return a training view. The original record, interval times and tags stay intact."""
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("missing messages array")
    answers = [(i, msg) for i, msg in enumerate(messages)
               if isinstance(msg, dict) and msg.get("role") == "assistant"]
    if len(answers) != 1 or not isinstance(answers[0][1].get("content"), str):
        raise ValueError("exactly one assistant transcript is required")
    index, source_answer = answers[0]
    marked = HEADER_TIME.sub("", source_answer["content"])
    raw = record.get("output")
    annotation = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(annotation, dict) or not isinstance(annotation.get("segments"), list):
        raise ValueError("Expected output.segments for direct 1s recast")
    if not isinstance(record.get("timestamps"), list):
        raise ValueError("Expected timestamps for direct 1s recast")
    # JSON string annotations are already detached. Copy object annotations and
    # timestamp metadata before editing text shared with the caller.
    annotation = annotation if isinstance(raw, str) else deepcopy(annotation)
    timestamps = deepcopy(record["timestamps"])
    segments = annotation["segments"]
    answer = {**source_answer, "content": marked}
    fields = [(answer, "content", 0), *_timed_text_fields(segments, timestamps)]
    if answer.get("content_original") is not None:
        if not isinstance(answer["content_original"], str):
            raise ValueError("content_original must be a string")
        fields.append((answer, "content_original", 0))
    breaks = set()
    for item, key, offset in fields:
        if LINE_BREAK.search(item[key]):
            breaks.update(_break_offsets(item[key], offset))
    if not breaks:
        return record

    letters = _nonlayout_text(marked)
    spaces = {pos for pos in breaks if _english_space(letters, pos)}
    # An uninterrupted spelling inside a reference or source segment is evidence
    # for a split word (or unit, e.g. 2dB). An adjacent segment supplies no such
    # evidence. Keep that spelling when another copy wraps it mid-word.
    spellings = [[marked], (s["source_text"] for s in segments)]
    if answer.get("content_original") is not None:
        spellings.append([answer["content_original"]])
    for texts in spellings:
        offset = 0
        for text in texts:
            if not spaces:
                break
            for part in MARKED_PART.finditer(text):
                if part.lastgroup == "body":
                    end = offset + len(part[0])
                    for pos in range(offset + 1, end):
                        spaces.discard(pos)
                    offset = end

    for item, key, offset in fields:
        item[key] = _render_linebreaks(item[key], offset, breaks, spaces)
    if answer.get("content_original") is not None and answer["content_original"] != answer["content"]:
        raise ValueError("content_original disagrees with the marked transcript")
    messages = list(messages)
    messages[index] = answer
    return {**record, "messages": messages, "output": annotation, "timestamps": timestamps}


def json_line(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"


def reference_body(spans):
    """Render the source independently of short-fragment merging and release scheduling."""
    expected = []
    last_speaker = None
    for _, speaker, text in spans:
        if text.strip():
            if last_speaker is not None and speaker != last_speaker:
                expected.append(SPK_CHANGE)
            last_speaker = speaker
        expected.append(text)
    return "".join(expected)


def checked_changes(expected, replies, protocol):
    observed = []
    changed = []
    for turn, reply in enumerate(replies, 1):
        body = protocol.parse_reply(reply["content"])
        observed.append(body)
        if SPK_CHANGE in body:
            changed.append(turn)
    if expected != "".join(observed):
        raise ValueError("Recast changed speaker order, transcript, punctuation or spacing")
    return changed


def prepare_record(record, audio_root, steps, number_format, release_on, *, protocol, short_ms, tail_policy="keep"):
    audios = record.get("audios") if isinstance(record, dict) else None
    if not (isinstance(audios, list) and len(audios) == 1 and
            isinstance(audios[0], str) and audios[0].strip()):
        raise ValueError("Expected one original audio path in audios")
    if release_on == "source-segment":
        record = normalize_source_record(record)
    path = (audio_root / audios[0]).resolve(strict=True)
    try:
        info = sf.info(str(path))
    except (RuntimeError, OSError) as exc:
        raise ValueError(f"Cannot read audio metadata for {path}: {exc}") from exc
    if info.frames < 1 or info.samplerate < 1:
        raise ValueError(f"Invalid audio length or sample rate: {path}")
    duration_ms = info.frames * 1000 / info.samplerate
    source_spans = reference_spans(record)
    expected = reference_body(source_spans)
    result = {}
    for step in steps:
        units, preparation = prepare_units(record, duration_ms, chunk_ms=step, short_ms=short_ms,
                                           release_on=release_on)
        audit = {}
        # Use the latest original timestamp even for overlapping speech. The file
        # name and timestamp of the last segment in text order can both mislead.
        last_annotated = max(t["end_ms"] for t in record["timestamps"])
        last_frame = end_frame(last_annotated, info.frames, info.samplerate)
        short_tail = info.frames - last_frame < info.samplerate
        open_end = tail_policy == "annotated-prefix" or (tail_policy == "short-open" and short_tail)
        frames, target, stripped = info.frames, expected, False
        if open_end:
            if short_tail:
                units, target, stripped = project_open_tail(units, source_spans[-1][2], expected)
            frames = end_frame(max(u.end_ms for u in units), info.frames, info.samplerate)
        row = build_sft_row(units, path, frames, info.samplerate, step, number_format, protocol=protocol,
                            audit=audit, release_completed=release_on in ("merged-unit", "source-segment"),
                            open_end=open_end)
        real_replies = row["messages"][2:2+2*len(row["audios"]):2]
        waits = sum(msg["content"] == WAIT for msg in real_replies)
        changes = checked_changes(target, real_replies, protocol)
        events = [{"turn": turn, "kind": kind}
                  for kind, turns in (("speaker_change", changes),
                                      ("switch_deferred", audit["switch_deferred_turns"]),
                                      ("punctuation_deferred", audit["punctuation_deferred_turns"]))
                  for turn in turns]
        events.sort(key=lambda event: event["turn"])
        # A real audio-end event keeps Arrow's nested event schema typed even
        # when the first shard contains only single-speaker, monolingual rows.
        events.append({"turn": len(real_replies), "kind": "annotation_end" if open_end else "audio_end"})
        row["recast"] = {
            "source_audio": str(path), "chunk_ms": step, "merged_units": preparation["merged_units"],
            "events": events,
        }
        if tail_policy != "keep":
            row["recast"].update(source_frames=info.frames, supervised_frames=frames,
                                 terminal_period_removed=stripped)
        stats = Counter(preparation)
        stats.update(audio_turns=len(real_replies), wait_audio_turns=waits,
                     direct_audio_turns=len(real_replies) - waits, speaker_change_turns=len(changes),
                     open_ended_rows=int(open_end), terminal_periods_removed=int(stripped),
                     switch_deferred_turns=len(audit["switch_deferred_turns"]),
                     punctuation_deferred_turns=len(audit["punctuation_deferred_turns"]))
        result[step] = (json_line(row).encode("utf-8"), stats)
    # Report gaps between original annotations; the audio there can contain speech.
    timestamps = sorted(record["timestamps"], key=lambda item: item["start_ms"])
    gaps = []
    covered = 0
    for stamp in timestamps:
        gaps.append((covered, stamp["start_ms"]))
        covered = max(covered, stamp["end_ms"])
    gaps.append((covered, duration_ms))
    longest_gap = max(gaps, key=lambda gap: gap[1] - gap[0])
    return result, longest_gap


def main(argv=None, *, source_segments_only=False):
    description = "Clean transcript line breaks and recast original timestamped segments into 1-second Fourth SFT turns."
    parser = argparse.ArgumentParser(description=description if source_segments_only else __doc__)
    parser.add_argument("input", type=Path, help="Source JSONL with per-unit timestamps in ms")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, help="Base for relative audio paths (default: input directory)")
    if source_segments_only:
        parser.set_defaults(chunk_ms=[1000], short_ms=0, release_on="source-segment")
    else:
        parser.add_argument("--chunk-ms", nargs="+", type=int, default=[1000], metavar="MS")
        parser.add_argument("--short-ms", type=int, default=300, help="Short-fragment merge threshold in ms")
        parser.add_argument("--release-on", choices=("semantic-unit", "timed-span", "merged-unit"), default="semantic-unit",
                            help="Use merged-unit after merge_timestamped_segments.py to emit completed groups immediately")
    parser.add_argument("--tail-policy", choices=("keep", "short-open", "annotated-prefix"), default="keep",
                        help="keep: original whole recording; short-open: treat a trailing gap under 1 s as a crop; "
                             "annotated-prefix: end all examples at the last annotation. Both open modes remove a "
                             "simple terminal period only for a gap under 1 s.")
    parser.add_argument("--number-format", choices=["normalized", "verbatim"], default="normalized")
    parser.add_argument("--protocol", choices=("speaker", "explicit"), default="speaker",
                        help="Fourth speaker-change protocol; explicit is a CLI alias")
    args = parser.parse_args(argv)
    protocol = get_protocol("speaker")
    steps = args.chunk_ms
    try:
        for step in steps:
            validate_chunk_ms(step)
    except ValueError as exc:
        parser.error(str(exc))
    if len(set(steps)) != len(steps):
        parser.error("--chunk-ms values must be unique")
    if args.short_ms < 0:
        parser.error("--short-ms must be nonnegative")
    if not args.input.is_file():
        parser.error(f"Missing input file: {args.input}")
    audio_root = args.audio_root or args.input.parent
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    paths = {step: output / f"train.{step}ms.jsonl" for step in steps}
    tokens_files = [output / "token.txt", *(Path(str(path) + ".tokens.txt") for path in paths.values())]
    rejected, summary = output / "rejected.jsonl", output / "summary.json"
    for path in [*paths.values(), *tokens_files, rejected, summary]:
        if path.exists():
            parser.error(f"Output already exists: {path}. Select a fresh --output-dir.")

    kept = bad = 0
    longest_gap = None
    counts = {step: Counter() for step in steps}
    with args.input.open("rb") as source, ExitStack() as stack:
        writers = {step: stack.enter_context(path.open("xb")) for step, path in paths.items()}
        errors = stack.enter_context(rejected.open("x", encoding="utf-8", newline="\n"))
        for line_number, raw in enumerate(source, 1):
            try:
                line = raw.decode("utf-8-sig" if line_number == 1 else "utf-8")
                record = json.loads(line)
                rows, (gap_start, gap_end) = prepare_record(
                    record, audio_root, steps, args.number_format, args.release_on,
                    protocol=protocol, short_ms=args.short_ms, tail_policy=args.tail_policy)
            except (ValueError, KeyError, TypeError, OSError, OverflowError, RecursionError) as exc:
                # Serialize errors as ASCII so even malformed Unicode input can be reported.
                errors.write(json.dumps({"line": line_number, "error": str(exc)}, ensure_ascii=True) + "\n")
                errors.flush()
                bad += 1
            else:
                # Prepare every grid before the first write, keeping their samples in sync.
                # Storage failures propagate rather than turning into data rejections.
                for step, (payload, stats) in rows.items():
                    writers[step].write(payload)
                    counts[step].update(stats)
                kept += 1
                gap_ms = gap_end - gap_start
                if gap_ms > 0 and (longest_gap is None or gap_ms > longest_gap["duration_ms"]):
                    longest_gap = {"line": line_number, "start_ms": gap_start, "end_ms": gap_end,
                                   "duration_ms": gap_ms}
            if line_number == 1 or line_number % 1000 == 0:
                for stream in writers.values():
                    stream.flush()
                print(f"Rows {line_number} | kept {kept} | rejected {bad}", file=sys.stderr, flush=True)
        for path in tokens_files:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write("".join(t + "\n" for t in protocol.tokens))
        report = {"input": str(args.input.resolve()), "number_format": args.number_format, "release_on": args.release_on,
                  "kept": kept, "rejected": bad, "chunk_ms": steps, "by_chunk_ms": counts,
                  "longest_unannotated_gap": longest_gap}
        report.update(protocol=protocol.name, tail_policy=args.tail_policy)
        if not source_segments_only:
            report["short_ms"] = args.short_ms
        with summary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(f"Finished | kept {kept} | rejected {bad} | {summary}", file=sys.stderr, flush=True)
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
