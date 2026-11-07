"""Join lossless text annotations to real acoustic word/character timestamps.

No proportional timing, ASR rewriting, or punctuation-time interpolation is used.
A source unit crossing an aligned word uses that word's full interval. This can
delay a commit; it cannot fabricate a sub-word completion time.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
import unicodedata

from .io import Journal, fingerprint, row_id, write_row

TAG = re.compile(r"<([^<>]+)>")
LANGUAGE = re.compile(r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*\Z")
SPEAKER = re.compile(r"spk[1-9][0-9]*\Z")


def source_rows(path):
    """Isolate malformed UTF-8, JSON, nonobjects and lone surrogate records."""
    import hashlib
    with open(path, "rb") as stream:
        for line, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                def invalid_constant(value):
                    raise ValueError("Non-finite JSON value: " + value)
                row = json.loads(raw.decode("utf-8-sig" if line == 1 else "utf-8"),
                                 parse_constant=invalid_constant)
                if not isinstance(row, dict):
                    raise ValueError("JSONL row must be an object")
                json.dumps(row, ensure_ascii=False, allow_nan=False).encode("utf-8")
                yield line, row, None
            except (ValueError, TypeError, UnicodeError) as exc:
                yield line, {"_malformed_line_sha256": hashlib.sha256(raw).hexdigest()}, str(exc)


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + " must be finite")
    return float(value)


def tagged_units(text):
    """Parse inherited attribution into one unit per original text character."""
    if not isinstance(text, str) or not text:
        raise ValueError("Tagged transcript must be nonempty text")
    language = speaker = None
    pending = {}
    units, position, leading = [], 0, []

    def append(value):
        nonlocal leading
        if not value:
            return
        if not value.strip() and (language is None or speaker is None):
            leading.append(value)
            return
        if language is None or speaker is None or "<" in value or ">" in value:
            raise ValueError("Missing initial attribution or malformed tags")
        if leading:
            value = "".join(leading) + value
            leading = []
        units.extend((language, speaker, char) for char in value)
        if value.strip():
            pending.clear()

    for match in TAG.finditer(text):
        append(text[position:match.start()])
        tag = match.group(1)
        kind = "speaker" if SPEAKER.fullmatch(tag) else "language" if LANGUAGE.fullmatch(tag) else None
        if kind is None:
            raise ValueError("Unknown transcript tag: " + tag)
        if kind in pending and pending[kind] != tag:
            raise ValueError("Conflicting attribution headers without speech")
        pending[kind] = tag
        if kind == "speaker":
            speaker = tag
        else:
            language = tag
        position = match.end()
    append(text[position:])
    if pending or not units:
        raise ValueError("Transcript ends with empty attribution")
    return units


def normalize_source(text):
    """Match annotate_asr.normalize_transcript, including inter-header spaces."""
    language = speaker = previous = None
    pending, result, position = {}, [], 0
    for match in list(TAG.finditer(text)) + [None]:
        stop = match.start() if match else len(text)
        body = text[position:stop]
        if "<" in body or ">" in body:
            raise ValueError("Malformed source tag")
        if body.strip():
            if language is None or speaker is None:
                raise ValueError("Missing initial attribution")
            if (language, speaker) != previous:
                result.append("<%s><%s>" % (language, speaker))
                previous = language, speaker
            result.append(body)
            pending.clear()
        elif body:
            result.append(body)
        if match is None:
            break
        tag = match.group(1)
        kind = "speaker" if SPEAKER.fullmatch(tag) else "language" if LANGUAGE.fullmatch(tag) else None
        if kind is None or (kind in pending and pending[kind] != tag):
            raise ValueError("Unknown or conflicting source attribution")
        pending[kind] = tag
        if kind == "speaker":
            speaker = tag
        else:
            language = tag
        position = match.end()
    if pending or previous is None:
        raise ValueError("Empty source attribution")
    return "".join(result)


def render_units(units):
    result, state = [], None
    for language, speaker, char in units:
        if (language, speaker) != state:
            result.append("<%s><%s>" % (language, speaker))
            state = language, speaker
        result.append(char)
    return "".join(result)


def group_units(units):
    result = []
    for language, speaker, char in units:
        if result and (language, speaker) == (result[-1]["language"], result[-1]["speaker"]):
            result[-1]["text"] += char
        else:
            result.append({"language": language, "speaker": speaker, "text": char})
    if any(not item["text"].strip() for item in result):
        raise ValueError("An attribution span contains only whitespace")
    return result


def raw_transcript(row):
    values = [message.get("content") for message in row.get("messages", [])
              if isinstance(message, dict) and message.get("role") == "assistant"]
    if len(values) != 1 or not isinstance(values[0], str) or not values[0].strip():
        raise ValueError("Exactly one nonempty assistant transcript is required")
    return values[0]


def annotation_units(row):
    annotation = row.get("output")
    if isinstance(annotation, str):
        if not annotation:
            raise ValueError("Annotation output is empty")
        annotation = json.loads(annotation)
    if not isinstance(annotation, dict) or set(annotation) != {"segments", "flush", "error"}:
        raise ValueError("Expected v2 annotation keys segments, flush, error")
    if annotation["error"] is not None or not isinstance(annotation["flush"], bool):
        raise ValueError("Failed annotation or invalid flush")
    segments = annotation["segments"]
    if not isinstance(segments, list) or not segments:
        raise ValueError("Annotation has no segments")
    source = normalize_source(raw_transcript(row))
    for segment in segments:
        if not isinstance(segment, dict) or set(segment) != {"source_text", "action"}:
            raise ValueError("Annotation segments require source_text and action only")
        if not isinstance(segment["source_text"], str) or not segment["source_text"]:
            raise ValueError("Empty source unit")
        if segment["action"] not in ("WAIT", "DIRECT"):
            raise ValueError("Unknown action")
        plain = TAG.sub("", segment["source_text"])
        if "<" in plain or ">" in plain or not any(c.isalnum() for c in plain):
            raise ValueError("Source unit cuts a tag or has no lexical content")
        if re.search(r"<[^<>]+>\s*\Z", segment["source_text"]):
            raise ValueError("Source unit ends with empty attribution")
    joined = "".join(s["source_text"] for s in segments)
    units = tagged_units(source)
    # Teacher input has change-only headers; each DIRECT output carries its own
    # complete header. Extra identical headers are metadata, not new speech or
    # a switch. Compare attribution for EVERY character, not stripped text alone.
    # This also admits a full header moved before preserved initial whitespace.
    if TAG.sub("", joined) != TAG.sub("", source) or tagged_units(joined) != units:
        raise ValueError("Annotation does not reconstruct source text and attribution exactly")
    if annotation["flush"] != (segments[-1]["action"] == "WAIT"):
        raise ValueError("Annotation flush disagrees with final action")
    result, offset, previous_state = [], 0, None
    for segment in segments:
        plain = TAG.sub("", segment["source_text"])
        if "<" in plain or ">" in plain or not any(c.isalnum() for c in plain):
            raise ValueError("Source unit cuts a tag or has no lexical content")
        local = units[offset:offset + len(plain)]
        if "".join(unit[2] for unit in local) != plain:
            raise ValueError("Source unit character mapping mismatch")
        switched = False
        for language, speaker, _ in local:
            if previous_state is not None and previous_state != (language, speaker):
                switched = True
            previous_state = language, speaker
        if switched and segment["action"] != "WAIT":
            raise ValueError("Actual attribution switch must be WAIT")
        result.append(dict(segment, attributed=group_units(local), char_start=offset,
                           char_end=offset + len(plain), switched=switched))
        offset += len(plain)
    return result, units, annotation["flush"]


def lexical_chars(text):
    # Keep combining marks (Arabic vowels, Thai tones, etc.) and mathematical
    # symbols. Ignore only punctuation/whitespace for alignment correspondence.
    # NFKC would silently conflate distinct numeric forms; use NFC instead.
    text = unicodedata.normalize("NFC", text).casefold()
    return "".join(c for i, c in enumerate(text) if not c.isspace() and (
        not unicodedata.category(c).startswith("P")
        or c in "%‰‱"
        or (c in ".,:/" and 0 < i < len(text) - 1 and text[i-1].isdigit() and text[i+1].isdigit())
        or (c in "-−" and i < len(text) - 1 and text[i+1].isdigit())))


def alignment_words(record, duration_ms):
    if record.get("status", "ok") != "ok":
        raise ValueError("Acoustic alignment failed: " + str(record.get("error", "unknown")))
    values = record.get("words", record.get("chars", record.get("word_segments")))
    if not isinstance(values, list) or not values:
        raise ValueError("Alignment needs nonempty words/chars with real timestamps")
    result, previous_start, previous_end = [], -1.0, -1.0
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("Alignment item must be an object")
        text = value.get("text", value.get("word", value.get("char")))
        if not isinstance(text, str) or not text:
            raise ValueError("Alignment item has no text")
        lexical = lexical_chars(text)
        if not lexical:
            continue
        if "start_ms" in value or "end_ms" in value:
            start = finite(value.get("start_ms"), "start_ms")
            end = finite(value.get("end_ms"), "end_ms")
        else:
            start = 1000 * finite(value.get("start"), "start seconds")
            end = 1000 * finite(value.get("end"), "end seconds")
        if not 0 <= start < end <= duration_ms + 1e-6:
            raise ValueError("Alignment interval outside audio or empty")
        if start < previous_start - 1e-6 or end < previous_end - 1e-6:
            raise ValueError("Alignment is not chronological; overlapping unordered speech needs separate review")
        result.append({"text": text, "lexical": lexical, "start_ms": start, "end_ms": end})
        previous_start, previous_end = start, end
    if not result:
        raise ValueError("No timed lexical alignment units")
    return result


def align_annotation(row, acoustic, audio_root=None, identifier=None):
    import soundfile as sf

    segments, units, flush = annotation_units(row)
    paths = row.get("audios")
    if not isinstance(paths, list) or len(paths) != 1 or not isinstance(paths[0], str):
        raise ValueError("Exactly one source audio path is required")
    audio = Path(paths[0]).expanduser()
    if not audio.is_absolute():
        audio = Path(audio_root or Path.cwd()) / audio
    audio = audio.resolve(strict=True)
    info = sf.info(str(audio))
    duration = info.frames * 1000.0 / info.samplerate
    if duration <= 0:
        raise ValueError("Empty source audio")
    if "duration_ms" in acoustic and abs(finite(acoustic["duration_ms"], "duration_ms") - duration) > 1:
        raise ValueError("Alignment duration differs from source audio")
    words = alignment_words(acoustic, duration)
    plain = "".join(unit[2] for unit in units)
    target = lexical_chars(plain)
    observed = "".join(word["lexical"] for word in words)
    if target != observed:
        raise ValueError("Alignment lexical text differs from transcript (no substitutions or guessed matching)")
    # Normalize each atomic unit and compare with the normalized whole once.
    # This rejects boundaries inside combining sequences/numeric punctuation
    # while remaining linear rather than normalizing every growing prefix.
    pieces = [lexical_chars(plain[s["char_start"]:s["char_end"]]) for s in segments]
    if "".join(pieces) != target:
        raise ValueError("Source boundary splits a normalized character or numeric punctuation")
    timed = []
    for word in words:
        timed.extend([(word["start_ms"], word["end_ms"])] * len(word["lexical"]))
    first = 0
    for segment, piece in zip(segments, pieces):
        last = first + len(piece)
        if not first < last:
            raise ValueError("Source unit boundary splits a normalized character or has no alignable content")
        segment["start_ms"] = min(t[0] for t in timed[first:last])
        segment["end_ms"] = max(t[1] for t in timed[first:last])
        first = last
    return {"schema_version": 2, "kind": "aligned_asr", "status": "ok",
            "id": identifier or str(row.get("id", row.get("key", audio))),
            "audio": str(audio), "duration_ms": duration, "segments": segments,
            "annotation_flush": flush, "transcript": render_units(units),
            "audio_info": {"sample_rate": info.samplerate, "frames": info.frames},
            "alignment": {"method": "exact-lexical-sequential-word-or-character-interval-union",
                          "timestamp_source": acoustic.get("backend", "external"),
                          "proportional_timing": False,
                          "note": "Word intervals are conservative bounds; accuracy depends on acoustic alignment."}}


def alignment_keys(record):
    keys = []
    if record.get("id", record.get("key")) is not None:
        keys.append(("id", str(record.get("id", record.get("key")))))
    audio = record.get("audio")
    if not audio and isinstance(record.get("audios"), list) and len(record["audios"]) == 1:
        audio = record["audios"][0]
    if isinstance(audio, str) and audio:
        keys.append(("audio", audio.replace("\\", "/")))
    return keys


def load_alignment_index(path):
    index = {}
    for line, record, error in source_rows(path):
        keys = alignment_keys(record)
        if not keys:
            print(f"[ALIGNMENT_REJECT] line={line}: missing id/audio or malformed JSON", file=sys.stderr)
            continue
        for key in keys:
            if key in index:
                index[key] = ValueError("Duplicate alignment key: " + str(key))
            elif error:
                index[key] = ValueError(error)
            else:
                index[key] = record
    return index


def find_alignment(row, index):
    # Prefer explicit record ID. Audio fallback permits datasets without IDs.
    for key in alignment_keys(row):
        if key in index:
            value = index[key]
            if isinstance(value, Exception):
                raise value
            row_audio = [v for k, v in alignment_keys(row) if k == "audio"]
            match_audio = [v for k, v in alignment_keys(value) if k == "audio"]
            if row_audio and match_audio and row_audio != match_audio:
                raise ValueError("Alignment ID matched but audio paths differ")
            return value
    raise ValueError("No acoustic alignment matched this record ID/audio")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="Annotated JSONL, with output.segments")
    p.add_argument("--timestamps", required=True, help="Real acoustic alignment JSONL keyed by id or audio")
    p.add_argument("--output", required=True)
    p.add_argument("--audio-root", help="Resolve relative source audio paths")
    p.add_argument("--resume", action="store_true")
    return p


def run(args):
    if len({Path(p).resolve() for p in (args.input, args.timestamps, args.output)}) != 3:
        raise ValueError("Input, timestamps and output must be distinct")
    index = load_alignment_index(args.timestamps)
    # Contents, not just timestamp filename, are part of resume identity.
    import hashlib
    timestamp_hash = hashlib.sha256(Path(args.timestamps).read_bytes()).hexdigest()
    journal = Journal(args.output, {"pipeline": "align-v2", "timestamps_sha256": timestamp_hash,
                                   "audio_root": str(args.audio_root)}, args.resume)
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
                    result = align_annotation(row, find_alignment(row, index),
                                              args.audio_root or Path(args.input).resolve().parent, identifier)
                except Exception as exc:
                    counts["failed"] += 1
                    result = {"schema_version": 2, "kind": "aligned_asr", "status": "error",
                              "id": identifier, "output": "", "error": str(exc)}
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
