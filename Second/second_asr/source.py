"""Read semantic units and real millisecond intervals from an annotated ASR record."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from itertools import groupby

from .protocol import LANGUAGE, SPEAKER

TAG = re.compile(r"<([^<>]+)>")
# The switch annotator inserts markers next to an attribution tag. A literal
# "[200ms]" in the middle of spoken text is never discarded.
HEADER_TIME = re.compile(rf"\[\d+(?:\.\d+)?ms\](?=<(?:{SPEAKER}|{LANGUAGE})>)")


@dataclass(frozen=True)
class Unit:
    start_ms: float
    end_ms: float
    spans: tuple[tuple[str, str, str], ...]


def _characters(text, initial=(None, None)):
    """Expand inherited headers, preserving the legacy normalizer's whitespace."""
    language, speaker = initial
    previous = initial
    pending, chars, leading = {}, [], ""
    cursor = 0
    for match in [*TAG.finditer(text), None]:
        body = text[cursor:match.start() if match else len(text)]
        if "<" in body or ">" in body:
            raise ValueError("malformed transcript header")
        if body.strip():
            if language is None or speaker is None:
                raise ValueError("transcript is missing its initial language or speaker")
            chars.extend((language, speaker, ch) for ch in leading + body)
            leading = ""
            previous = language, speaker
            pending.clear()
        elif body:
            if None in previous:
                leading += body
            else:
                chars.extend((*previous, ch) for ch in body)
        if match is None:
            break
        tag = match.group(1)
        if re.fullmatch(LANGUAGE, tag):
            kind, language = "language", tag
        elif re.fullmatch(SPEAKER, tag):
            kind, speaker = "speaker", tag
        else:
            raise ValueError(f"unknown transcript header <{tag}>")
        if kind in pending and pending[kind] != tag:
            raise ValueError(f"conflicting {kind} headers without speech")
        pending[kind] = tag
        cursor = match.end()
    if pending or not chars:
        raise ValueError("empty transcript or trailing attribution header")
    return chars


def _reference(record):
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("missing messages array")
    answers = [m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
    if len(answers) != 1 or not isinstance(answers[0].get("content"), str):
        raise ValueError("exactly one assistant transcript is required")
    content = answers[0]["content"]
    original = answers[0].get("content_original")
    if original == content:
        return _characters(content)
    clean = HEADER_TIME.sub("", content)
    if original is not None and clean != original:
        raise ValueError("content_original disagrees with the marked transcript")
    return _characters(clean)


def _finite_number(value):
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _interval(item, lower, upper, context):
    start, end = item.get("start_ms"), item.get("end_ms")
    for value in (start, end):
        if not _finite_number(value):
            raise ValueError(f"{context} requires finite numeric start_ms and end_ms")
    if not lower <= start < end <= upper:
        raise ValueError(f"{context} interval is empty or outside its audio/segment")
    return start, end


def _check_text(text, chars, context):
    """Check lexical fidelity and any explicit headers against the known reference."""
    if not isinstance(text, str) or not chars:
        raise ValueError(f"{context} is empty or has no corresponding reference text")
    if TAG.sub("", text) != "".join(ch for _, _, ch in chars):
        raise ValueError(f"{context} differs from reference text")
    if TAG.search(text) and _characters(text, initial=chars[0][:2]) != chars:
        raise ValueError(f"{context} attribution differs from reference")


def _check_metadata(item, chars, context):
    for key in ("source_text", "text"):
        if key in item:
            _check_text(item[key], chars, f"{context}.{key}")
    if "text_plain" in item and item["text_plain"] != "".join(ch for _, _, ch in chars):
        raise ValueError(f"{context}.text_plain differs from reference text")
    for pos, field in enumerate(("language", "speaker")):
        values = {entry[pos] for entry in chars}
        # A scalar cannot describe all states of a mixed segment.
        if len(values) == 1 and field in item and item[field] != chars[0][pos]:
            raise ValueError(f"{context}.{field} differs from reference")
        plural = field + "s"
        if plural in item:
            entries = item[plural]
            if (not isinstance(entries, list) or any(not isinstance(v, str) for v in entries)
                    or set(entries) != values):
                raise ValueError(f"{context}.{plural} differs from reference")


def _check_spans(spans, chars, begin, end, context):
    if not isinstance(spans, list) or not spans:
        raise ValueError(f"{context}.spans must be a nonempty array")
    offset, previous = 0, (begin, begin)
    for span in spans:
        if not isinstance(span, dict):
            raise ValueError(f"{context} has an invalid timed span")
        bounds = _interval(span, begin, end, f"{context} span")
        if bounds[0] < previous[0] or bounds[1] < previous[1]:
            raise ValueError(f"{context} spans are out of chronological order")
        previous = bounds
        raw = span.get("text_plain", span.get("text"))
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"{context} span requires text or text_plain")
        length = len(TAG.sub("", raw))
        part = chars[offset:offset + length]
        if not part or any((span.get("language"), span.get("speaker")) != c[:2] for c in part):
            raise ValueError(f"{context} span attribution differs from reference")
        _check_metadata(span, part, f"{context} span")
        offset += length
    if offset != len(chars):
        raise ValueError(f"{context} spans do not cover the entire unit")


def parse_record(record, duration_ms):
    """Recover complete timed semantic units; old action/flush labels do not gate release."""
    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    if not _finite_number(duration_ms) or duration_ms <= 0:
        raise ValueError("audio duration must be positive and finite")
    audios = record.get("audios")
    if not isinstance(audios, list) or len(audios) != 1 or not isinstance(audios[0], str) or not audios[0].strip():
        raise ValueError("exactly one source audio path is required")
    reference = _reference(record)
    annotation = record.get("output")
    if isinstance(annotation, str):
        annotation = json.loads(annotation)
    if not isinstance(annotation, dict) or annotation.get("error") is not None:
        raise ValueError("missing or unsuccessful segment annotation in output")
    segments = annotation.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("output.segments must be a nonempty array")
    times = record.get("timestamps")
    if not isinstance(times, list) or len(times) != len(segments) or any(not isinstance(t, dict) for t in times):
        raise ValueError("timestamps must supply one start_ms/end_ms interval per semantic segment")
    if any("index" in t for t in times):
        indices = [t.get("index") for t in times]
        if any(type(i) is not int for i in indices) or sorted(indices) != list(range(1, len(times) + 1)):
            raise ValueError("timestamp indices must uniquely cover 1..N")
        times = sorted(times, key=lambda t: t["index"])

    units, offset, previous = [], 0, (0, 0)
    for i, (segment, timestamp) in enumerate(zip(segments, times), 1):
        context = f"segment {i}"
        if not isinstance(segment, dict) or not isinstance(segment.get("source_text"), str):
            raise ValueError(f"{context} requires source_text")
        if "action" in segment and segment["action"] not in ("WAIT", "DIRECT"):
            raise ValueError(f"{context} has an unknown legacy action")
        raw = segment["source_text"]
        plain = TAG.sub("", raw)
        if not plain.strip() or "<" in plain or ">" in plain:
            raise ValueError(f"{context} has empty text or a malformed header")
        chars = reference[offset:offset + len(plain)]
        _check_text(raw, chars, context)
        begin, end = _interval(timestamp, 0, duration_ms, context)
        if begin < previous[0] or end < previous[1]:
            raise ValueError(f"{context} is out of chronological order")
        _check_metadata(timestamp, chars, f"{context} timestamp")
        if timestamp.get("spans") is not None:
            _check_spans(timestamp["spans"], chars, begin, end, context)
        spans = tuple((language, speaker, "".join(ch for _, _, ch in group))
                      for (language, speaker), group in groupby(chars, key=lambda c: c[:2]))
        units.append(Unit(begin, end, spans))
        offset += len(plain)
        previous = begin, end
    if offset != len(reference):
        raise ValueError("semantic segments do not cover the entire reference transcript")
    return units
