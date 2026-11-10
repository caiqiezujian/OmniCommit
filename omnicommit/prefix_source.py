"""Parse switch-annotated transcripts and plan causal, language-local queries.

All times are milliseconds relative to the sole ``audios`` file. Text offsets
refer to Python Unicode characters, not UTF-8 bytes. Repeated attribute headers
are merely old semantic boundaries: they neither reset the teacher nor create
switches. Primary teacher queries restart only at a real language change;
auxiliary local queries can recover later speakers without rewriting that track.
"""
from __future__ import annotations

import math
import re

from .protocol import LANGUAGE, SPEAKER, TAG

MARKER = re.compile(r"\[(\d+(?:\.\d+)?)ms\]")


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    return float(value)


def _tagged_spans(text):
    """Keep exact body text, plus original header locations for audit checks."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Assistant transcription must be nonempty text")
    result, pending, pending_space = [], [], ""
    language = speaker = None
    position = 0

    def append_body(body):
        nonlocal pending, pending_space
        if not body:
            return
        if "<" in body or ">" in body:
            raise ValueError("Malformed tag or literal angle bracket in reference text")
        if not body.strip():
            if pending or not result:
                pending_space += body
            else:
                result[-1]["text"] += body
            return
        if language is None or speaker is None:
            raise ValueError("Initial transcription requires language and speaker headers")
        body = pending_space + body
        if result and (result[-1]["language"], result[-1]["speaker"]) == (language, speaker):
            result[-1]["text"] += body
        else:
            result.append({"language": language, "speaker": speaker, "text": body,
                           "_headers": list(pending)})
        pending, pending_space = [], ""

    for match in TAG.finditer(text):
        append_body(text[position:match.start()])
        tag = match.group(1)
        if SPEAKER.fullmatch(tag):
            kind, speaker = "speaker", tag
        elif LANGUAGE.fullmatch(tag):
            kind, language = "language", tag
        else:
            raise ValueError("Unknown reference tag: " + tag)
        if any(header["kind"] == kind for header in pending):
            raise ValueError("Consecutive " + kind + " headers without speech text")
        pending.append({"kind": kind, "char_start": match.start(),
                        "char_end": match.end(), "tag": match.group(0)})
        position = match.end()
    append_body(text[position:])
    if pending:
        raise ValueError("Reference ends with attributes but no speech text")
    if pending_space:
        if not result:
            raise ValueError("Reference contains no speech text")
        result[-1]["text"] += pending_space
    if not result:
        raise ValueError("Reference contains no speech text")
    offset = 0
    for span in result:
        span["char_start"], span["char_end"] = offset, offset + len(span["text"])
        offset = span["char_end"]
    return result


def _remove_markers(content):
    """Return de-marked text and each marker's position in that original text."""
    markers, removed = [], 0
    for match in MARKER.finditer(content):
        markers.append({"time_ms": float(match.group(1)), "position": match.start() - removed})
        removed += len(match.group(0))
    return MARKER.sub("", content), markers


def parse_source(record, duration_ms):
    """Validate the user's ``audios/messages/timestamps`` schema without audio I/O.

    A missing timestamps list is valid only if there are no real changes.
    ``tag_positions`` are checked against ``content_original`` (or the content
    with verified inline time markers removed), never the shifted marked text.
    No transcript normalization, speaker renumbering, or time-origin inference
    from filenames is performed.
    """
    duration_ms = _number(duration_ms, "duration_ms")
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    if not isinstance(record, dict):
        raise ValueError("Source row must be an object")
    audios = record.get("audios")
    if (not isinstance(audios, list) or len(audios) != 1
            or not isinstance(audios[0], str) or not audios[0].strip()):
        raise ValueError("Source requires exactly one audio path")
    messages = record.get("messages")
    if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
        raise ValueError("messages must be an array of objects")
    assistants = [(i, message) for i, message in enumerate(messages) if message.get("role") == "assistant"]
    if len(assistants) != 1:
        raise ValueError("Source requires exactly one assistant transcription")
    message_index, message = assistants[0]
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Assistant content must be text")
    original, markers = _remove_markers(content)
    supplied_original = message.get("content_original")
    if supplied_original is not None and supplied_original != original:
        raise ValueError("content_original differs from content after removing inline timestamp markers")
    spans = _tagged_spans(original)
    timestamps = record.get("timestamps", [])
    if not isinstance(timestamps, list) or any(not isinstance(t, dict) for t in timestamps):
        raise ValueError("timestamps must be an array of objects")
    if len(timestamps) != len(spans) - 1:
        raise ValueError("Every real language/speaker change needs exactly one timestamp; repeated headers do not")
    events = []
    previous_time = 0.0
    for index, (before, after, raw) in enumerate(zip(spans, spans[1:], timestamps), 1):
        changed = [key for key in ("language", "speaker") if before[key] != after[key]]
        if "message_index" in raw and (type(raw["message_index"]) is not int or raw["message_index"] != message_index):
            raise ValueError("Timestamp message_index does not identify the assistant transcription")
        if "changed" in raw:
            supplied = raw["changed"]
            if not isinstance(supplied, list) or len(supplied) != len(set(supplied)) or set(supplied) != set(changed):
                raise ValueError("Timestamp changed fields disagree with actual attribution change")
        for key in ("language", "speaker"):
            if key in raw and raw[key] != after[key]:
                raise ValueError("Timestamp " + key + " disagrees with reference text")
            if "previous_" + key in raw and raw["previous_" + key] != before[key]:
                raise ValueError("Timestamp previous_" + key + " disagrees with reference text")
        if "text" in raw and raw["text"] != after["text"]:
            raise ValueError("Timestamp text differs from the complete attributed span")
        time_ms = _number(raw.get("time_ms"), "time_ms")
        if not previous_time < time_ms < duration_ms:
            raise ValueError("Switch times must strictly increase inside the current audio duration")
        previous_time = time_ms
        positions = raw.get("tag_positions", [])
        if not isinstance(positions, list) or any(not isinstance(p, dict) for p in positions):
            raise ValueError("tag_positions must be an array of objects")
        seen_positions = set()
        for supplied in positions:
            kind = supplied.get("kind")
            if kind not in changed or kind in seen_positions:
                raise ValueError("tag_positions must identify each supplied changed attribute once")
            seen_positions.add(kind)
            if "message_index" in supplied and (type(supplied["message_index"]) is not int or supplied["message_index"] != message_index):
                raise ValueError("tag_positions message_index is inconsistent")
            expected = next((p for p in after["_headers"] if p["kind"] == kind), None)
            if expected is None or any(supplied.get(key) != expected[key] for key in ("tag", "char_start", "char_end")):
                raise ValueError("tag_positions offsets must refer to the actual switch header in content_original")
            if type(supplied.get("char_start")) is not int or type(supplied.get("char_end")) is not int:
                raise ValueError("tag_positions offsets must be integer character positions")
        events.append({"index": index, "time_ms": time_ms, "char_offset": after["char_start"],
                       "changed": changed, "language": after["language"], "speaker": after["speaker"],
                       "previous_language": before["language"], "previous_speaker": before["speaker"],
                       "_headers": after["_headers"]})
    # Inline markers are not interpreted as spoken text or silently trusted.
    marker_events = set()
    for marker in markers:
        matches = [event for event in events if abs(event["time_ms"] - marker["time_ms"]) < 1e-6
                   and any(marker["position"] == header["char_start"]
                           for header in event["_headers"])]
        if len(matches) != 1 or matches[0]["index"] in marker_events:
            raise ValueError("Inline timestamp marker must match one switch time and its original header position")
        marker_events.add(matches[0]["index"])
    starts = [0.0] + [event["time_ms"] for event in events]
    ends = starts[1:] + [duration_ms]
    for span, start_ms, end_ms in zip(spans, starts, ends):
        span.update(start_ms=start_ms, end_ms=end_ms)
        span.pop("_headers")
    for event in events:
        event.pop("_headers")
    runs = []
    for span in spans:
        if runs and runs[-1]["language"] == span["language"]:
            runs[-1]["text"] += span["text"]
            runs[-1]["char_end"], runs[-1]["end_ms"] = span["char_end"], span["end_ms"]
        else:
            runs.append({key: span[key] for key in ("language", "text", "char_start", "char_end", "start_ms", "end_ms")})
            runs[-1]["index"] = len(runs) - 1
    return {"audio": audios[0], "text": "".join(span["text"] for span in spans),
            "message_index": message_index, "duration_ms": duration_ms,
            "spans": spans, "events": events, "language_runs": runs}


def plan_queries(source, duration_samples, sample_rate, chunk_seconds):
    """Plan unique cumulative language-run prefixes on the source sample grid.

    Chunk ends use an integer sample step, and timestamp boundaries are rounded
    to the nearest source sample using Python's round-to-even rule. Each query
    is sliced BEFORE resampling. Adjacent language runs share exactly one sample
    boundary: no sample from the next language is included in the previous run.
    Exact switch anchors are included even off the training grid; same-language
    speaker changes keep the language-run start rather than resetting context.
    """
    if type(duration_samples) is not int or duration_samples <= 0 or type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("duration_samples and sample_rate must be positive integers")
    chunk_seconds = _number(chunk_seconds, "chunk_seconds")
    step = round(chunk_seconds * sample_rate)
    if chunk_seconds <= 0 or step <= 0:
        raise ValueError("chunk_seconds must be positive and cover at least one sample")
    if abs(source["duration_ms"] - duration_samples * 1000.0 / sample_rate) > 0.5 * 1000 / sample_rate + 1e-6:
        raise ValueError("Parsed source duration differs from actual source sample count")
    boundaries = [0] + [round(e["time_ms"] * sample_rate / 1000) for e in source["events"]] + [duration_samples]
    if any(b <= a for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("Switches collapse or lie outside the source sample grid")
    ends = list(range(step, duration_samples, step)) + [duration_samples]
    result = []
    for run in source["language_runs"]:
        start = round(run["start_ms"] * sample_rate / 1000)
        end = round(run["end_ms"] * sample_rate / 1000)
        planned = {}

        def add(endpoint, kind):
            if start < endpoint <= end:
                planned.setdefault(endpoint, set()).add(kind)

        for endpoint in ends:
            if endpoint > start:
                add(min(endpoint, end), "prefix")
                if endpoint >= end:
                    break
        for boundary in boundaries[1:-1]:
            add(boundary, "switch_anchor")
        add(end, "full")
        for endpoint, kinds in sorted(planned.items()):
            kind = "full" if "full" in kinds else "switch_anchor" if "switch_anchor" in kinds else "prefix"
            result.append({"query_id": f"run{run['index']}:{start}:{endpoint}",
                           "run_index": run["index"], "start_sample": start, "end_sample": endpoint,
                           "end_ms": endpoint * 1000.0 / sample_rate, "kind": kind, "kinds": sorted(kinds)})
    return result


def plan_span_queries(source, duration_samples, sample_rate, chunk_seconds):
    """Plan auxiliary local prefixes after same-language speaker boundaries.

    Primary language-run queries retain their context. These independent extra
    hypotheses let later attributed spans recover when an earlier missed word
    makes the cumulative hypothesis disagree with the full reference. A span
    beginning at the language-run start already has equivalent primary queries
    and is not duplicated. All coordinates use the same source sample rounding.
    """
    # Keep the primary schedule and validation unchanged for legacy traces.
    plan_queries(source, duration_samples, sample_rate, chunk_seconds)
    step = round(chunk_seconds * sample_rate)
    global_ends = list(range(step, duration_samples, step)) + [duration_samples]
    result = []
    for span_index, span in enumerate(source["spans"]):
        runs = [run for run in source["language_runs"]
                if run["char_start"] <= span["char_start"] < span["char_end"] <= run["char_end"]
                and run["language"] == span["language"]]
        if len(runs) != 1:
            raise ValueError("Attributed span must belong to exactly one language run")
        run = runs[0]
        start = round(span["start_ms"] * sample_rate / 1000)
        end = round(span["end_ms"] * sample_rate / 1000)
        if start == round(run["start_ms"] * sample_rate / 1000):
            continue
        if not round(run["start_ms"] * sample_rate / 1000) < start < end <= round(run["end_ms"] * sample_rate / 1000):
            raise ValueError("Local speaker query is outside its language run")
        endpoints = {value: {"prefix"} for value in global_ends if start < value <= end}
        endpoints.setdefault(end, set()).add("full")
        for endpoint, kinds in sorted(endpoints.items()):
            result.append({"query_id": f"span{span_index}:{start}:{endpoint}",
                           "run_index": run["index"], "span_index": span_index,
                           "start_sample": start, "end_sample": endpoint,
                           "end_ms": endpoint * 1000.0 / sample_rate,
                           "kind": "full" if "full" in kinds else "prefix",
                           "kinds": sorted(kinds)})
    return result
