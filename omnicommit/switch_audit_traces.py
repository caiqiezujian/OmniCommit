"""Align CLI and Web file-session observations to exact SFT source intervals."""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from contextlib import closing
import json
import math
from pathlib import Path

from .io import source_rows


# Smaller than one 16 kHz sample. Ignore arithmetic noise, not a shifted source grid.
EPS = 1e-6


def _seconds(value, field):
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be finite nonnegative seconds")
    return value


def _records(path):
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8-sig") as stream:
            rows = json.load(stream)
        if isinstance(rows, dict):
            rows = rows.get("events")  # Web transcript export, including display corrections.
        if not isinstance(rows, list):
            raise ValueError("expected an array of events or a Web export containing events")
        yield from rows
    else:
        for number, row, error in source_rows(path):
            if error:
                raise ValueError(f"line {number}: {error}")
            yield row


class SwitchTrace:
    def __init__(self, path, source, header):
        self.path = str(path)
        self.source = source
        self.header = header
        self.terminal = {"status": "incomplete"}
        self._starts = []
        self._audio = []
        self._eos = None

    def _append(self, obs):
        start, end = obs["source_start_seconds"], obs["source_end_seconds"]
        if self._audio and start < self._audio[-1]["source_end_seconds"] - EPS:
            raise ValueError("overlapping or out-of-order trace intervals")
        if obs["event"] == "end_of_audio":
            if abs(end - start) > EPS:
                raise ValueError("EOS must have zero duration")
            self._eos = obs
        else:
            if end <= start:
                raise ValueError("audio interval must have positive duration")
            self._starts.append(start)
            self._audio.append(obs)

    def lookup(self, start_seconds, end_seconds, event="audio"):
        start = _seconds(start_seconds, "start_seconds")
        end = _seconds(end_seconds, "end_seconds")
        if event not in ("audio", "end_of_audio"):
            raise ValueError("unknown requested event")
        if event == "audio" and end <= start or event == "end_of_audio" and abs(end-start) > EPS:
            raise ValueError("invalid requested interval/event")
        if event == "end_of_audio":
            if self._eos is not None and abs(self._eos["source_start_seconds"] - start) <= EPS:
                return self._eos
            return {"status": "missing"}
        i = bisect_left(self._starts, start - EPS)
        if i < len(self._audio) and abs(self._starts[i] - start) <= EPS:
            if abs(self._audio[i]["source_end_seconds"] - end) <= EPS:
                return self._audio[i]
        # Last interval starting strictly before the requested end. A partial
        # overlap cannot tell us which words belong to the requested SFT turn.
        i = bisect_left(self._starts, end - EPS) - 1
        if i >= 0 and self._audio[i]["source_end_seconds"] > start + EPS:
            return {"status": "grid_mismatch"}
        return {"status": "missing"}


def _interval(row, offset, start_key, end_key):
    if "source_start_seconds" in row or "source_end_seconds" in row:
        return (_seconds(row.get("source_start_seconds"), "source_start_seconds"),
                _seconds(row.get("source_end_seconds"), "source_end_seconds"))
    return (_seconds(row.get(start_key), start_key) + offset, _seconds(row.get(end_key), end_key) + offset)


def _load(path, rows, source, offset_seconds):
    head = next(rows, None)
    if not isinstance(head, dict) or head.get("type") not in ("ready", "error"):
        raise ValueError("expected a single session starting with a ready or error record")
    header = head if head["type"] == "ready" else {}
    if header.get("source_kind", "file") != "file":
        raise ValueError("expected file input")
    if source is None:
        source = header.get("input")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("no source path in this trace; supply the SFT recording path explicitly")
    offset = _seconds(header.get("file_start_seconds", 0.0), "file_start_seconds")
    if offset_seconds is not None:
        supplied = _seconds(offset_seconds, "offset_seconds")
        if "file_start_seconds" in header and abs(supplied - offset) > EPS:
            raise ValueError("offset conflicts with the Web session's original file_start_seconds")
        offset = supplied
    trace = SwitchTrace(path, source, header)
    if head["type"] == "error":
        trace.terminal = {"status": "error", "error": head.get("error"), "stage": head.get("stage")}
    previous_turn = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("expected a trace event object")
        kind = row.get("type")
        if kind == "ready":
            raise ValueError("multiple sessions in one trace")
        if kind in ("file_wait", "file_done", "control_ack", "control_error", "local_correction"):
            continue
        if trace.terminal["status"] != "incomplete":
            raise ValueError("unexpected event after the end of the session")
        if kind == "file_end_empty":
            trace.terminal = {"status": "empty", "source_heard_seconds": row.get("source_heard_seconds", offset)}
        elif kind == "error":
            failure = {"status": "error", "error": row.get("error")}
            for key in ("stage", "error_type", "backend_called", "raw_model_text"):
                if key in row:
                    failure[key] = row[key]
            if "source_heard_seconds" in row:
                failure["source_heard_seconds"] = _seconds(row["source_heard_seconds"], "source_heard_seconds")
            elif "heard_seconds" in row:
                failure["source_heard_seconds"] = _seconds(row["heard_seconds"], "heard_seconds") + offset
            if "source_end_seconds" in row or "attempted_end_seconds" in row:
                start_key = "attempted_start_seconds" if "attempted_start_seconds" in row else "heard_seconds"
                start, end = _interval(row, offset, start_key, "attempted_end_seconds")
                failure.update(event="end_of_audio" if start == end else "audio", source_start_seconds=start,
                               source_end_seconds=end)
                trace._append(failure)
            trace.terminal = failure
        elif kind == "result":
            event, action = row.get("event"), row.get("action")
            if action not in ("DIRECT", "WAIT") or event not in ("audio", "end_of_audio"):
                raise ValueError("unsupported result action/event")
            turn = row.get("turn")
            if type(turn) is not int or turn <= previous_turn:
                raise ValueError("non-monotonic or invalid turn number")
            previous_turn = turn
            start, end = _interval(row, offset, "start_seconds", "end_seconds")
            raw = row.get("raw_model_text", row.get("text"))
            if not isinstance(raw, str):
                raise ValueError(f"turn {turn} is missing model output")
            segments = row.get("segments")
            if segments is not None:
                if not isinstance(segments, list) or (action == "WAIT") != (len(segments) == 0):
                    raise ValueError(f"turn {turn} has invalid segments for {action}")
                if any(not isinstance(s, dict) or any(not isinstance(s.get(k), str)
                       for k in ("language", "speaker", "text")) for s in segments):
                    raise ValueError(f"turn {turn} has invalid segment fields")
            obs = {"status": "result", "event": event, "action": action, "turn": turn,
                   "source_start_seconds": start, "source_end_seconds": end,
                   "raw_model_text": raw, "segments": segments}
            for key in ("generation_seconds", "service_seconds", "queue_wait_seconds"):
                if row.get(key) is not None:
                    obs[key] = _seconds(row[key], key)
            trace._append(obs)
            if event == "end_of_audio":
                trace.terminal = {**obs, "status": "eos"}
        else:
            raise ValueError(f"unsupported trace event type: {kind!r}")
    return trace


def load_trace(path, *, source=None, offset_seconds=None):
    """Index a CLI/Web trace; a Web export requires an explicit source identity.

    A rebased CLI clip can specify its original source and a clip offset. For
    new Web exports the explicit source times are already on the recording clock.
    Completed records in an unclosed JSONL remain available as observations.
    """
    path = Path(path)
    if offset_seconds and source is None:
        raise ValueError(f"{path}: an explicit offset requires the original SFT source path")
    try:
        with closing(_records(path)) as rows:
            return _load(path, rows, source, offset_seconds)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def summarize_switch(trace, turns, *, language, kind, anchor_seconds, window_end_seconds, next_onset_seconds=None):
    """Count exact live observations; the first-tag metric uses effective span attribution.

    Each result is emitted at the end of a chunk. Include the chunk containing
    the onset; a chunk ending at the onset contains no speech from the new span.
    Exclude submissions after the next known onset to avoid crediting a later span.
    """
    stop = _seconds(window_end_seconds, "window_end_seconds")
    if next_onset_seconds is not None:
        stop = min(stop, _seconds(next_onset_seconds, "next_onset_seconds"))
    if anchor_seconds is not None:
        _seconds(anchor_seconds, "anchor_seconds")
    count = Counter()
    languages = Counter()
    waits = 0
    wait_run = longest_wait = 0.0
    previous_wait_end = None
    unlabeled_directs = 0
    first_time = first_model_turn = first_gold_turn = None
    for turn in turns:
        end = turn["end_seconds"]
        if turn["event"] != "audio" or end > stop + EPS:
            continue
        if anchor_seconds is not None and end <= anchor_seconds + EPS:
            continue
        obs = trace.lookup(turn["start_seconds"], end)
        count[obs["status"]] += 1
        if obs["status"] != "result" or obs["action"] != "WAIT":
            previous_wait_end = None
        if obs["status"] != "result":
            continue
        if obs["action"] == "WAIT":
            waits += 1
            start = turn["start_seconds"]
            if previous_wait_end is None or abs(start - previous_wait_end) > EPS:
                wait_run = 0.0
            # The onset may sit inside the first WAIT chunk.
            wait_run += end - max(start, anchor_seconds if anchor_seconds is not None else start)
            longest_wait = max(wait_run, longest_wait)
            previous_wait_end = end
            continue
        if obs["segments"] is None:
            unlabeled_directs += 1
            continue
        emitted_languages = {s["language"] for s in obs["segments"] if s["text"].strip()}
        languages.update(sorted(emitted_languages))
        if kind in ("language", "both") and anchor_seconds is not None and first_time is None and language in emitted_languages:
            first_time, first_model_turn, first_gold_turn = obs["source_end_seconds"], obs["turn"], turn["turn"]
    if not count:
        status = "no_eligible_audio_turns"
    elif not count["result"]:
        status = "no_exact_results"
    elif count["result"] != sum(count.values()):
        status = "partial_coverage"
    else:
        status = "matched"
    return {"status": status, "terminal": trace.terminal,
            "scope": "review_window_no_reference_onset" if anchor_seconds is None else "from_reference_onset",
            "through_seconds": stop, "eligible_audio_turns": sum(count.values()),
            "exact_audio_turns": count["result"], "coverage_counts": dict(count),
            "observed_wait_turns": waits, "longest_observed_wait_seconds": longest_wait,
            "direct_turns_by_language": dict(languages), "direct_turns_without_segment_metadata": unlabeled_directs,
            "first_target_tag_seconds": first_time, "first_target_tag_turn": first_model_turn,
            "first_target_tag_gold_turn": first_gold_turn,
            "first_target_tag_delay_seconds": first_time-anchor_seconds if first_time is not None else None}
