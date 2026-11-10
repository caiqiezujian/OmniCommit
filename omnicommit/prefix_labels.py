"""Reference-checked local-language prefix supervision, with switch gates.

This is an OmniCommit adaptation, NOT token-level R2T2 reproduction. A whole
normalized hypothesis must be a prefix of the reference. In strict mode the
full-run teacher must verify that reference; completed-spans mode can supervise
closed, switched spans from operator-trusted reference text. Speaker/language
supervision comes from the reference, never guessed teacher IDs.
"""
from __future__ import annotations

import re
import unicodedata

from .protocol import LANGUAGE, SPEAKER
from .prefix_source import parse_source, plan_queries, plan_span_queries

ALGORITHM = "local-language-reference-completed-span-v2"


def teacher_body(text, allow_empty_labels=False):
    if not isinstance(text, str):
        raise ValueError("Teacher output is not text")
    def remove(match):
        tag = match.group(1)
        if LANGUAGE.fullmatch(tag) or SPEAKER.fullmatch(tag):
            return ""
        raise ValueError("Unexpected teacher markup: " + match.group(0))
    body = re.sub(r"<([^<>]+)>", remove, text)
    if "<" in body or ">" in body or re.search(r"\[\d+(?:\.\d+)?ms\]", body):
        raise ValueError("Unexpected teacher markup/timestamp")
    if text.strip() and not body.strip() and not allow_empty_labels:
        raise ValueError("Teacher returned tags only, without transcription")
    return body


def comparison_units(text, mode="surface"):
    """Return comparison string and original character-end mapping.

    surface collapses whitespace and ignores ordinary punctuation, preserving case,
    diacritics, symbols, apostrophes/hyphens, and punctuation adjacent to digits.
    In particular 1.5 != 15, -5 != 5, and don't != dont. Unicode decomposition
    permits canonically equivalent spelling without changing emitted reference.
    exact compares literal characters, including spaces and punctuation.
    """
    if mode not in ("surface", "exact"):
        raise ValueError("comparison must be surface or exact")
    units, ends = [], []
    for i, char in enumerate(text):
        if mode == "surface":
            adjacent_digit = ((i > 0 and text[i - 1].isdigit()) or
                              (i + 1 < len(text) and text[i + 1].isdigit()))
            punctuation = unicodedata.category(char).startswith("P")
            if char.isspace():
                if units and units[-1] != " ":
                    units.append(" ")
                    ends.append(i + 1)
                continue
            if punctuation and not adjacent_digit and char not in "'’ʼ-‐‑":
                continue
            values = unicodedata.normalize("NFD", char)
        else:
            values = char
        units.extend(values)
        ends.extend([i + 1] * len(values))
    if mode == "surface" and units and units[-1] == " ":
        units.pop()
        ends.pop()
    return "".join(units), ends


def _slice_segments(source, start, end):
    segments = []
    for span in source["spans"]:
        a, b = max(start, span["char_start"]), min(end, span["char_end"])
        if a < b:
            value = source["text"][a:b]
            if not value.strip():
                raise ValueError("A release would create whitespace-only attribution")
            segments.append({"language": span["language"], "speaker": span["speaker"], "text": value})
    return segments


def _reference_end(source, run, count, mode):
    normalized, ends = comparison_units(run["text"], mode)
    if not count:
        return run["char_start"]
    if count == len(normalized):
        return run["char_end"]
    # Include ignored punctuation/space in the same attribution only. Never
    # create a whitespace-only prefix belonging to the next speaker.
    end = run["char_start"] + ends[count - 1]
    span_end = next(s["char_end"] for s in source["spans"]
                    if s["char_start"] < end <= s["char_end"])
    while end < span_end and not comparison_units(source["text"][end], mode)[0]:
        # Numeric-adjacent punctuation must use the original context.
        before = comparison_units(source["text"][run["char_start"]:end + 1], mode)[0]
        if len(before) != count:
            break
        end += 1
    return end


def _validate_plan(rows, planned, name):
    if (not isinstance(rows, list) or any(not isinstance(q, dict) for q in rows)
            or len(rows) != len(planned) or any(
                any(actual.get(key) != value for key, value in expected.items())
                for actual, expected in zip(rows, planned))):
        raise ValueError(name + " plan is incomplete or modified; refusing inferred WAIT labels")
    if any(q.get("error") for q in rows):
        raise ValueError("Teacher query failed; not a valid WAIT or reference-fallback example")


def _observations(source, scope, rows, comparison, allow_empty_labels=False):
    reference, ends = comparison_units(scope["text"], comparison)
    if not reference:
        raise ValueError("Reference span has no comparison text")
    full = comparison_units(teacher_body(rows[-1]["text"], allow_empty_labels), comparison)[0]
    stable, observations = 0, []
    for row in rows:
        hypothesis = comparison_units(teacher_body(row["text"], allow_empty_labels), comparison)[0]
        complete_character = (not hypothesis or len(hypothesis) == len(reference) or
                              (len(hypothesis) < len(reference) and
                               ends[len(hypothesis) - 1] != ends[len(hypothesis)] and
                               not unicodedata.combining(reference[len(hypothesis)])))
        admissible = reference.startswith(hypothesis) and complete_character
        if admissible:
            stable = max(stable, len(hypothesis))
        observations.append({"end_sample": row["end_sample"], "stable_units": stable,
                             "reference_end": _reference_end(source, scope, stable, comparison),
                             "whole_hypothesis_matches": admissible,
                             "hypothesis_equals_lsp": admissible and len(hypothesis) == stable})
    intervals, active = [], None
    for observation in observations:
        if not observation["hypothesis_equals_lsp"]:
            active = None
        elif active is not None and active["stable_units"] == observation["stable_units"]:
            active["end_sample"] = observation["end_sample"]
        else:
            active = {"start_sample": observation["end_sample"], "end_sample": observation["end_sample"],
                      "stable_units": observation["stable_units"]}
            intervals.append(active)
    return full == reference, observations, intervals


def _validate_source(source, duration_ms):
    """Recheck trace geometry before trusted reference text can be released."""
    text = "".join(f"<{s['language']}><{s['speaker']}>{s['text']}" for s in source["spans"])
    rebuilt = parse_source({"audios": [source["audio"]],
                            "messages": [{"role": "assistant", "content": text}],
                            "timestamps": source["events"]}, duration_ms)
    for field in ("text", "spans", "events", "language_runs"):
        if rebuilt[field] != source[field]:
            raise ValueError("Trace source geometry disagrees with its reference: " + field)


def build_trajectory(trace, comparison="surface", reference_fallback="completed-spans"):
    """Build ordered deltas; fallback is explicitly trusted reference supervision.

    A completed-span fallback requires the whole attributed span's audio to have
    arrived AND its onset chunk gate to have expired. New events never re-freeze
    old pending spans. Technical failures and unknown markup remain failures;
    completed known attribution-only output counts as semantic empty only in
    fallback mode. Single-attribution data retain strict teacher verification.
    """
    if reference_fallback not in ("completed-spans", "off"):
        raise ValueError("reference_fallback must be completed-spans or off")
    if trace.get("kind") != "local_prefix_trace" or trace.get("status") != "ok":
        raise ValueError("Expected a successful local_prefix_trace")
    source, info = trace["source"], trace["audio_info"]
    rate, frames = info["sample_rate"], info["frames"]
    if type(rate) is not int or type(frames) is not int or rate <= 0 or frames <= 0:
        raise ValueError("Invalid source sample geometry")
    duration = frames * 1000 / rate
    _validate_source(source, duration)
    expected = plan_queries(source, frames, rate, trace["chunk_seconds"])
    queries = trace["queries"]
    _validate_plan(queries, expected, "Query")
    step = round(trace["chunk_seconds"] * rate)
    endpoints = list(range(step, frames, step)) + [frames]
    fallback_enabled = reference_fallback == "completed-spans" and bool(source["events"])
    local_expected = plan_span_queries(source, frames, rate, trace["chunk_seconds"])
    has_local_queries = "local_queries" in trace
    teacher_metadata = trace.get("teacher", {})
    if not isinstance(teacher_metadata, dict):
        raise ValueError("Teacher metadata must be an object")
    local_policy = teacher_metadata.get("local_query_policy")
    if local_policy not in (None, "later-speakers-in-language-run-v1"):
        raise ValueError("Unsupported local query policy")
    if local_policy is not None and not has_local_queries:
        raise ValueError("Declared local query policy requires local_queries; not a legacy trace")
    local_queries = trace.get("local_queries", [])
    if has_local_queries:
        _validate_plan(local_queries, local_expected, "Local span")

    diagnostics, run_observations, full_mismatches = [], {}, 0
    for run in source["language_runs"]:
        rows = [q for q in queries if q["run_index"] == run["index"]]
        full_match, observations, intervals = _observations(source, run, rows, comparison, fallback_enabled)
        if not full_match:
            full_mismatches += 1
            if not fallback_enabled:
                raise ValueError(f"Run {run['index']}: full teacher/reference mismatch; no reference filling")
        run_observations[run["index"]] = observations
        diagnostics.append({"run_index": run["index"], "full_reference_match": full_match,
                            "observations": observations, "stable_intervals": intervals})
    local_observations, local_diagnostics = {}, []
    for index, span in enumerate(source["spans"]):
        rows = [q for q in local_queries if q["span_index"] == index]
        if rows:
            full_match, observations, intervals = _observations(source, span, rows, comparison, fallback_enabled)
            local_observations[index] = observations
            local_diagnostics.append({"span_index": index, "full_reference_match": full_match,
                                      "observations": observations, "stable_intervals": intervals})
    span_runs = [next(run["index"] for run in source["language_runs"]
                      if run["char_start"] <= span["char_start"] < run["char_end"])
                 for span in source["spans"]]
    events = [{**e, "sample": round(e["time_ms"] * rate / 1000)} for e in source["events"]]
    fallback_events, fallback_spans, language_fallback = [], set(), {}

    def latest(observations, endpoint, default):
        heard = [o for o in observations if o["end_sample"] <= endpoint]
        return heard[-1]["reference_end"] if heard else default

    def readiness(endpoint):
        teacher_limits, ready_limits = [], []
        for index, span in enumerate(source["spans"]):
            start, end = span["char_start"], span["char_end"]
            primary = latest(run_observations[span_runs[index]], endpoint, start)
            local = (latest(local_observations.get(index, []), endpoint, start)
                     if fallback_enabled else start)
            teacher = min(end, max(start, primary, local))
            closed = round(span["end_ms"] * rate / 1000) <= endpoint
            teacher_limits.append(teacher)
            ready_limits.append(end if fallback_enabled and closed else teacher)
        candidate = 0
        for span, limit in zip(source["spans"], ready_limits):
            candidate = limit
            if limit != span["char_end"]:
                break  # Never skip an unresolved span to release a later one.
        return candidate, teacher_limits

    def track_fallback(start, release, teacher_limits, endpoint, turn_index, event):
        added = []
        for index, (span, teacher_end) in enumerate(zip(source["spans"], teacher_limits)):
            a, b = max(start, span["char_start"], teacher_end), min(release, span["char_end"])
            if a >= b:
                continue
            if not fallback_enabled or round(span["end_ms"] * rate / 1000) > endpoint:
                raise ValueError("Reference fallback attempted before complete span audio")
            item = {"source": "reference_completed_span", "span_index": index,
                    "language": span["language"], "speaker": span["speaker"],
                    "char_start": a, "char_end": b, "text": source["text"][a:b],
                    "span_start_ms": span["start_ms"], "span_end_ms": span["end_ms"],
                    "eligible_audio_end_sample": round(span["end_ms"] * rate / 1000),
                    "emitted_at_ms": endpoint * 1000 / rate, "turn_index": turn_index,
                    "event": event, "reason": "teacher_prefix_did_not_confirm_emitted_suffix"}
            fallback_events.append(item)
            added.append(item)
            fallback_spans.add(index)
            language_fallback[span["language"]] = language_fallback.get(span["language"], 0) + b - a
        return added

    turns, committed, previous_sample = [], 0, 0
    for endpoint in endpoints:
        candidate, teacher_limits = readiness(endpoint)
        # [start,end): the onset chunk blocks new spans, including future ones;
        # it never blocks older spans merely because another change occurs now.
        blocked = [e for e in events if e["sample"] >= previous_sample]
        cap = min((e["char_offset"] for e in blocked), default=len(source["text"]))
        release = min(candidate, cap)
        if release < committed:
            raise ValueError("Non-monotone commit position; refusing hidden rollback")
        segments = _slice_segments(source, committed, release) if release > committed else []
        crossing = [e for e in events if previous_sample <= e["sample"] < endpoint]
        added = track_fallback(committed, release, teacher_limits, endpoint, len(turns), "audio")
        turns.append({"event": "audio", "start_ms": previous_sample * 1000 / rate,
                      "end_ms": endpoint * 1000 / rate,
                      "action": "DIRECT" if segments else "WAIT", "segments": segments,
                      "provenance": {"candidate_characters": candidate, "committed_characters": release,
                                     "switch_gate": bool(crossing), "blocked_new_text": release < candidate,
                                     "switch_times_ms": [e["time_ms"] for e in crossing],
                                     "reference_fallback_characters": sum(e["char_end"] - e["char_start"] for e in added)}})
        committed, previous_sample = release, endpoint

    # Explicit real EOS removes the last onset gate. In fallback mode all audio
    # has now arrived; reference-supervised tails stay distinct from teacher LSP.
    final_candidate, teacher_limits = readiness(frames)
    if final_candidate != len(source["text"]):
        raise ValueError("Incomplete final teacher evidence")
    remaining = _slice_segments(source, committed, final_candidate) if committed < final_candidate else []
    added = track_fallback(committed, final_candidate, teacher_limits, frames, len(turns), "end_of_audio")
    turns.append({"event": "end_of_audio", "start_ms": duration, "end_ms": duration,
                  "action": "DIRECT" if remaining else "WAIT", "segments": remaining,
                  "provenance": {"reference_fallback_characters": sum(e["char_end"] - e["char_start"] for e in added)}})
    if "".join(s["text"] for turn in turns for s in turn["segments"]) != source["text"]:
        raise ValueError("Reference reconstruction failed")
    warnings = (["legacy_trace_without_local_speaker_queries"] if local_expected and not has_local_queries else [])
    return {"schema_version": 2, "id": trace["id"], "status": "ok", "audio": trace["audio"],
            "duration_ms": duration, "audio_info": info, "turns": turns,
            "policy": {"number_format": trace["number_format"], "chunk_seconds": trace["chunk_seconds"],
                       "algorithm": ALGORITHM, "comparison": comparison,
                       "switch_policy": "old-confirmed-tail-allowed-new-span-one-chunk-gate",
                       "reference_fallback": reference_fallback, "reference_fallback_enabled": fallback_enabled,
                       "reference_trust": "operator-supplied-reference" if fallback_enabled else "teacher-checked",
                       "labels_from": "reference", "protocol": "direct-wait-v1"},
            "provenance": {"trace_input_hash": trace["input_hash"], "run_diagnostics": diagnostics,
                           "local_span_diagnostics": local_diagnostics, "has_local_queries": has_local_queries,
                           "reference_fallback_events": fallback_events, "warnings": warnings,
                           "scope": "Reference-supervised normalized-character adaptation, not exact R2T2 or acoustic timing truth."},
            "metrics": {"audio_turns": len(turns) - 1,
                        "wait_turns": sum(t["action"] == "WAIT" for t in turns[:-1]),
                        "eos_characters": sum(len(s["text"]) for s in remaining),
                        "total_characters": len(source["text"]),
                        "languages": sorted({s["language"] for s in source["spans"]}),
                        "reference_fallback_spans": len(fallback_spans),
                        "reference_fallback_characters": sum(language_fallback.values()),
                        "teacher_full_mismatch_runs": full_mismatches,
                        "language_fallback_characters": language_fallback}}
