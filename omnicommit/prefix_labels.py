"""Reference-checked local-language prefix supervision, with switch gates.

This is an OmniCommit adaptation, NOT token-level R2T2 reproduction. A whole
normalized hypothesis must be a prefix of the verified full-run hypothesis.
Speaker/language supervision comes from the reference, never guessed teacher IDs.
"""
from __future__ import annotations

import re
import unicodedata

from .protocol import LANGUAGE, SPEAKER
from .prefix_source import plan_queries

ALGORITHM = "local-language-whole-hypothesis-reference-checked-v1"


def teacher_body(text):
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
    if text.strip() and not body.strip():
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


def build_trajectory(trace, comparison="surface"):
    if trace.get("kind") != "local_prefix_trace" or trace.get("status") != "ok":
        raise ValueError("Expected a successful local_prefix_trace")
    source = trace["source"]
    info = trace["audio_info"]
    rate, frames = info["sample_rate"], info["frames"]
    step = round(trace["chunk_seconds"] * rate)
    if step < 1:
        raise ValueError("Invalid chunk step")
    endpoints = list(range(step, frames, step)) + [frames]
    queries = trace["queries"]
    if not queries or any(q.get("error") for q in queries):
        raise ValueError("Teacher query failed; not a valid WAIT example")
    expected = plan_queries(source, frames, rate, trace["chunk_seconds"])
    if len(queries) != len(expected) or any(
            any(actual.get(key) != value for key, value in planned.items())
            for actual, planned in zip(queries, expected)):
        raise ValueError("Query plan is incomplete or modified; refusing inferred WAIT labels")
    diagnostics, runs = [], []
    for run in source["language_runs"]:
        rows = sorted((q for q in queries if q["run_index"] == run["index"]),
                      key=lambda q: q["end_sample"])
        if not rows or len({q["end_sample"] for q in rows}) != len(rows):
            raise ValueError("Missing or duplicate run queries")
        end_sample = round(run["end_ms"] * rate / 1000)
        start_sample = round(run["start_ms"] * rate / 1000)
        if any(q["start_sample"] != start_sample or not start_sample < q["end_sample"] <= end_sample
               for q in rows):
            raise ValueError("Teacher query outside its language run")
        if rows[-1]["end_sample"] != end_sample:
            raise ValueError("Missing complete-language-run teacher anchor")
        full = comparison_units(teacher_body(rows[-1]["text"]), comparison)[0]
        reference, reference_ends = comparison_units(run["text"], comparison)
        if not reference or full != reference:
            raise ValueError(f"Run {run['index']}: full teacher/reference mismatch; no reference filling")
        stable, observations = 0, []
        for q in rows:
            hypothesis = comparison_units(teacher_body(q["text"]), comparison)[0]
            complete_character = (not hypothesis or len(hypothesis) == len(reference) or
                                  (len(hypothesis) < len(reference) and
                                   reference_ends[len(hypothesis) - 1] != reference_ends[len(hypothesis)] and
                                   not unicodedata.combining(reference[len(hypothesis)])))
            admissible = full.startswith(hypothesis) and complete_character
            if admissible:
                stable = max(stable, len(hypothesis))
            observations.append({"end_sample": q["end_sample"], "stable_units": stable,
                                 "reference_end": _reference_end(source, run, stable, comparison),
                                 "whole_hypothesis_matches": admissible,
                                 "hypothesis_equals_lsp": admissible and len(hypothesis) == stable})
        # Retain stable runs for diagnosis. These are sampled, normalized-text
        # intervals, not the paper's 80ms token-level intervals or WAIT labels.
        intervals = []
        active = None
        for o in observations:
            if not o["hypothesis_equals_lsp"]:
                active = None
            elif active is not None and active["stable_units"] == o["stable_units"]:
                active["end_sample"] = o["end_sample"]
            else:
                active = {"start_sample": o["end_sample"], "end_sample": o["end_sample"],
                          "stable_units": o["stable_units"]}
                intervals.append(active)
        runs.append((run, observations))
        diagnostics.append({"run_index": run["index"], "full_reference_match": True,
                            "observations": observations, "stable_intervals": intervals})

    events = [{**e, "sample": round(e["time_ms"] * rate / 1000)} for e in source["events"]]
    turns, committed, previous_sample = [], 0, 0
    for endpoint in endpoints:
        candidate = 0
        for run, observations in runs:
            heard = [o for o in observations if o["end_sample"] <= endpoint]
            if not heard:
                break
            current = heard[-1]["reference_end"]
            candidate = current
            if current != run["char_end"]:
                break  # Never jump over uncommitted text from an earlier run.
        # [start,end): a switch exactly at endpoint belongs to the next chunk.
        # Include future boundaries too, so teacher anticipation cannot leak a
        # future language/speaker label. The gate expires after its own chunk.
        blocked = [e for e in events if e["sample"] >= previous_sample]
        cap = min((e["char_offset"] for e in blocked), default=len(source["text"]))
        release = min(candidate, cap)
        if release < committed:
            raise ValueError("Non-monotone commit position; refusing hidden rollback")
        segments = _slice_segments(source, committed, release) if release > committed else []
        crossing = [e for e in events if previous_sample <= e["sample"] < endpoint]
        turns.append({"event": "audio", "start_ms": previous_sample * 1000 / rate,
                      "end_ms": endpoint * 1000 / rate,
                      "action": "DIRECT" if segments else "WAIT", "segments": segments,
                      "provenance": {"candidate_characters": candidate, "committed_characters": release,
                                     "switch_gate": bool(crossing),
                                     "blocked_new_text": release < candidate,
                                     "switch_times_ms": [e["time_ms"] for e in crossing]}})
        committed, previous_sample = release, endpoint
    # A verified full-run anchor covers each run. End-of-audio only removes the
    # final chunk's switch gate; it never repairs a failed/partial transcription.
    if any(obs[-1]["reference_end"] != run["char_end"] for run, obs in runs):
        raise ValueError("Incomplete final teacher evidence")
    remaining = _slice_segments(source, committed, len(source["text"])) if committed < len(source["text"]) else []
    duration = frames * 1000 / rate
    turns.append({"event": "end_of_audio", "start_ms": duration, "end_ms": duration,
                  "action": "DIRECT" if remaining else "WAIT", "segments": remaining})
    reconstructed = "".join(s["text"] for turn in turns for s in turn["segments"])
    if reconstructed != source["text"]:
        raise ValueError("Reference reconstruction failed")
    audio_turns = turns[:-1]
    return {"schema_version": 2, "id": trace["id"], "status": "ok", "audio": trace["audio"],
            "duration_ms": duration, "audio_info": info, "turns": turns,
            "policy": {"number_format": trace["number_format"], "chunk_seconds": trace["chunk_seconds"],
                       "algorithm": ALGORITHM, "comparison": comparison,
                       "switch_policy": "old-confirmed-tail-allowed-new-span-one-chunk-gate",
                       "labels_from": "reference", "protocol": "direct-wait-v1"},
            "provenance": {"trace_input_hash": trace["input_hash"], "run_diagnostics": diagnostics,
                           "scope": "Reference-checked normalized-character adaptation, not exact R2T2 or acoustic timing truth."},
            "metrics": {"audio_turns": len(audio_turns),
                        "wait_turns": sum(t["action"] == "WAIT" for t in audio_turns),
                        "eos_characters": sum(len(s["text"]) for s in remaining),
                        "total_characters": len(source["text"]),
                        "languages": sorted({s["language"] for s in source["spans"]})}}
