"""Offline, meeting-global transcript/identity checks (not a DER estimator)."""
from __future__ import annotations

import math
import unicodedata

from .protocol import canonical_segments, parse_tagged_text


def lexical_units(segments):
    """Keep letters, numbers and combining marks; preserve their attribution."""
    result = []
    for segment in canonical_segments(segments):
        text = unicodedata.normalize("NFC", segment["text"]).casefold()
        result.extend((char, segment["language"], segment["speaker"])
                      for char in text if unicodedata.category(char)[0] in "LNM")
    return result


def source_segments(row):
    if "messages" in row:
        replies = [m["content"] for m in row["messages"] if m.get("role") == "assistant"]
        if len(replies) != 1:
            raise ValueError("Reference must contain one complete assistant transcript")
        return parse_tagged_text(replies[0])
    if "turns" in row:
        return canonical_segments([s for t in row["turns"] for s in t["segments"]])
    if row.get("kind") == "aligned_asr":
        return canonical_segments([s for unit in row["segments"] for s in unit["attributed"]])
    return canonical_segments(row["segments"])


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    lo, hi = math.floor(rank), math.ceil(rank)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def evaluate_events(reference, events, *, opcodes=None):
    """Match text once for the whole meeting, never permute speaker IDs.

    Identity accuracy is conditional on correctly recognized characters; it
    must be read alongside CER and matched coverage, not in isolation. Numeric
    normalization is NOT applied: changing 100 to one hundred remains an error.
    """
    if opcodes is None:
        try:
            from rapidfuzz.distance import Levenshtein
        except ImportError as exc:
            raise RuntimeError("Install requirements-eval.txt in the CPU tools environment") from exc
        opcodes = Levenshtein.opcodes
    ref = lexical_units(source_segments(reference))
    if not ref:
        raise ValueError("Reference has no lexical characters")
    emitted, latencies, rtf_values = [], [], []
    waits = direct = heuristic_evictions = 0
    last_end, ended, runtime_error = 0.0, False, None
    for index, event in enumerate(events):
        if ended or runtime_error is not None:
            raise ValueError("Prediction contains events after termination")
        kind = event.get("event")
        if kind == "error":
            runtime_error = str(event.get("error", "Unknown runtime error"))
            continue
        if kind not in ("audio", "end_of_audio"):
            raise ValueError(f"Prediction event {index}: unsupported event")
        start, end = event.get("start_seconds"), event.get("end_seconds")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in (start, end)):
            raise ValueError("Prediction timestamps must be finite seconds")
        if abs(start - last_end) > 1e-5 or end < start:
            raise ValueError("Prediction audio timeline must be contiguous")
        if kind == "audio" and end <= start:
            raise ValueError("Audio event must have positive duration")
        if kind == "end_of_audio":
            if abs(end - start) > 1e-5:
                raise ValueError("EOS must not invent audio")
            ended = True
        action, segments = event.get("action"), event.get("segments")
        if action == "WAIT":
            if segments != []:
                raise ValueError("WAIT contains transcript")
            waits += 1
        elif action == "DIRECT":
            parsed = canonical_segments(segments)
            if not parsed:
                raise ValueError("DIRECT is empty")
            emitted.extend(parsed)
            direct += 1
        else:
            raise ValueError("Unknown action in prediction")
        elapsed = event.get("generation_seconds")
        if elapsed is not None:
            if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("Invalid generation_seconds")
            latencies.append(elapsed)
            if kind == "audio":
                rtf_values.append(elapsed / (end - start))
        heuristic_evictions += bool(event.get("evicted_this_turn", False))
        last_end = end
    if not events:
        raise ValueError("Empty inference result")
    hyp = lexical_units(emitted)
    ref_text, hyp_text = "".join(x[0] for x in ref), "".join(x[0] for x in hyp)
    matched = speaker_ok = language_ok = 0
    edits = substitutions = deletions = insertions = 0
    per_speaker = {}
    matches = {}
    for tag, i, j, a, b in opcodes(ref_text, hyp_text):
        if tag == "equal":
            for r, h in zip(range(i, j), range(a, b)):
                matches[r] = h
                matched += 1
                language_ok += ref[r][1] == hyp[h][1]
                speaker_ok += ref[r][2] == hyp[h][2]
                bucket = per_speaker.setdefault(ref[r][2], {"matched_characters": 0, "correct_speaker": 0})
                bucket["matched_characters"] += 1
                bucket["correct_speaker"] += ref[r][2] == hyp[h][2]
        else:
            common = min(j - i, b - a)
            substitutions += common
            deletions += j - i - common
            insertions += b - a - common
    edits = substitutions + deletions + insertions
    probes = []
    # Probe offsets refer to normalized lexical characters, not UTF-8 bytes.
    for probe in reference.get("speaker_return_probes", []):
        begin, end = probe["start_char"], probe["end_char"]
        if type(begin) is not int or type(end) is not int or not 0 <= begin < end <= len(ref):
            raise ValueError("Invalid speaker_return_probes lexical character range")
        indices = [r for r in range(begin, end) if r in matches]
        correct = sum(ref[r][2] == hyp[matches[r]][2] for r in indices)
        probes.append({"name": probe.get("name", "return"), "reference_characters": end - begin,
                       "matched_characters": len(indices), "correct_speaker": correct,
                       "speaker_accuracy_on_matches": correct / len(indices) if indices else None,
                       "matched_coverage": len(indices) / (end - begin)})
    observed = {x[2] for x in hyp}
    expected = {x[2] for x in ref}
    return {
        "id": reference.get("id", reference.get("key")),
        "complete_eos": ended,
        "runtime_error": runtime_error,
        "audio_seconds": last_end,
        "character_error_rate": edits / len(ref),
        "normalization": "NFC + casefold + letters/numbers/marks; whitespace/punctuation excluded",
        "substitutions": substitutions, "deletions": deletions, "insertions": insertions,
        "reference_characters": len(ref), "hypothesis_characters": len(hyp),
        "matched_character_coverage": matched / len(ref),
        "speaker_accuracy_on_text_matches": speaker_ok / matched if matched else None,
        "language_accuracy_on_text_matches": language_ok / matched if matched else None,
        "speaker_ids_remapped": False, "not_a_diarization_error_rate": True,
        "unexpected_speaker_ids": sorted(observed - expected),
        "per_speaker": per_speaker, "speaker_return_probes": probes,
        "direct_turns": direct, "wait_turns": waits,
        "wait_fraction": waits / (waits + direct) if waits + direct else None,
        "generation_seconds_total": sum(latencies) if latencies else None,
        "generation_rtf": sum(latencies) / last_end if latencies and last_end else None,
        "generation_p50_seconds": percentile(latencies, .5),
        "generation_p95_seconds": percentile(latencies, .95),
        "fraction_audio_calls_slower_than_step": sum(v > 1 for v in rtf_values) / len(rtf_values) if rtf_values else None,
        "heuristic_eviction_events": heuristic_evictions,
        "latency_scope": "model generation including backend preparation if reported; excludes network/queue unless backend includes them",
    }


def transcript_events(rows):
    """Accept CLI results or a Web download including transport acknowledgements."""
    result = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Event rows must be objects")
        if "event" in row:
            result.append(row)
        elif row.get("type") in ("ready", "control_ack", "control_error"):
            continue
        elif row.get("type") == "error":
            result.append(dict(row, event="error"))
        else:
            raise ValueError("Unknown event row; do not silently skip malformed model results")
    return result
