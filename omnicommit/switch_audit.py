"""Read real SFT targets and independently join attributed reference text to their source clock."""
from __future__ import annotations

from bisect import bisect_right
import math
from pathlib import Path
import re
import sys

from Second.second_asr import protocol as second_protocol
from Second.second_asr.source import parse_record
from .prefix_source import parse_source
from .runtime_protocol import END_OF_AUDIO, LIVE_AUDIO_PREFIX, SPEAKER_REFERENCE_INTRO
from .streaming import DIRECT, WAIT, parse_reply
from .switch_audit_sidecars import pathkey

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "SFT-Scripts"))
from omnicommit_audio import decode_chunk_ref


def _messages(row):
    items = row.get("messages") if isinstance(row, dict) else None
    if (not isinstance(items, list) or len(items) < 3 or len(items) % 2 != 1
            or not all(isinstance(m, dict) and isinstance(m.get("content"), str) for m in items)
            or items[0].get("role") != "system"):
        raise ValueError("Expected one system message followed by user/assistant pairs")
    pairs = list(zip(items[1::2], items[2::2]))
    if any(u.get("role") != "user" or a.get("role") != "assistant" for u, a in pairs):
        raise ValueError("Expected alternating user and assistant messages")
    return items[0]["content"], [(u["content"], a["content"]) for u, a in pairs]


def parse_sft_row(row, protocol="auto"):
    """Decode virtual references only. Speaker prompts, input padding and EOS add no live audio."""
    system, pairs = _messages(row)
    advertised = None
    if system.startswith("Transcribe new speech. Mark first text and changes"):
        advertised = "second"
    elif DIRECT in system:
        advertised = "first"
    if protocol == "auto":
        answers = [raw for _, raw in pairs if raw != WAIT]
        if advertised:
            protocol = advertised
        elif answers:
            protocol = "first" if answers[0].startswith(DIRECT) else "second"
        else:
            raise ValueError("Ambiguous all-WAIT protocol; choose first or second")
    if protocol not in ("first", "second"):
        raise ValueError("Unknown SFT protocol")
    if advertised and advertised != protocol:
        raise ValueError("Requested SFT protocol conflicts with the training system message")
    initial = re.search(r"(?:Last committed live attribution: |the last committed voice was )"
                        r"<([a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*)><(spk[1-9][0-9]*)>\.", system)
    initial_state = initial.groups() if initial else None
    state = initial_state
    refs = row.get("audios")
    if not isinstance(refs, list) or not refs:
        raise ValueError("Missing SFT virtual audio chunks")
    refs = [decode_chunk_ref(value) for value in refs]
    turns = []
    ref_index = 0
    audio_path = None
    previous_end = None
    for turn, (user, raw) in enumerate(pairs, 1):
        if user == END_OF_AUDIO:
            if turn != len(pairs) or previous_end is None or (protocol == "second" and raw != WAIT):
                raise ValueError("Invalid end_of_audio turn")
            event = "end_of_audio"
            start = end = previous_end
        else:
            if user != "<audio>":
                if (turn != 1 or not user.startswith(SPEAKER_REFERENCE_INTRO)
                        or not user.endswith(LIVE_AUDIO_PREFIX + "<audio>")
                        or user.count(SPEAKER_REFERENCE_INTRO) != user.count("<audio>") - 1):
                    raise ValueError("Unknown live/reference audio layout")
                ref_index += user.count("<audio>") - 1
            if ref_index >= len(refs):
                raise ValueError("Too few virtual audio references")
            ref = refs[ref_index]
            ref_index += 1
            event = "audio"
            start, end = ref["start"], ref["end"]
            if audio_path is None:
                audio_path = ref["path"]
                if any(pathkey(r["path"]) != pathkey(audio_path) or r["end"] > start + 1e-7
                       for r in refs[:ref_index - 1]):
                    raise ValueError("Speaker reference must precede the live window in the same audio")
            if (pathkey(ref["path"]) != pathkey(audio_path)
                    or previous_end is not None and abs(start - previous_end) > 1e-7):
                raise ValueError("Live audio chunks need one file and a contiguous source clock")
            previous_end = end
        if protocol == "first":
            if raw != WAIT and not raw.startswith(DIRECT):
                raise ValueError("Expected a first-protocol SFT target")
            segments = parse_reply(raw)["segments"]
            if row.get("schema_version") == 1:
                # Legacy multi-span first-protocol targets use a newline separator.
                for segment in segments[:-1]:
                    segment["text"] = segment["text"].removesuffix("\n")
        else:
            parsed, state = second_protocol.parse_reply(raw, state)
            segments = [dict(language=lang, speaker=spk, text=text) for lang, spk, text in parsed]
        turns.append(dict(turn=turn, event=event, start_seconds=start, end_seconds=end,
                          action="DIRECT" if segments else "WAIT", raw=raw, segments=segments))
    if ref_index != len(refs):
        raise ValueError("Unused SFT audio references")
    window = row.get("window", {})
    if not isinstance(window, dict):
        raise ValueError("Invalid SFT window")
    if window:
        bounds = [window.get("start_ms"), window.get("end_ms")]
        if (any(type(x) not in (int, float) or not math.isfinite(x) for x in bounds)
                or type(window.get("ends_meeting")) is not bool
                or abs(bounds[0] - turns[0]["start_seconds"] * 1000) > 1e-4
                or abs(bounds[1] - previous_end * 1000) > 1e-4
                or window["ends_meeting"] != (turns[-1]["event"] == "end_of_audio")):
            raise ValueError("Window metadata disagrees with virtual audio or EOS")
    complete = (turns[0]["start_seconds"] == 0 and
                (turns[-1]["event"] == "end_of_audio" or row.get("schema_version") == 1)
                and window.get("ends_meeting", True))
    return dict(audio=audio_path, protocol=protocol, start_seconds=turns[0]["start_seconds"],
                end_seconds=previous_end, turns=turns, initial_state=initial_state,
                complete=complete, duration_ms=row.get("duration_ms"))


def _state(span):
    return span["language"], span["speaker"]


def _segments(turns):
    offset = 0
    for t in turns:
        for i, segment in enumerate(t["segments"]):
            yield dict(segment, offset=offset, turn=t["turn"], end=t["end_seconds"], index=i)
            offset += len(segment["text"])


def _raw_reference(row, duration):
    """A unit end is timing evidence for a complete unit, never for an individual word."""
    timestamps = row.get("timestamps", [])
    if not isinstance(timestamps, list) or not all(isinstance(t, dict) for t in timestamps):
        raise ValueError("Invalid reference timestamps")
    if not any("start_ms" in t or "end_ms" in t for t in timestamps):
        source = parse_source(row, duration)
        return [dict(language=s["language"], speaker=s["speaker"], text=s["text"],
                     onset_seconds=s["start_ms"] / 1000, first_unit_end_seconds=None,
                     anchor_source="switch_timestamp") for s in source["spans"]]
    units = parse_record(row, duration)
    if any("index" in t for t in timestamps):
        timestamps = sorted(timestamps, key=lambda t: t["index"])
    result = []
    for unit, timestamp in zip(units, timestamps):
        if timestamp.get("spans") is not None:
            for span in timestamp["spans"]:
                text = re.sub(r"<[^<>]+>", "", span.get("text_plain", span.get("text")))
                result.append(dict(language=span["language"], speaker=span["speaker"], text=text,
                                   onset_seconds=span["start_ms"] / 1000,
                                   first_unit_end_seconds=span["end_ms"] / 1000,
                                   anchor_source="aligned_attribution_span"))
        else:
            for i, (lang, spk, text) in enumerate(unit.spans):
                result.append(dict(language=lang, speaker=spk, text=text,
                                   onset_seconds=None if i else unit.start_ms / 1000,
                                   first_unit_end_seconds=unit.end_ms / 1000,
                                   anchor_source="untimed_internal_change" if i else "aligned_semantic_unit"))
    return result


def _aligned_reference(row, duration):
    # Adapt the native alignment schema to the existing validated semantic-unit reader.
    if row.get("status") != "ok" or not isinstance(row.get("segments"), list):
        raise ValueError("Expected successful aligned_asr")
    times = [dict(start_ms=s["start_ms"], end_ms=s["end_ms"], spans=s.get("attribution_spans"))
             for s in row["segments"]]
    segments = []
    for s in row["segments"]:
        text = "".join(f"<{p['language']}><{p['speaker']}>{p['text']}" for p in s["attributed"])
        segments.append({"source_text": text})
    raw = dict(audios=[row["audio"]], messages=[dict(role="assistant", content=row["transcript"])],
               output=dict(segments=segments), timestamps=times)
    return _raw_reference(raw, duration)


def _reference_spans(row, parsed):
    if not isinstance(row, dict):
        raise ValueError("Expected reference record")
    aligned = row.get("kind") == "aligned_asr"
    if aligned:
        path = row.get("audio")
    else:
        audios = row.get("audios")
        if not isinstance(audios, list) or len(audios) != 1:
            raise ValueError("Reference needs exactly one source audio path")
        path = audios[0]
    if not isinstance(path, str) or pathkey(path) != pathkey(parsed["audio"]):
        raise ValueError("Reference and training row are from different audio paths")
    duration = row.get("duration_ms", parsed.get("duration_ms"))
    if duration is None:
        if parsed["complete"]:
            duration = parsed["end_seconds"] * 1000
        else:
            # No source duration available for compact subwindows. A finite
            # parser bound allows validation of source text and timestamp order.
            # The unverified bound is NEVER used as event timing or a gold unit end.
            duration = sys.float_info.max
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid reference duration")
    end_ms = parsed["end_seconds"] * 1000
    # Input timestamps are in milliseconds. A submitted duration can be rounded
    # to an integer millisecond, while the virtual ref can end on a source sample.
    if end_ms > duration + 1 or parsed["complete"] and abs(duration - end_ms) > 1:
        raise ValueError("Reference duration disagrees with the SFT source clock")
    try:
        return _aligned_reference(row, duration) if aligned else _raw_reference(row, duration)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Invalid reference structure: {exc}") from exc


def _kind(before, after):
    if before[0] == after[0]:
        return "speaker"
    return "language" if before[1] == after[1] else "both"


def _make_case(before, after, offset, segment, snippet, **extra):
    matches = _state(segment) == after
    return dict(kind=_kind(before, after), previous_language=before[0], previous_speaker=before[1],
                language=after[0], speaker=after[1], char_offset=offset,
                gold_segment_index=segment["index"], gold_turn=segment["turn"], gold_seconds=segment["end"],
                gold_language=segment["language"], gold_speaker=segment["speaker"],
                gold_text=snippet, first_commit_characters=len(snippet) if matches else 0,
                gold_attribution_matches=matches, **extra)


def _first_commit(case, segments, offsets, stop, expected):
    """Count only text inside this reference state and this first assistant turn."""
    start = case["char_offset"]
    i = bisect_right(offsets, start) - 1
    pieces, correct = [], 0
    case.update(gold_expected_tag_turn=None, gold_expected_tag_seconds=None)
    while i < len(segments) and offsets[i] < stop:
        segment = segments[i]
        left, right = max(start - offsets[i], 0), stop - offsets[i]
        text = segment["text"][left:right]
        match = _state(segment) == expected
        if match and case["gold_expected_tag_turn"] is None:
            case.update(gold_expected_tag_turn=segment["turn"], gold_expected_tag_seconds=segment["end"])
        if segment["turn"] == case["gold_turn"]:
            pieces.append(text)
            if match:
                correct += len(text)
        elif case["gold_expected_tag_turn"] is not None:
            break
        i += 1
    case.update(gold_text="".join(pieces), first_commit_characters=correct)


def find_switches(parsed, reference=None):
    """Localize switches by exact character offsets. Preserve wrong/missing gold tags as evidence."""
    segments = list(_segments(parsed["turns"]))
    offsets = [s["offset"] for s in segments]
    gold = {}
    state = parsed["initial_state"]
    for s in segments:
        next_state = _state(s)
        if state and next_state != state:
            gold[s["offset"]] = _make_case(
                state, next_state, s["offset"], s, s["text"], event_basis="sft",
                onset_seconds=None, first_unit_end_seconds=None, next_onset_seconds=None,
                reference_char_offset=None, reference_text=None, anchor_source="sft_commit_only")
        state = next_state
    transcript = "".join(s["text"] for s in segments)
    gold_offsets = list(gold)
    for i, offset in enumerate(gold_offsets):
        stop = gold_offsets[i + 1] if i + 1 < len(gold_offsets) else len(transcript)
        _first_commit(gold[offset], segments, offsets, stop, _state(gold[offset]))
    if reference is None:
        return list(gold.values())

    timed = _reference_spans(reference, parsed)
    source_text = "".join(s["text"] for s in timed)
    begin = 0
    if parsed["complete"]:
        if transcript != source_text:
            raise ValueError("Full reference text differs from the concatenated SFT text")
    else:
        if not transcript:
            for before, after in zip(timed, timed[1:]):
                onset = after["onset_seconds"]
                if (_state(before) != _state(after) and (onset is None or
                        parsed["start_seconds"] <= onset < parsed["end_seconds"])):
                    raise ValueError("All-WAIT window has no text to associate with a possible reference switch")
            return []
        begin = source_text.find(transcript)
        if begin < 0 or source_text.find(transcript, begin + 1) != -1:
            raise ValueError("SFT window text does not match a unique exact reference substring")
    end = begin + len(transcript)
    # Consecutive semantic units with unchanged headers do not represent real changes.
    changes = []
    ref_offset, previous = 0, None
    for s in timed:
        new = _state(s)
        if previous is not None and new != previous:
            changes.append((ref_offset, previous, s))
        ref_offset += len(s["text"])
        previous = new
    cases = {}
    change_offsets = [position for position, _, _ in changes]
    for i, (position, before, entry) in enumerate(changes):
        following = changes[i + 1] if i + 1 < len(changes) else None
        if not begin <= position < end:
            continue
        offset = position - begin
        observed = segments[bisect_right(offsets, offset) - 1]
        stop = min(following[0] if following else end, end) - begin
        case = _make_case(
            before, _state(entry), offset, observed, "", event_basis="reference",
            onset_seconds=entry["onset_seconds"], first_unit_end_seconds=entry["first_unit_end_seconds"],
            next_onset_seconds=following[2]["onset_seconds"] if following else None,
            next_reference_char_offset=following[0] if following else None,
            reference_char_offset=position, reference_text=entry["text"], anchor_source=entry["anchor_source"])
        if entry["first_unit_end_seconds"] is not None:
            case.update(evidence_source=entry["anchor_source"], evidence_text=entry["text"])
        _first_commit(case, segments, offsets, stop, _state(entry))
        case["gold_transition_matches"] = (None if offset == 0 and begin > 0 and parsed["initial_state"] is None
            else bool(offset in gold and
                (gold[offset]["previous_language"], gold[offset]["previous_speaker"]) == before
                and _state(gold[offset]) == _state(entry)))
        case["onset_in_window"] = (None if entry["onset_seconds"] is None else
                                   parsed["start_seconds"] <= entry["onset_seconds"] < parsed["end_seconds"])
        cases[offset] = case
    for off, case in gold.items():
        if off in cases:
            continue
        i = bisect_right(change_offsets, begin + off)
        following = changes[i][2] if i < len(changes) else None
        current = changes[i - 1][2] if i else timed[0]
        matches = _state(current) == _state(case)
        case.update(reference_char_offset=begin + off, anchor_source="sft_change_without_reference_switch",
                    next_onset_seconds=following["onset_seconds"] if following else None,
                    reference_language=current["language"], reference_speaker=current["speaker"],
                    gold_attribution_matches=matches, gold_transition_matches=False)
        if not matches:
            case["first_commit_characters"] = 0
            case.update(gold_expected_tag_turn=None, gold_expected_tag_seconds=None)
        cases[off] = case
    return [cases[k] for k in sorted(cases)]


def enrich_with_alignment(primary_events, parsed, aligned):
    """Join independent switch anchors with verified aligned first-unit timing where available."""
    try:
        auxiliary = find_switches(parsed, aligned)
    except (ValueError, TypeError, KeyError) as exc:
        return primary_events, [f"Rejected secondary alignment: {exc}"]

    def identity(e):
        return (e["reference_char_offset"], e["previous_language"], e["previous_speaker"],
                e["language"], e["speaker"])

    auxiliary = {identity(e): e for e in auxiliary if e["event_basis"] == "reference"}
    result, warnings = [], []
    for raw in primary_events:
        case = dict(raw)
        result.append(case)
        if case["event_basis"] != "reference":
            continue
        position = case["reference_char_offset"]
        other = auxiliary.pop(identity(case), None)
        if other is None:
            warnings.append(f"No secondary switch at reference offset {position} with the same old and new attribution")
            continue
        onset = case["onset_seconds"]
        secondary_onset = other["onset_seconds"]
        if secondary_onset is not None:
            case["trajectory_onset_seconds"] = secondary_onset
            if onset is None:
                case.update(onset_seconds=secondary_onset, onset_in_window=other["onset_in_window"],
                            anchor_source="trajectory_" + other["anchor_source"])
            elif not math.isclose(secondary_onset, onset, rel_tol=0, abs_tol=1e-6):
                warnings.append(f"Reference offset {position}: source onset {onset}s, trajectory onset {secondary_onset}s")
        evidence = other["first_unit_end_seconds"]
        if case["first_unit_end_seconds"] is None and evidence is not None:
            new_state_end = case["next_reference_char_offset"]
            evidence_end = position + len(other["reference_text"])
            if new_state_end is not None and evidence_end > new_state_end:
                warnings.append(f"Secondary evidence crosses the next primary attribution at reference offset {position}")
            elif case["onset_seconds"] is not None and evidence < case["onset_seconds"]:
                warnings.append(f"Secondary evidence ends before the primary switch at reference offset {position}")
            else:
                case.update(first_unit_end_seconds=evidence, evidence_text=other["reference_text"],
                            evidence_source="trajectory_" + other["anchor_source"])
    if auxiliary:
        warnings.append(f"Secondary alignment has {len(auxiliary)} additional unmatched attribution switch(es)")
    known_onsets = {e["reference_char_offset"]: e["onset_seconds"] for e in result if e["event_basis"] == "reference"}
    previous_onset = None
    for e in result:
        if e["event_basis"] != "reference":
            continue
        if e["next_onset_seconds"] is None:
            e["next_onset_seconds"] = known_onsets.get(e["next_reference_char_offset"])
        onset = e["onset_seconds"]
        if onset is not None:
            if previous_onset is not None and onset < previous_onset:
                return primary_events, [*warnings, "Combined source and trajectory switch times are not chronological"]
            previous_onset = onset
    return result, warnings
