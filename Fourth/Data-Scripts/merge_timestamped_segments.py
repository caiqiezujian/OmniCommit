"""Step 1: group intact timed ASR segments before streaming actions are assigned."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path, PureWindowsPath
from tempfile import NamedTemporaryFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_asr.semantic_preparation import STANDALONE, TRAILING, closes_clause, numeric_continuation
from fourth_asr.timestamped_source import parse_record


INPUT = Path("/data/s50062929/FLEX/SFT_data/full_10000h_augmented.jsonl")
OUTPUT = INPUT.with_name("full_10000h_augmented.merged_v2.jsonl")
SHORT_BELOW_MS = 800
PROTECT_THROUGH_MS = 1200
MAX_GAP_MS = 350
MAX_GROUP_MS = 3000
PROGRESS_EVERY = 1000

WEAK = frozenset("，,、،：:")
SUFFIX_PUNCT = frozenset(".!?。！？；;，,、،؛؟…:：") | frozenset(TRAILING)


TITLES = {"es": {"sr.", "sra.", "srta."}, "pt": {"sr.", "sra."},
          "fr": {"m.", "mme.", "mlle."}, "it": {"sig.", "sig.ra."}}


def end_kind(text, following, language=""):
    """Sentence/semicolon ends close immediately; short comma clauses can continue."""
    if not closes_clause(text, following):
        return None
    tail = text.rstrip().rstrip(TRAILING).rstrip()
    word = tail.split()[-1].casefold()
    next_char = following.lstrip()[:1]
    if (word == "art." and next_char.isdecimal()) or (
            word in TITLES.get(language.split("-")[0], ()) and next_char.isupper()):
        return None
    if tail[-1] not in WEAK:
        return "strong"
    body = tail[:-1].strip().casefold()
    # Repetitions such as “你，你，你们” provide poor internal comma breaks.
    if body and len(body) <= 3 and following.lstrip().casefold().startswith(body):
        return None
    return "weak"


def synthesis_boundaries(record, units, stats):
    """Only join units wholly inside the same documented source clip."""
    if "synthesis" not in record:
        return set()
    synthesis = record["synthesis"]
    clips = synthesis.get("clips") if isinstance(synthesis, dict) else None
    rate = synthesis.get("sample_rate") if isinstance(synthesis, dict) else None
    intervals = []
    if type(rate) is int and rate > 0 and isinstance(clips, list):
        for clip in clips:
            if not isinstance(clip, dict):
                break
            start, end = clip.get("start_frame"), clip.get("end_frame")
            if (type(start) is not int or type(end) is not int or not 0 <= start < end
                    or (intervals and intervals[-1][1] > start)):
                break
            intervals.append((start, end))
    if not intervals or len(intervals) != len(clips):
        stats["synthesis_rows_without_usable_clip_times"] += 1
        return set(range(1, len(units)))
    barriers = set()
    pos = 0
    previous = None
    for i, unit in enumerate(units):
        # Timestamps are relative to the composed recording, not the source WAV.
        begin, end = unit.start_ms * rate, unit.end_ms * rate
        while pos < len(intervals) and intervals[pos][1] * 1000 <= begin:
            pos += 1
        owner = None
        if pos < len(intervals) and intervals[pos][0] * 1000 <= begin and end <= intervals[pos][1] * 1000:
            owner = pos
        if owner is None:
            stats["units_without_unique_synthesis_clip"] += 1
        if i and (owner is None or previous is None or previous != owner):
            barriers.add(i)
        previous = owner
    stats["synthesis_boundaries_protected"] += len(barriers)
    return barriers


def group_ranges(units, barriers):
    start = 0
    while start < len(units):
        first = units[start]
        stop = start + 1
        last_weak = None
        has_protected = first.end_ms - first.start_ms >= SHORT_BELOW_MS
        while stop < len(units):
            if stop in barriers:
                break
            left, right = units[stop-1], units[stop]
            punct_only = bool(right.text.strip()) and all(ch in SUFFIX_PUNCT for ch in right.text.strip())
            if (right.text.strip() in ('"', "'") and right.text[0].isspace() and stop+1 < len(units)
                    and units[stop+1].text[:1].isalnum()):
                break  # A separated opening quote precedes the next word.
            left_text = left.text
            if stop-start > 1 and all(ch in SUFFIX_PUNCT for ch in left_text.strip()):
                left_text = "".join(unit.text for unit in units[start:stop])
            decimal = numeric_continuation(left_text, right.text)
            # Even if separately timestamped, closing punctuation belongs with its text.
            if ((has_protected and not (punct_only or decimal))
                    or right.end_ms - right.start_ms > PROTECT_THROUGH_MS
                    or len(left.spans) != 1 or len(right.spans) != 1
                    or left.spans[0][:2] != right.spans[0][:2]
                    or not 0 <= right.start_ms - left.end_ms <= MAX_GAP_MS):
                break
            boundary = end_kind(left_text, right.text, left.spans[-1][0])
            if boundary == "strong" and not punct_only:
                break
            if boundary == "weak" and not punct_only:
                last_weak = stop
                if (left.end_ms - first.start_ms >= SHORT_BELOW_MS
                        or left.text.strip().rstrip("".join(WEAK)).strip().casefold() in STANDALONE):
                    break
            if right.end_ms - first.start_ms > MAX_GROUP_MS:
                # If the group cannot reach the next clause, use an existing comma if available.
                stop = last_weak or stop
                break
            has_protected |= right.end_ms - right.start_ms >= SHORT_BELOW_MS
            stop += 1
        yield start, stop
        start = stop


def child_intervals(unit, stamp):
    if unit.children:
        return stamp["spans"]
    (language, speaker, text), = unit.spans
    return [{"start_ms": unit.start_ms, "end_ms": unit.end_ms, "text_plain": text,
             "language": language, "speaker": speaker}]


def merge_record(record):
    times = record.get("timestamps") if isinstance(record, dict) else None
    if not isinstance(times, list) or not times or any(
            not isinstance(t, dict) or type(t.get("end_ms")) not in (int, float) for t in times):
        raise ValueError("expected one start_ms/end_ms per source segment")
    upper = max(t["end_ms"] for t in times)
    encoded = isinstance(record.get("output"), str)
    annotation = json.loads(record["output"]) if encoded else record.get("output")
    units = parse_record({**record, "output": annotation}, upper)
    if "index" in times[0]:
        times = sorted(times, key=lambda item: item["index"])
    segments, stamps = [], []
    stats = Counter(source_segments=len(units))
    previous_state = None
    barriers = synthesis_boundaries(record, units, stats)
    covered_until = 0
    for i, u in enumerate(units):
        following = units[i+1].text if i+1 < len(units) else ""
        boundary = end_kind(u.text, following, u.spans[-1][0])
        stats["source_ends_at_punctuation"] += boundary is not None
        stats[f"source_end_{boundary or 'unmarked'}"] += 1
        if (covered_until > u.start_ms or (i+1 < len(units) and units[i+1].start_ms < u.end_ms)):
            # Isolate both ends of an overlap participant, including an interjection
            # followed by solo speech from the same speaker.
            barriers.update((i, i+1))
            stats["units_with_overlapping_intervals"] += 1
        covered_until = max(covered_until, u.end_ms)
        if len(u.spans) > 1:
            stats["mixed_state_units_with_child_times" if u.children else "mixed_state_units_without_child_times"] += 1
        for language, speaker, text in u.spans:
            if not text.strip():
                continue
            if previous_state is not None and previous_state != (language, speaker):
                old_language, old_speaker = previous_state
                if language == old_language:
                    stats["speaker_only_changes"] += 1
                elif speaker == old_speaker:
                    stats["language_only_changes"] += 1
                else:
                    stats["language_and_speaker_changes"] += 1
            previous_state = language, speaker
    for start, stop in group_ranges(units, barriers):
        first, last = units[start], units[stop-1]
        if stop == start + 1:
            body = first.text
            marked = annotation["segments"][start]["source_text"]
            stamp = dict(times[start])
        else:
            lang, speaker, _ = first.spans[0]
            body = "".join(u.text for u in units[start:stop])
            marked = f"<{lang}><{speaker}>" + body
            spans = []
            origins = []
            for i in range(start, stop):
                children = child_intervals(units[i], times[i])
                origins.append({"index": i+1, "start_ms": units[i].start_ms, "end_ms": units[i].end_ms,
                                "span_count": len(children)})
                spans.extend(children)
            stamp = {"start_ms": first.start_ms, "end_ms": last.end_ms, "source_text": marked,
                     "text": marked, "text_plain": body, "language": lang, "speaker": speaker,
                     "languages": [lang], "speakers": [speaker], "spans": spans,
                     "merge_source_units": origins}
            backend = times[start].get("backend")
            if backend is not None and all(times[i].get("backend") == backend for i in range(start+1, stop)):
                stamp["backend"] = backend
            stats["joined_groups"] += 1
            stats["short_prefix_into_protected"] += any(
                u.end_ms - u.start_ms >= SHORT_BELOW_MS for u in units[start+1:stop])
        stamp["index"] = len(stamps) + 1
        stamps.append(stamp)
        segments.append({"source_text": marked})
        following = units[stop].text if stop < len(units) else ""
        boundary = end_kind(body, following, last.spans[-1][0])
        stats["merged_ends_at_punctuation"] += boundary is not None
        stats[f"merged_end_{boundary or 'unmarked'}"] += 1
    stats["merged_segments"] = len(segments)
    out = dict(record)
    # Legacy per-segment actions and EOS flush belong to the old grouping.
    out.pop("tags", None)
    new_annotation = {k: v for k, v in annotation.items() if k not in ("segments", "flush")}
    new_annotation["segments"] = segments
    out.update(output=new_annotation, timestamps=stamps)
    if stats["joined_groups"]:
        # Checks text including spaces, speaker/language order and every child timestamp.
        parse_record(out, upper)
    if encoded:
        out["output"] = json.dumps(new_annotation, ensure_ascii=False, separators=(",", ":"))
    return out, stats


@contextmanager
def publish_on_success(path):
    with NamedTemporaryFile("wb", dir=path.parent, prefix=path.name + ".", suffix=".partial",
                            delete=False) as stream:
        tmp = Path(stream.name)
        try:
            yield stream
            stream.close()
            os.replace(tmp, path)
        finally:
            stream.close()
            tmp.unlink(missing_ok=True)


def run(source, output):
    errors = output.with_suffix(".rejected.jsonl")
    if source.resolve() in (output.resolve(), errors.resolve()):
        raise ValueError("choose a distinct destination")
    totals = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    source_dir = source.resolve().parent
    relocate = source_dir != output.resolve().parent
    with source.open("rb") as inp, publish_on_success(output) as out, publish_on_success(errors) as bad:
        for line, raw in enumerate(inp, 1):
            totals["rows"] += 1
            try:
                record = json.loads(raw.decode("utf-8-sig" if line == 1 else "utf-8"))
                merged, stats = merge_record(record)
                audio = merged["audios"][0]
                if relocate and not (audio.startswith("/") or PureWindowsPath(audio).is_absolute()):
                    merged["audios"] = [str(source_dir / audio)]
                data = json.dumps(merged, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                payload = (data + "\n").encode("utf-8")
            except (ValueError, KeyError, TypeError, OverflowError, RecursionError) as exc:
                bad.write((json.dumps({"line": line, "reason": str(exc)}, ensure_ascii=True) + "\n").encode("ascii"))
                totals["rejected"] += 1
            else:
                out.write(payload)
                totals.update(stats)
                totals["kept"] += 1
            if line == 1 or line % PROGRESS_EVERY == 0:
                print(f"[{line:,}] kept={totals['kept']:,} rejected={totals['rejected']:,} "
                      f"segments={totals['source_segments']:,} -> {totals['merged_segments']:,}", flush=True)
    print(json.dumps(dict(totals), ensure_ascii=False), flush=True)
    print(f"Output: {output}\nRejected: {errors}", flush=True)
    return totals


if __name__ == "__main__":
    raise SystemExit(0 if run(INPUT, OUTPUT)["kept"] else 1)
