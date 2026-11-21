"""Convert full-recording Fourth annotations to causal ASR GSPO prompts and scorer metadata."""
from __future__ import annotations

import json
import math
import re
import sys
import unicodedata
from pathlib import Path

import soundfile as sf

from .audio import encode_chunk_ref
from .protocol import initial_messages
from .tail_source import tail_text_metadata

from fourth_asr.recast import validate_chunk_ms, end_frame
from fourth_asr.annotation_tail import project_open_tail
from fourth_asr.semantic_preparation import prepare_units
from fourth_asr.timestamped_source import HEADER_TIME, LANGUAGE, SPEAKER, TAG, parse_record, reference_spans


def audio_chunks(frames, rate, chunk_ms, max_turns=65, *, with_eof=True):
    """Same absolute frame grid and tail padding as Fourth SFT, with no copied audio."""
    validate_chunk_ms(chunk_ms)
    if rate < 1 or frames < 1 or rate * chunk_ms < 1000:
        raise ValueError("Empty audio or chunk shorter than one source sample")
    required_turns = (frames - 1) * 1000 // (chunk_ms * rate) + 1 + int(with_eof)
    if type(max_turns) is not int or max_turns < 2 or required_turns > max_turns:
        suffix = "including EOF" if with_eof else "ending at annotation_end"
        raise ValueError(f"Audio requires {required_turns} turns {suffix}; max_turns={max_turns}")
    start = 0
    chunks = []
    while start < frames:
        end = min(frames, ((len(chunks) + 1) * chunk_ms * rate + 999) // 1000)
        duration = (end - start) / rate
        chunks.append({"start_seconds": start / rate, "end_seconds": end / rate,
                       "input_seconds": max(duration, 0.5 if end == frames else 0.01)})
        start = end
    return chunks


def transcript_parts(row):
    answers = [m for m in row.get("messages", []) if isinstance(m, dict) and m.get("role") == "assistant"]
    if len(answers) != 1 or not isinstance(answers[0].get("content"), str):
        raise ValueError("Expected one full assistant transcript in messages")
    answer = answers[0]
    cleaned = HEADER_TIME.sub("", answer["content"])
    if "content_original" in answer and cleaned != answer["content_original"]:
        raise ValueError("content_original differs from the timestamp-stripped transcript")
    if re.search(rf"<{SPEAKER}>", cleaned) and re.search(rf"<{LANGUAGE}>", cleaned):
        # Use Fourth's exact attribution rules for fully labelled data,
        # including rejecting contradictory adjacent headers without speech.
        if not TAG.sub("", cleaned).strip():
            raise ValueError("Transcript contains only attribution tags")
        return list(reference_spans(row))

    lang = row.get("lang", "und")
    if not isinstance(lang, str) or not re.fullmatch(LANGUAGE, lang):
        raise ValueError("Invalid fallback language code")
    speaker = None
    parts = []
    leading = ""
    pos = 0
    for match in [*TAG.finditer(cleaned), None]:
        text = cleaned[pos:(match.start() if match else len(cleaned))]
        if "<" in text or ">" in text:
            raise ValueError("Malformed transcript tag")
        if text.strip():
            text = leading + text
            leading = ""
            if parts and parts[-1][:2] == (lang, speaker):
                parts[-1] = (lang, speaker, parts[-1][2] + text)
            else:
                parts.append((lang, speaker, text))
        elif text:
            if parts:
                old_lang, old_speaker, body = parts[-1]
                parts[-1] = (old_lang, old_speaker, body + text)
            else:
                leading += text
        if match is None:
            break
        tag = match.group(1)
        if re.fullmatch(SPEAKER, tag):
            speaker = tag
        elif re.fullmatch(LANGUAGE, tag):
            lang = tag
        else:
            raise ValueError(f"Unknown transcript tag <{tag}>")
        pos = match.end()
    if not "".join(t for _, _, t in parts).strip():
        if cleaned.strip():
            raise ValueError("Transcript contains only attribution tags")
        return []
    return parts


def _number(value, field):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    return value


def _timing(item, edge):
    key = edge + "_ms"
    if key in item:
        return None if item[key] is None else _number(item[key], key)
    if edge in item:
        return None if item[edge] is None else 1000 * _number(item[edge], edge)
    return None


def _ignorable(ch):
    return ch.isspace() or unicodedata.category(ch).startswith("P")


def _checked_alignment(entries, reference, duration_ms):
    if not isinstance(entries, list):
        raise ValueError("Expected a list of timed units")
    alignment = []
    cursor = 0
    for i, item in enumerate(entries):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid alignment entry {i}")
        text = item.get("text", item.get("word", item.get("char")))
        if not isinstance(text, str) or not text:
            raise ValueError(f"Missing alignment text at entry {i}")
        if "char_start" in item or "char_end" in item:
            start, end = item.get("char_start"), item.get("char_end")
            if (type(start) is not int or type(end) is not int or
                    not cursor <= start < end <= len(reference) or reference[start:end] != text):
                raise ValueError(f"Invalid reference offsets or mismatched text at entry {i}")
        else:
            start = cursor
            # Standard CTC output may omit punctuation and spaces. Never jump over
            # a lexical mismatch to find a later instance of a repeated word.
            while not reference.startswith(text, start) and start < len(reference) and _ignorable(reference[start]):
                start += 1
            end = start + len(text)
            if not reference.startswith(text, start):
                raise ValueError(f"Alignment text {text!r} disagrees at reference offset {start}")
        cursor = end
        if item.get("trusted", True) is False:
            continue
        begin, finish = _timing(item, "start"), _timing(item, "end")
        if begin is None and finish is None:
            continue
        if ((begin is not None and not 0 <= begin <= duration_ms + 1e-6) or
            (finish is not None and not 0 <= finish <= duration_ms + 1e-6) or
            (finish is not None and begin is not None and begin >= finish)):
            raise ValueError(f"Invalid chronological/audio bounds at alignment entry {i}")
        # Transcript order is checked by offsets. Overlapping speakers can
        # have nested intervals, so acoustic ends need not be monotonic.
        alignment.append(dict(text=text, start_ms=begin, end_ms=finish, char_start=start, char_end=end))
    return alignment


def _alignment(row, reference, duration_ms):
    acoustic = row.get("alignment")
    if isinstance(acoustic, dict) and acoustic.get("status", "ok") != "ok":
        raise ValueError("Unsuccessful acoustic alignment")
    containers = [acoustic, row] if isinstance(acoustic, dict) else [row]
    if acoustic is not None and not isinstance(acoustic, (list, dict)):
        raise ValueError("alignment must be an array or object")
    if isinstance(acoustic, list):
        return _checked_alignment(acoustic, reference, duration_ms)
    for container in containers:
        for key in ("chars", "words", "word_segments"):
            if key in container:
                return _checked_alignment(container[key], reference, duration_ms)

    timestamps = row.get("timestamps", [])
    if not isinstance(timestamps, list) or not all(isinstance(t, dict) for t in timestamps):
        raise ValueError("timestamps must be a list of objects")
    if timestamps and any("start_ms" in t or "end_ms" in t for t in timestamps):
        # The SFT path already verifies fidelity, inheritance, attribution,
        # interval ordering and nested spans for distilled semantic units.
        if row.get("output") is not None:
            units = parse_record(row, duration_ms, release_on="timed-span")
            entries = [dict(text="".join(s[2] for s in u.spans),
                            start_ms=u.start_ms, end_ms=u.end_ms) for u in units]
        else:
            entries = timestamps
        return _checked_alignment(entries, reference, duration_ms)
    return []


def _switches(row, reference, duration_ms):
    # A change is the onset of a new attribution. It provides neither the
    # previous word's end (possible silence) nor the new first word's end.
    timestamps = row.get("timestamps", [])
    if not isinstance(timestamps, list) or not all(isinstance(s, dict) for s in timestamps):
        raise ValueError("timestamps must be a list of objects")
    if not any("time_ms" in s for s in timestamps):
        return []
    # Enforce the source annotator's existing checks for original header
    # offsets, inline markers, complete span text and attribution fields.
    repo = str(Path(__file__).resolve().parents[2])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from omnicommit.prefix_source import parse_source

    source = parse_source(row, duration_ms)
    if source["text"] != reference:
        raise ValueError("Switch annotations disagree with the reference text")
    return [dict(char_start=e["char_offset"], time_ms=e["time_ms"], lang=e["language"], speaker=e["speaker"])
            for e in source["events"]]


def _source_ids(row, path, root):
    ids = {str(path)}
    extras = row.get("source_ids", [])
    if not isinstance(extras, list) or not all(isinstance(v, str) and v.strip() for v in extras):
        raise ValueError("source_ids must be an array of nonempty strings")
    ids.update(extras)
    for field, prefix in (("meeting_id", "meeting"), ("source_recording_id", "recording"),
                          ("recording_id", "recording"), ("source_id", "source")):
        value = row.get(field)
        if value is not None:
            if type(value) not in (str, int) or not str(value).strip():
                raise ValueError(f"Invalid {field}")
            ids.add(f"{prefix}:{value}")
    if "synthesis" in row:
        synthesis = row["synthesis"]
        clips = synthesis.get("clips") if isinstance(synthesis, dict) else None
        if not isinstance(clips, list) or not clips:
            raise ValueError("synthesis.clips must identify the donor audios")
        for clip in clips:
            if not isinstance(clip, dict) or not isinstance(clip.get("audio"), str) or not clip["audio"].strip():
                raise ValueError("synthesis.clips entry missing donor audio path")
            ids.add(str((root / clip["audio"]).resolve()))
    return sorted(ids)


def _check_audio_identity(metadata, path, root, duration_ms):
    declared = metadata.get("audio")
    if declared is not None and (not isinstance(declared, str) or (root / declared).resolve() != path):
        raise ValueError("Alignment audio path differs from the actual source recording")
    if "duration_ms" in metadata and abs(_number(metadata["duration_ms"], "duration_ms") - duration_ms) > 1:
        raise ValueError("Declared duration_ms differs from the actual source recording")


def _native_aligned_row(row):
    """Read the repository's schema-v2 alignment output through the Fourth validator."""
    if row.get("status") != "ok" or row.get("schema_version") != 2:
        raise ValueError("Expected successful schema_version=2 aligned_asr output")
    segments = row.get("segments")
    if not isinstance(segments, list) or not segments or not all(isinstance(s, dict) for s in segments):
        raise ValueError("aligned_asr requires a nonempty segments array")
    stamps = []
    for segment in segments:
        stamp = {k: segment.get(k) for k in ("source_text", "start_ms", "end_ms")}
        if "attribution_spans" in segment:
            stamp["spans"] = segment["attribution_spans"]
        stamps.append(stamp)
    return dict(row, audios=[row.get("audio")], messages=[{"role": "assistant", "content": row.get("transcript")}],
                output={"segments": segments}, timestamps=stamps)


def _tail_projection(row, parts, info, chunk_ms, policy):
    """Use the same unit parser and tail projection as Fourth build_sft."""
    if policy not in ("keep", "short-open", "annotated-prefix"):
        raise ValueError("Unknown tail-policy")
    if policy == "keep" or not parts:
        return parts, info.frames, False, None
    stamps = row.get("timestamps")
    if not stamps or not all(isinstance(s, dict) and "end_ms" in s for s in stamps):
        raise ValueError("Open tail policies require per-unit timestamps, not switch-onset-only annotations")
    duration_ms = info.frames * 1000 / info.samplerate
    units, _ = prepare_units(row, duration_ms, chunk_ms=chunk_ms, short_ms=0, release_on="merged-unit")
    last = end_frame(max(t["end_ms"] for t in stamps), info.frames, info.samplerate)
    short = info.frames - last < info.samplerate
    opened = policy == "annotated-prefix" or short
    if not opened:
        return parts, info.frames, False, None
    text = "".join(p[2] for p in parts)
    target = text
    if short:
        units, target, stripped = project_open_tail(units, parts[-1][2], text)
    else:
        stripped = False
    removed = None
    if stripped:
        removed = next(i for i, (a, b) in enumerate(zip(text + "\0", target + "\0")) if a != b)
        old_parts, parts, cursor = parts, [], 0
        for lang, speaker, body in old_parts:
            lo, hi = cursor, cursor + len(body)
            cursor = hi
            if lo <= removed < hi:
                body = body[:removed-lo] + body[removed-lo+1:]
            parts.append((lang, speaker, body))
    frames = end_frame(max(u.end_ms for u in units), info.frames, info.samplerate)
    return parts, frames, True, removed


def prepare_record(row, audio_root, chunk_ms=1000, max_turns=65, *, tail_policy="keep", timing_mode="tail-only"):
    if not isinstance(row, dict):
        raise ValueError("Expected a JSON object")
    if timing_mode not in ("tail-only", "aligned"):
        raise ValueError("timing_mode must be tail-only or aligned")
    if timing_mode == "tail-only" and tail_policy != "keep":
        raise ValueError("tail-only timing requires --tail-policy keep; open-tail projection needs --timing-mode aligned")
    if type(row.get("punctuation_trusted", True)) is not bool:
        raise ValueError("punctuation_trusted must be a JSON boolean")
    if row.get("kind") == "aligned_asr":
        row = _native_aligned_row(row)
    paths = row.get("audios")
    if not isinstance(paths, list) or len(paths) != 1 or not isinstance(paths[0], str) or not paths[0].strip():
        raise ValueError("Expected exactly one original recording in audios")
    root = Path(audio_root)
    path = (root / paths[0]).resolve(strict=True)
    info = sf.info(str(path))
    parts = transcript_parts(row)
    original_reference = "".join(p[2] for p in parts)
    # The final distilled DIRECT may omit a stop still present in the source.
    # Preserve source text; withhold only that uncertain terminal-stop reward.
    terminal_trusted = (tail_text_metadata(row, original_reference)["terminal_punctuation_trusted"]
                        if timing_mode == "tail-only" and any(c.isalnum() for c in original_reference) else True)
    parts, supervised_frames, opened, removed = _tail_projection(row, parts, info, chunk_ms, tail_policy)
    chunks = audio_chunks(supervised_frames, info.samplerate, chunk_ms, max_turns, with_eof=not opened)
    spans = []
    for lang, _, body in parts:
        if spans and spans[-1]["lang"] == lang:
            spans[-1]["text"] += body
        else:
            spans.append(dict(text=body, lang=lang))
    reference = "".join(s["text"] for s in spans)
    duration_ms = 1000 * info.frames / info.samplerate
    _check_audio_identity(row, path, root, duration_ms)
    if timing_mode == "aligned" and isinstance(row.get("alignment"), dict):
        _check_audio_identity(row["alignment"], path, root, duration_ms)
        if row.get("id") is not None and row["alignment"].get("id") is not None:
            if str(row["id"]) != str(row["alignment"]["id"]):
                raise ValueError("Acoustic alignment belongs to another record ID")
    # In the default source-transcript path, legacy semantic chunks/actions are
    # not policy targets and their timestamps are not latency supervision. The
    # independent filter uses just the final endpoint when punctuation needs it.
    # Keep an opt-in path for experiments with validated word/semantic timing.
    alignment = _alignment(row, original_reference, duration_ms) if timing_mode == "aligned" else []
    switches = _switches(row, original_reference, duration_ms) if timing_mode == "aligned" else []
    if removed is not None:
        def projected_offset(i):
            return i - int(i > removed)
        projected = []
        for item in alignment:
            lo, hi = projected_offset(item["char_start"]), projected_offset(item["char_end"])
            if lo < hi:
                projected.append(dict(item, char_start=lo, char_end=hi, text=reference[lo:hi]))
        alignment = projected
        switches = [dict(s, char_start=projected_offset(s["char_start"])) for s in switches]
    effective_ms = supervised_frames * 1000 / info.samplerate
    if opened:
        # A separately timed final period may have shortened the supervised
        # range. Do not invent a new acoustic end for its enclosing text span.
        alignment = [dict(a, end_ms=(None if a.get("end_ms") is not None and a["end_ms"] > effective_ms + .001 else a.get("end_ms")))
                     for a in alignment if a.get("start_ms") is None or a["start_ms"] < effective_ms]
    attributed, speaker_changes, cursor, previous = [], [], 0, None
    for lang, speaker, body in parts:
        attributed.append(dict(text=body, lang=lang, speaker=speaker, char_start=cursor, char_end=cursor+len(body)))
        if body.strip():
            if previous is not None and speaker != previous:
                speaker_changes.append(cursor)
            previous = speaker
        cursor += len(body)
    bucket = row.get("bucket", "switch" if len(spans) > 1 else ("general" if reference else "stability"))
    if bucket not in ("general", "switch", "stability"):
        raise ValueError(f"Unknown sampling bucket {bucket!r}")
    identity = row.get("id", row.get("key", str(path)))
    if type(identity) not in (str, int) or not str(identity).strip():
        raise ValueError("An id or key must be a nonempty string or integer")
    stat = path.stat()
    task = dict(id=str(identity), audio_path=str(path), audio_size_bytes=stat.st_size,
                audio_mtime_ns=stat.st_mtime_ns, source_ids=_source_ids(row, path, root),
                protocol="fourth_speaker_v1", end_policy="annotation_end" if opened else "eof",
                tail_policy=tail_policy, timing_mode=timing_mode, terminal_period_removed=removed is not None,
                source_duration_seconds=info.frames / info.samplerate,
                bucket=bucket, duration_seconds=supervised_frames / info.samplerate,
                chunks=chunks, reference_text=reference, spans=spans, alignment=alignment)
    task.update(attribution_spans=attributed, speaker_changes=speaker_changes,
                speaker_scorable=bool(parts) and all(s is not None for _, s, body in parts if body.strip()),
                punctuation_trusted=row.get("punctuation_trusted", True),
                terminal_punctuation_trusted=terminal_trusted)
    if switches:
        task["switches"] = switches
    return dict(messages=initial_messages(), audios=[encode_chunk_ref(task, 0)],
                asr_task=json.dumps(task, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
