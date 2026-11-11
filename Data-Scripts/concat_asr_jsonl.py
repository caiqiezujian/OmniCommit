#!/usr/bin/env python3
"""Pair A-tail != B-head, join their WAVs, and reuse pre-timestamp ASR annotations.

python concat_asr_jsonl.py A.jsonl B.jsonl joined.jsonl [--audio-dir DIR] [--seed 42]
Dependencies: numpy, soundfile, scipy (scipy is used for resampling).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import re
import sys
from uuid import uuid4

import numpy as np
import soundfile as sf


DEFAULT_AUDIO_DIR = "/data/data_process/zhh/multilingual_asr/data/codeswitch/audios"
RATE = 16000
TAG = re.compile(r"<([^<>]+)>")
LANG = re.compile(r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*")
SPK = re.compile(r"spk[1-9][0-9]*")
SPK_TAG = re.compile(r"<(spk[1-9][0-9]*)>")
HEAD = re.compile(r"^((?:\s*<[^<>]+>)+)")
DIRECT_HEAD = re.compile(r"^<" + LANG.pattern + r"><" + SPK.pattern + r">")
ENTITY_PREFIX = "<audio>**Entity Reference:**"


def tagged_runs(text, *, original=False):
    """Return attributed body text, preserving every space and semantic character."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty transcript")
    runs = []
    language = speaker = None
    pending, leading = {}, ""
    pos = 0
    for match in [*TAG.finditer(text), None]:
        body = text[pos:match.start() if match else len(text)]
        if "<" in body or ">" in body:
            raise ValueError("broken attribution tag")
        if body:
            if not body.strip() and (original or not (language and speaker)):
                # The teacher normalizes whitespace-only gaps before a new header.
                if runs:
                    runs[-1][2] += body
                else:
                    leading += body
            else:
                if not (language and speaker):
                    raise ValueError("missing initial language or speaker")
                if body.strip():
                    pending.clear()
                if runs and runs[-1][:2] == [language, speaker]:
                    runs[-1][2] += body
                else:
                    runs.append([language, speaker, leading + body])
                    leading = ""
        if match is None:
            break
        tag = match[1]
        if SPK.fullmatch(tag):
            kind, speaker = "speaker", tag
        elif LANG.fullmatch(tag):
            kind, language = "language", tag
        else:
            raise ValueError(f"unknown transcript tag <{tag}>")
        if kind in pending and pending[kind] != tag:
            raise ValueError("conflicting headers without speech")
        pending[kind] = tag
        pos = match.end()
    if not runs or pending:
        raise ValueError("dangling attribution or no speech")
    return runs


def messages(row):
    items = row["messages"]
    if not isinstance(items, list) or [m["role"] for m in items] != ["system", "user", "assistant"]:
        raise ValueError("expected system/user/assistant messages")
    if any(not isinstance(m["content"], str) for m in items) or items[1]["content"].count("<audio>") != 1:
        raise ValueError("expected text messages with one <audio> in user content")
    return items


def decode_line(raw):
    row = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(row, dict):
        raise ValueError("row must be an object")
    return row


def checked_segments(row, runs):
    output = row["output"]
    if isinstance(output, str):
        output = json.loads(output)
    if not isinstance(output, dict) or output.get("error", True) is not None:
        raise ValueError("failed or missing annotation")
    segments = output["segments"]
    if not isinstance(segments, list) or not segments:
        raise ValueError("empty segments")
    changes = []
    total = 0
    for _, _, body in runs[:-1]:
        total += len(body)
        changes.append(total)
    change_index = offset = 0
    texts = []
    for seg in segments:
        if not isinstance(seg, dict) or set(seg) != {"source_text", "action"}:
            raise ValueError("expected source_text/action in each segment")
        text, action = seg["source_text"], seg["action"]
        if not isinstance(text, str) or action not in ("DIRECT", "WAIT"):
            raise ValueError("invalid segment text or action")
        body = TAG.sub("", text)
        if "<" in body or ">" in body or re.search(r"<[^<>]+>\s*$", text):
            raise ValueError("broken or dangling tag inside a segment")
        if not any(c.isalnum() for c in body):
            raise ValueError("segment has no lexical content")
        if action == "DIRECT" and not DIRECT_HEAD.match(text):
            raise ValueError("DIRECT lacks a complete language/speaker header")
        start = 0
        while change_index < len(changes) and changes[change_index] < offset + len(body):
            if action != "WAIT":
                raise ValueError("attribution switch requires WAIT")
            end = changes[change_index] - offset
            if start < end and not body[start:end].strip():
                raise ValueError("attribution span contains only whitespace")
            start = end
            change_index += 1
        if not body[start:].strip():
            raise ValueError("attribution span contains only whitespace")
        offset += len(body)
        texts.append(text)
    if tagged_runs("".join(texts)) != runs:
        raise ValueError("annotation changes text or attribution")
    if output.get("flush") is not (segments[-1]["action"] == "WAIT"):
        raise ValueError("flush disagrees with last segment")
    return segments


@dataclass
class Clip:
    messages: list
    runs: list
    segments: list
    path: Path


def load_clip(stream, offset):
    stream.seek(offset)
    row = decode_line(stream.readline())
    items = messages(row)
    # Escaped lone surrogates are accepted by json.loads but cannot be written to UTF-8 JSONL.
    json.dumps(items, ensure_ascii=False, allow_nan=False).encode("utf-8")
    runs = tagged_runs(items[2]["content"], original=True)
    paths = row["audios"]
    if not isinstance(paths, list) or len(paths) != 1 or not isinstance(paths[0], str) or not paths[0].strip():
        raise ValueError("expected one source audio path")
    path = (Path(stream.name).parent / Path(paths[0]).expanduser()).resolve()
    return Clip(items, runs, checked_segments(row, runs), path)


def merged_user(first, second):
    if first == second:
        return first
    refs = []
    for prompt in (first, second):
        if prompt.strip() == "<audio>":
            continue
        if not prompt.startswith(ENTITY_PREFIX):
            raise ValueError("cannot combine different user prompts with unknown format")
        entities = json.loads(prompt[len(ENTITY_PREFIX):])
        if not isinstance(entities, list):
            raise ValueError("expected an entity reference list")
        for entity in entities:
            if entity not in refs:
                refs.append(entity)
    return ENTITY_PREFIX + " " + json.dumps(refs, ensure_ascii=False)


def join_labels(a, b):
    if a.runs[-1][0] == b.runs[0][0]:
        raise ValueError("same language at the join")
    highest = max(int(speaker[3:]) for _, speaker, _ in a.runs)
    speakers = dict.fromkeys(speaker for _, speaker, _ in b.runs)
    mapping = {speaker: f"spk{highest + i}" for i, speaker in enumerate(speakers, 1)}
    lang, first_speaker, _ = b.runs[0]
    prefix = f"<{lang}><{mapping[first_speaker]}>"

    def remap(text, first=False):
        text = SPK_TAG.sub(lambda m: f"<{mapping[m[1]]}>", text)
        if first:
            # Anchor even B's initial whitespace to B when it follows A.
            head = HEAD.match(text)
            if not head:
                raise ValueError("B is missing its initial attribution")
            text = prefix + TAG.sub("", head[0]) + text[head.end():]
        return text

    transcript = a.messages[2]["content"] + remap(b.messages[2]["content"], first=True)
    segments = a.segments + [
        {"source_text": remap(s["source_text"], first=(i == 0)),
         "action": "WAIT" if i == 0 else s["action"]}
        for i, s in enumerate(b.segments)
    ]
    output = {"segments": segments, "flush": segments[-1]["action"] == "WAIT", "error": None}
    items = [dict(a.messages[0]),
             {"role": "user", "content": merged_user(a.messages[1]["content"], b.messages[1]["content"])},
             {"role": "assistant", "content": transcript}]
    checked_segments({"output": output}, tagged_runs(transcript, original=True))
    return {"messages": items, "output": json.dumps(output, ensure_ascii=False, separators=(",", ":"))}


def read_audio(path):
    with sf.SoundFile(str(path)) as audio:
        rate, frames = audio.samplerate, len(audio)
        data = audio.read(dtype="float32", always_2d=True)
    if frames == 0 or len(data) != frames or not np.isfinite(data).all():
        raise ValueError(f"empty, truncated or nonfinite audio: {path}")
    mono = data.mean(axis=1, dtype=np.float32)
    if rate != RATE:
        from scipy.signal import resample_poly
        gcd = math.gcd(rate, RATE)
        mono = resample_poly(mono, RATE // gcd, rate // gcd)
    return mono


def write_pair(stream, audio_dir, a_line, b_line, left, right, labels):
    path = audio_dir / f"cs_A{a_line}_B{b_line}_{uuid4().hex}.wav"
    row = {"audios": [str(path)], **labels}
    line = json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
    file = path.open("x+b")
    try:
        with file:
            with sf.SoundFile(file, "w", samplerate=RATE, channels=1, format="WAV", subtype="PCM_16") as wav:
                wav.write(left)
                wav.write(right)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    # Publish only a closed, complete WAV. Disk/write errors must surface.
    stream.write(line)
    stream.flush()


BAD_ROW = (ValueError, KeyError, IndexError, TypeError, UnicodeError, OSError, sf.LibsndfileError)


def rejected(counts, key, line, exc):
    counts[key] += 1
    if counts[key] <= 10:
        print(f"[{key}] {line}: {exc}", file=sys.stderr)


def index_rows(stream, side, counts):
    rows = []
    number = 0
    while True:
        offset = stream.tell()
        raw = stream.readline()
        if not raw:
            break
        number += 1
        if not raw.strip():
            continue
        try:
            row = decode_line(raw)
            runs = tagged_runs(messages(row)[2]["content"], original=True)
            language = runs[-1 if side == "A" else 0][0]
            rows.append((language, number, offset))
        except BAD_ROW as exc:
            rejected(counts, f"invalid_{side}", number, exc)
    return rows


def build(args):
    counts = Counter(written=0)
    switches = Counter()
    rng = random.Random(args.seed)
    audio_dir = args.audio_dir.expanduser().resolve()
    destination = args.output.expanduser().resolve()
    audio_dir.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with args.a.open("rb") as fa, args.b.open("rb") as fb, destination.open("x", encoding="utf-8") as out:
        left = index_rows(fa, "A", counts)
        rng.shuffle(left)
        remaining = Counter(lang for lang, _, _ in left)
        buckets = defaultdict(list)
        for lang, line, offset in index_rows(fb, "B", counts):
            buckets[lang].append((line, offset))
        for bucket in buckets.values():
            rng.shuffle(bucket)
        print(f"Indexed A={len(left)}, B={sum(map(len, buckets.values()))}", flush=True)
        for lang_a, a_line, offset in left:
            remaining[lang_a] -= 1
            if not any(k != lang_a and v for k, v in buckets.items()):
                counts["unmatched_A"] += 1
                continue
            try:
                a = load_clip(fa, offset)
                pcm_a = read_audio(a.path)
            except BAD_ROW as exc:
                rejected(counts, "invalid_A", a_line, exc)
                continue
            deferred = []
            while True:
                allowed = [k for k, v in buckets.items() if k != lang_a and v]
                if not allowed:
                    counts["unmatched_A"] += 1
                    break
                # Favor the language with most remaining rows across BOTH sides:
                # any rows of that language can only match rows of other languages.
                lang_b = max(allowed, key=lambda k: remaining[k] + len(buckets[k]))
                b_line, b_offset = buckets[lang_b].pop()
                try:
                    b = load_clip(fb, b_offset)
                    if a.path == b.path:
                        deferred.append((lang_b, b_line, b_offset))
                        continue
                except BAD_ROW as exc:
                    rejected(counts, "invalid_B", b_line, exc)
                    continue
                try:
                    labels = join_labels(a, b)
                except BAD_ROW as exc:
                    rejected(counts, "incompatible_pair", f"A{a_line}/B{b_line}", exc)
                    deferred.append((lang_b, b_line, b_offset))
                    continue
                try:
                    pcm_b = read_audio(b.path)
                except BAD_ROW as exc:
                    rejected(counts, "invalid_B", b_line, exc)
                    continue
                write_pair(out, audio_dir, a_line, b_line, pcm_a, pcm_b, labels)
                counts["written"] += 1
                switches[f"{lang_a}->{lang_b}"] += 1
                if counts["written"] % 1000 == 0:
                    print(f"Joined {counts['written']} pairs", flush=True)
                break
            for lang, line, pos in deferred:
                buckets[lang].append((line, pos))
        counts["unused_B"] = sum(map(len, buckets.values()))
    print(f"Wrote {counts['written']} pairs to {destination}; WAVs in {audio_dir}")
    print(json.dumps({**counts, "switches": dict(switches)}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path, help="A JSONL, first clips")
    parser.add_argument("b", type=Path, help="B JSONL, second clips")
    parser.add_argument("output", type=Path, help="new output JSONL")
    parser.add_argument("--audio-dir", type=Path, default=Path(DEFAULT_AUDIO_DIR))
    parser.add_argument("--seed", type=int, default=42)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
