#!/usr/bin/env python3
"""Convert reviewed OmniCommit trajectories to virtual-audio multi-turn SFT JSONL.

No teacher calls, alignment or target rewriting are performed here. Each audio
range is a NEW chunk, while prior audio/text remain earlier conversation turns.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

from omnicommit_audio import encode_chunk_ref, MAX_CHUNK_SECONDS

ACTION_TOKENS = ["<|WAIT|>", "<|DIRECT|>"]
LANGUAGE = re.compile(r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*")
SPEAKER = re.compile(r"spk[1-9][0-9]*")
SYSTEM_PROMPT = """You transcribe speech incrementally in its original language and identify its speaker.
Each user turn provides only the next audio chunk; use earlier audio and text as context.
Return only newly committed text, never repeat or revise previously committed text.
If new text and its language/speaker attribution are sufficiently supported, begin with <|DIRECT|>.
For every emitted segment, write <language><speaker>text, for example <zh><spk1>你好。
Use the same speaker ID across this entire conversation; changing language does not change speaker identity.
If no new segment can be committed, return exactly <|WAIT|>. Do not output empty speaker/language segments.
Release a confirmed prefix while deferring uncertain names, numeric expressions or attribution.
Preserve the spoken language and disfluencies. {number_instruction}
Use punctuation supported by the available audio. A chunk boundary alone is not a sentence boundary.
Within a DIRECT response, separate segments by a newline. Do not translate, explain or include timestamps."""


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def render_segments(segments):
    if not isinstance(segments, list) or not segments:
        raise ValueError("DIRECT requires nonempty segments")
    rendered, tokens = [], set()
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("segment must be an object")
        language, speaker, text = (segment.get(k) for k in ("language", "speaker", "text"))
        if not isinstance(language, str) or not LANGUAGE.fullmatch(language):
            raise ValueError(f"Invalid language code: {language!r}")
        if not isinstance(speaker, str) or not SPEAKER.fullmatch(speaker):
            raise ValueError(f"Invalid speaker ID: {speaker!r}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Each segment needs nonempty transcription text")
        if "<" in text or ">" in text or "\n" in text or "\r" in text:
            raise ValueError("Transcription text must not contain markup or embedded line breaks")
        rendered.append(f"<{language}><{speaker}>{text}")
        tokens.update((f"<{language}>", f"<{speaker}>"))
    return "\n".join(rendered), tokens


def prepare_row(row, audio_root=None):
    import soundfile as sf

    if not isinstance(row, dict) or row.get("schema_version") != 1:
        raise ValueError("Expected schema_version=1 trajectory object")
    if row.get("status", "ok") != "ok":
        raise ValueError("Cannot train on a failed trajectory")
    source = row.get("audio")
    if not isinstance(source, str) or not source:
        raise ValueError("audio must be a local source path")
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = Path(audio_root or Path.cwd()) / path
    path = path.resolve(strict=True)
    info = sf.info(str(path))
    actual_ms = info.frames * 1000 / info.samplerate
    duration_ms = number(row.get("duration_ms"), "duration_ms")
    # One source sample tolerance handles serialized timestamp rounding only.
    if duration_ms <= 0 or abs(duration_ms - actual_ms) > 1000 / info.samplerate + 1e-6:
        raise ValueError(f"duration_ms differs from actual audio duration: {duration_ms} vs {actual_ms}")
    turns = row.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("turns must be a nonempty array")
    policy = row.get("policy", {})
    if not isinstance(policy, dict):
        raise ValueError("policy must be an object")
    number_format = policy.get("number_format", "verbatim")
    if number_format == "verbatim":
        number_instruction = "Transcribe numbers in their spoken form (verbatim); do not convert them to digits."
    elif number_format == "normalized":
        number_instruction = "Normalize numerical expressions to digits only when their complete value is determined."
    else:
        raise ValueError("policy.number_format must be verbatim or normalized")
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(number_instruction=number_instruction)}]
    refs, durations, audio_segments, provenance = [], [], [], []
    inventory, previous_end, direct_count = set(ACTION_TOKENS), 0.0, 0
    for i, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError(f"turn {i} must be an object")
        start = number(turn.get("start_ms"), "start_ms")
        end = number(turn.get("end_ms"), "end_ms")
        if abs(start - previous_end) > 1e-6 or end <= start or end > duration_ms + 1e-6:
            raise ValueError(f"turn {i}: ranges must be contiguous, increasing and within duration")
        if end - start > MAX_CHUNK_SECONDS * 1000:
            raise ValueError(f"turn {i}: chunk exceeds {MAX_CHUNK_SECONDS} seconds")
        action, segments = turn.get("action"), turn.get("segments")
        if action == "WAIT":
            if segments != []:
                raise ValueError(f"turn {i}: WAIT requires segments=[]")
            content = ACTION_TOKENS[0]
        elif action == "DIRECT":
            text, tokens = render_segments(segments)
            content = ACTION_TOKENS[1] + text
            inventory.update(tokens)
            direct_count += 1
        else:
            raise ValueError(f"turn {i}: action must be DIRECT or WAIT")
        # Last timestamp can be rounded by <=1 source sample. Never read future
        # audio beyond the actual file. Pad the final tail to >=500 ms in memory
        # so a one-sample tail does not become an almost-empty encoder input.
        source_end = actual_ms / 1000 if i == len(turns) - 1 else min(end, actual_ms) / 1000
        if round(source_end * info.samplerate) <= round(start * info.samplerate / 1000):
            raise ValueError(f"turn {i}: range contains no source samples")
        input_seconds = max((source_end - start / 1000), .5 if i == len(turns) - 1 else .01)
        ref = encode_chunk_ref(path, start / 1000, source_end, input_seconds)
        refs.append(ref)
        durations.append(input_seconds)
        audio_segments.append({"path": str(path), "start": start / 1000,
                               "end": source_end, "input_seconds": input_seconds})
        messages.extend([{"role": "user", "content": "<audio>"},
                         {"role": "assistant", "content": content}])
        provenance.append({"turn_index": i, "start_ms": start, "end_ms": end,
                           "action": action, "segments": segments})
        previous_end = end
    if abs(previous_end - duration_ms) > 1e-6:
        raise ValueError("Final turn must cover the end of the audio")
    if not direct_count:
        raise ValueError("All-WAIT trajectory has no transcription supervision")
    prepared = {"schema_version": 1, "id": row.get("id"), "source_audio": str(path),
                "duration_ms": duration_ms, "audios": refs, "messages": messages,
                "audio_segments": audio_segments, "audio_chunk_durations": durations,
                "turn_provenance": provenance, "policy": dict(policy, number_format=number_format)}
    for field in ("provenance", "reference_audit"):
        if field in row:
            prepared[field] = row[field]
    return prepared, inventory


def sort_tokens(inventory):
    rest = set(inventory) - set(ACTION_TOKENS)
    languages = sorted(t for t in rest if not SPEAKER.fullmatch(t[1:-1]))
    speakers = sorted((t for t in rest if SPEAKER.fullmatch(t[1:-1])), key=lambda t: int(t[4:-1]))
    return ACTION_TOKENS + languages + speakers


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audio-root", type=Path, help="Resolve relative audio paths here (default: input parent)")
    parser.add_argument("--tokens-output", type=Path, help="Default: OUTPUT.tokens.txt")
    args = parser.parse_args(argv)
    tokens_path = args.tokens_output or Path(str(args.output) + ".tokens.txt")
    rejected_path = Path(str(args.output) + ".rejected.jsonl")
    summary_path = Path(str(args.output) + ".summary.json")
    paths = [args.output, tokens_path, rejected_path, summary_path]
    if len({p.resolve() for p in paths + [args.input]}) != len(paths) + 1:
        parser.error("Input/output/token/report paths must be distinct")
    if any(p.exists() for p in paths):
        parser.error("Output or sidecar already exists; choose a new output path")
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
    kept, rejected, inventory = 0, 0, set(ACTION_TOKENS)
    with args.input.open(encoding="utf-8-sig") as source, args.output.open("x", encoding="utf-8") as out, rejected_path.open("x", encoding="utf-8") as rejects:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = None
            try:
                row = json.loads(line)
                prepared, tokens = prepare_row(row, args.audio_root or args.input.resolve().parent)
                serialized = json.dumps(prepared, ensure_ascii=False, allow_nan=False)
            except Exception as exc:
                rejected += 1
                rejects.write(json.dumps({"line": line_no, "id": row.get("id") if isinstance(row, dict) else None,
                                          "error": str(exc)}, ensure_ascii=False) + "\n")
                rejects.flush()
                print(f"[reject] line={line_no}: {exc}", file=sys.stderr, flush=True)
                continue
            out.write(serialized + "\n")
            out.flush()
            kept += 1
            inventory.update(tokens)
    tokens_path.write_text("\n".join(sort_tokens(inventory)) + "\n", encoding="utf-8")
    summary = {"kept": kept, "rejected": rejected, "tokens": str(tokens_path),
               "full_dataset_checked": True, "gpu_training_verified": False}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
