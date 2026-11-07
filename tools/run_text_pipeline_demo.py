#!/usr/bin/env python3
"""CPU wiring demo: scripted annotation/times -> recast -> SFT -> stream replay.

The waveform is a synthetic tone, NOT the spoken transcript. Outputs are for
testing code only, never training or measuring recognition/diarization quality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SFT-Scripts"))

import numpy as np
import soundfile as sf

from omnicommit.alignment import align_annotation
from omnicommit.recast import recast_row
from omnicommit.streaming import ReplayBackend, StreamOptions, StreamSession
from prepare_sft import prepare_row, sort_tokens
from check_sft import check_row


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def build_demo(directory, chunk_seconds=.5):
    directory = Path(directory).resolve()
    if directory.exists():
        raise ValueError("Choose a new demo output directory; existing results are not overwritten")
    directory.mkdir(parents=True)
    audio = directory / "synthetic_not_speech.wav"
    rate = 16000
    pcm = (.04 * np.sin(2 * np.pi * 220 * np.arange(rate * 6) / rate)).astype("float32")
    sf.write(str(audio), pcm, rate, subtype="FLOAT")
    # Uneven semantic times deliberately cross .5/1s physical boundaries.
    units = [
        ("<zh><spk1>请联系", "DIRECT", .08, .31),
        ("张伟", "WAIT", .32, .82),
        ("确认预算。", "DIRECT", .83, 1.19),
        ("100万美元", "WAIT", 1.20, 2.27),
        ("已经批准。", "DIRECT", 2.28, 2.43),
        ("<en><spk1>Please confirm ", "WAIT", 2.60, 2.98),
        ("the schedule.", "DIRECT", 3.01, 3.43),
        ("<en><spk2>I agree. ", "WAIT", 3.60, 3.87),
        ("We can proceed.", "DIRECT", 3.88, 4.30),
        ("<zh><spk1>明天下午3点", "WAIT", 4.60, 5.23),
        ("再讨论。", "DIRECT", 5.24, 5.80),
    ]
    source = "".join(u[0] for u in units)
    annotation = {"segments": [{"source_text": text, "action": action} for text, action, _, _ in units],
                  "flush": False, "error": None}
    row = {"id": "synthetic-wiring-only", "synthetic_not_for_training": True,
           "audios": [str(audio)], "messages": [
               {"role": "system", "content": "Synthetic wiring fixture only."},
               {"role": "user", "content": "<audio>"},
               {"role": "assistant", "content": source}],
           "output": json.dumps(annotation, ensure_ascii=False)}
    import re
    timings = {"id": row["id"], "audio": str(audio), "backend": "synthetic-scripted-not-an-aligner",
               "words": [{"text": re.sub(r"<[^<>]+>", "", text), "start": start, "end": end}
                         for text, _, start, end in units]}
    write_jsonl(directory / "annotated.jsonl", [row])
    write_jsonl(directory / "timestamps.jsonl", [timings])
    aligned = align_annotation(row, timings)
    trajectory = recast_row(aligned, chunk_seconds=chunk_seconds, number_format="normalized")
    prepared, inventory = prepare_row(trajectory, directory)
    check_row(prepared, decode=True, token_inventory=inventory)
    write_jsonl(directory / "aligned.jsonl", [aligned])
    write_jsonl(directory / "trajectory.jsonl", [trajectory])
    write_jsonl(directory / "sft.jsonl", [prepared])
    (directory / "sft.jsonl.tokens.txt").write_text("\n".join(sort_tokens(inventory)) + "\n", encoding="utf-8")
    replies = [m["content"] for m in prepared["messages"] if m["role"] == "assistant"]
    (directory / "replay-replies.json").write_text(json.dumps(replies, ensure_ascii=False, indent=2), encoding="utf-8")
    session = StreamSession(ReplayBackend(replies), StreamOptions(chunk_seconds=chunk_seconds, number_format="normalized"))
    events = []
    try:
        step = round(rate * chunk_seconds)
        for start in range(0, len(pcm), step):
            events.append(session.step(pcm[start:start + step]))
        events.append(session.step(end_of_audio=True))
    finally:
        session.close()
    write_jsonl(directory / "replay-events.jsonl", events)
    (directory / "DO_NOT_TRAIN.txt").write_text(
        "Synthetic tone + scripted transcript/timestamps/replies. Code wiring only.\n"
        "No ASR, voice identity, forced alignment, GPU training or real-time performance was measured.\n", encoding="utf-8")
    report = {"synthetic_not_for_training": True, "chunk_seconds": chunk_seconds,
              "audio_rounds": len(events) - 1, "eos_rounds": 1,
              "direct_rounds": sum(e["action"] == "DIRECT" for e in events),
              "wait_rounds": sum(e["action"] == "WAIT" for e in events),
              "source_preserved": "".join(s["text"] for e in events for s in e["segments"]) == re.sub(r"<[^<>]+>", "", source),
              "output_dir": str(directory)}
    (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-seconds", type=float, choices=[.5, 1.0], default=.5)
    args = parser.parse_args(argv)
    print(json.dumps(build_demo(args.output_dir, args.chunk_seconds), ensure_ascii=False))


if __name__ == "__main__":
    main()
