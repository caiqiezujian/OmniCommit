#!/usr/bin/env python3
"""Sample known non-speech recordings and build all-WAIT Fourth streaming SFT rows."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
from stat import S_ISREG
import sys
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_asr.protocol import get_protocol
from fourth_asr.recast import build_sft_row, validate_chunk_ms


AUDIO_SUFFIXES = {".wav", ".wave", ".flac", ".ogg", ".oga", ".opus", ".mp3", ".aif", ".aiff", ".au", ".caf"}


def audio_dimensions(path):
    """Check that the training reader can seek and decode this file, bounded in memory."""
    with sf.SoundFile(str(path)) as audio:
        frames, rate = audio.frames, audio.samplerate
        if frames < 1 or rate < 1 or not audio.seekable():
            raise ValueError("expected nonempty, seekable audio")
        decoded = 0
        for block in audio.blocks(blocksize=65536, dtype="float32", always_2d=True):
            if not np.isfinite(block).all():
                raise ValueError("non-finite audio samples")
            decoded += len(block)
        if decoded != frames:
            raise ValueError(f"decoded {decoded} frames, header declares {frames}")
    return frames, rate


def generate(folder, output_dir, count, seed, chunk_ms, number_format):
    validate_chunk_ms(chunk_ms)
    if count < 1 or not folder.is_dir():
        raise ValueError("provide an existing audio directory and a positive --count")
    output = output_dir / f"train.{chunk_ms}ms.jsonl"
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    protocol = get_protocol("speaker")
    candidates = [Path(root) / name for root, _, files in os.walk(folder)
                  for name in files if Path(name).suffix.lower() in AUDIO_SUFFIXES]
    candidates.sort(key=Path.as_posix)
    random.Random(seed).shuffle(candidates)
    print(f"[silence] {len(candidates)} candidate files | target {count} | seed {seed}", flush=True)

    seen = set()
    kept = rejected = audio_turns = 0
    seconds = 0.0
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir, prefix=".silence-sft-") as staging:
        dataset = Path(staging) / "dataset"
        with dataset.open("w", encoding="utf-8", newline="\n") as writer:
            for candidate in candidates:
                try:
                    path = candidate.resolve(strict=True)
                    stat = path.stat()
                    if not S_ISREG(stat.st_mode):
                        raise ValueError("expected a regular audio file")
                    identity = (stat.st_dev, stat.st_ino) if stat.st_ino else path
                    if identity in seen:
                        continue
                    seen.add(identity)
                    frames, rate = audio_dimensions(path)
                    row = build_sft_row([], path, frames, rate, chunk_ms, number_format,
                                        protocol=protocol, silence_only=True)
                except (ValueError, OSError, RuntimeError) as exc:
                    rejected += 1
                    if rejected <= 10:
                        print(f"[skip] {candidate}: {exc}", file=sys.stderr)
                    continue
                writer.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                kept += 1
                audio_turns += len(row["audios"])
                seconds += frames / rate
                if kept % 500 == 0:
                    print(f"[silence] {kept}/{count} recordings", flush=True)
                if kept == count:
                    break
        if kept < count:
            raise ValueError(f"Only {kept} usable distinct recordings for --count {count} ({rejected} invalid)")

        # Match build_sft.py's two token files and publish the complete dataset last.
        inventory = ("\n".join(protocol.tokens) + "\n").encode("utf-8")
        for destination in (output_dir / "token.txt", Path(str(output) + ".tokens.txt")):
            pending = Path(staging) / destination.name
            pending.write_bytes(inventory)
            try:
                os.link(pending, destination)
            except FileExistsError:
                if destination.read_text(encoding="utf-8-sig").splitlines() != list(protocol.tokens):
                    raise ValueError(f"Existing token inventory differs from Fourth speaker protocol: {destination}")
        os.link(dataset, output)
    print(f"[silence] {kept} recordings | {seconds:.2f} s | {audio_turns} audio WAITs + {kept} end WAITs "
          f"| {rejected} invalid encountered | {output}", flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_dir", type=Path, help="Directory of known non-speech audio, searched recursively")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for train.<chunk>ms.jsonl and its token files")
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-ms", type=int, default=1000)
    parser.add_argument("--number-format", choices=("normalized", "verbatim"), default="normalized")
    args = parser.parse_args(argv)
    try:
        generate(args.audio_dir, args.output_dir, args.count, args.seed, args.chunk_ms, args.number_format)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"[silence] {exc}\n")


if __name__ == "__main__":
    main()
