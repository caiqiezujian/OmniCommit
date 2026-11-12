#!/usr/bin/env python3
"""Validate the compact Second SFT format, optionally reading the actual audio chunks."""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

from omnicommit_audio import SAMPLE_RATE, chunk_wav_bytes, decode_chunk_ref

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from second_asr.protocol import EOS, WAIT, parse_reply, sort_tokens, system_prompt


def read_tokens(path):
    tokens = path.read_text(encoding="utf-8-sig").splitlines()
    if not tokens or tokens[0] != WAIT or len(tokens) != len(set(tokens)):
        raise ValueError("Token file must begin with WAIT and contain unique tokens")
    return set(sort_tokens(tokens))


def _content(message, role):
    if not isinstance(message, dict) or message.get("role") != role:
        raise ValueError(f"Expected a {role} message")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError(f"{role} content must be a string")
    return content


def validate_row(row, token_inventory=None, decode=False):
    """Validate one recording from fresh state and return the special tokens it uses."""
    if not isinstance(row, dict) or set(row) != {"audios", "messages"}:
        raise ValueError("Expected a compact Second row with audios and messages")
    refs, messages = row["audios"], row["messages"]
    if not isinstance(refs, list) or not refs:
        raise ValueError("audios must be a nonempty array of virtual references")
    if not isinstance(messages, list) or len(messages) != 1 + 2 * (len(refs) + 1):
        raise ValueError("Expected system, one message pair per audio, and an EOS pair")
    if _content(messages[0], "system") not in (system_prompt("verbatim"), system_prompt("normalized")):
        raise ValueError("Unexpected Second system instruction")
    if _content(messages[-2], "user") != EOS or _content(messages[-1], "assistant") != WAIT:
        raise ValueError("Expected final text-only end_of_audio followed by standalone WAIT")

    source_path = decode_chunk_ref(refs[0])["path"]
    if not (PurePosixPath(source_path).is_absolute() or PureWindowsPath(source_path).is_absolute()):
        raise ValueError("Source audio path must be absolute on the training host")
    if decode:
        import soundfile as sf
        source_info = sf.info(source_path)
    tokens, state, previous_end = {WAIT}, None, 0
    for i, ref in enumerate(refs):
        try:
            segment = decode_chunk_ref(ref)
            if segment["path"] != source_path or abs(segment["start"] - previous_end) > 1e-9:
                raise ValueError("Audio references must cover one recording contiguously from zero")
            previous_end = segment["end"]
            if _content(messages[1 + 2*i], "user") != "<audio>":
                raise ValueError("Expected one live audio placeholder")
            spans, state = parse_reply(_content(messages[2 + 2*i], "assistant"), state)
            for lang, speaker, _ in spans:
                tokens.update((f"<{lang}>", f"<{speaker}>"))
            if decode:
                with sf.SoundFile(io.BytesIO(chunk_wav_bytes(ref))) as chunk:
                    if (chunk.samplerate != SAMPLE_RATE or chunk.channels != 1
                            or len(chunk) != round(segment["input_seconds"] * SAMPLE_RATE)):
                        raise ValueError("Unexpected decoded audio format or frame count")
        except (ValueError, OSError, RuntimeError) as exc:
            raise ValueError(f"audio turn {i+1}: {exc}") from exc
    if state is None:
        raise ValueError("Recording contains no transcribed speech")
    if decode and round(previous_end * source_info.samplerate) != source_info.frames:
        raise ValueError("Final audio interval ends before or beyond the source recording end")
    if token_inventory is not None and not tokens.issubset(token_inventory):
        raise ValueError(f"Tokens missing from training token file: {sorted(tokens - set(token_inventory))}")
    return tokens


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--tokens-file", type=Path, help="Defaults to DATASET.tokens.txt")
    parser.add_argument("--rows", type=int, help="Check only the first N records (default: entire file)")
    parser.add_argument("--decode", action="store_true", help="Read and validate every audio crop in checked rows")
    parser.add_argument("--framework", action="store_true", help="Verify that ms-swift registers the Second template")
    args = parser.parse_args(argv)
    if args.rows is not None and args.rows < 1:
        parser.error("--rows must be a positive integer")
    count = 0
    full_dataset = True
    try:
        tokens = read_tokens(args.tokens_file or Path(str(args.dataset) + ".tokens.txt"))
        with args.dataset.open(encoding="utf-8-sig") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                if args.rows is not None and count >= args.rows:
                    full_dataset = False
                    break
                try:
                    validate_row(json.loads(line), tokens, args.decode)
                except Exception as exc:
                    raise ValueError(f"line {line_no}: {exc}") from exc
                count += 1
        if not count:
            raise ValueError("Dataset contains no records")
        if args.framework:
            from omnicommit_sft_plugin import TEMPLATE_MAPPING
            if "omnicommit_second_qwen3_omni" not in TEMPLATE_MAPPING:
                raise ValueError("Second training template was not registered")
    except Exception as exc:
        print(f"[check] {exc}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps({"checked_rows": count, "full_dataset_checked": full_dataset,
                      "decoded": args.decode, "framework_checked": args.framework}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
