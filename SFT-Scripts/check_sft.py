#!/usr/bin/env python3
"""Read-only SFT sample preflight; no weights, teacher calls or GPU training."""
import argparse
import importlib.metadata
import io
import json
from pathlib import Path
import re
import sys

from omnicommit_audio import SAMPLE_RATE, decode_chunk_ref, chunk_wav_bytes
from prepare_sft import ACTION_TOKENS, LANGUAGE, SPEAKER, number


def check_content(content):
    if content == "<|WAIT|>":
        return {"<|WAIT|>"}
    if not isinstance(content, str) or not content.startswith("<|DIRECT|>"):
        raise ValueError("Assistant must return DIRECT with tagged text or exactly WAIT")
    tokens = {"<|DIRECT|>"}
    body = content[len("<|DIRECT|>"):]
    for line in body.split("\n"):
        match = re.fullmatch(r"<([^<>]+)><([^<>]+)>([^<>\r\n]+)", line)
        if not match:
            raise ValueError("Each DIRECT segment must be <language><speaker>nonempty-text")
        language, speaker, text = match.groups()
        if not LANGUAGE.fullmatch(language) or not SPEAKER.fullmatch(speaker) or not text.strip():
            raise ValueError("Invalid segment language, speaker or empty text")
        tokens.update((f"<{language}>", f"<{speaker}>"))
    return tokens


def check_row(row, decode=True, token_inventory=None):
    import soundfile as sf

    if not isinstance(row, dict) or row.get("schema_version") != 1:
        raise ValueError("Expected a schema_version=1 SFT object")
    refs, durations, messages = (row.get(k) for k in ("audios", "audio_chunk_durations", "messages"))
    if not isinstance(refs, list) or not refs:
        raise ValueError("audios must be a nonempty array of virtual references")
    if not isinstance(durations, list) or len(refs) != len(durations):
        raise ValueError("audio_chunk_durations must match audios")
    if not isinstance(messages, list) or len(messages) != 1 + 2 * len(refs):
        raise ValueError("Expected one system and one user/assistant pair per audio")
    if messages[0].get("role") != "system" or not isinstance(messages[0].get("content"), str) or not messages[0]["content"].strip():
        raise ValueError("Missing nonempty system message")
    previous_end, source_path, direct_count = 0., None, 0
    samples, used_tokens = [], set()
    actual_seconds = None
    for i, ref in enumerate(refs):
        data = decode_chunk_ref(ref)
        if source_path is None:
            source_path = data["path"]
            if not Path(source_path).is_absolute():
                raise ValueError("Source audio must use an absolute training-host path")
            info = sf.info(source_path)
            actual_seconds = info.frames / info.samplerate
        if data["path"] != source_path or abs(data["start"] - previous_end) > 1e-6:
            raise ValueError(f"turn {i}: audio ranges must be contiguous and use one source")
        if data["end"] > actual_seconds + 1e-8:
            raise ValueError(f"turn {i}: source range exceeds actual duration")
        if abs(number(durations[i], "audio_chunk_durations") - data["input_seconds"]) > 1e-8:
            raise ValueError(f"turn {i}: virtual and declared durations differ")
        user, assistant = messages[1 + 2*i:3 + 2*i]
        if user.get("role") != "user" or user.get("content") != "<audio>":
            raise ValueError(f"turn {i}: expected exactly one user <audio> placeholder")
        if assistant.get("role") != "assistant":
            raise ValueError(f"turn {i}: expected assistant")
        tokens = check_content(assistant.get("content"))
        used_tokens.update(tokens)
        direct_count += "<|DIRECT|>" in tokens
        if decode:
            pcm, sr = sf.read(io.BytesIO(chunk_wav_bytes(ref)), dtype="float32")
            if sr != SAMPLE_RATE or pcm.ndim != 1 or len(pcm) != round(data["input_seconds"] * sr):
                raise ValueError(f"turn {i}: invalid materialized waveform")
            samples.append(len(pcm))
        previous_end = data["end"]
    if abs(previous_end - actual_seconds) > 1e-6:
        raise ValueError("Final audio range must cover the actual recording end")
    if not direct_count:
        raise ValueError("All-WAIT sample has no transcription supervision")
    if token_inventory is not None and not used_tokens.issubset(token_inventory):
        raise ValueError(f"Missing special tokens: {sorted(used_tokens - token_inventory)}")
    return {"id": row.get("id"), "audio_turns": len(refs), "direct_turns": direct_count,
            "wait_turns": len(refs) - direct_count, "decoded": decode,
            "input_seconds": sum(durations), "sample_counts": samples}


def read_tokens(path):
    tokens = path.read_text(encoding="utf-8-sig").splitlines()
    if tokens[:2] != ACTION_TOKENS or len(tokens) != len(set(tokens)):
        raise ValueError("Token file must begin WAIT, DIRECT and contain no duplicate tokens")
    for token in tokens[2:]:
        if not token.startswith("<") or not token.endswith(">") or not (LANGUAGE.fullmatch(token[1:-1]) or SPEAKER.fullmatch(token[1:-1])):
            raise ValueError(f"Invalid language/speaker special token: {token!r}")
    return set(tokens)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--tokens-file", type=Path)
    parser.add_argument("--framework", action="store_true", help="Import SWIFT plugin without loading weights")
    args = parser.parse_args(argv)
    if args.rows < 1:
        parser.error("--rows must be positive")
    try:
        inventory = read_tokens(args.tokens_file or Path(str(args.dataset) + ".tokens.txt"))
        count = 0
        with args.dataset.open(encoding="utf-8-sig") as source:
            for line_no, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    result = check_row(json.loads(line), token_inventory=inventory)
                except Exception as exc:
                    raise ValueError(f"line {line_no}: {exc}") from exc
                print(json.dumps(result, ensure_ascii=False), flush=True)
                count += 1
                if count == args.rows:
                    break
        if not count:
            raise ValueError("Dataset has no samples")
        if args.framework:
            import omnicommit_sft_plugin
            print(f"[check] ms-swift={importlib.metadata.version('ms-swift')} "
                  f"template={omnicommit_sft_plugin._meta.template_type}", flush=True)
        print(json.dumps({"ok": True, "sampled_rows_checked": count, "full_dataset_checked": False,
                          "framework_import_checked": args.framework,
                          "real_template_encoding_checked": False, "gpu_training_verified": False}))
        return 0
    except Exception as exc:
        print(f"[check] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
