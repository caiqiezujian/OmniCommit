"""Validate Fourth streaming ASR targets, audio references and optional decoded audio."""
import argparse
import io
import json
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

from .chunk_refs import decode_chunk_ref
from .protocol import DIRECT, EOS, WAIT, get_protocol, protocol_for_tokens
from .training_audio import SAMPLE_RATE, chunk_wav_bytes


def read_tokens(path):
    tokens = Path(path).read_text(encoding="utf-8-sig").splitlines()
    return set(protocol_for_tokens(tokens).tokens)


def _content(message, role):
    if not isinstance(message, dict) or message.get("role") != role:
        raise ValueError(f"Expected a {role} message")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError(f"{role} content must be a string")
    return content


def validate_row(row, token_inventory=None, decode=False):
    """Validate one recording from fresh state and return the special tokens it uses."""
    explicit = token_inventory is not None and DIRECT in token_inventory
    profile = get_protocol("explicit" if explicit else "plain")
    if token_inventory is not None and set(token_inventory) != set(profile.tokens):
        raise ValueError("Unexpected Fourth token inventory")

    if not isinstance(row, dict) or set(row) != {"audios", "messages"}:
        raise ValueError("Expected a compact Fourth row with audios and messages")
    refs, messages = row["audios"], row["messages"]
    if not isinstance(refs, list) or not refs:
        raise ValueError("audios must be a nonempty array of virtual references")
    if not isinstance(messages, list) or len(messages) != 1 + 2 * (len(refs) + 1):
        raise ValueError("Expected system, one message pair per audio, and an EOS pair")
    if _content(messages[0], "system") not in (profile.system_prompt("verbatim"), profile.system_prompt("normalized")):
        raise ValueError("Unexpected Fourth system instruction")
    if _content(messages[-2], "user") != EOS or _content(messages[-1], "assistant") != WAIT:
        raise ValueError("Expected final text-only end_of_audio followed by standalone WAIT")

    source_path = decode_chunk_ref(refs[0])["path"]
    if not (PurePosixPath(source_path).is_absolute() or PureWindowsPath(source_path).is_absolute()):
        raise ValueError("Source audio path must be absolute on the training host")
    if decode:
        import soundfile as sf
        source_info = sf.info(source_path)
    tokens, has_text, previous_end = {WAIT}, False, 0
    for i, ref in enumerate(refs):
        try:
            segment = decode_chunk_ref(ref)
            if segment["path"] != source_path or abs(segment["start"] - previous_end) > 1e-9:
                raise ValueError("Audio references must cover one recording contiguously from zero")
            previous_end = segment["end"]
            if _content(messages[1 + 2*i], "user") != "<audio>":
                raise ValueError("Expected one live audio placeholder")
            text = profile.parse_reply(_content(messages[2 + 2*i], "assistant"))
            if text:
                has_text = True
                if explicit:
                    tokens.add(DIRECT)
            if decode:
                with sf.SoundFile(io.BytesIO(chunk_wav_bytes(ref))) as chunk:
                    if (chunk.samplerate != SAMPLE_RATE or chunk.channels != 1
                            or len(chunk) != round(segment["input_seconds"] * SAMPLE_RATE)):
                        raise ValueError("Unexpected decoded audio format or frame count")
        except (ValueError, OSError, RuntimeError) as exc:
            raise ValueError(f"audio turn {i+1}: {exc}") from exc
    if not has_text:
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
    parser.add_argument("--framework", action="store_true", help="Verify that ms-swift registers the training template")
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
            from importlib import import_module
            template_name = "omnicommit_third_qwen3_omni"
            if template_name not in import_module("omnicommit_fourth_sft_plugin").TEMPLATE_MAPPING:
                raise ValueError(f"Training template was not registered: {template_name}")
    except Exception as exc:
        print(f"[check] {exc}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps({"checked_rows": count, "full_dataset_checked": full_dataset,
                      "decoded": args.decode, "framework_checked": args.framework}))
    return 0
