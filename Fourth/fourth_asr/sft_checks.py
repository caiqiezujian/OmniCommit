"""Validate Fourth streaming ASR targets, audio references and optional decoded audio."""
import argparse
import io
import json
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

from .chunk_refs import decode_chunk_ref
from .protocol import DIRECT, EOS, PROTOCOLS, SPK_CHANGE, WAIT, protocol_for_tokens
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


def _check_recast(meta, path, turns):
    if not isinstance(meta, dict) or meta.get("source_audio") != path:
        raise ValueError("recast.source_audio must match the virtual audio references")
    for name, minimum in (("chunk_ms", 1), ("merged_units", 0)):
        if type(meta.get(name)) is not int or meta[name] < minimum:
            raise ValueError(f"Invalid recast.{name}")
    events = meta.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("recast.events must include an audio_end or annotation_end event")
    previous_turn = 0
    by_kind = {kind: set() for kind in ("speaker_change", "switch_deferred", "punctuation_deferred",
                                         "audio_end", "annotation_end")}
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Invalid recast.events entry")
        turn, kind = event.get("turn"), event.get("kind")
        if (type(turn) is not int or not 1 <= turn <= turns or turn < previous_turn
                or not isinstance(kind, str) or kind not in by_kind or turn in by_kind[kind]):
            raise ValueError("recast.events requires unique known events in audio turn order")
        previous_turn = turn
        by_kind[kind].add(turn)
    if turns in by_kind["switch_deferred"] or turns in by_kind["punctuation_deferred"]:
        raise ValueError("recast.events cannot defer text beyond the final audio turn")
    open_end = bool(by_kind["annotation_end"])
    if open_end:
        if by_kind["audio_end"] or by_kind["annotation_end"] != {turns} or events[-1]["kind"] != "annotation_end":
            raise ValueError("annotation_end must be the sole end event and occur last on the final audio turn")
        source_frames, supervised_frames = meta.get("source_frames"), meta.get("supervised_frames")
        if (type(source_frames) is not int or type(supervised_frames) is not int
                or not 0 < supervised_frames <= source_frames):
            raise ValueError("annotation_end requires integer frame counts with 0 < supervised_frames <= source_frames")
    elif by_kind["audio_end"] != {turns}:
        raise ValueError("recast.events must contain exactly one audio_end on the last audio turn")
    return open_end, by_kind["speaker_change"]


def validate_row(row, token_inventory=None, decode=False):
    """Validate one recording from fresh state and return the special tokens it uses."""
    if not isinstance(row, dict) or not {"audios", "messages"} <= row.keys() <= {"audios", "messages", "recast"}:
        raise ValueError("Expected a Fourth row with audios and messages and optional recast metadata")
    refs, messages = row["audios"], row["messages"]
    if not isinstance(refs, list) or not refs:
        raise ValueError("audios must be a nonempty array of virtual references")
    source_path = decode_chunk_ref(refs[0])["path"]
    if not (PurePosixPath(source_path).is_absolute() or PureWindowsPath(source_path).is_absolute()):
        raise ValueError("Source audio path must be absolute on the training host")
    open_end, expected_switches = False, None
    if "recast" in row:
        open_end, expected_switches = _check_recast(row["recast"], source_path, len(refs))
    expected_pairs = len(refs) if open_end else len(refs) + 1
    if not isinstance(messages, list) or len(messages) != 1 + 2 * expected_pairs:
        raise ValueError("Expected one message pair per audio and an EOS pair only for closed recordings")
    instruction = _content(messages[0], "system")
    profiles = PROTOCOLS.values()
    if token_inventory is not None:
        profiles = [p for p in profiles if set(p.tokens) == set(token_inventory)]
        if not profiles:
            raise ValueError("Unexpected Fourth token inventory")
    profile = next((p for p in profiles if instruction in (p.system_prompt(), p.system_prompt("normalized"))), None)
    if profile is None:
        raise ValueError("Unexpected Fourth system instruction")
    if not open_end and (_content(messages[-2], "user") != EOS or _content(messages[-1], "assistant") != WAIT):
        raise ValueError("Expected final text-only end_of_audio followed by standalone WAIT")
    if decode:
        import soundfile as sf
        source_info = sf.info(source_path)
    tokens = set() if open_end else {WAIT}
    has_text, previous_end = False, 0
    switch_turns = []
    for i, ref in enumerate(refs):
        try:
            segment = decode_chunk_ref(ref)
            if segment["path"] != source_path or abs(segment["start"] - previous_end) > 1e-9:
                raise ValueError("Audio references must cover one recording contiguously from zero")
            previous_end = segment["end"]
            if _content(messages[1 + 2*i], "user") != "<audio>":
                raise ValueError("Expected one live audio placeholder")
            reply = _content(messages[2 + 2*i], "assistant")
            text = profile.parse_reply(reply)
            if open_end and i == len(refs) - 1 and not (text and reply.startswith(DIRECT)):
                raise ValueError("annotation_end must end with a DIRECT containing transcript text")
            if text:
                if SPK_CHANGE in text:
                    if not has_text and text.lstrip().startswith(SPK_CHANGE):
                        raise ValueError("The initial speaker must not have a change marker")
                    switch_turns.append(i + 1)
                    tokens.add(SPK_CHANGE)
                has_text = True
                if profile.prefix:
                    tokens.add(DIRECT)
            else:
                tokens.add(WAIT)
            if decode:
                with sf.SoundFile(io.BytesIO(chunk_wav_bytes(ref))) as chunk:
                    if (chunk.samplerate != SAMPLE_RATE or chunk.channels != 1
                            or len(chunk) != round(segment["input_seconds"] * SAMPLE_RATE)):
                        raise ValueError("Unexpected decoded audio format or frame count")
        except (ValueError, OSError, RuntimeError) as exc:
            raise ValueError(f"audio turn {i+1}: {exc}") from exc
    if not has_text:
        raise ValueError("Recording contains no transcribed speech")
    if expected_switches is not None and expected_switches != set(switch_turns):
        raise ValueError("recast speaker_change events differ from the assistant replies")
    if decode:
        if open_end:
            meta = row["recast"]
            if meta["source_frames"] != source_info.frames:
                raise ValueError("annotation_end source_frames differ from the original audio header")
            if abs(previous_end * source_info.samplerate - meta["supervised_frames"]) > 1e-5:
                raise ValueError("Last audio reference does not end at supervised_frames of the original audio")
        elif round(previous_end * source_info.samplerate) != source_info.frames:
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
            template_name = "omnicommit_fourth_qwen3_omni"
            if template_name not in import_module("omnicommit_fourth_sft_plugin").TEMPLATE_MAPPING:
                raise ValueError(f"Training template was not registered: {template_name}")
    except Exception as exc:
        print(f"[check] {exc}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps({"checked_rows": count, "full_dataset_checked": full_dataset,
                      "decoded": args.decode, "framework_checked": args.framework}))
    return 0
