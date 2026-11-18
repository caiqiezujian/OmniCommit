#!/usr/bin/env python3
"""Convert compact Second or Third SFT to a Fourth protocol without reopening the WAVs."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

FOURTH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FOURTH))

from fourth_asr.chunk_refs import decode_chunk_ref
from fourth_asr.protocol import EOS, PROTOCOLS, WAIT, get_protocol
from fourth_asr.timestamped_source import LANGUAGE, SPEAKER

SOURCE_PROMPTS = {}
for fmt, numbers in (("normalized", "Use digits for numbers."), ("verbatim", "Preserve spoken numbers.")):
    legacy = ("Transcribe new speech. Mark first text and changes with <language><speaker>. "
              f"Emit {WAIT} if nothing is ready. {numbers}")
    SOURCE_PROMPTS[legacy] = ("second", fmt)
    for profile in PROTOCOLS.values():
        SOURCE_PROMPTS[profile.system_prompt(fmt)] = (profile.name, fmt)

SECOND_HEADER = re.compile(rf"<({LANGUAGE})><({SPEAKER})>")


def second_text(reply, state):
    """Read the attributed Second wire format, including inherited headers."""
    if reply == WAIT:
        return "", state
    if not isinstance(reply, str) or not reply:
        raise ValueError("Expected Second text or standalone WAIT")
    pieces = []
    pos = 0
    while pos < len(reply):
        if reply[pos] == "<":
            header = SECOND_HEADER.match(reply, pos)
            if not header or state == header.groups():
                raise ValueError("Invalid or redundant Second language/speaker header")
            state = header.groups()
            pos = header.end()
        if state is None:
            raise ValueError("First Second transcript text requires language and speaker")
        end = reply.find("<", pos)
        end = len(reply) if end == -1 else end
        part = reply[pos:end]
        if not part or ">" in part:
            raise ValueError("Invalid text after Second header")
        pieces.append(part)
        pos = end
    return "".join(pieces), state


def convert_record(row, protocol=get_protocol("explicit")):
    if not isinstance(row, dict) or set(row) != {"messages", "audios"}:
        raise ValueError("Expected a compact SFT row with audios and messages")
    refs, messages = row["audios"], row["messages"]
    if not isinstance(refs, list) or not refs:
        raise ValueError("Expected at least one virtual audio reference")
    if (not isinstance(messages, list) or len(messages) != 1 + 2 * (len(refs) + 1) or
            not all(isinstance(message, dict) for message in messages)):
        raise ValueError("Expected one system message, one pair per audio and one EOS pair")
    header = messages[0]
    instruction = header.get("content")
    if header.get("role") != "system" or not isinstance(instruction, str) or instruction not in SOURCE_PROMPTS:
        raise ValueError("Unrecognized Second or Third system message")
    source, number_format = SOURCE_PROMPTS[instruction]
    end_user, end_assistant = messages[-2:]
    if (end_user.get("role") != "user" or end_user.get("content") != EOS or
            end_assistant.get("role") != "assistant" or end_assistant.get("content") != WAIT):
        raise ValueError("Expected end_of_audio followed by standalone WAIT")
    converted = [{"role": "system", "content": protocol.system_prompt(number_format)}]
    source_path = None
    previous_end = 0.
    state = None
    has_speech = False
    for i, ref in enumerate(refs):
        segment = decode_chunk_ref(ref)
        if i == 0:
            source_path = segment["path"]
            if not (PurePosixPath(source_path).is_absolute() or PureWindowsPath(source_path).is_absolute()):
                raise ValueError("Source audio path must be absolute on the training host")
        if segment["path"] != source_path or abs(segment["start"] - previous_end) > 1e-9:
            raise ValueError("Audio references must cover one recording contiguously from zero")
        previous_end = segment["end"]
        user, assistant = messages[1 + 2*i:3 + 2*i]
        if user.get("role") != "user" or user.get("content") != "<audio>" or assistant.get("role") != "assistant":
            raise ValueError(f"Invalid user/assistant pair at audio turn {i + 1}")
        if source == "second":
            text, state = second_text(assistant.get("content"), state)
        else:
            text = get_protocol(source).parse_reply(assistant.get("content"))
        has_speech = has_speech or bool(text)
        converted.extend((user, {"role": "assistant", "content": protocol.format_text(text) if text else WAIT}))
    if not has_speech:
        raise ValueError("Recording contains no transcribed speech")
    converted.extend((end_user, {"role": "assistant", "content": WAIT}))
    return {"audios": refs, "messages": converted}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Existing compact Second or Third SFT JSONL")
    parser.add_argument("--output", type=Path, required=True, help="New Fourth SFT JSONL")
    parser.add_argument("--protocol", choices=tuple(PROTOCOLS), default="explicit",
                        help="Training output format (default: %(default)s)")
    args = parser.parse_args(argv)
    protocol = get_protocol(args.protocol)
    if not args.input.is_file():
        parser.error(f"Input does not exist: {args.input}")
    output = args.output
    tokens = Path(str(output) + ".tokens.txt")
    rejected = Path(str(output) + ".rejected.jsonl")
    for path in (output, tokens, rejected):
        if path.exists():
            parser.error(f"Output already exists: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    kept = bad = 0
    with args.input.open("rb") as source, output.open("xb") as target, \
            rejected.open("x", encoding="utf-8", newline="\n") as errors:
        for line, raw in enumerate(source, 1):
            try:
                row = convert_record(json.loads(raw.decode("utf-8-sig" if line == 1 else "utf-8")), protocol)
                payload = (json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                           + "\n").encode("utf-8")
            except (ValueError, TypeError, RecursionError, OverflowError) as exc:
                errors.write(json.dumps({"line": line, "error": str(exc)}, ensure_ascii=True) + "\n")
                bad += 1
            else:
                target.write(payload)
                kept += 1
            if line == 1 or line % 1000 == 0:
                target.flush()
                errors.flush()
                print(f"Rows {line} | kept {kept} | rejected {bad}", file=sys.stderr, flush=True)
    if kept:
        with tokens.open("x", encoding="utf-8", newline="\n") as handle:
            handle.writelines(token + "\n" for token in protocol.tokens)
    print(f"Finished | kept {kept} | rejected {bad} | {output}", file=sys.stderr, flush=True)
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
