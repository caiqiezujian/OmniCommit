#!/usr/bin/env python3
"""Convert compact Second or Third SFT to a Third protocol without reopening the WAVs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIRD = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(THIRD), str(THIRD.parent / "Second")]

from second_asr.protocol import parse_reply as parse_second
from second_asr.protocol import system_prompt as second_prompt
from third_asr.protocol import PROTOCOLS, WAIT, get_protocol
from third_asr.sft_checks import validate_row

SOURCE_PROMPTS = {}
for fmt in ("normalized", "verbatim"):
    SOURCE_PROMPTS[second_prompt(fmt)] = ("second", fmt)
    for profile in PROTOCOLS.values():
        SOURCE_PROMPTS[profile.system_prompt(fmt)] = (profile.name, fmt)


def convert_record(row, protocol=get_protocol("explicit")):
    if not isinstance(row, dict) or set(row) != {"messages", "audios"}:
        raise ValueError("Expected a compact SFT row with audios and messages")
    messages = row["messages"]
    if (not isinstance(messages, list) or not messages or
            not all(isinstance(message, dict) for message in messages)):
        raise ValueError("Expected SFT chat messages")
    header = messages[0]
    instruction = header.get("content")
    if header.get("role") != "system" or not isinstance(instruction, str) or instruction not in SOURCE_PROMPTS:
        raise ValueError("Unrecognized Second or Third system message")
    source, number_format = SOURCE_PROMPTS[instruction]
    converted = [{"role": "system", "content": protocol.system_prompt(number_format)}]
    state = None
    for message in messages[1:]:
        if message.get("role") != "assistant":
            converted.append(message)
            continue
        if source == "second":
            spans, state = parse_second(message.get("content"), state)
            text = "".join(span[2] for span in spans)
        else:
            text = get_protocol(source).parse_reply(message.get("content"))
        converted.append({"role": "assistant", "content": protocol.format_text(text) if text else WAIT})
    result = {"audios": row["audios"], "messages": converted}
    validate_row(result, token_inventory=set(protocol.tokens))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Existing compact Second or Third SFT JSONL")
    parser.add_argument("--output", type=Path, required=True, help="New Third SFT JSONL")
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
