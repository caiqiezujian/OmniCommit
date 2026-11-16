#!/usr/bin/env python3
"""Validate the compact Second SFT format, optionally reading the actual audio chunks."""
from __future__ import annotations

import sys
from pathlib import Path

from omnicommit_streaming_checks import check_streaming_row, run_check

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from second_asr.protocol import WAIT, parse_reply, sort_tokens, system_prompt


def read_tokens(path):
    tokens = path.read_text(encoding="utf-8-sig").splitlines()
    if not tokens or tokens[0] != WAIT or len(tokens) != len(set(tokens)):
        raise ValueError("Token file must begin with WAIT and contain unique tokens")
    return set(sort_tokens(tokens))


def _inspect_reply(reply, state):
    spans, state = parse_reply(reply, state)
    tokens = set()
    for lang, speaker, _ in spans:
        tokens.update((f"<{lang}>", f"<{speaker}>"))
    return tokens, state


def validate_row(row, token_inventory=None, decode=False):
    return check_streaming_row(row, token_inventory, decode, system_prompt=system_prompt,
                               inspect_reply=_inspect_reply, protocol_name="Second")


def main(argv=None):
    return run_check(argv, read_tokens=read_tokens, validate_row=validate_row,
                     template_name="omnicommit_second_qwen3_omni",
                     plugin_name="omnicommit_sft_plugin", description=__doc__)


if __name__ == "__main__":
    sys.exit(main())
