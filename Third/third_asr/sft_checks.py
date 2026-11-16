"""Validate the exact transcript-or-WAIT supervision used by Third."""
from pathlib import Path
import sys

from .protocol import WAIT, parse_reply, system_prompt

SHARED_SFT = Path(__file__).resolve().parents[2] / "Second" / "SFT-Scripts"
sys.path.append(str(SHARED_SFT))
from omnicommit_streaming_checks import check_streaming_row, run_check


def read_tokens(path):
    tokens = Path(path).read_text(encoding="utf-8-sig").splitlines()
    if tokens != [WAIT]:
        raise ValueError("Third requires exactly <|WAIT|> in the SFT token file")
    return {WAIT}


def _inspect_reply(reply, state):
    text = parse_reply(reply)
    return (), True if text else state


def validate_row(row, token_inventory=None, decode=False):
    if token_inventory is not None and set(token_inventory) != {WAIT}:
        raise ValueError("Third token inventory must contain only <|WAIT|>")
    return check_streaming_row(row, token_inventory, decode, system_prompt=system_prompt,
                               inspect_reply=_inspect_reply, protocol_name="Third")


def main(argv=None):
    return run_check(argv, read_tokens=read_tokens, validate_row=validate_row,
                     template_name="omnicommit_third_qwen3_omni",
                     plugin_name="omnicommit_third_sft_plugin", description=__doc__)
