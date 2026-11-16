"""Validate Third supervision against the token inventory used for training."""
from pathlib import Path
import sys

from .protocol import DIRECT, get_protocol, protocol_for_tokens

SHARED_SFT = Path(__file__).resolve().parents[2] / "Second" / "SFT-Scripts"
sys.path.append(str(SHARED_SFT))
from omnicommit_streaming_checks import check_streaming_row, run_check


def read_tokens(path):
    tokens = Path(path).read_text(encoding="utf-8-sig").splitlines()
    return set(protocol_for_tokens(tokens).tokens)


def validate_row(row, token_inventory=None, decode=False):
    # Dataset/file entry points always supply the sidecar; legacy in-process
    # callers without an inventory use the original plain-text protocol.
    explicit = token_inventory is not None and DIRECT in token_inventory
    profile = get_protocol("explicit" if explicit else "plain")
    if token_inventory is not None and set(token_inventory) != set(profile.tokens):
        raise ValueError("Unexpected Third token inventory")

    def inspect_reply(reply, state):
        text = profile.parse_reply(reply)
        return ((DIRECT,) if explicit and text else ()), True if text else state

    return check_streaming_row(row, token_inventory, decode, system_prompt=profile.system_prompt,
                               inspect_reply=inspect_reply, protocol_name="Third")


def main(argv=None):
    return run_check(argv, read_tokens=read_tokens, validate_row=validate_row,
                     template_name="omnicommit_third_qwen3_omni",
                     plugin_name="omnicommit_third_sft_plugin", description=__doc__)
