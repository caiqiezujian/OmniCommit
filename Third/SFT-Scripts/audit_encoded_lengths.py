#!/usr/bin/env python3
"""Validate every Third assistant payload and loss mask with the SWIFT processor."""
from pathlib import Path
import runpy
import sys

THIRD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIRD))
from third_asr import sft_checks


def main(argv=None):
    shared = runpy.run_path(str(sft_checks.SHARED_SFT / "audit_encoded_lengths.py"))
    return shared["main"](argv, checks=sft_checks, template_name="omnicommit_third_qwen3_omni",
                          plugin_name="omnicommit_third_sft_plugin")


if __name__ == "__main__":
    sys.exit(main())
