#!/usr/bin/env python3
"""Export a Third PEFT checkpoint and verify each trained action-token row."""
from pathlib import Path
import runpy
import sys

THIRD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIRD))
from third_asr.service_backend import read_export_protocol


def main(argv=None):
    shared = runpy.run_path(str(THIRD.parent / "SFT-Scripts" / "merge_lora.py"))
    return shared["main"](argv, protocol_reader=read_export_protocol)


if __name__ == "__main__":
    sys.exit(main())
