#!/usr/bin/env python3
"""Build 1-second Fourth SFT turns using the original timestamped segment boundaries."""
from build_sft import main as build_sft


def main(argv=None):
    return build_sft(argv, source_segments_only=True)


if __name__ == "__main__":
    raise SystemExit(main())
