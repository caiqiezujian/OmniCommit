#!/usr/bin/env python3
"""Build fixed 0.5/1 second streaming supervision with explicit EOS."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omnicommit.recast import main

if __name__ == "__main__":
    raise SystemExit(main())
