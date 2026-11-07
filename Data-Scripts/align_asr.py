#!/usr/bin/env python3
"""Join annotated source units with real acoustic timestamps."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omnicommit.alignment import main

if __name__ == "__main__":
    raise SystemExit(main())
