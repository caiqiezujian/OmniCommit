"""Online reinforcement learning for OmniCommit Fourth streaming ASR."""

from pathlib import Path
import sys

# Use the SFT and serving protocol from the same OmniCommit checkout.
_fourth_dir = str(Path(__file__).resolve().parents[2] / "Fourth")
if _fourth_dir not in sys.path:
    sys.path.insert(0, _fourth_dir)
