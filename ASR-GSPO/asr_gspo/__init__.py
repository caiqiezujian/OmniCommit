"""Online reinforcement learning for OmniCommit Third streaming ASR."""

from pathlib import Path
import sys

# Use the SFT and serving protocol from the same OmniCommit checkout.
_third_dir = str(Path(__file__).resolve().parents[2] / "Third")
if _third_dir not in sys.path:
    sys.path.insert(0, _third_dir)
