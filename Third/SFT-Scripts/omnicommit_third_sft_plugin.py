"""Third's SWIFT registration for streaming Qwen3-Omni SFT."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Second" / "SFT-Scripts"))
from omnicommit_streaming_template import TEMPLATE_MAPPING, register_streaming_template

register_streaming_template("omnicommit_third_qwen3_omni")
