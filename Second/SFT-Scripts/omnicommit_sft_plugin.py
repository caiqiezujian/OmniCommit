"""Second's SWIFT registration for streaming Qwen3-Omni SFT."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from omnicommit_streaming_template import (
    TEMPLATE_MAPPING, OmniCommitQwen3OmniTemplate, register_streaming_template,
)

register_streaming_template("omnicommit_second_qwen3_omni")
