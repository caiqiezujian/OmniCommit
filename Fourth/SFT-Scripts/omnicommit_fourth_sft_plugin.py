"""Fourth's streaming Qwen3-Omni SWIFT template and audio loader."""
from pathlib import Path
from copy import deepcopy
import os
import sys

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir.parent))
from fourth_asr.chunk_refs import PREFIX
from fourth_asr.training_audio import SAMPLE_RATE, materialize_chunk_refs

if os.environ.get("OMNICOMMIT_ACTION_ROWS") == "1":
    from fourth_asr.mcore_action_rows import install_action_rows
    install_action_rows()

if os.environ.get("OMNICOMMIT_SFT_ATTENTION_COMPAT") == "1":
    from omnicommit_attention_compat import install_attention_compat, install_swift_loader_compat
    install_attention_compat()
    install_swift_loader_compat()

if os.environ.get("OMNICOMMIT_PACKED_NO_FLASH") == "1":
    from omnicommit_packed_adapter import install_packed_no_flash
    install_packed_no_flash()

try:
    from swift.template.register import TEMPLATE_MAPPING, register_template
    from swift.template.templates.qwen import Qwen3OmniTemplate
except ModuleNotFoundError as exc:
    # Only fall back when the new namespace is absent. Do not hide a missing
    # dependency inside an installed new-style SWIFT package.
    if exc.name not in ("swift.template", "swift.template.register", "swift.template.templates", "swift.template.templates.qwen"):
        raise
    from swift.llm.template.register import TEMPLATE_MAPPING, register_template
    from swift.llm.template.template.qwen import Qwen3OmniTemplate

if not hasattr(Qwen3OmniTemplate, "_preprocess_inputs"):
    raise RuntimeError("Installed SWIFT lacks the required _preprocess_inputs template hook")


class OmniCommitQwen3OmniTemplate(Qwen3OmniTemplate):
    def _preprocess_inputs(self, inputs):
        audios = inputs.audios
        virtual = [isinstance(value, str) and value.startswith(PREFIX) for value in audios]
        if any(virtual):
            if not all(virtual):
                raise ValueError("Do not mix full recordings and virtual chunks in a single session")
            sr = getattr(self, "sampling_rate", SAMPLE_RATE)
            if sr != SAMPLE_RATE:
                raise ValueError(f"OMNICOMMIT virtual chunks require a 16000 Hz processor, got {sr}")
            # Only the selected waveform enters the native feature extractor.
            # Keeping bytes here also preserves old SWIFT loader compatibility.
            inputs.audios = materialize_chunk_refs(audios)
        return super()._preprocess_inputs(inputs)


meta = deepcopy(TEMPLATE_MAPPING["qwen3_omni"])
meta.template_type = "omnicommit_third_qwen3_omni"
meta.template_cls = OmniCommitQwen3OmniTemplate
register_template(meta)
