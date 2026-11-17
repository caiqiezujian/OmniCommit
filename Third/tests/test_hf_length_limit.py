"""Verify the HF stopping token at the length boundary, without loading weights."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from third_asr.runtime.inference_backend import HFBackend
from third_asr.runtime.streaming import GenerationTruncated


def test_hf_length_limit_preserves_last_valid_stop_and_rejects_unterminated_reply():
    class Inputs(dict):
        def to(self, device):
            return self

    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, **kwargs):
            return Inputs(input_ids=np.array([[1, 2]]))

    backend = HFBackend.__new__(HFBackend)
    backend.processor = Processor()
    backend.max_context_tokens, backend.max_new_tokens = 16, 3
    backend.stop_ids, backend.wait_id = [98], 99
    backend.torch = SimpleNamespace(inference_mode=nullcontext)
    tokens = np.array([[1, 2, 7, 8, 9]])
    backend.model = SimpleNamespace(device="cpu", generate=lambda **kwargs: tokens)
    backend.tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: str(ids))
    messages = [{"role": "user", "content": "<audio>"}]
    audios = [np.zeros(8000)]
    with pytest.raises(GenerationTruncated) as caught:
        backend.generate(messages, audios)
    assert caught.value.as_dict() == {"backend": "HF", "generated_tokens": 3, "prompt_tokens": 2,
                                      "max_new_tokens": 3, "max_context_tokens": 16}
    tokens[0, -1] = backend.stop_ids[0]
    assert backend.generate(messages, audios)["text"] == "[7, 8]"
