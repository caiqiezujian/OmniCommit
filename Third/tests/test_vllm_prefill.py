"""Exercise forced decoding on one concatenated acoustic window, without a GPU."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from third_asr.runtime.streaming import ContextBudgetError
from third_asr.runtime.vllm_backend import VLLMBackend


def test_vllm_direct_prefix_counts_against_context_and_preserves_continuation():
    direct = "<|DIRECT|>"
    backend = VLLMBackend.__new__(VLLMBackend)
    backend.allowed_tokens = {direct, "<|WAIT|>"}
    backend.max_audio_turns = 1
    backend.max_audio_seconds = 6.4
    backend.max_context_tokens = 32
    backend.max_new_tokens = 8
    backend.TokensPrompt = dict
    backend.stop_ids = [99]
    backend.sampling = object()
    backend.tokenizer = SimpleNamespace(
        apply_chat_template=lambda *a, **k: "assistant\n",
        encode=lambda text, **k: {"assistant\n": [1], "assistant\n" + direct: [1, 2]}[text],
        decode=lambda ids, **k: " Morgen" if ids == [20] else "",
    )
    backend.audio_counter = SimpleNamespace(count=lambda *args: (7, False))
    requests = []

    def run(batch, sampling, **kwargs):
        request, = batch
        requests.append(request)
        completion = SimpleNamespace(token_ids=[20, 99], finish_reason="stop")
        return [SimpleNamespace(prompt_token_ids=[0] * 9, outputs=[completion])]

    backend.llm = SimpleNamespace(generate=run)
    audio = np.zeros(16000 * 3, dtype=np.float32)
    messages = [{"role": "user", "content": "<audio>"}]
    result = backend.generate_with_prefix(messages, [audio], direct)
    assert result["text"] == direct + " Morgen" and result["prompt_tokens"] == 9
    assert requests[0]["prompt_token_ids"] == [1, 2]
    assert requests[0]["multi_modal_data"]["audio"][0] is audio
    backend.max_context_tokens = 16
    with pytest.raises(ContextBudgetError):
        backend.generate_with_prefix(messages, [audio], direct)
    assert len(requests) == 1
