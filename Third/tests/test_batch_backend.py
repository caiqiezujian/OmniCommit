"""Cross-meeting batching with the real Third prompt/decoder path and a fake engine."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from third_asr.batch_backend import BatchBackend
from third_asr.runtime.streaming import ContextBudgetError
from third_asr.runtime.vllm_backend import VLLMBackend


DIRECT = "<|DIRECT|>"
MESSAGES = [{"role": "user", "content": "<audio>"}]


def engine_fixture():
    backend = VLLMBackend.__new__(VLLMBackend)
    backend.allowed_tokens = {DIRECT, "<|WAIT|>"}
    backend.max_audio_turns = 1
    backend.max_audio_seconds = 1
    backend.max_context_tokens, backend.max_new_tokens = 16, 3
    backend.TokensPrompt = dict
    backend.stop_ids = [99]
    backend.sampling = object()
    backend.tokenizer = SimpleNamespace(
        apply_chat_template=lambda *a, **kwargs: "assistant",
        encode=lambda text, **kwargs: [0, 1] if text.endswith(DIRECT) else [0],
        decode=lambda ids, **kwargs: str(ids[0]),
    )
    backend.audio_counter = SimpleNamespace(count=lambda *args: (1, False))
    batches = []

    def generate(requests, params, **kwargs):
        batches.append(requests)
        outputs = []
        for req in requests:
            token = int(req["multi_modal_data"]["audio"][0][0])
            completion = SimpleNamespace(token_ids=[token, 99], finish_reason="length" if token == 4 else "stop")
            outputs.append(SimpleNamespace(prompt_token_ids=req["prompt_token_ids"]+[2],
                                           outputs=[completion], num_cached_tokens=token))
        return outputs

    backend.llm = SimpleNamespace(generate=generate)
    return backend, batches


def test_vllm_validates_each_prompt_and_maps_plain_and_prefill_results_in_order():
    backend, batches = engine_fixture()
    replies = backend.generate_batch([
        (MESSAGES, [np.array([1])], ""),
        (MESSAGES, [], ""),
        (MESSAGES, [np.array([2])], DIRECT),
        (MESSAGES, [np.zeros(16001)], ""),
        (MESSAGES, [np.array([4])], ""),
        (MESSAGES, [np.array([5])], "<invalid>"),
    ])
    batch, = batches
    assert len(batch) == backend.last_batch_size == 3
    assert replies[0] == {"text": "1", "prompt_tokens": 2, "num_cached_tokens": 1}
    assert isinstance(replies[1], ValueError)
    assert replies[2] == {"text": DIRECT + "2", "prompt_tokens": 3, "num_cached_tokens": 2}
    assert isinstance(replies[3], ContextBudgetError)
    assert isinstance(replies[4], RuntimeError) and "truncated" in str(replies[4])
    assert isinstance(replies[5], ValueError)


def test_overlapping_meetings_use_one_native_generate_with_correct_return_values():
    backend, batches = engine_fixture()
    adapter = BatchBackend(lambda: backend, max_batch_size=4, batch_wait_ms=250)
    barrier = threading.Barrier(4)

    def send(n):
        barrier.wait(timeout=3)
        audio = [np.array([n])]
        return (adapter.generate_with_prefix(MESSAGES, audio, DIRECT) if n % 2 else
                adapter.generate(MESSAGES, audio))

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = [r["text"] for r in pool.map(send, range(4))]
        assert actual == ["0", DIRECT+"1", "2", DIRECT+"3"]
        assert list(map(len, batches)) == [4]
        assert adapter.stats() == {"model_batches": 1, "largest_batch": 4, "queued_decodes": 0}
    finally:
        adapter.close()
    with pytest.raises(RuntimeError, match="closed"):
        adapter.generate([], [])


def test_serial_fallback_creates_decodes_and_closes_on_the_same_owner_thread():
    owner = []

    class Single:
        def __init__(self):
            owner.append(threading.get_ident())

        def generate(self, messages, audios):
            owner.append(threading.get_ident())
            return "plain"

        def generate_with_prefix(self, messages, audios, prefix):
            owner.append(threading.get_ident())
            return prefix+"!"

        def close(self):
            owner.append(threading.get_ident())

    adapter = BatchBackend(Single, max_batch_size=8, batch_wait_ms=500)
    try:
        assert adapter.max_batch_size == 1 and adapter.batch_wait_ms == 0
        assert adapter.generate([], []) == "plain"
        assert adapter.generate_with_prefix([], [], DIRECT) == DIRECT+"!"
    finally:
        adapter.close()
    assert len(owner) == 4 and len(set(owner)) == 1 and owner[0] != threading.get_ident()
