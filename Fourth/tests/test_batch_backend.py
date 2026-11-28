"""Cross-meeting batching with the real Fourth prompt/decoder path and a fake engine."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_asr.batch_backend import BatchBackend
from fourth_asr.http_vllm import HttpVLLMWorker
from fourth_asr.runtime.streaming import ContextBudgetError, GenerationTruncated
from fourth_asr.runtime.vllm_backend import VLLMBackend


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
            completion = (SimpleNamespace(token_ids=[token] * backend.max_new_tokens, finish_reason="length")
                          if token == 4 else SimpleNamespace(token_ids=[token, 99], finish_reason="stop"))
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
    assert isinstance(replies[4], GenerationTruncated) and "truncated" in str(replies[4])
    assert replies[4].as_dict() == {"backend": "vLLM", "generated_tokens": 3, "prompt_tokens": 2,
                                    "max_new_tokens": 3, "max_context_tokens": 16}
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


class ControlledStepEngine:
    """vLLM 0.11.1 add_request/step interface. Tests choose which meetings may finish."""
    def __init__(self):
        self.requests = {}
        self.gates = {}
        self.aborted = []
        self.max_running = 0

    def hold(self, marker):
        events = threading.Event(), threading.Event()
        self.gates[marker] = events
        return events

    def add_request(self, request_id, prompt, sampling):
        wave = prompt["multi_modal_data"]["audio"][-1]
        marker = round(float(wave[0]) * 32768)
        self.requests[request_id] = (marker, prompt)
        self.max_running = max(self.max_running, len(self.requests))
        if marker == 7:
            raise ValueError("failed after registration")

    def abort_request(self, request_ids):
        for request_id in request_ids:
            marker, _ = self.requests.pop(request_id)
            self.aborted.append(marker)

    def get_num_unfinished_requests(self):
        return len(self.requests)

    def step(self):
        outputs = []
        for request_id, (marker, prompt) in list(self.requests.items()):
            if marker in self.gates:
                started, release = self.gates[marker]
                if not started.is_set():
                    outputs.append(SimpleNamespace(request_id=request_id, finished=False))
                    started.set()
                if not release.is_set():
                    continue
            completion = SimpleNamespace(token_ids=[marker, 99], finish_reason="length" if marker == 4 else "stop")
            count = len(prompt["multi_modal_data"]["audio"])
            outputs.append(SimpleNamespace(request_id=request_id, finished=True, outputs=[completion],
                                           prompt_token_ids=prompt["prompt_token_ids"] + [2] * count,
                                           num_cached_tokens=0))
            del self.requests[request_id]
        if not outputs:
            time.sleep(.002)  # Simulate a decode iteration that has no final outputs yet.
        return outputs


def stepped_backend():
    backend, _ = engine_fixture()
    engine = ControlledStepEngine()
    backend.llm = SimpleNamespace(llm_engine=engine)
    backend.max_batch_size = 2
    backend.max_audio_turns = 6
    texts = {0: "<|WAIT|>", 1: DIRECT + "long reply", 2: DIRECT + "new meeting", 3: "continuation"}
    backend.tokenizer.decode = lambda ids, **kwargs: texts[ids[0]]
    return backend, engine


def test_http_wait_and_new_meeting_finish_while_long_reply_is_still_running():
    from fastapi.testclient import TestClient
    from fourth_asr.http_service import create_app
    from fourth_asr.service_session import FourthStreamOptions

    backend, engine = stepped_backend()
    long_started, finish_long = engine.hold(1)
    short_started, finish_short = engine.hold(0)
    app = create_app(lambda: HttpVLLMWorker(lambda: backend, max_batch_size=2, batch_wait_ms=0),
                     FourthStreamOptions(enable_recovery=False, chunk_seconds=1))
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=3) as callers:
        def post(meeting, *markers):
            pcm = b"".join(np.full(16000, marker, dtype="<i2").tobytes() for marker in markers)
            return client.post(f"/v1/meetings/{meeting}/chunks", headers={"Content-Type": "audio/pcm"},
                               content=pcm)

        long = callers.submit(post, "B", 1)
        try:
            assert long_started.wait(3)
            short = callers.submit(post, "A", 0)
            assert short_started.wait(3)
            newcomer = callers.submit(post, "C", 2)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                health = client.get("/health").json()
                if health["batch_stats"]["queued_decodes"] == 1:
                    break
                time.sleep(.005)
            assert health["scheduling_mode"] == "continuous" and health["batch_stats"]["queued_decodes"] == 1
            assert post("B", 2).status_code == 409
            finish_short.set()
            for job, expected in [(short, ""), (newcomer, "new meeting")]:
                response = job.result(timeout=3)
                assert response.status_code == 200 and response.json() == {"src": expected}
            response = callers.submit(post, "C", 2, 0).result(timeout=3)
            assert response.status_code == 200 and response.json() == {"src": "new meeting"}
            assert not long.done() and engine.max_running == 2
        finally:
            finish_short.set()
            finish_long.set()
        assert long.result(timeout=3).json() == {"src": "long reply"}


def test_bad_request_and_truncated_completion_only_fail_their_own_turn():
    backend, engine = stepped_backend()
    started, release = engine.hold(1)
    worker = HttpVLLMWorker(lambda: backend, max_batch_size=2, batch_wait_ms=0)

    def wave(marker):
        return [np.array([marker / 32768], dtype=np.float32)]

    with ThreadPoolExecutor(max_workers=1) as caller:
        long = caller.submit(worker.generate, MESSAGES, wave(1))
        try:
            assert started.wait(3)
            with pytest.raises(ValueError, match="failed after registration"):
                worker.generate(MESSAGES, wave(7))
            assert engine.aborted == [7]
            with pytest.raises(GenerationTruncated):
                worker.generate(MESSAGES, wave(4))
            with pytest.raises(ValueError, match="Forced decoding"):
                worker.generate_with_prefix(MESSAGES, wave(3), "invalid")
            assert worker.generate_with_prefix(MESSAGES, wave(3), DIRECT)["text"] == DIRECT + "continuation"
            assert not long.done()
        finally:
            release.set()
            worker.close()
        assert long.result(timeout=3)["text"] == DIRECT + "long reply"


@pytest.mark.parametrize("failure_site", ["admission", "step", "output_socket"])
def test_unusable_engine_fails_active_and_queued_http_calls(failure_site):
    from fastapi.testclient import TestClient
    from fourth_asr.http_service import create_app
    from fourth_asr.service_session import FourthStreamOptions

    class DeadCore(Exception):
        pass

    backend, engine = stepped_backend()
    backend.engine_dead_error = DeadCore
    first_started, release_first = engine.hold(1)
    failing, release_error = threading.Event(), threading.Event()
    original_add, original_step = engine.add_request, engine.step

    def fail():
        failing.set()
        assert release_error.wait(5)
        if failure_site == "output_socket":
            raise OSError("output socket closed")
        raise DeadCore("model process exited")

    def add(*args):
        original_add(*args)
        if failure_site == "admission" and len(engine.requests) == 2:
            fail()

    def step():
        if failure_site in ("step", "output_socket") and len(engine.requests) == 2:
            fail()
        return original_step()

    engine.add_request, engine.step = add, step
    engine.abort_request = lambda ids: None  # Native MP client ignores aborts on a dead core.
    app = create_app(lambda: HttpVLLMWorker(lambda: backend, max_batch_size=2, batch_wait_ms=0),
                     FourthStreamOptions(enable_recovery=False, chunk_seconds=1))
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=3) as callers:
        def post(key, marker):
            return client.post(f"/v1/meetings/{key}/chunks", headers={"Content-Type": "audio/pcm"},
                               content=np.full(16000, marker, dtype="<i2").tobytes())

        try:
            first = callers.submit(post, "first", 1)
            assert first_started.wait(3)
            second = callers.submit(post, "second", 2)
            assert failing.wait(3)
            queued = callers.submit(post, "queued", 0)
            deadline = time.monotonic() + 3
            while client.get("/health").json()["batch_stats"]["queued_decodes"] != 1:
                assert time.monotonic() < deadline, "third request did not reach the queue"
                time.sleep(.005)
        finally:
            release_error.set()
            release_first.set()
        assert [f.result(timeout=3).status_code for f in (first, second, queued)] == [500, 500, 500]
        assert client.get("/health").json()["ready"] is False
        assert post("new", 2).status_code == 503


def test_dropped_final_reply_fails_the_waiter_instead_of_stepping_an_idle_engine():
    backend, engine = stepped_backend()

    def drop_final():
        assert engine.requests, "called step after the engine ran out of work"
        engine.requests.clear()
        return []

    engine.step = drop_final
    worker = HttpVLLMWorker(lambda: backend, max_batch_size=1)
    try:
        with pytest.raises(RuntimeError, match="unfinished request count"):
            worker.generate(MESSAGES, [np.zeros(1)])
    finally:
        worker.close()
    assert not worker.ready
