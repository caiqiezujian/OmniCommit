"""Check the acoustic content IDs submitted to vLLM without loading a model."""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_asr.protocol import DIRECT, WAIT
from fourth_asr.runtime import audio_identity
from fourth_asr.runtime.streaming import SAMPLE_RATE
from fourth_asr.runtime.vllm_backend import VLLMBackend
from fourth_asr.service_session import FourthStreamOptions, FourthStreamSession


def legacy_id(audio):
    return "omnicommit-16k-" + hashlib.sha256(np.asarray(audio, dtype="<f4").tobytes()).hexdigest()


@pytest.fixture
def compute(monkeypatch):
    spy = Mock(wraps=audio_identity._compute_uuid)
    monkeypatch.setattr(audio_identity, "_compute_uuid", spy)
    return spy


class RecordingBackend(VLLMBackend):
    def __init__(self, replies):
        self.max_audio_turns = 4
        self.max_audio_seconds = 1
        self.max_new_tokens = 8
        self.max_context_tokens = 2048
        self.TokensPrompt = dict
        self.stop_ids = []
        self.requests = []
        self.tokenizer = SimpleNamespace(
            apply_chat_template=lambda messages, **kwargs: repr(messages),
            encode=lambda prompt, **kwargs: list(prompt.encode()),
            decode=Mock(side_effect=replies),
        )
        self.audio_counter = SimpleNamespace(count=lambda *args: (0, False))

    def _run_requests(self, requests):
        request, = requests
        self.requests.append(request)
        completion = SimpleNamespace(finish_reason="stop", token_ids=[1])
        return [SimpleNamespace(prompt_token_ids=request["prompt_token_ids"], outputs=[completion])]


def test_retained_audio_hashes_once_across_packet_edit_rollover_and_eos(compute):
    backend = RecordingBackend([DIRECT + "I agree.", WAIT, WAIT, WAIT])
    session = FourthStreamSession(backend, FourthStreamOptions(
        chunk_seconds=1, window_seconds=2, discard_seconds=1, enable_recovery=False))
    pcm = np.full(SAMPLE_RATE, .25, dtype=np.float32)
    first_id = legacy_id(pcm)
    previous = session.packet_text_state()
    event = session.step(pcm)
    assert session.apply_packet_period(event, *previous, silence_ms=300, comma_ms=450, remove_ms=150)
    history_audio = session.history[0].audio
    with pytest.raises(ValueError):
        history_audio.setflags(write=True)

    pcm.fill(-.5)
    second_id = legacy_id(pcm)
    session.step(pcm)
    second_request = backend.requests[-1]
    assert "I agree," in bytes(second_request["prompt_token_ids"]).decode()
    assert np.all(second_request["multi_modal_data"]["audio"][0] == .25)

    tail = pcm[:200]
    tail_id = legacy_id(np.pad(tail, (0, SAMPLE_RATE // 2 - len(tail))))
    event = session.step(tail, final_chunk=True)
    assert event["discarded_seconds"] == 1
    assert event["end_seconds"] == pytest.approx(2 + len(tail) / SAMPLE_RATE)
    assert compute.call_count == 3
    session.step(end_of_audio=True)

    assert [r["multi_modal_uuids"]["audio"] for r in backend.requests] == [
        [first_id], [first_id, second_id], [second_id, tail_id], [second_id, tail_id]]
    assert compute.call_count == 3


def test_interleaved_sessions_own_independent_snapshots(compute):
    backend = RecordingBackend([WAIT] * 4)
    options = FourthStreamOptions(enable_recovery=False)
    first, second = FourthStreamSession(backend, options), FourthStreamSession(backend, options)
    pcm = np.full(SAMPLE_RATE, -.25, dtype=np.float32)
    first_id = legacy_id(pcm)
    first.step(pcm)
    pcm.fill(.75)
    second_id = legacy_id(pcm)
    second.step(pcm)
    pcm.fill(0)
    first.step(end_of_audio=True)
    second.step(end_of_audio=True)
    assert [r["multi_modal_uuids"]["audio"] for r in backend.requests] == [
        [first_id], [second_id], [first_id], [second_id]]
    assert compute.call_count == 2


@pytest.mark.parametrize("kind", ["writable", "readonly_alias", "strided_float64"])
def test_direct_caller_arrays_are_rehashed_after_mutation(compute, kind):
    backend = RecordingBackend([WAIT, WAIT])
    pcm = np.arange(8, dtype=">f8" if kind == "strided_float64" else np.float32)
    view = pcm
    if kind == "readonly_alias":
        view = pcm.view()
        view.setflags(write=False)
    elif kind == "strided_float64":
        view = pcm[::2]
    messages = [{"role": "user", "content": "<audio>"}]
    for value in (-.5, .75):
        pcm[0] = value
        backend.generate(messages, [view])
        assert backend.requests[-1]["multi_modal_uuids"]["audio"] == [legacy_id(view)]
    assert compute.call_count == 2
