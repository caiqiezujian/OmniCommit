"""Web protocol integration with synthetic audio; no real model or identity claims."""
import io
import shutil
import threading
import time
import types

import numpy as np
import pytest
import soundfile as sf

from omnicommit.streaming import DIRECT, WAIT, ReplayBackend, StreamOptions

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from omnicommit.web_service import create_app


def test_meeting_page_serves_its_log_scripts():
    with TestClient(create_app(lambda: ReplayBackend([]), StreamOptions())) as client:
        page = client.get("/")
        assert page.status_code == 200
        for name in ("log-archive", "turn-log"):
            url = f"/web/{name}.js"
            assert f'src="{url}"' in page.text
            script = client.get(url)
            assert script.status_code == 200
            assert "javascript" in script.headers["content-type"]


def test_web_800ms_step_is_advertised_and_accepts_12800_samples():
    replies = [DIRECT + "<en><spk1>Hello.", WAIT, WAIT]
    with TestClient(create_app(lambda: ReplayBackend(replies), StreamOptions(chunk_seconds=.8))) as client:
        health = client.get("/health").json()
        assert health["protocol"] == "first" and health["chunk_seconds"] == .8
        assert health["chunk_ms_choices"] == [500, 700, 800, 1000]
        assert (health["chunk_ms_min"], health["chunk_ms_max"]) == (200, 5000)
        with client.websocket_connect("/stream") as socket:
            socket.send_json({"type": "start", "sample_rate": 16000})
            ready = socket.receive_json()
            assert ready["type"] == "ready" and ready["chunk_seconds"] == .8
            assert ready["chunk_ms_choices"] == health["chunk_ms_choices"]
            assert (ready["chunk_ms_min"], ready["chunk_ms_max"]) == (200, 5000)
            socket.send_bytes(np.zeros(12800, dtype="<i2").tobytes())
            first = socket.receive_json()
            assert first["type"] == "result" and first["end_seconds"] == .8
            socket.send_bytes(np.zeros(1600, dtype="<i2").tobytes())
            assert socket.receive_json()["end_seconds"] == pytest.approx(.9)
            socket.send_json({"type": "end"})
            final = socket.receive_json()
            assert final["event"] == "end_of_audio" and final["end_seconds"] == pytest.approx(.9)


@pytest.mark.parametrize("ms", [200, 333, 700, 1500, 5000])
def test_first_meetings_use_independent_steps_and_keep_direct_replies(ms):
    reply = DIRECT + "<en><spk1>Hello."
    app = create_app(lambda: ReplayBackend([reply, WAIT]), StreamOptions(), max_sessions=2)
    with TestClient(app) as client:
        with client.websocket_connect("/stream") as mic, client.websocket_connect("/stream") as system:
            mic.send_json({"type": "start", "sample_rate": 16000, "chunk_ms": ms})
            assert mic.receive_json()["chunk_seconds"] == ms / 1000
            system.send_json({"type": "start", "sample_rate": 16000, "source_kind": "system", "chunk_ms": 1000})
            assert system.receive_json()["chunk_seconds"] == 1
            mic.send_bytes(bytes(32 * ms))
            result = mic.receive_json()
            assert result["action"] == "DIRECT" and result["text"] == reply
            assert result["end_seconds"] == ms / 1000
            mic.send_bytes(bytes(32 * ms + 2))
            assert mic.receive_json()["type"] == "error"
            system.send_bytes(bytes(2 * 16000))
            assert system.receive_json()["end_seconds"] == 1
        assert client.get("/health").json()["chunk_seconds"] == .5


@pytest.mark.parametrize("step", [199, 5001, "700", None, True, 200.5])
def test_first_service_rejects_invalid_step_before_audio(step):
    with TestClient(create_app(ReplayBackend, StreamOptions())) as client, client.websocket_connect("/stream") as socket:
        socket.send_json({"type": "start", "sample_rate": 16000, "chunk_ms": step})
        error = socket.receive_json()
        assert error["type"] == "error" and error["heard_seconds"] == 0


def test_speaker_corrections_change_display_and_preserve_original_identity():
    replies = [DIRECT + "<zh><spk1>你好。", DIRECT + "<en><spk2>Hi.", WAIT]
    with TestClient(create_app(lambda: ReplayBackend(replies), StreamOptions())) as client:
        with client.websocket_connect("/stream") as socket:
            socket.send_json({"type": "start", "sample_rate": 16000})
            assert socket.receive_json()["type"] == "ready"
            for _ in range(2):
                socket.send_bytes(np.zeros(8000, dtype="<i2").tobytes())
                result = socket.receive_json()
                assert result["new_segments"][0]["segment_id"] == result["turn"]
            socket.send_json({"type": "control", "op": "rename_speaker", "speaker": "spk1", "name": "Alice"})
            assert socket.receive_json()["snapshot"]["speakers"][0]["name"] == "Alice"
            socket.send_json({"type": "control", "op": "correct_segment", "segment_id": 2, "speaker": "spk1"})
            snapshot = socket.receive_json()["snapshot"]
            assert snapshot["segments"][1]["model_speaker"] == "spk2"
            assert snapshot["segments"][1]["display_speaker"] == "spk1"
            socket.send_json({"type": "control", "op": "merge_speakers", "source": "spk2", "target": "spk1"})
            assert socket.receive_json()["snapshot"]["aliases"] == {"spk2": "spk1"}
            socket.send_json({"type": "control", "op": "reference", "speaker": "spk2",
                              "language": "en", "text": "Hi.", "start": .5, "end": 1.,
                              "verified": True, "clean": True, "overlap": False})
            assert "merged" in socket.receive_json()["error"]
            socket.send_json({"type": "end"})
            assert socket.receive_json()["action"] == "WAIT"


def test_optional_experimental_reference_is_wired_into_next_model_turn():
    from omnicommit.auto_reference import QwenAutoReference
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    class Aligner:
        def align(self, *, audio, text, language):
            assert len(audio[0]) == 16000 and text == 'Hello.' and language == 'English'
            started.set()
            assert release.wait(2)
            finished.set()
            return [types.SimpleNamespace(items=[types.SimpleNamespace(text='Hello', start_time=.05, end_time=.95)])]
    class Backend(ReplayBackend):
        def __init__(self):
            super().__init__([WAIT, DIRECT + '<en><spk1>Hello.'] + [WAIT] * 20)
            self.messages = None
        def generate(self, messages, audios):
            self.messages = messages
            return super().generate(messages, audios)
    backend = Backend()
    provider = QwenAutoReference('fake', model=Aligner())
    with TestClient(create_app(lambda: backend, StreamOptions(), auto_reference_factory=lambda: provider)) as client:
        assert client.get('/health').json()['automatic_reference'] == 'experimental_qwen'
        with client.websocket_connect('/stream') as socket:
            socket.send_json({'type': 'start', 'sample_rate': 16000})
            assert socket.receive_json()['auto_reference'] == 'experimental_qwen'
            audio = np.full(8000, 3000, dtype='<i2').tobytes()
            socket.send_bytes(audio)
            assert socket.receive_json()['action'] == 'WAIT'
            socket.send_bytes(audio)
            result = socket.receive_json()
            assert result['action'] == 'DIRECT'
            assert started.wait(2)
            began = time.monotonic()
            socket.send_bytes(audio)
            assert socket.receive_json()['action'] == 'WAIT'
            assert time.monotonic() - began < 1  # ASR continues while aligner is blocked
            release.set()
            assert finished.wait(2)
            time.sleep(.02)  # allow the event-loop future to publish its result
            socket.send_bytes(audio)
            result = socket.receive_json()
            assert result['automatic_reference']['status'] == 'experimental'
            assert result['automatic_reference']['identity_verified'] is False
            assert 'Speaker reference' in backend.messages[1]['content']  # applied before this turn
            socket.send_bytes(audio)
            assert socket.receive_json()['action'] == 'WAIT'
            assert 'Speaker reference' in backend.messages[1]['content']
            socket.send_json({'type': 'end'})
            socket.receive_json()


def test_reference_failure_disables_only_auto_reference_for_session():
    class BrokenProvider:
        def __init__(self): self.calls = 0
        def prepare(self, ledger, result): return {}
        def evaluate(self, snapshot):
            self.calls += 1
            return {'status': 'error', 'verified': False, 'error': 'missing alignment model'}
        def apply(self, ledger, result): return result
    provider = BrokenProvider()
    backend = ReplayBackend([DIRECT + '<en><spk1>One.', DIRECT + '<en><spk1>Two.', WAIT])
    with TestClient(create_app(lambda: backend, StreamOptions(), auto_reference_factory=lambda: provider)) as client:
        with client.websocket_connect('/stream') as socket:
            socket.send_json({'type': 'start', 'sample_rate': 16000}); socket.receive_json()
            audio = np.zeros(8000, dtype='<i2').tobytes()
            socket.send_bytes(audio)
            assert socket.receive_json()['action'] == 'DIRECT'
            time.sleep(.02)
            socket.send_bytes(audio)
            result = socket.receive_json()
            assert result['action'] == 'DIRECT'
            assert result['automatic_reference']['status'] == 'error'
            assert provider.calls == 1
            socket.send_json({'type': 'end'}); socket.receive_json()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
@pytest.mark.parametrize(("default", "choice_ms"), [(.5, None), (.8, None), (.5, 700), (.5, 200)])
def test_file_upload_playback_gating_tail_and_cleanup(default, choice_ms):
    class Recording(ReplayBackend):
        def __init__(self):
            super().__init__([WAIT, WAIT, WAIT, WAIT])
            self.last = None
        def generate(self, messages, audios):
            self.last = audios[-1].copy() if messages[-1]["content"] != "<end_of_audio>" else None
            return super().generate(messages, audios)

    step = default if choice_ms is None else choice_ms / 1000
    frames = round(step * 16000)
    final_playhead = (2 * frames + 2400) / 16000
    source = np.linspace(-.8, .8, 2 * frames + 3200, dtype=np.float32)
    file = io.BytesIO()
    sf.write(file, source, 16000, format="WAV", subtype="PCM_16")
    backend = Recording()
    app = create_app(lambda: backend, StreamOptions(chunk_seconds=default))
    with TestClient(app) as client:
        upload = client.post("/api/media", content=file.getvalue())
        assert upload.status_code == 200
        media_id = upload.json()["media_id"]
        with client.websocket_connect("/stream") as socket:
            fields = {} if choice_ms is None else {"chunk_ms": choice_ms}
            socket.send_json({"type": "start", "sample_rate": 16000,
                              "source_kind": "file", "media_id": media_id, **fields})
            ready = socket.receive_json()
            assert ready["type"] == "ready" and ready["chunk_seconds"] == step
            socket.send_json({"type": "file_next", "release_seconds": step - .01})
            assert socket.receive_json()["type"] == "file_wait"
            socket.send_json({"type": "file_next", "release_seconds": step})
            assert socket.receive_json()["end_seconds"] == step
            assert np.max(np.abs(backend.last - source[:frames])) < .0001
            socket.send_json({"type": "file_next", "release_seconds": final_playhead, "final": True})
            assert socket.receive_json()["end_seconds"] == 2 * step
            socket.send_json({"type": "file_next", "release_seconds": final_playhead, "final": True})
            result = socket.receive_json()
            assert result["end_seconds"] == pytest.approx(final_playhead)
            assert np.max(np.abs(backend.last[:2400] - source[2*frames:2*frames+2400])) < .0001
            assert np.all(backend.last[2400:] == 0)  # final tail is padded only in memory
            socket.send_json({"type": "file_next", "release_seconds": final_playhead, "final": True})
            assert socket.receive_json()["type"] == "file_done"
            socket.send_json({"type": "end"})
            assert socket.receive_json()["event"] == "end_of_audio"
        assert media_id not in app.state.media_store.uploads
