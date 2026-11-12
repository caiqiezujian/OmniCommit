"""Exercise the Second wire protocol through the shared meeting page and websocket."""
import io
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omnicommit.streaming import ReplayBackend
from second_asr.protocol import WAIT
from second_asr.service_session import SecondStreamOptions
from second_asr.service_web import create_app


def start(socket, source_kind="microphone", **fields):
    socket.send_json({"type": "start", "sample_rate": 16000, "source_kind": source_kind, **fields})
    ready = socket.receive_json()
    assert ready["type"] == "ready" and ready["protocol"] == "second"
    return ready


def send_audio(socket, frames=11200):
    socket.send_bytes(np.zeros(frames, dtype="<i2").tobytes())
    return socket.receive_json()


def test_concurrent_meetings_negotiate_independent_chunk_lengths():
    app = create_app(ReplayBackend, SecondStreamOptions(chunk_seconds=.701), max_sessions=2)
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["chunk_ms_choices"] == [500, 700, 701, 800, 1000]
        assert (health["chunk_ms_min"], health["chunk_ms_max"]) == (200, 5000)
        assert health["chunk_seconds"] == .701
        with client.websocket_connect("/stream") as mic, client.websocket_connect("/stream") as system:
            assert start(mic, chunk_ms=200)["chunk_seconds"] == .2
            assert start(system, "system", chunk_ms=1000)["chunk_seconds"] == 1
            assert send_audio(mic, 3200)["end_seconds"] == .2
            assert send_audio(system, 16000)["end_seconds"] == 1
            assert send_audio(mic, 3200)["end_seconds"] == .4
            # 700 ms fits the other meeting but exceeds the negotiated 200 ms.
            rejected = send_audio(mic, 11200)
            assert rejected["type"] == "error" and rejected["heard_seconds"] == .4
            assert send_audio(system, 16000)["end_seconds"] == 2
            system.send_json({"type": "end"})
            assert system.receive_json()["text"] == WAIT
        with client.websocket_connect("/stream") as default:
            assert start(default)["chunk_seconds"] == .701
            assert send_audio(default, 11216)["end_seconds"] == .701
            default.send_json({"type": "end", "chunk_ms": 500})
            late = default.receive_json()
            assert late["type"] == "error" and late["heard_seconds"] == .701


@pytest.mark.parametrize("invalid_ms", [None, True, "500", 500.0, 0, 199, 5001, 8001])
def test_invalid_chunk_choice_only_rejects_that_connection(invalid_ms):
    app = create_app(ReplayBackend, SecondStreamOptions(), max_sessions=2)
    with TestClient(app) as client, client.websocket_connect("/stream") as valid:
        assert start(valid, chunk_ms=800)["chunk_seconds"] == .8
        with client.websocket_connect("/stream") as other:
            other.send_json({"type": "start", "sample_rate": 16000, "chunk_ms": invalid_ms})
            error = other.receive_json()
            assert error["type"] == "error" and error["heard_seconds"] == 0
        assert send_audio(valid, 12800)["end_seconds"] == .8


@pytest.mark.parametrize("source_kind", ["microphone", "system"])
def test_live_sparse_text_switch_correction_and_isolation(source_kind):
    backend = ReplayBackend(["<zh><spk3>请联系", WAIT, "张伟。<en><spk3>OK?",
                             "<en><spk2>Sure,", " I will.", WAIT, "orphan"])
    with TestClient(create_app(lambda: backend, SecondStreamOptions(chunk_seconds=.7))) as client:
        assert client.get("/health").json()["protocol"] == "second"
        assert client.get("/").status_code == 200
        with client.websocket_connect("/stream") as socket:
            assert start(socket, source_kind)["chunk_seconds"] == .7
            assert send_audio(socket)["text"] == "<zh><spk3>请联系"
            assert send_audio(socket)["new_segments"] == []
            switch = send_audio(socket)
            assert switch["text"] == "张伟。<en><spk3>OK?"
            assert [(s["language"], s["model_speaker"], s["text"]) for s in switch["new_segments"]] == [
                ("zh", "spk3", "张伟。"), ("en", "spk3", "OK?")]
            speaker = send_audio(socket)["new_segments"][0]
            socket.send_json({"type": "control", "op": "correct_segment",
                              "segment_id": speaker["segment_id"], "speaker": "spk3"})
            corrected = socket.receive_json()["snapshot"]["segments"][-1]
            assert (corrected["model_speaker"], corrected["display_speaker"]) == ("spk2", "spk3")
            last = send_audio(socket)
            assert last["text"] == " I will." and last["new_segments"][0]["model_speaker"] == "spk2"
            socket.send_json({"type": "end"})
            assert socket.receive_json()["text"] == WAIT
        with client.websocket_connect("/stream") as socket:
            start(socket, source_kind)
            result = send_audio(socket)
            assert result["type"] == "error" and result["heard_seconds"] == 0


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires FFmpeg")
@pytest.mark.parametrize(("default_ms", "choice_ms"), [(701, None), (500, 700), (500, 200), (500, 333)])
def test_file_audio_respects_training_sample_grid_and_flushes_tail(default_ms, choice_ms):
    # At 22.05 kHz 701 ms ends at sample 15457.05. The last speech
    # sample of this cell can only be delivered after sample 15458.
    rate, ms = 22050, choice_ms or default_ms
    boundary = (rate * ms + 999) // 1000
    samples = np.zeros(boundary + 300, dtype="<i2")
    samples[boundary:] = 13000
    wav = io.BytesIO()
    sf.write(wav, samples, rate, format="WAV", subtype="PCM_16")

    class Recorder(ReplayBackend):
        def generate(self, messages, audios):
            self.last = audios[-1] if messages[-1]["content"] == "<audio>" else None
            return super().generate(messages, audios)

    backend = Recorder(["<en><spk1>Yes", ".", WAIT])
    opts = SecondStreamOptions(chunk_seconds=default_ms / 1000)
    with TestClient(create_app(lambda: backend, opts)) as client:
        upload = client.post("/api/media", content=wav.getvalue())
        assert upload.status_code == 200, upload.text
        item = upload.json()
        assert item["sample_rate"] == rate
        with client.websocket_connect("/stream") as socket:
            fields = {"chunk_ms": choice_ms} if choice_ms is not None else {}
            assert start(socket, "file", media_id=item["media_id"], **fields)["chunk_seconds"] == ms / 1000
            for position, event in (((boundary - 1) / rate, "file_wait"), (boundary / rate, "result")):
                socket.send_json({"type": "file_next", "release_seconds": position})
                assert socket.receive_json()["type"] == event
            assert backend.last is not None and not backend.last.any()
            socket.send_json({"type": "file_next", "release_seconds": item["duration_seconds"]})
            tail = socket.receive_json()
            assert tail["new_segments"][0]["text"] == "."
            assert backend.last.any() and len(backend.last) == 8000
            socket.send_json({"type": "end"})
            assert socket.receive_json()["text"] == WAIT


def test_largest_live_frame():
    app = create_app(lambda: ReplayBackend(["<zh><spk7>开始", WAIT]), SecondStreamOptions(chunk_seconds=8))
    with TestClient(app) as client, client.websocket_connect("/stream") as socket:
        assert start(socket)["chunk_seconds"] == 8
        assert send_audio(socket, 8 * 16000)["end_seconds"] == 8
        socket.send_json({"type": "end"})
        assert socket.receive_json()["text"] == WAIT


def test_untagged_speech_can_seed_reference_and_merge_invalidates_it():
    from omnicommit.auto_reference import QwenAutoReference

    class Aligner:
        def align(self, *, audio, text, language):
            assert language == "English" and text == "Okay." and len(audio[0]) == 16000
            return [SimpleNamespace(items=[SimpleNamespace(text="Okay", start_time=.05, end_time=.95)])]

    provider = QwenAutoReference("test-aligner", model=Aligner(), max_window_seconds=1)
    backend = ReplayBackend(["<zh><spk1>开始。<en><spk2>Hello.", "Okay."] + [WAIT] * 50)
    app = create_app(lambda: backend, SecondStreamOptions(chunk_seconds=1),
                     auto_reference_factory=lambda: provider)
    block = np.full(16000, 4000, dtype="<i2").tobytes()
    with TestClient(app) as client, client.websocket_connect("/stream") as socket:
        assert start(socket)["automatic_reference"] == "experimental_qwen"
        for expected in (2, 1):
            socket.send_bytes(block)
            assert len(socket.receive_json()["new_segments"]) == expected
        for _ in range(40):
            socket.send_bytes(block)
            result = socket.receive_json()
            if result["reference_speakers"]:
                break
            time.sleep(.01)
        assert result["reference_speakers"] == ["spk2"]
        socket.send_json({"type": "control", "op": "merge_speakers", "source": "spk2", "target": "spk1"})
        assert socket.receive_json()["snapshot"]["aliases"] == {"spk2": "spk1"}
        socket.send_json({"type": "end"})
        assert socket.receive_json()["reference_speakers"] == []
