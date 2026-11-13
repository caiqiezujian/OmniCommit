"""File playback uses the source clock; the model starts a fresh session at zero."""
import io
import shutil

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from omnicommit.streaming import ReplayBackend, StreamOptions
from omnicommit.web_service import create_app
from Second.second_asr.service_session import SecondStreamOptions
from Second.second_asr.service_web import create_app as create_second_app


requires_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires FFmpeg")
RATE = 16000


@pytest.fixture(params=[
    pytest.param((create_app, StreamOptions), id="first"),
    pytest.param((create_second_app, SecondStreamOptions), id="second"),
])
def protocol(request):
    return request.param


def upload(client, samples):
    wav = io.BytesIO()
    sf.write(wav, samples, RATE, format="WAV", subtype="PCM_16")
    response = client.post("/api/media", content=wav.getvalue())
    assert response.status_code == 200, response.text
    return response.json()["media_id"]


def exchange(socket, kind, **fields):
    socket.send_json({"type": kind, **fields})
    return socket.receive_json()


@requires_ffmpeg
@pytest.mark.parametrize(("bounds", "start", "end", "early_end"), [
    ({}, 0.0, 2.0, None),
    ({"file_start_seconds": .375}, .375, 2.0, None),
    ({"file_end_seconds": 1.125}, 0.0, 1.125, None),
    ({"file_start_seconds": .375, "file_end_seconds": 1.5}, .375, 1.5, None),
    ({"file_start_seconds": .375, "file_end_seconds": 1.5}, .375, 1.5, 1.075),
])
def test_file_range_only_submits_selected_samples(protocol, bounds, start, end, early_end):
    class Recorder(ReplayBackend):
        def __init__(self):
            super().__init__()
            self.packets = []

        def generate(self, messages, audios):
            if messages[-1]["content"] != "<end_of_audio>":
                self.packets.append(audios[-1].copy())
            return super().generate(messages, audios)

    factory, options = protocol
    backend = Recorder()
    source = np.arange(RATE * 2, dtype=np.int16)
    with TestClient(factory(lambda: backend, options())) as client:
        media_id = upload(client, source)
        with client.websocket_connect("/stream") as socket:
            ready = exchange(socket, "start", source_kind="file", sample_rate=RATE, media_id=media_id, **bounds)
            assert ready["type"] == "ready"
            assert (ready["file_start_seconds"], ready["file_end_seconds"]) == (start, end)
            assert exchange(socket, "file_next", release_seconds=start + .49)["type"] == "file_wait"
            assert not backend.packets
            stop = end if early_end is None else early_end
            # The player may have advanced past the selected end while waiting
            # for inference. On manual stop only the already played tail is released.
            release = len(source) / RATE if early_end is None else early_end
            previous = start
            while previous < stop:
                result = exchange(socket, "file_next", release_seconds=release, final=early_end is not None)
                assert result["type"] == "result", result
                current = min(previous + .5, stop)
                assert (result["source_start_seconds"], result["source_end_seconds"]) == pytest.approx((previous, current))
                assert (result["start_seconds"], result["end_seconds"]) == pytest.approx((previous - start, current - start))
                assert result["file_complete"] is (current == stop)
                expected = source[round(previous * RATE):round(current * RATE)]
                np.testing.assert_array_equal(backend.packets[-1][:len(expected)], expected.astype(np.float32) / 32768)
                assert not backend.packets[-1][len(expected):].any()
                previous = current
            assert exchange(socket, "file_next", release_seconds=release, final=early_end is not None)["type"] == "file_done"
            eos = exchange(socket, "end")
            assert eos["event"] == "end_of_audio"
            assert (eos["start_seconds"], eos["end_seconds"]) == pytest.approx((stop - start, stop - start))
            assert (eos["source_start_seconds"], eos["source_end_seconds"]) == pytest.approx((stop, stop))


@pytest.mark.parametrize("kind", ["microphone", "system"])
def test_live_sources_reject_file_bounds(protocol, kind):
    factory, options = protocol
    with TestClient(factory(ReplayBackend, options())) as client, client.websocket_connect("/stream") as socket:
        error = exchange(socket, "start", source_kind=kind, sample_rate=RATE, file_start_seconds=.375)
        assert error["type"] == "error" and error["heard_seconds"] == 0


@requires_ffmpeg
def test_stop_before_first_sample_ends_without_generation(protocol):
    class NoAudioBackend:
        def generate(self, messages, audios):
            pytest.fail("No audio was released")

    factory, options = protocol
    with TestClient(factory(NoAudioBackend, options())) as client:
        media_id = upload(client, np.zeros(RATE, dtype=np.int16))
        with client.websocket_connect("/stream") as socket:
            assert exchange(socket, "start", source_kind="file", sample_rate=RATE,
                            media_id=media_id, file_start_seconds=.375)["type"] == "ready"
            assert exchange(socket, "file_next", release_seconds=.375, final=True)["type"] == "file_done"
            assert exchange(socket, "end") == {"type": "file_end_empty", "heard_seconds": 0, "source_heard_seconds": .375}


@requires_ffmpeg
def test_file_range_backend_failure_logs_original_position(protocol):
    class FailAfterFirst(ReplayBackend):
        def generate(self, messages, audios):
            if len(audios) > 1:
                raise RuntimeError("synthetic inference failure")
            return super().generate(messages, audios)

    factory, options = protocol
    with TestClient(factory(FailAfterFirst, options())) as client:
        media_id = upload(client, np.zeros(RATE * 2, dtype=np.int16))
        with client.websocket_connect("/stream") as socket:
            assert exchange(socket, "start", source_kind="file", sample_rate=RATE, media_id=media_id,
                            file_start_seconds=.375, file_end_seconds=1.5)["type"] == "ready"
            assert exchange(socket, "file_next", release_seconds=.875)["type"] == "result"
            error = exchange(socket, "file_next", release_seconds=1.375)
            assert error["type"] == "error" and "synthetic inference failure" in error["error"]
            assert error["heard_seconds"] == .5 and error["source_heard_seconds"] == .875
            assert (error["source_start_seconds"], error["source_end_seconds"]) == (.875, 1.375)
