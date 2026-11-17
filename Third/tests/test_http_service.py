"""Exercise meeting admission, timeouts and incremental audio over the real ASGI routes."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import sys
import threading
import time
import wave

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from third_asr.http_service import create_app
from third_asr.runtime.streaming import GenerationTruncated, ReplayBackend
from third_asr.service_session import ThirdStreamOptions


BASE = "/v1/meetings/"
STEP = 8000
RAW = {"Content-Type": "audio/pcm"}
WAIT = "<|WAIT|>"


def app(backend, **settings):
    return create_app(lambda: backend, ThirdStreamOptions(chunk_seconds=.5), **settings)


def audio(client, key, count=STEP, value=0):
    return client.post(BASE + key + "/chunks", content=np.full(count, value, dtype="<i2").tobytes(), headers=RAW)


def wav(data, rate=16000):
    f = BytesIO()
    with wave.open(f, "wb") as out:
        out.setparams((1, 2, rate, 0, "NONE", ""))
        out.writeframes(data)
    return f.getvalue()


def test_reblocking_buffered_partial_and_wav_tail_with_eos():
    class Model(ReplayBackend):
        def __init__(self):
            super().__init__(["a", WAIT, " b", WAIT])
            self.heard = []

        def generate(self, messages, audios):
            if messages[-1]["content"] == "<audio>":
                self.heard.append(audios[-1].copy())
            return super().generate(messages, audios)

    model = Model()
    with TestClient(app(model)) as client:
        buffered = audio(client, "abc", 100, 33).json()
        assert (buffered["status"], buffered["action"], buffered["heard_seconds"]) == ("buffered", None, 0)
        body = np.full(2 * STEP + 100, 44, dtype="<i2").tobytes()
        row = client.post(BASE+"abc/chunks", headers={"Content-Type": "audio/wav"}, content=wav(body)).json()
        assert (row["status"], row["action"], row["text"]) == ("ok", "DIRECT", "a")
        assert [t["action"] for t in row["turns"]] == ["DIRECT", "WAIT"]
        assert row["buffered_seconds"] == 200 / 16000 and row["heard_seconds"] == 1
        assert client.get(BASE+"abc/result").json() == row
        end = client.post(BASE+"abc/end").json()
        assert end["status"] == "ended" and end["text"] == " b"
        assert end["heard_seconds"] == pytest.approx(1 + 200 / 16000)
        assert [t["event"] for t in end["turns"]] == ["audio", "end_of_audio"]
        np.testing.assert_array_equal(model.heard[0][:100], np.full(100, 33 / 32768))
        assert np.all(model.heard[0][100:] == 44/32768)
        assert np.all(model.heard[2][:200] == 44/32768) and not np.any(model.heard[2][200:])
        assert client.get(BASE+"abc/result").json() == end
        assert audio(client, "abc").status_code == 409
        assert client.get("/health").json()["active_meetings"] == 0


def test_late_result_is_pollable_and_charged_to_admission_until_finished():
    class Gate:
        release = threading.Event()
        finished = threading.Event()
        def generate(self, messages, audios):
            assert self.release.wait(3)
            self.finished.set()
            return "reply"
    gate = Gate()
    with TestClient(app(gate, response_timeout_ms=15, max_in_flight=1)) as client:
        try:
            late = audio(client, "late")
            assert late.status_code == 202 and late.json()["action"] is None
            url = late.headers["Location"]
            assert client.get(url).status_code == 202
            assert audio(client, "late").status_code == 409
            assert audio(client, "other").status_code == 429
            assert client.get(BASE+"other/result").status_code == 404
        finally:
            gate.release.set()
        assert gate.finished.wait(2)
        deadline = time.monotonic() + 2
        while client.get("/health").json()["in_flight"] and time.monotonic() < deadline:
            time.sleep(.005)
        assert audio(client, "late").status_code == 409  # original result still unread
        assert client.get(url).json()["text"] == "reply"
        assert client.get(url).json()["text"] == "reply"
        for key, seconds in [("late", 1), ("other", .5)]:
            response = audio(client, key)
            deadline = time.monotonic() + 2
            while response.status_code == 202 and time.monotonic() < deadline:
                response = client.get(response.headers["Location"])
            assert response.json()["heard_seconds"] == seconds


def test_meeting_workers_concurrent_and_histories_isolated():
    class Model:
        barrier = threading.Barrier(2)
        def generate(self, messages, audios):
            self.barrier.wait(timeout=3)
            assert all(np.array_equal(audios[0], x) for x in audios)
            return str(round(float(audios[-1][0]) * 32768))
    with TestClient(app(Model(), response_timeout_ms=3000, max_meetings=2)) as client, ThreadPoolExecutor(2) as pool:
        for _ in range(2):
            futures = [pool.submit(audio, client, "m" + str(v), STEP, v) for v in (12, 34)]
            assert [f.result().json()["text"] for f in futures] == ["12", "34"]
        assert audio(client, "over").status_code == 429


def test_inference_error_preserves_partial_commits_and_capacity():
    with TestClient(app(ReplayBackend(["one", "invalid<tag>"]), max_meetings=1)) as client:
        failed = audio(client, "failed", 2 * STEP)
        assert failed.status_code == 500
        assert failed.json()["text"] == "one" and len(failed.json()["turns"]) == 1
        assert failed.json()["status"] == "error" and failed.json()["unprocessed_seconds"] == .5
        assert client.get(BASE+"failed/result").json() == failed.json()
        assert client.get("/health").json()["active_meetings"] == 0
        assert audio(client, "new", 2).status_code == 200


def test_length_gap_in_a_multi_turn_packet_preserves_commits_and_continues():
    class Capped:
        replies = iter(["before", WAIT, None, None, " after", None, None])
        def generate(self, messages, audios):
            text = next(self.replies)
            if text is None:
                raise GenerationTruncated("vLLM", 8, 60, 8, 4096)
            return text

    with TestClient(app(Capped())) as client:
        row = audio(client, "long", count=4 * STEP).json()
        assert row["status"] == "partial" and row["action"] == "DIRECT" and row["text"] == "before after"
        assert [t["action"] for t in row["turns"]] == ["DIRECT", "WAIT", None, "DIRECT"]
        gap, = row["gaps"]
        assert (gap["turn"], gap["event"], gap["start_seconds"], gap["end_seconds"]) == (3, "audio", .5, 1.5)
        assert audio(client, "long", count=100).json()["status"] == "buffered"
        end = client.post(BASE+"long/end").json()
        assert end["status"] == "ended" and end["action"] is None and len(end["gaps"]) == 1
        assert end["gaps"][0]["start_seconds"] == 2
        assert end["gaps"][0]["end_seconds"] == pytest.approx(2 + 100/16000)
        assert end["turns"][-1]["end_reason"] == "gap_already_reported"
        assert client.get(BASE+"long/result").json() == end


def test_gap_only_response_and_following_end_do_not_claim_model_wait():
    class Capped:
        def generate(self, messages, audios):
            raise GenerationTruncated("vLLM", 8, 60, 8, 4096)
    with TestClient(app(Capped())) as client:
        row = audio(client, "only-gap").json()
        assert (row["status"], row["action"], row["text"]) == ("partial", None, "")
        assert len(row["gaps"]) == 1
        end = client.post(BASE+"only-gap/end").json()
        assert end["status"] == "ended" and end["action"] is None and end["gaps"] == []


def test_invalid_audio_does_not_allocate_or_modify_a_meeting():
    with TestClient(app(ReplayBackend(), max_packet_seconds=1.0)) as client:
        assert client.post(BASE+"test/end").status_code == 404
        for content, kind, expected in [
            (b"bad", "audio/pcm", 400), (b"", "audio/wav", 400), (b"bad", "audio/mp3", 415),
            (wav(bytes(200), 8000), "audio/wav", 415), (bytes(32002), "audio/pcm", 413),
            (wav(bytes(200))[:-1], "audio/wav", 400), (wav(bytes(32002)), "audio/wav", 413),
        ]:
            r = client.post(BASE+"test/chunks", content=content, headers={"Content-Type":kind})
            assert r.status_code == expected, r.text
        assert client.get("/health").json()["active_meetings"] == 0
        assert audio(client, "test", 100).json()["status"] == "buffered"


def test_idle_meeting_expiry_and_bounded_terminal_cache():
    with TestClient(app(ReplayBackend(), idle_timeout_seconds=.1, max_meetings=1)) as client:
        assert audio(client, "inactive", count=2).status_code == 200
        time.sleep(.12)
        assert client.get(BASE+"inactive/result").status_code == 410
        for i in range(3):
            key = str(i)
            assert audio(client, key).status_code == 200
            assert client.post(BASE+key+"/end").status_code == 200
        assert client.get(BASE+"0/result").status_code == 404


def test_slow_upload_does_not_hold_an_inference_slot_or_consume_audio():
    import httpx

    async def exercise():
        api = app(ReplayBackend(), max_in_flight=1, upload_timeout_seconds=.05)
        async def slow_body():
            yield bytes(100)
            await asyncio.sleep(1)
            yield bytes(100)
        async with api.router.lifespan_context(api), httpx.AsyncClient(
                transport=httpx.ASGITransport(app=api), base_url="http://local") as client:
            url = BASE+"slow/chunks"
            response = await client.post(url, content=slow_body(), headers=RAW)
            assert response.status_code == 408
            health = (await client.get("/health")).json()
            assert health["active_meetings"] == health["in_flight"] == 0
            response = await client.post(url, content=bytes(STEP * 2), headers=RAW)
            assert response.json()["heard_seconds"] == .5
    asyncio.run(exercise())
