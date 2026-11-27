"""Exercise meeting admission, timeouts and incremental audio over the real ASGI routes."""
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from io import BytesIO
import json
import logging
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
from fourth_asr.http_service import create_app
from fourth_asr.runtime.streaming import GenerationTruncated, ReplayBackend
from fourth_asr.service_session import FourthStreamOptions


BASE = "/v1/meetings/"
STEP = 8000
RAW = {"Content-Type": "audio/pcm"}
WAIT = "<|WAIT|>"


@pytest.fixture
def request_records(caplog, monkeypatch):
    name = "fourth_asr.http_requests"
    monkeypatch.setattr(logging.getLogger(name), "propagate", True)
    caplog.set_level(logging.INFO, logger=name)

    def read(event, *, response=None, key=None):
        rows = [json.loads(record.getMessage()) for record in caplog.records if record.name == name]
        return [row for row in rows if row["event"] == event
                and (response is None or row.get("request_id") == response.headers["X-Request-ID"])
                and (key is None or row.get("meeting_id") == key)]

    return read


def app(backend, chunk_seconds=.5, **settings):
    return create_app(lambda: backend, FourthStreamOptions(protocol="plain", chunk_seconds=chunk_seconds), **settings)


def audio(client, key, count=STEP, value=0):
    return client.post(BASE + key + "/chunks", content=np.full(count, value, dtype="<i2").tobytes(), headers=RAW)


def wav(data, rate=16000):
    f = BytesIO()
    with wave.open(f, "wb") as out:
        out.setparams((1, 2, rate, 0, "NONE", ""))
        out.writeframes(data)
    return f.getvalue()


def assert_src(response, expected):
    assert response.status_code == 200, response.text
    assert response.json() == {"src": expected}


def test_integer_second_json_packets_only_return_new_transcript():
    # The caller sends 1, 2 and 3 seconds without querying server state.
    replies = [WAIT, "Please", " confirm", WAIT, " the time", ".", WAIT]
    with TestClient(app(ReplayBackend(replies), chunk_seconds=1)) as client:
        path = BASE + "meeting-001"
        for seconds, text in [(1, ""), (2, "Please confirm"), (3, " the time.")]:
            data = base64.b64encode(bytes(seconds * 16000 * 4)).decode("ascii")
            assert_src(client.post(path + "/chunks", json={"data": data}), text)
        assert_src(client.post(path + "/end"), "")


def test_speaker_http_src_and_detailed_history():
    raw = "<|DIRECT|><|SPK_CHANGE|>能。<|SPK_CHANGE|>好。"
    backend = ReplayBackend(["<|DIRECT|>能听到吗？", raw, WAIT])
    options = FourthStreamOptions(enable_recovery=False)
    with TestClient(create_app(lambda: backend, options)) as client:
        assert client.get("/health").json()["action_protocol"] == "speaker"
        assert_src(audio(client, "s", count=32000), "能听到吗？能。好。")
        row = client.get(BASE + "s/result").json()["turns"][1]
        assert row["text"] == raw
        assert row["transcript_parts"] == [{"text": "能。", "speaker_change": True},
                                           {"text": "好。", "speaker_change": True}]
        assert_src(client.post(BASE + "s/end"), "")


def test_json_float32_is_lossless_across_json_wav_and_pcm_packets():
    class Capture(ReplayBackend):
        def __init__(self):
            super().__init__(["first", " second", WAIT])
            self.blocks = []

        def generate(self, messages, audios):
            if messages[-1]["content"] == "<audio>":
                self.blocks.append(audios[-1].copy())
            return super().generate(messages, audios)

    def encoded(samples):
        return {"data": base64.b64encode(samples.tobytes()).decode("ascii")}

    model = Capture()
    # Include floats that would change or disappear if converted through int16.
    start = np.resize(np.array([-.12345679, 1e-9, -1., 1.], dtype="<f4"), STEP - 2)
    middle = np.array([32767], dtype="<i2")
    last = np.array([-32768], dtype="<i2")
    tail = np.array([-.9876543, 1e-9], dtype="<f4")
    path = BASE + "mixed"
    with TestClient(app(model, max_packet_seconds=.5)) as client:
        first = client.post(path + "/chunks", json=encoded(start))
        assert_src(first, "")
        buffered = client.post(path + "/chunks", content=wav(middle.tobytes()),
                               headers={"Content-Type": "audio/wav"})
        assert_src(buffered, "")
        assert_src(client.post(path + "/chunks", headers=RAW, content=last.tobytes()), "first")
        assert_src(client.post(path + "/chunks", json=encoded(tail)), "")
        assert_src(client.post(path + "/end"), " second")
        ended = client.get(path + "/result").json()
    assert ended["text"] == " second" and ended["heard_seconds"] == pytest.approx(.5 + 2 / 16000)
    full, padded = model.blocks
    expected = np.concatenate([start, middle.astype("f4") / 32768, last.astype("f4") / 32768])
    np.testing.assert_array_equal(full, expected)
    np.testing.assert_array_equal(padded[:len(tail)], tail)
    assert full.dtype == padded.dtype == np.float32 and not np.any(padded[len(tail):])


def test_invalid_float32_requests_leave_buffer_and_capacity_intact():
    headers = {"Content-Type": "application/json; charset=utf-8"}
    path = BASE + "malformed"
    def payload(raw):
        return json.dumps({"data": base64.b64encode(raw).decode("ascii")}).encode("utf-8")

    bad_bodies = [
        b"{", b"[]", b"{}", b'{"data":null}', b'{"data":[]}', b'{"data":""}',
        b'{"data":"AAAAAA==!"}', payload(bytes(5)),
        *(payload(np.array([value], dtype="<f4").tobytes()) for value in [np.nan, np.inf, -np.inf, -4.]),
    ]
    with TestClient(app(ReplayBackend(), max_packet_seconds=.5, max_meetings=1)) as client:
        assert client.post(path + "/chunks", content=bad_bodies[0], headers=headers).status_code == 400
        assert client.get("/health").json()["active_meetings"] == 0
        assert_src(client.post(path + "/chunks", content=payload(bytes(4)), headers=headers), "")
        initial = client.get(path + "/result").json()
        assert initial["buffered_seconds"] == 1 / 16000
        for body in bad_bodies:
            assert client.post(path + "/chunks", content=body, headers=headers).status_code == 400
        oversized = payload(bytes(4 * (STEP + 1)))
        assert client.post(path + "/chunks", content=oversized, headers=headers).status_code == 413
        # The preceding bad packets neither advanced the meeting nor destroyed its previous tail.
        assert client.get(path + "/result").json() == initial
        assert_src(client.post(path + "/chunks", content=payload(bytes(4 * STEP)), headers=headers), "")
        limit = client.get(path + "/result").json()
        assert limit["buffered_seconds"] == 1 / 16000 and limit["heard_seconds"] == .5


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
        assert_src(audio(client, "abc", 100, 33), "")
        buffered = client.get(BASE+"abc/result").json()
        assert (buffered["status"], buffered["action"], buffered["heard_seconds"]) == ("buffered", None, 0)
        body = np.full(2 * STEP + 100, 44, dtype="<i2").tobytes()
        assert_src(client.post(BASE+"abc/chunks", headers={"Content-Type": "audio/wav"}, content=wav(body)), "a")
        row = client.get(BASE+"abc/result").json()
        assert (row["status"], row["action"], row["text"]) == ("ok", "DIRECT", "a")
        assert [t["action"] for t in row["turns"]] == ["DIRECT", "WAIT"]
        assert row["buffered_seconds"] == 200 / 16000 and row["heard_seconds"] == 1
        assert_src(client.post(BASE+"abc/end"), " b")
        end = client.get(BASE+"abc/result").json()
        assert end["status"] == "ended" and end["text"] == " b"
        assert end["heard_seconds"] == pytest.approx(1 + 200 / 16000)
        assert [t["event"] for t in end["turns"]] == ["audio", "end_of_audio"]
        np.testing.assert_array_equal(model.heard[0][:100], np.full(100, 33 / 32768))
        assert np.all(model.heard[0][100:] == 44/32768)
        assert np.all(model.heard[2][:200] == 44/32768) and not np.any(model.heard[2][200:])
        assert audio(client, "abc").status_code == 409
        assert client.get("/health").json()["active_meetings"] == 0


def test_slow_inference_waits_on_same_post_and_keeps_its_capacity():
    class Gate:
        release = threading.Event()
        entered = threading.Event()
        def generate(self, messages, audios):
            self.entered.set()
            assert self.release.wait(3)
            return "reply"
    gate = Gate()
    with TestClient(app(gate, max_in_flight=1)) as client, ThreadPoolExecutor(1) as pool:
        request = pool.submit(audio, client, "late")
        try:
            assert gate.entered.wait(2)
            assert audio(client, "late").status_code == 409
            assert audio(client, "other").status_code == 429
            with pytest.raises(FutureTimeout):
                request.result(timeout=1.0)  # Longer than the former 900 ms return-202 budget.
        finally:
            gate.release.set()
        assert_src(request.result(timeout=2), "reply")
        assert_src(audio(client, "late"), "reply")
        assert_src(audio(client, "other"), "reply")


def test_meeting_workers_concurrent_and_histories_isolated():
    class Model:
        barrier = threading.Barrier(2)
        def generate(self, messages, audios):
            self.barrier.wait(timeout=3)
            assert all(np.array_equal(audios[0], x) for x in audios)
            return str(round(float(audios[-1][0]) * 32768))
    with TestClient(app(Model(), max_meetings=2)) as client, ThreadPoolExecutor(2) as pool:
        for _ in range(2):
            futures = [pool.submit(audio, client, "m" + str(v), STEP, v) for v in (12, 34)]
            assert [f.result().json() for f in futures] == [{"src": "12"}, {"src": "34"}]
        assert audio(client, "over").status_code == 429


def test_cancelled_http_waiter_does_not_discard_audio_or_lock_the_meeting(request_records):
    import httpx

    entered, release = threading.Event(), threading.Event()

    class Model:
        def generate(self, messages, audios):
            entered.set()
            assert release.wait(3)
            return str(len(audios))

    async def exercise():
        api = app(Model(), max_in_flight=1)
        async with api.router.lifespan_context(api), httpx.AsyncClient(
                transport=httpx.ASGITransport(app=api), base_url="http://local") as client:
            url = BASE + "cancel/chunks"
            body = bytes(STEP * 2)
            request = asyncio.create_task(client.post(url, content=body, headers=RAW))
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                request.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await request
                conflict = await client.post(url, content=body, headers=RAW)
                assert conflict.status_code == 409
                rejected, = request_records("request_rejected", response=conflict)
                assert (rejected["reason"], rejected["blocking_operation"],
                        rejected["blocking_phase"]) == ("in_flight", "chunks", "processing")
                assert rejected["blocking_elapsed_ms"] >= 0
                original_id = rejected["blocking_request_id"]
                assert original_id != conflict.headers["X-Request-ID"]
            finally:
                release.set()
            # Once the first operation completes, a caller can keep sending directly.
            deadline = time.monotonic() + 2
            while (await client.get("/health")).json()["in_flight"] and time.monotonic() < deadline:
                await asyncio.sleep(.005)
            finished, = request_records("request_finished", key="cancel")
            assert (finished["request_id"], finished["status_code"], finished["src_chars"]) == (original_id, 200, 1)
            assert_src(await client.post(url, content=body, headers=RAW), "2")

    asyncio.run(exercise())


def test_inference_error_preserves_partial_commits_and_capacity(request_records):
    with TestClient(app(ReplayBackend(["one", "invalid<tag>"]), max_meetings=1)) as client:
        failed = audio(client, "failed", 2 * STEP)
        assert failed.status_code == 500
        diagnostic = client.get(BASE+"failed/result").json()
        assert diagnostic["text"] == "one" and len(diagnostic["turns"]) == 1
        assert diagnostic["status"] == "error" and diagnostic["unprocessed_seconds"] == .5
        assert client.get("/health").json()["active_meetings"] == 0
        conflict = audio(client, "failed")
        assert conflict.status_code == 409
        rejection, = request_records("request_rejected", response=conflict)
        assert (rejection["reason"], rejection["last_request_id"]) == ("failed", failed.headers["X-Request-ID"])
        trace, = request_records("processing_error", response=failed)
        assert trace["error_type"] and trace["stack"]
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
        assert_src(audio(client, "long", count=4 * STEP), "before after")
        row = client.get(BASE+"long/result").json()
        assert row["status"] == "partial" and row["action"] == "DIRECT" and row["text"] == "before after"
        assert [t["action"] for t in row["turns"]] == ["DIRECT", "WAIT", None, "DIRECT"]
        gap, = row["gaps"]
        assert (gap["turn"], gap["event"], gap["start_seconds"], gap["end_seconds"]) == (3, "audio", .5, 1.5)
        assert_src(audio(client, "long", count=100), "")
        assert_src(client.post(BASE+"long/end"), "")
        end = client.get(BASE+"long/result").json()
        assert end["status"] == "ended" and end["action"] is None and len(end["gaps"]) == 1
        assert end["gaps"][0]["start_seconds"] == 2
        assert end["gaps"][0]["end_seconds"] == pytest.approx(2 + 100/16000)
        assert end["turns"][-1]["end_reason"] == "gap_already_reported"


def test_gap_only_response_and_following_end_do_not_claim_model_wait(tmp_path):
    class Capped:
        def generate(self, messages, audios):
            raise GenerationTruncated("vLLM", 8, 60, 8, 4096)
    with TestClient(app(Capped(), debug_dir=tmp_path)) as client:
        assert_src(audio(client, "only-gap"), "")
        row = client.get(BASE+"only-gap/result").json()
        assert (row["status"], row["action"], row["text"]) == ("partial", None, "")
        assert len(row["gaps"]) == 1
        assert_src(client.post(BASE+"only-gap/end"), "")
        end = client.get(BASE+"only-gap/result").json()
        assert end["status"] == "ended" and end["action"] is None and end["gaps"] == []
        log, = tmp_path.rglob("events.jsonl")
        rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        assert rows[1]["动作"] == "解码缺口" and rows[2]["解码缺口轮"] == [1]
        assert rows[-2]["动作"] == "跳过收尾（缺口已记录）"


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
        assert_src(audio(client, "test", 100), "")


def test_idle_meeting_expiry_and_bounded_terminal_cache(request_records):
    with TestClient(app(ReplayBackend(), idle_timeout_seconds=.1, max_meetings=1)) as client:
        first = audio(client, "inactive", count=2)
        assert first.status_code == 200
        time.sleep(.12)
        assert client.get(BASE+"inactive/result").status_code == 410
        conflict = client.post(BASE+"inactive/end")
        assert conflict.status_code == 409
        rejection, = request_records("request_rejected", response=conflict)
        assert (rejection["reason"], rejection["last_request_id"]) == ("expired", first.headers["X-Request-ID"])
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
            assert_src(response, "")
    asyncio.run(exercise())


def test_conflict_during_upload_links_both_requests_and_terminal_state(request_records, caplog, tmp_path):
    import httpx

    async def exercise():
        waiting, release = asyncio.Event(), asyncio.Event()
        transcript = "Confidential transcript."
        pcm = np.full(STEP, .12345, dtype="<f4").tobytes()
        encoded = base64.b64encode(pcm).decode("ascii")
        packet = json.dumps({"data": encoded}).encode("ascii")

        async def upload():
            yield packet[:50]
            waiting.set()
            await release.wait()
            yield packet[50:]

        api = app(ReplayBackend([transcript, WAIT]), debug_dir=tmp_path)
        async with api.router.lifespan_context(api), httpx.AsyncClient(
                transport=httpx.ASGITransport(app=api), base_url="http://local") as client:
            path = BASE + "upload"
            task = asyncio.create_task(client.post(path + "/chunks", content=upload(),
                                                   headers={"Content-Type": "application/json"}))
            try:
                await asyncio.wait_for(waiting.wait(), timeout=2)
                busy = await client.post(path + "/end")
                assert busy.status_code == 409
            finally:
                release.set()
                accepted = await task
            assert_src(accepted, transcript)
            busy_log, = request_records("request_rejected", response=busy)
            assert (busy_log["reason"], busy_log["operation"], busy_log["blocking_operation"],
                    busy_log["blocking_phase"]) == ("in_flight", "end", "chunks", "upload")
            assert busy_log["blocking_request_id"] == accepted.headers["X-Request-ID"]
            assert busy_log["blocking_elapsed_ms"] >= 0
            finished, = request_records("request_finished", response=accepted)
            assert finished["meeting_id"] == "upload" and finished["status_code"] == 200
            assert finished["duration_ms"] >= finished["upload_ms"] >= 0
            assert finished["src_chars"] == len(transcript)
            end = await client.post(path + "/end")
            assert_src(end, "")
            closed = await client.post(path + "/end")
            assert closed.status_code == 409
            closed_log, = request_records("request_rejected", response=closed)
            assert (closed_log["reason"], closed_log["last_request_id"]) == ("ended", end.headers["X-Request-ID"])
            assert len({r.headers["X-Request-ID"] for r in (accepted, busy, end, closed)}) == 4
            timeline, = (tmp_path / "upload").glob("*/events.jsonl")
            rows = [json.loads(line) for line in timeline.read_text(encoding="utf-8").splitlines()]
            rejected, = [r for r in rows if r.get("请求号") == busy.headers["X-Request-ID"]]
            assert (rejected["占用阶段"], rejected["占用请求号"]) == ("接收上传", accepted.headers["X-Request-ID"])
        for record in caplog.records:
            if record.name == "fourth_asr.http_requests":
                line = record.getMessage()
                assert transcript not in line and encoded[:96] not in line
                assert set(json.loads(line)).isdisjoint({"src", "text", "data", "audio"})

    asyncio.run(exercise())
