"""Cross-check a captured meeting against what the HTTP client and model actually saw."""
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys
import threading

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_asr.http_service import create_app
from fourth_asr.protocol import DIRECT, EOS, WAIT
from fourth_asr.runtime.streaming import ReplayBackend
from fourth_asr.service_session import FourthStreamOptions


STEP = 8000
URL = "/v1/meetings/replay-001"


def make_app(root, replies):
    options = FourthStreamOptions(protocol="explicit", chunk_seconds=.5, enable_recovery=False)
    return create_app(lambda: ReplayBackend(replies), options, debug_dir=root, event_log_mode="detailed")


def post_samples(client, samples, **metadata):
    encoded = base64.b64encode(samples.astype("<f4").tobytes()).decode("ascii")
    return client.post(URL + "/chunks", json={"data": encoded, **metadata})


def read_archive(root):
    trace, = root.rglob("trace.jsonl")
    assert set(p.name for p in trace.parent.iterdir()) == {"trace.jsonl", "events.jsonl"}
    return [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]


def read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def for_request(events, response, name):
    request_id = response.headers["X-Request-ID"]
    return [row for row in events if row.get("request_id") == request_id and row["event"] == name]


def check_response(events, response):
    saved, = for_request(events, response, "response")
    assert saved["status_code"] == response.status_code
    assert saved["body"] == response.json()


def test_text_archive_preserves_packet_boundaries_turns_and_http_results(tmp_path):
    # One model turn spans both packets.
    first = np.resize(np.array([1e-9, -.12345678], dtype="f4"), 128)
    second = np.resize(np.array([.99999, -.00000034, .25], dtype="f4"), 2 * STEP)
    replies = [WAIT, DIRECT + "谢谢。", DIRECT + " Bye.", WAIT]
    with TestClient(make_app(tmp_path, replies)) as client:
        buffered = post_samples(client, first)
        spoken = post_samples(client, second)
        ended = client.post(URL + "/end")
        rejected = post_samples(client, second)

    assert [r.status_code for r in [buffered, spoken, ended, rejected]] == [200, 200, 200, 409]
    assert [r.json() for r in [buffered, spoken, ended]] == [{"src": ""}, {"src": "谢谢。"}, {"src": " Bye."}]
    events = read_archive(tmp_path)
    start = 0
    for response, samples in [(buffered, first), (spoken, second)]:
        packet, = for_request(events, response, "packet_audio")
        assert (packet["start_sample"], packet["end_sample"]) == (start, start + len(samples))
        assert "audio_file" not in packet
        body, = for_request(events, response, "request_body")
        assert "body_file" not in body
        assert base64.b64encode(samples.astype("<f4").tobytes()).decode("ascii") not in json.dumps(events)
        start += len(samples)
    assert not for_request(events, rejected, "packet_audio")
    assert not for_request(events, ended, "packet_audio")
    assert not for_request(events, buffered, "model_call")
    for response in (buffered, spoken, ended, rejected):
        check_response(events, response)
    conflict, = for_request(events, rejected, "request_rejected")
    assert conflict["reason"] == "ended"
    turns = [row["turn_result"] for row in events if row["event"] == "turn_finished"]
    assert [turn["text"] for turn in turns] == replies
    assert [turn["event"] for turn in turns] == ["audio", "audio", "audio", "end_of_audio"]
    assert turns[-1]["end_seconds"] == pytest.approx((len(first) + len(second)) / 16000)
    starts = [row for row in events if row["event"] == "turn_started"]
    assert [(row["start_sample"], row["end_sample"]) for row in starts] == [
        (0, STEP), (STEP, 2 * STEP), (2 * STEP, start), (start, start)]
    calls = [row for row in events if row["event"] == "model_call"]
    assert len(calls) == 4
    assert [call["audio_samples"] for call in calls] == [
        [STEP], [STEP, STEP], [STEP, STEP, STEP], [STEP, STEP, STEP]]
    assert calls[-1]["messages"][-1] == {"role": "user", "content": EOS}
    assert [m["content"] for m in calls[-1]["messages"] if m["role"] == "assistant"] == replies[:-1]
    raw = [row for row in events if row["event"] == "model_reply"]
    assert [row["raw_text"] for row in raw] == replies
    assert [row["call_id"] for row in raw] == [row["call_id"] for row in calls]
    assert len({row["call_id"] for row in calls}) == 4
    # Reuse the caller's original inputs with the archived model replies.
    with TestClient(make_app(None, [row["raw_text"] for row in raw])) as replay:
        recovered = [post_samples(replay, data).json() for data in (first, second)]
        recovered.append(replay.post(URL + "/end").json())
    assert recovered == [r.json() for r in (buffered, spoken, ended)]


def test_malformed_model_reply_keeps_partial_commit_and_failed_response(tmp_path):
    replies = [DIRECT + "已经提交", "missing action token"]
    with TestClient(make_app(tmp_path, replies)) as client:
        failed = post_samples(client, np.zeros(2 * STEP, dtype="f4"))
        rejected = client.post(URL + "/end")
    assert failed.status_code == 500
    assert failed.json()["text"] == "已经提交"
    assert rejected.status_code == 409
    events = read_archive(tmp_path)
    finished, = for_request(events, failed, "turn_finished")
    assert finished["turn_result"] == failed.json()["turns"][0]
    assert len(for_request(events, failed, "model_call")) == 2
    assert [row["raw_text"] for row in for_request(events, failed, "model_reply")] == replies
    turn_error, = for_request(events, failed, "turn_error")
    assert turn_error["turn"] == 2 and turn_error["error"] == failed.json()["error"]
    check_response(events, failed)
    conflict, = for_request(events, rejected, "request_rejected")
    assert conflict["reason"] == "failed"
    assert conflict["last_request_id"] == failed.headers["X-Request-ID"]
    readable = read_events(next(tmp_path.rglob("events.jsonl")))
    returned, = [r for r in readable if r["事件"] == "返回"]
    assert (returned["HTTP"], returned["已提交正文"], returned["原因"]) == (500, "已经提交", failed.json()["error"])
    assert "src" not in returned and readable[-1]["前次请求号"] == failed.headers["X-Request-ID"]


def test_invalid_json_logs_size_and_hash_only(tmp_path):
    raw = b' {\r\n "data": "private-malformed-payload"   '
    with TestClient(make_app(tmp_path, [])) as client:
        invalid = client.post(URL + "/chunks", content=raw, headers={"Content-Type": "application/json"})
    assert invalid.status_code == 400
    events = read_archive(tmp_path)
    body, = for_request(events, invalid, "request_body")
    assert body["body_bytes"] == len(raw)
    assert body["body_sha256"] == hashlib.sha256(raw).hexdigest()
    assert "body_file" not in body
    assert "private-malformed-payload" not in json.dumps(events)
    assert not for_request(events, invalid, "packet_audio")
    assert not for_request(events, invalid, "model_call")
    check_response(events, invalid)
    rejection, = read_events(next(tmp_path.rglob("events.jsonl")))
    assert (rejection["事件"], rejection["HTTP"], rejection["原因"]) == ("拒绝", 400, "音频包无效")


def test_busy_payload_is_rejected_before_ingestion(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class GatedBackend(ReplayBackend):
        def generate(self, messages, audios):
            entered.set()
            assert release.wait(5)
            return super().generate(messages, audios)

    options = FourthStreamOptions(protocol="explicit", chunk_seconds=.5, enable_recovery=False)
    app = create_app(lambda: GatedBackend([WAIT]), options, debug_dir=tmp_path, event_log_mode="detailed")
    block = np.full(STEP, .05, dtype="f4")
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(post_samples, client, block)
        try:
            assert entered.wait(3)
            busy = post_samples(client, block)
            assert busy.status_code == 409
        finally:
            release.set()
        accepted = pending.result(timeout=3)
    events = read_archive(tmp_path)
    conflict, = for_request(events, busy, "request_rejected")
    assert conflict["reason"] == "in_flight"
    assert conflict["blocking_request_id"] == accepted.headers["X-Request-ID"]
    assert not for_request(events, busy, "request_body")
    assert not for_request(events, busy, "packet_audio")
    assert not for_request(events, busy, "model_call")
    check_response(events, busy)


@pytest.mark.parametrize("detailed", [False, True])
@pytest.mark.parametrize("silence,expected,note,threshold", [
    (0.03, "我觉得可以", "删除句号", 100), (0.13, "我觉得可以，", "句号改逗号", 150),
])
def test_live_packet_journal_tracks_turns_conflict_and_exact_returned_src(tmp_path, detailed, monkeypatch,
                                                                         silence, expected, note, threshold):
    entered, release = threading.Event(), threading.Event()

    class GatedBackend(ReplayBackend):
        def generate(self, messages, audios):
            if len(audios) == 3 and not entered.is_set():
                entered.set()
                assert release.wait(5)
            return super().generate(messages, audios)

    settings = {"event_log_mode": "detailed"} if detailed else {}
    if not detailed:
        # Full prompts and hashes must not be collected for the readable timeline.
        from fourth_asr import http_service
        monkeypatch.setattr(http_service, "DebugBackend", lambda *a: pytest.fail("verbose wrapper was installed"))
        monkeypatch.setattr(http_service.hashlib, "sha256", lambda *a: pytest.fail("normal packet body was hashed"))
    options = FourthStreamOptions(protocol="explicit", chunk_seconds=1, enable_recovery=False)
    api = create_app(lambda: GatedBackend([DIRECT + "我觉得", DIRECT + "可以。", WAIT, WAIT]), options,
                     debug_dir=tmp_path, packet_period_to_comma=True, **settings)
    pcm = np.zeros(3 * 16000, dtype="f4")
    with TestClient(api) as client, ThreadPoolExecutor(max_workers=1) as pool:
        assert client.get("/health").json()["event_log_mode"] == ("detailed" if detailed else "readable")
        pending = pool.submit(post_samples, client, pcm, tail_silence=silence)
        try:
            assert entered.wait(3)
            path, = (tmp_path / "replay-001").glob("*/events.jsonl")
            rows = read_events(path)
            assert [r["事件"] for r in rows] == ["收包", "逐轮", "逐轮"]
            assert rows[0]["录音区间_s"] == [0, 3] and rows[0]["上报静音_ms"] == silence * 1000
            assert rows[2]["模型正文"] == "可以。" and rows[2]["新增音频_s"] == [1, 2]
            busy = post_samples(client, pcm)
            assert busy.status_code == 409
            rejected = read_events(path)[-1]
            assert (rejected["事件"], rejected["HTTP"], rejected["包"]) == ("拒绝", 409, 2)
            assert rejected["占用请求号"] == rows[0]["请求号"] and rejected["占用阶段"] == "转录中"
        finally:
            release.set()
        reply = pending.result(timeout=3)
        assert reply.json() == {"src": expected}
        assert client.post(URL + "/end").json() == {"src": ""}
        assert client.post(URL + "/end").status_code == 409

    rows = read_events(path)
    assert list(tmp_path.rglob("events.jsonl")) == [path]
    assert {p.name for p in path.parent.iterdir()} == ({"events.jsonl", "trace.jsonl"} if detailed else {"events.jsonl"})
    assert all(datetime.fromisoformat(r["时间"]).utcoffset() == timedelta(hours=8) for r in rows)
    assert path.parent.name.endswith("_" + reply.headers["X-Request-ID"][:12])
    turns = [r for r in rows if r["事件"] == "逐轮"]
    assert [(r["包"], r["轮"], r["动作"]) for r in turns] == [(1, 1, "DIRECT"), (1, 2, "DIRECT"), (1, 3, "WAIT")]
    assert "模型正文" not in turns[-1]
    response, = [r for r in rows if r["事件"] == "返回" and r.get("包") == 1]
    assert response["src"] == reply.json()["src"] and response["请求号"] == reply.headers["X-Request-ID"]
    assert response["标点处理"] == f"第2轮{note}，{silence * 1000} ms < {threshold} ms"
    if detailed:
        event = "packet_period_removed" if silence < .1 else "packet_period_to_comma"
        edit, = [r for r in read_events(path.with_name("trace.jsonl")) if r["event"] == event]
        assert edit["threshold_ms"] == threshold
        assert edit["turn"] == 2 and edit["text"] == DIRECT + expected.removeprefix("我觉得")
    assert [r["事件"] for r in rows[-4:]] == ["结束请求", "收尾轮", "返回", "拒绝"]
    assert rows[-1]["原因"] == "会议已结束"


def test_readable_journal_separates_reused_id_and_flushes_buffer(tmp_path):
    options = FourthStreamOptions(protocol="explicit", chunk_seconds=1, enable_recovery=False)
    api = create_app(ReplayBackend, options, debug_dir=tmp_path, max_meetings=1)
    pcm = bytes(16000)  # half a second of PCM16
    with TestClient(api) as client:
        for name in ("again", "other1", "other2", "again"):
            url = f"/v1/meetings/{name}"
            assert client.post(url + "/chunks", content=pcm, headers={"Content-Type": "audio/pcm"}).json() == {"src": ""}
            assert client.post(url + "/end").json() == {"src": ""}
    runs = list((tmp_path / "again").glob("*/events.jsonl"))
    assert len(runs) == 2
    for file in runs:
        packet, buffered, end, flushed, eos, reply = read_events(file)
        assert packet["包"] == 1 and packet["上报静音_ms"] is None and packet["录音区间_s"] == [0, .5]
        assert buffered["暂存音频_s"] == .5 and buffered["src"] == ""
        assert end["事件"] == "结束请求" and flushed["新增音频_s"] == [0, .5]
        assert eos["事件"] == "收尾轮" and "新增音频_s" not in eos and reply["操作"] == "结束"
