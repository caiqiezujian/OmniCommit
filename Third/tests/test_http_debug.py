"""Cross-check a captured meeting against what the HTTP client and model actually saw."""
import base64
from concurrent.futures import ThreadPoolExecutor
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
from third_asr.http_service import create_app
from third_asr.protocol import DIRECT, EOS, WAIT
from third_asr.runtime.streaming import ReplayBackend
from third_asr.service_session import ThirdStreamOptions


STEP = 8000
URL = "/v1/meetings/replay-001"


def make_app(root, replies):
    options = ThirdStreamOptions(protocol="explicit", chunk_seconds=.5, enable_recovery=False)
    return create_app(lambda: ReplayBackend(replies), options, debug_dir=root)


def post_samples(client, samples):
    encoded = base64.b64encode(samples.astype("<f4").tobytes()).decode("ascii")
    return client.post(URL + "/chunks", json={"data": encoded})


def read_archive(root):
    events_file, = (path for path in root.rglob("*") if path.is_file())
    assert events_file.name == "events.jsonl"
    return [json.loads(line) for line in events_file.read_text(encoding="utf-8").splitlines()]


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


def test_busy_payload_is_rejected_before_ingestion(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class GatedBackend(ReplayBackend):
        def generate(self, messages, audios):
            entered.set()
            assert release.wait(5)
            return super().generate(messages, audios)

    options = ThirdStreamOptions(protocol="explicit", chunk_seconds=.5, enable_recovery=False)
    app = create_app(lambda: GatedBackend([WAIT]), options, debug_dir=tmp_path)
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
