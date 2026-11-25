"""Exercise client silence measurements at the HTTP and model-history boundaries."""
import base64
import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_asr.http_service import create_app
from fourth_asr.protocol import DIRECT, SPK_CHANGE, WAIT
from fourth_asr.runtime.streaming import ReplayBackend, SAMPLE_RATE
from fourth_asr.service_session import FourthStreamOptions


MEETING = "/v1/meetings/punctuation"


class Capture(ReplayBackend):
    def __init__(self, replies):
        super().__init__(replies)
        self.prompts = []

    def generate(self, messages, audios):
        self.prompts.append(([(m["role"], m["content"]) for m in messages], [a.copy() for a in audios]))
        return super().generate(messages, audios)


def client_for(model, **settings):
    options = FourthStreamOptions(chunk_seconds=1, protocol="speaker", enable_recovery=False)
    return TestClient(create_app(lambda: model, options, **settings))


def packet(samples=SAMPLE_RATE, value=0., **metadata):
    data = np.full(samples, value, dtype="<f4").tobytes()
    return {"data": base64.b64encode(data).decode("ascii"), **metadata}


def assert_src(response, expected):
    assert response.status_code == 200, response.text
    assert response.json() == {"src": expected}


def test_disabled_by_default_even_when_clients_send_silence():
    model = Capture([DIRECT + "第一句。", DIRECT + " Second.", DIRECT + "继续。"])
    with client_for(model) as client:
        assert client.get("/health").json()["packet_period_to_comma"] is False
        assert_src(client.post(MEETING + "/chunks", json=packet(silence_ms=0)), "第一句。")
        assert_src(client.post(MEETING + "/chunks", json=packet(trailing_silence_ms=200)), " Second.")
        assert_src(client.post(MEETING + "/chunks", json=packet()), "继续。")
    assert model.prompts[-1][0][-4] == ("assistant", DIRECT + "第一句。")
    assert model.prompts[-1][0][-2] == ("assistant", DIRECT + " Second.")


def test_five_second_packet_edits_only_its_final_text_and_keeps_trailing_wait():
    replies = [DIRECT + "开始。", WAIT, DIRECT + "三。", DIRECT + "知道。", WAIT]
    model = Capture(replies + [WAIT, WAIT])
    with client_for(model, packet_period_to_comma=True) as client:
        assert client.get("/health").json()["packet_period_to_comma"] is True
        response = client.post(MEETING + "/chunks", json=packet(5 * SAMPLE_RATE, silence_ms=120))
        assert_src(response, "开始。三。知道，")
        turns = client.get(MEETING + "/result").json()["turns"]
        assert [t["text"] for t in turns] == replies[:3] + [DIRECT + "知道，", WAIT]
        # An all-WAIT packet has no text of its own, even with a short silence measurement.
        assert_src(client.post(MEETING + "/chunks", json=packet(silence_ms=0)), "")
        assert_src(client.post(MEETING + "/chunks", json=packet()), "")
    history = [text for role, text in model.prompts[-1][0] if role == "assistant"]
    assert history == replies[:3] + [DIRECT + "知道，", WAIT, WAIT]


def test_five_second_packet_leaves_internal_periods_when_last_turn_is_direct():
    model = Capture([DIRECT + "开始。", WAIT, DIRECT + "好。", DIRECT + "收到。", DIRECT + "Next."])
    with client_for(model, packet_period_to_comma=True) as client:
        response = client.post(MEETING + "/chunks", json=packet(5 * SAMPLE_RATE, silence_ms=200))
        assert_src(response, "开始。好。收到。Next,")
        turns = client.get(MEETING + "/result").json()["turns"]
        assert [t.get("terminal_period_to_comma", False) for t in turns] == [False] * 4 + [True]


def test_new_packet_of_waits_cannot_change_an_already_returned_sentence():
    model = Capture([DIRECT + "结束。", WAIT, WAIT])
    with client_for(model, packet_period_to_comma=True) as client:
        assert_src(client.post(MEETING + "/chunks", json=packet(silence_ms=500)), "结束。")
        assert_src(client.post(MEETING + "/chunks", json=packet(silence_ms=100)), "")
        assert_src(client.post(MEETING + "/end"), "")
    assert [text for role, text in model.prompts[-1][0] if role == "assistant"] == [DIRECT + "结束。", WAIT]


def test_corrected_turn_is_the_single_source_for_client_and_next_model_prompt():
    raw = DIRECT + "他们来了。" + SPK_CHANGE + "我知道。"
    corrected = raw[:-1] + "，"
    model = Capture([DIRECT + "第一句。", DIRECT + "第二句。", raw, WAIT])
    with client_for(model, packet_period_to_comma=True) as client:
        for i, metadata, expected in [
            (1, {}, "第一句。"), (2, {"silence_ms": 300}, "第二句。"),
            (3, {"silence_ms": 199.5}, "他们来了。我知道，"),
        ]:
            assert_src(client.post(MEETING + "/chunks", json=packet(value=i / 8, **metadata)), expected)
        turn = client.get(MEETING + "/result").json()["turns"][0]
        assert turn["action"] == "DIRECT" and turn["text"] == corrected
        assert turn["transcript_parts"] == [{"text": "他们来了。", "speaker_change": False},
                                            {"text": "我知道，", "speaker_change": True}]
        assert_src(client.post(MEETING + "/chunks", json=packet(value=.5)), "")

    messages, audios = model.prompts[3]
    assert messages[1:] == [
        ("user", "<audio>"), ("assistant", DIRECT + "第一句。"),
        ("user", "<audio>"), ("assistant", DIRECT + "第二句。"),
        ("user", "<audio>"), ("assistant", corrected), ("user", "<audio>"),
    ]
    assert len(audios) == 4 and len(model.prompts) == 4
    for index, audio in enumerate(audios, 1):
        assert len(audio) == SAMPLE_RATE and np.all(audio == index / 8)


@pytest.mark.parametrize("silence,text,expected", [
    (0, "We can.", "We can,"), (299.99, "我知道。", "我知道，"), (300, "好了。", "好了。"),
    (310, "OK.", "OK."), (120, "Version 3.14", "Version 3.14"),
    (120, "Ask Dr.", "Ask Dr."), (120, "Well...", "Well..."),
    (120, "明白吗？", "明白吗？"), (100, "“好的。”", "“好的，”"),
])
def test_punctuation_and_threshold(silence, text, expected):
    model = Capture([DIRECT + text, WAIT])
    with client_for(model, packet_period_to_comma=True) as client:
        assert_src(client.post(MEETING + "/chunks", json=packet(silence_ms=silence)), expected)
        row = client.get(MEETING + "/result").json()["turns"][0]
        assert row["text"] == DIRECT + expected
        assert row["action"] == "DIRECT"
        assert_src(client.post(MEETING + "/end"), "")
    assert model.prompts[-1][0][-2] == ("assistant", row["text"])


def test_legacy_measurements_follow_model_endpoints_across_packet_splits_and_flush():
    model = Capture([DIRECT + "One.", DIRECT + " Two.", DIRECT + " Three.", DIRECT + " Four.", WAIT])
    with client_for(model, packet_period_to_comma=True) as client:
        # A short first upload ends inside turn 1. The next upload spans the
        # remainder of turn 1 and all of turn 2; its measurement belongs to turn 2.
        assert_src(client.post(MEETING + "/chunks", json=packet(4000, trailing_silence_ms=200)), "")
        assert_src(client.post(MEETING + "/chunks", json=packet(28000, trailing_silence_ms=100)), "One. Two,")
        # Turn 3 ends before the packet's measurement; only /end consumes the measured tail.
        assert_src(client.post(MEETING + "/chunks", json=packet(20000, trailing_silence_ms=100)), " Three.")
        assert_src(client.post(MEETING + "/end"), " Four,")
    assert [body for role, body in model.prompts[-1][0] if role == "assistant"] == [
        DIRECT + "One.", DIRECT + " Two,", DIRECT + " Three.", DIRECT + " Four,",
    ]


def test_separately_emitted_dot_completes_abbreviation_in_preceding_turn():
    model = Capture([DIRECT + "Please ask Dr", DIRECT + ".", WAIT])
    with client_for(model, packet_period_to_comma=True) as client:
        assert_src(client.post(MEETING + "/chunks", json=packet()), "Please ask Dr")
        assert_src(client.post(MEETING + "/chunks", json=packet(trailing_silence_ms=200)), ".")
        assert_src(client.post(MEETING + "/end"), "")
    assert model.prompts[-1][0][-2] == ("assistant", DIRECT + ".")


def test_invalid_measurements_do_not_consume_or_relabel_buffered_audio():
    model = Capture([DIRECT + "你好。"])
    with client_for(model, packet_period_to_comma=True) as client:
        assert_src(client.post(MEETING + "/chunks", json=packet(8000, trailing_silence_ms=100)), "")
        for invalid in (-1, "200", True, float("nan"), float("inf")):
            body = json.dumps(packet(8000, silence_ms=invalid)).encode()
            response = client.post(MEETING + "/chunks", content=body, headers={"Content-Type": "application/json"})
            assert 400 <= response.status_code < 500, response.text
        conflict = client.post(MEETING + "/chunks", json=packet(8000, silence_ms=500, trailing_silence_ms=100))
        assert conflict.status_code == 400, conflict.text
        assert not model.prompts
        # Omitting a measurement at the current endpoint keeps the punctuation.
        assert_src(client.post(MEETING + "/chunks", json=packet(8000)), "你好。")


@pytest.mark.parametrize("previous,period,src", [
    ("你好", "。", "，"), ("Ask Dr", ".", "."), ("All done", ".", ","),
])
def test_a_separately_decoded_period(previous, period, src):
    with client_for(Capture([DIRECT + previous, DIRECT + period]), packet_period_to_comma=True) as client:
        assert_src(client.post(MEETING + "/chunks", json=packet()), previous)
        assert_src(client.post(MEETING + "/chunks", json=packet(trailing_silence_ms=120)), src)
