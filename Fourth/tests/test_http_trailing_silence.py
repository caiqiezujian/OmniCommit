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
        assert_src(client.post(MEETING + "/chunks", json=packet(tail_silence=0.0)), "第一句。")
        assert_src(client.post(MEETING + "/chunks", json=packet(trailing_silence_ms=200)), " Second.")
        assert_src(client.post(MEETING + "/chunks", json=packet()), "继续。")
    assert model.prompts[-1][0][-4] == ("assistant", DIRECT + "第一句。")
    assert model.prompts[-1][0][-2] == ("assistant", DIRECT + " Second.")


@pytest.mark.parametrize("silence,ending", [(0.12, "知道"), (0.2, "知道，"), (0.45, "知道。")])
def test_five_second_packet_edits_only_its_final_text_and_keeps_trailing_wait(silence, ending):
    replies = [DIRECT + "开始。", WAIT, DIRECT + "三。", DIRECT + "知道。", WAIT]
    model = Capture(replies + [WAIT, WAIT])
    with client_for(model, packet_period_to_comma=True) as client:
        assert client.get("/health").json()["packet_period_to_comma"] is True
        response = client.post(MEETING + "/chunks", json=packet(5 * SAMPLE_RATE, tail_silence=silence))
        assert_src(response, "开始。三。" + ending)
        turns = client.get(MEETING + "/result").json()["turns"]
        assert [t["text"] for t in turns] == replies[:3] + [DIRECT + ending, WAIT]
        # An all-WAIT packet has no text of its own, even with a short silence measurement.
        assert_src(client.post(MEETING + "/chunks", json=packet(silence_ms=0)), "")
        assert_src(client.post(MEETING + "/chunks", json=packet()), "")
    history = [text for role, text in model.prompts[-1][0] if role == "assistant"]
    assert history == replies[:3] + [DIRECT + ending, WAIT, WAIT]


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


@pytest.mark.parametrize("ms,suffix", [(149.5, ""), (150, "，"), (449.5, "，")])
def test_corrected_turn_is_the_single_source_for_client_and_next_model_prompt(ms, suffix):
    raw = DIRECT + "他们来了。" + SPK_CHANGE + "我知道。"
    corrected = raw[:-1] + suffix
    model = Capture([DIRECT + "第一句。", DIRECT + "第二句。", raw, WAIT])
    with client_for(model, packet_period_to_comma=True) as client:
        for i, metadata, expected in [
            (1, {}, "第一句。"), (2, {"silence_ms": 450}, "第二句。"),
            (3, {"silence_ms": ms}, "他们来了。我知道" + suffix),
        ]:
            assert_src(client.post(MEETING + "/chunks", json=packet(value=i / 8, **metadata)), expected)
        turn = client.get(MEETING + "/result").json()["turns"][0]
        assert turn["action"] == "DIRECT" and turn["text"] == corrected
        assert turn["transcript_parts"] == [{"text": "他们来了。", "speaker_change": False},
                                            {"text": "我知道" + suffix, "speaker_change": True}]
        assert_src(client.post(MEETING + "/chunks", json=packet(value=.5)), "")

    messages, audios = model.prompts[3]
    assert messages[:len(model.prompts[2][0])] == model.prompts[2][0]
    assert messages[1:] == [
        ("user", "<audio>"), ("assistant", DIRECT + "第一句。"),
        ("user", "<audio>"), ("assistant", DIRECT + "第二句。"),
        ("user", "<audio>"), ("assistant", corrected), ("user", "<audio>"),
    ]
    assert len(audios) == 4 and len(model.prompts) == 4
    for index, audio in enumerate(audios, 1):
        assert len(audio) == SAMPLE_RATE and np.all(audio == index / 8)


@pytest.mark.parametrize("silence,text,expected", [
    (0, "We can.", "We can"), (299.99, "我知道。", "我知道，"), (300, "好了。", "好了，"),
    (450, "好了。", "好了。"), (310, "OK.", "OK."), (120, "Version 3.14", "Version 3.14"),
    (120, "Ask Dr.", "Ask Dr."), (120, "Well...", "Well..."),
    (120, "明白吗？", "明白吗？"), (100, "“好的。”", "“好的”"),
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


@pytest.mark.parametrize("measurement,expected_ms,expected_src", [
    ({"tail_silence": 0.0}, 0, "收到"),
    ({"tail_silence": 0.149999}, 149.999, "收到"),
    ({"tail_silence": 0.150}, 150, "收到，"),
    ({"tail_silence": 0.3}, 300, "收到，"),
    ({"tail_silence": 0.449999}, 449.999, "收到，"),
    ({"tail_silence": 0.450}, 450, "收到。"),
    ({"tail_silence": 0.451}, 451, "收到。"),
    ({"silence_ms": 0.3}, 0.3, "收到"),
    ({"trailing_silence_ms": 150}, 150, "收到，"),
])
def test_json_field_units_at_both_thresholds(measurement, expected_ms, expected_src):
    with client_for(Capture([DIRECT + "收到。"]), packet_period_to_comma=True) as client:
        assert_src(client.post(MEETING + "/chunks", json=packet(**measurement)), expected_src)
        result = client.get(MEETING + "/result").json()
        assert result["silence_ms"] == pytest.approx(expected_ms)


@pytest.mark.parametrize("threshold_ms,measurement,expected", [
    (500, {"tail_silence": 0.499}, "收到，"),
    (500, {"tail_silence": 0.500}, "收到。"),
    (500, {"tail_silence": 0.501}, "收到。"),
    (125.5, {"silence_ms": 125.5}, "收到。"),
    (125.5, {"trailing_silence_ms": 125}, "收到"),
    (1001, {"tail_silence": 1.001}, "收到。"),
    (0, {"tail_silence": 0}, "收到。"),
])
def test_configured_threshold_applies_to_packet_even_when_it_ends_in_wait(threshold_ms, measurement, expected):
    model = Capture([DIRECT + "收到。", WAIT])
    with client_for(model, packet_period_to_comma=True, packet_period_threshold_ms=threshold_ms) as client:
        assert client.get("/health").json()["packet_period_threshold_ms"] == threshold_ms
        assert_src(client.post(MEETING + "/chunks", json=packet(2 * SAMPLE_RATE, **measurement)), expected)


@pytest.mark.parametrize("invalid", [-1, float("inf"), float("nan")])
def test_invalid_configured_threshold_is_rejected_at_startup(invalid):
    with pytest.raises(ValueError, match="packet_period_threshold_ms"):
        client_for(Capture([]), packet_period_threshold_ms=invalid)


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
        for field in ("silence_ms", "tail_silence"):
            for invalid in (-1, "0.3", True, None, float("nan"), float("inf")):
                body = json.dumps(packet(8000, **{field: invalid})).encode()
                response = client.post(MEETING + "/chunks", content=body, headers={"Content-Type": "application/json"})
                assert response.status_code == 400, response.text
        for invalid in (
            {"silence_ms": 500, "trailing_silence_ms": 100},
            {"tail_silence": 0.3, "silence_ms": 300},
            {"tail_silence": 0.3, "trailing_silence_ms": 300},
            {"tail_silence": 1e308},  # Seconds that overflow when expressed as milliseconds.
        ):
            response = client.post(MEETING + "/chunks", json=packet(8000, **invalid))
            assert response.status_code == 400, response.text
        assert not model.prompts
        # Omitting a measurement at the current endpoint keeps the punctuation.
        assert_src(client.post(MEETING + "/chunks", json=packet(8000)), "你好。")


@pytest.mark.parametrize("previous,period,src", [
    ("你好", "。", ""), ("Ask Dr", ".", "."), ("All done", ".", ""),
])
def test_a_separately_decoded_period(previous, period, src):
    with client_for(Capture([DIRECT + previous, DIRECT + period]), packet_period_to_comma=True) as client:
        assert_src(client.post(MEETING + "/chunks", json=packet()), previous)
        assert_src(client.post(MEETING + "/chunks", json=packet(trailing_silence_ms=120)), src)


@pytest.mark.parametrize("previous,following,expected", [
    ("I know.", "what you mean", " what you mean"),
    ("I know.", " what you mean", " what you mean"),
    ("I know. ", " what you mean", "what you mean"),
    ("I know.", ", actually", ", actually"),
    ("I know.", "中文", "中文"),
    ("We can.", "’t go", "’t go"),
    ("I know.", SPK_CHANGE + "Hello", " Hello"),
    ("我知道。", "这件事", "这件事"),
])
def test_next_visible_text_resolves_only_english_word_spacing(previous, following, expected):
    model = Capture([DIRECT + previous, WAIT, WAIT, DIRECT + following, WAIT])
    with client_for(model, packet_period_to_comma=True) as client:
        first = client.post(MEETING + "/chunks", json=packet(tail_silence=0.1))
        first_text = previous.replace(".", "").replace("。", "")
        assert_src(first, first_text)
        assert_src(client.post(MEETING + "/chunks", json=packet(2 * SAMPLE_RATE)), "")
        assert_src(client.post(MEETING + "/chunks", json=packet()), expected)
        row = client.get(MEETING + "/result").json()["turns"][0]
        changed_speaker = following.startswith(SPK_CHANGE)
        marked = (SPK_CHANGE if changed_speaker else "") + expected
        assert row["text"] == DIRECT + marked
        assert row["transcript_parts"] == [{"text": expected, "speaker_change": changed_speaker}]
        assert_src(client.post(MEETING + "/end"), "")
    history = [text for role, text in model.prompts[-1][0] if role == "assistant"]
    assert history == [DIRECT + first_text, WAIT, WAIT, DIRECT + marked]


def test_removed_standalone_period_is_a_wait_with_its_original_audio():
    model = Capture([DIRECT + "I know", DIRECT + ".", WAIT, DIRECT + "why", WAIT])
    with client_for(model, packet_period_to_comma=True) as client:
        assert_src(client.post(MEETING + "/chunks", json=packet(value=.1)), "I know")
        assert_src(client.post(MEETING + "/chunks", json=packet(2 * SAMPLE_RATE, value=.2, tail_silence=.05)), "")
        result = client.get(MEETING + "/result").json()
        assert result["action"] == "WAIT"
        assert [t["text"] for t in result["turns"]] == [WAIT, WAIT]
        assert result["turns"][0]["transcript_parts"] == []
        assert_src(client.post(MEETING + "/chunks", json=packet(value=.3)), " why")
        assert_src(client.post(MEETING + "/end"), "")
    messages, audios = model.prompts[3]
    assert [text for role, text in messages if role == "assistant"] == [DIRECT + "I know", WAIT, WAIT]
    assert len(audios) == 4 and np.all(audios[1] == np.float32(.2))


def test_standalone_dot_cannot_borrow_language_from_an_earlier_speaker():
    model = Capture([DIRECT + "We know ", DIRECT + SPK_CHANGE + "Paris", DIRECT + ".", WAIT])
    with client_for(model, packet_period_to_comma=True) as client:
        assert_src(client.post(MEETING + "/chunks", json=packet(2 * SAMPLE_RATE)), "We know Paris")
        assert_src(client.post(MEETING + "/chunks", json=packet(tail_silence=.05)), ",")
        assert_src(client.post(MEETING + "/end"), "")


def test_english_comma_followed_by_a_word_gets_one_separator():
    model = Capture([DIRECT + "I know.", DIRECT + "what happened", WAIT])
    with client_for(model, packet_period_to_comma=True) as client:
        assert_src(client.post(MEETING + "/chunks", json=packet(tail_silence=.4)), "I know,")
        assert_src(client.post(MEETING + "/chunks", json=packet()), " what happened")
        assert_src(client.post(MEETING + "/end"), "")
    assert model.prompts[-1][0][-2] == ("assistant", DIRECT + " what happened")


def test_partial_packet_applies_its_measurement_before_eos():
    model = Capture([DIRECT + "我知道。", WAIT])
    with client_for(model, packet_period_to_comma=True) as client:
        assert_src(client.post(MEETING + "/chunks", json=packet(8000, tail_silence=.09)), "")
        assert_src(client.post(MEETING + "/end"), "我知道")
    assert model.prompts[-1][0][-2] == ("assistant", DIRECT + "我知道")


@pytest.mark.parametrize("ms,expected", [(99.99, "知道"), (100, "知道，"), (399.99, "知道，"), (400, "知道。")])
def test_custom_remove_and_comma_thresholds(ms, expected):
    with client_for(Capture([DIRECT + "知道。"]), packet_period_to_comma=True,
                    packet_period_threshold_ms=400, packet_period_remove_threshold_ms=100) as client:
        health = client.get("/health").json()
        assert health["packet_period_threshold_ms"] == 400 and health["packet_period_remove_threshold_ms"] == 100
        assert_src(client.post(MEETING + "/chunks", json=packet(silence_ms=ms)), expected)


@pytest.mark.parametrize("invalid", [-1, 451, float("inf"), float("nan"), True])
def test_invalid_remove_threshold_is_rejected(invalid):
    with pytest.raises(ValueError, match="packet_period_remove_threshold_ms"):
        client_for(Capture([]), packet_period_remove_threshold_ms=invalid)
