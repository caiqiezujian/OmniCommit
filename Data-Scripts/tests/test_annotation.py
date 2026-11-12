import base64
import io
import json

import numpy as np
import pytest
import soundfile as sf

from omnicommit.audio import inspect_audio, prefix_endpoints, prefix_wav
from omnicommit.collection import main as collect_main
from omnicommit.protocol import parse_tagged_text, parse_teacher, plain_text, to_units
from omnicommit.teacher import AudioTeacher
from omnicommit.trajectory import make_trajectory, main as build_main


def seg(text, language="en", speaker="spk1"):
    return {"language": language, "speaker": speaker, "text": text}


def trace(full, hypotheses, number_format="verbatim"):
    return {"schema_version": 1, "status": "ok", "id": "sample", "audio": "/fake/audio.wav",
            "duration_ms": len(hypotheses) * 500, "full": {"segments": full},
            "teacher": {"number_format": number_format}, "reference_segments": full,
            "prefixes": [{"end_ms": (i + 1) * 500, "segments": hypothesis}
                         for i, hypothesis in enumerate(hypotheses)]}


def emitted(trajectory):
    return [s for t in trajectory["turns"] for s in t["segments"]]


def test_whole_output_not_partial_lcp():
    full = [seg("a b c d")]
    result = make_trajectory(trace(full, [[seg("a b x")], [seg("a b")], full]))
    assert [t["action"] for t in result["turns"]] == ["WAIT", "DIRECT", "DIRECT"]
    assert plain_text(result["turns"][1]["segments"]) == "a b"
    assert to_units(emitted(result)) == to_units(full)


def test_retains_earlier_valid_prefix_after_teacher_revision():
    full = [seg("a b c")]
    result = make_trajectory(trace(full, [[seg("a b")], [seg("a x")], full]))
    assert result["turns"][1]["action"] == "WAIT"
    assert plain_text(result["turns"][2]["segments"]) == " c"


def test_segment_split_is_not_lexical_disagreement():
    full = [seg("hello world")]
    result = make_trajectory(trace(full, [[seg("hello"), seg(" world")], full]))
    assert plain_text(result["turns"][0]["segments"]) == "hello world"
    assert result["turns"][-1]["action"] == "WAIT"  # legal final silence


@pytest.mark.parametrize("changed", [seg("hello", speaker="spk2"), seg("hello", language="fr")])
def test_joint_attribution_affects_commit(changed):
    full = [seg("hello world")]
    result = make_trajectory(trace(full, [[changed], full]))
    assert result["turns"][0]["action"] == "WAIT"


def test_whitespace_delta_is_carried_not_dropped():
    full = [seg("Hello world")]
    result = make_trajectory(trace(full, [[seg("Hello")], [seg("Hello ")], full]))
    assert result["turns"][1]["action"] == "WAIT"
    assert plain_text(result["turns"][2]["segments"]) == " world"
    assert plain_text(emitted(result)) == "Hello world"


def test_holdback_flushes_final_and_preserves_tags():
    full = [seg("你好", "zh"), seg(" OK", "en"), seg("好", "zh", "spk2")]
    result = make_trajectory(trace(full, [full, full]), holdback_chars=2)
    assert to_units(emitted(result)) == to_units(full)
    assert result["turns"][-1]["action"] == "DIRECT"


def test_normalized_digit_extension_does_not_commit_incomplete_number():
    full = [seg("金额是100000000元", "zh")]
    result = make_trajectory(trace(full, [[seg("金额是100", "zh")], full], "normalized"))
    assert plain_text(result["turns"][0]["segments"]) == "金额是"
    assert plain_text(result["turns"][1]["segments"]) == "100000000元"
    assert result["policy"]["number_format"] == "normalized"


def test_reference_mismatch_is_reported_not_rewritten():
    full = [seg("cat")]
    value = trace(full, [full])
    value["reference_segments"] = [seg("dog")]
    result = make_trajectory(value)
    assert plain_text(emitted(result)) == "cat"
    assert result["reference_audit"]["text_agreement"] == 0
    with pytest.raises(ValueError, match="threshold"):
        make_trajectory(value, min_reference_agreement=0.5)


@pytest.mark.parametrize("text", ["<zh><spk2><spk3>呃", "<zh><spk1>你好<spk2>",
                                   "你好", "<zh><en><spk1>Hello"])
def test_invalid_reference_headers_are_not_repaired(text):
    with pytest.raises(ValueError):
        parse_tagged_text(text)


def test_reference_change_only_headers_and_repeated_headers():
    values = parse_tagged_text("<zh><spk1>你好<en>hello<spk2>yes<zh><spk2>好的")
    assert [(s["language"], s["speaker"]) for s in values] == [
        ("zh", "spk1"), ("en", "spk1"), ("en", "spk2"), ("zh", "spk2")]


def test_teacher_requires_ordered_speaker_ids():
    with pytest.raises(ValueError, match="first occurrence"):
        parse_teacher({"segments": [seg("hello", speaker="spk3")]})


def test_crop_before_resample_has_no_future_dependency(tmp_path):
    rate = 44100
    a = np.sin(np.arange(rate) / 17).astype("float32") * 0.2
    b = a.copy()
    b[rate // 2:] = 0.9
    one, two = tmp_path / "a.wav", tmp_path / "b.wav"
    sf.write(one, a, rate, subtype="FLOAT")
    sf.write(two, b, rate, subtype="FLOAT")
    assert prefix_wav(one, 500) == prefix_wav(two, 500)
    decoded, sr = sf.read(io.BytesIO(prefix_wav(one, 500)))
    assert sr == 16000 and len(decoded) == 8000


def test_tail_is_kept_at_source_sample_boundary(tmp_path):
    wav = tmp_path / "tiny.wav"
    sf.write(wav, np.zeros(16001), 16000)
    ends = prefix_endpoints(inspect_audio(wav), 0.5)
    assert ends == [500.0, 1000.0, 1000.0625]


class Response:
    def __init__(self, content, reason="stop"):
        self.content, self.reason, self.closed = content, reason, False

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"finish_reason": self.reason, "message": {"content": self.content}}]}

    def close(self):
        self.closed = True


class Session:
    def __init__(self, response):
        self.response, self.calls = response, []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response

    def close(self):
        pass


def test_http_sends_audio_and_preserves_explicit_parameters():
    response = Response(json.dumps({"segments": [seg("hello")]}))
    session = Session(response)
    generation = {"temperature": 0.1, "top_p": 0.8, "max_tokens": 2048}
    teacher = AudioTeacher("https://example.invalid/v1", "test-omni", generation,
                           "Transcribe only this audio.", 300, session=session)
    assert teacher.transcribe(b"RIFFtest") == [seg("hello")]
    args, kwargs = session.calls[0]
    assert args == ("https://example.invalid/v1/chat/completions",)
    assert kwargs["timeout"] == 300
    body = kwargs["json"]
    assert all(body[k] == v for k, v in generation.items())
    assert len(body["messages"]) == 1
    audio = body["messages"][0]["content"][0]["input_audio"]
    assert base64.b64decode(audio["data"]) == b"RIFFtest"
    assert body["messages"][0]["content"][1]["text"] == "Transcribe only this audio."
    assert response.closed


@pytest.mark.parametrize("content, reason", [("[]", "stop"), ("", "stop"), ("{}", "length"), ("```json\n{}\n```", "stop")])
def test_bad_teacher_results_fail_and_close_response(content, reason):
    response = Response(content, reason)
    teacher = AudioTeacher("http://localhost/v1", "omni", {}, "prompt", session=Session(response))
    with pytest.raises((ValueError, KeyError)):
        teacher.transcribe(b"audio")
    assert response.closed


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_replay_collection_failure_continues_and_resume_skips(tmp_path):
    wav = tmp_path / "audio.wav"
    sf.write(wav, np.zeros(16000), 16000)
    source, replay = tmp_path / "source.jsonl", tmp_path / "replay.jsonl"
    output, trajectory = tmp_path / "trace.jsonl", tmp_path / "trajectory.jsonl"
    write_jsonl(source, [
        {"id": "bad", "audios": [str(wav)], "messages": [{"role": "assistant", "content": "<zh><spk1><spk2>错"}]},
        {"id": "good", "audios": [str(wav)], "messages": [{"role": "assistant", "content": "<en><spk1>Hello world"}]}])
    write_jsonl(replay, [{"id": "good", "full": {"segments": [seg("Hello world")]},
                         "prefixes": [{"end_ms": 500, "segments": [seg("Hello")]}]}])
    arguments = [str(source), "--output", str(output), "--replay-responses", str(replay), "--strict-reference"]
    assert collect_main(arguments) == 0
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["status"] == "error" and rows[0]["output"] == ""
    assert rows[1]["status"] == "ok" and len(rows[1]["prefixes"]) == 2
    before = output.read_bytes()
    assert collect_main(arguments + ["--resume"]) == 0
    assert output.read_bytes() == before
    assert collect_main(arguments + ["--resume", "--chunk-seconds", "0.25"]) == 2
    assert output.read_bytes() == before
    assert build_main([str(output), "--output", str(trajectory)]) == 0
    values = [json.loads(line) for line in trajectory.read_text(encoding="utf-8").splitlines()]
    assert values[0]["output"] == ""
    assert plain_text(emitted(values[1])) == "Hello world"


def test_ambiguous_reference_is_preserved_and_audio_teacher_continues(tmp_path):
    wav = tmp_path / "audio.wav"
    sf.write(wav, np.zeros(16000), 16000)
    source, replay = tmp_path / "source.jsonl", tmp_path / "replay.jsonl"
    output, trajectory = tmp_path / "trace.jsonl", tmp_path / "trajectory.jsonl"
    filtered = tmp_path / "filtered.jsonl"
    ambiguous = "<zh><spk1>我不建议。<zh><spk2><spk3>呃<zh><spk3>你说这个情况可能发生。"
    full = [seg("我不建议。", "zh"), seg("呃，你说这个情况可能发生。", "zh", "spk2")]
    good = [seg("Hello")]
    write_jsonl(source, [
        {"id": "ambiguous", "audios": [str(wav)], "messages": [{"role": "assistant", "content": ambiguous}]},
        {"id": "good", "audios": [str(wav)], "messages": [{"role": "assistant", "content": "<en><spk1>Hello"}]}])
    write_jsonl(replay, [
        {"id": "ambiguous", "full": {"segments": full}, "prefixes": [{"end_ms": 500, "segments": full[:1]}]},
        {"id": "good", "full": {"segments": good}, "prefixes": [{"end_ms": 500, "segments": good}]}])
    arguments = [str(source), "--output", str(output), "--replay-responses", str(replay)]
    assert collect_main(arguments) == 0
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [r["status"] for r in rows] == ["ok", "ok"]
    assert rows[0]["reference"] == ambiguous
    assert rows[0]["reference_segments"] is None
    assert rows[0]["reference_parse_error"]
    assert rows[0]["full"]["segments"] == full
    assert rows[1]["reference_parse_error"] is None
    before = output.read_bytes()
    assert collect_main(arguments + ["--resume"]) == 0
    assert collect_main(arguments + ["--resume", "--strict-reference"]) == 2
    assert output.read_bytes() == before
    assert build_main([str(output), "--output", str(trajectory)]) == 0
    values = [json.loads(line) for line in trajectory.read_text(encoding="utf-8").splitlines()]
    assert values[0]["reference"] == ambiguous
    assert values[0]["reference_parse_error"] == rows[0]["reference_parse_error"]
    audit = values[0]["reference_audit"]
    assert audit["status"] == "unavailable" and audit["text_agreement"] is None
    assert audit["reference_languages"] is None and audit["reference_speakers"] is None
    assert audit["teacher_speakers"] == ["spk1", "spk2"]
    assert to_units(emitted(values[0])) == to_units(full)
    assert values[1]["reference_audit"]["status"] == "available"
    assert build_main([str(output), "--output", str(filtered), "--min-reference-agreement", "0.8"]) == 0
    selected = [json.loads(line) for line in filtered.read_text(encoding="utf-8").splitlines()]
    assert selected[0]["status"] == "error" and selected[0]["output"] == ""
    assert "unavailable" in selected[0]["error"]
    assert selected[1]["status"] == "ok"


def test_missing_reference_and_valid_empty_reference_are_distinct():
    value = trace([seg("Hello")], [[seg("Hello")]])
    del value["reference_segments"]
    assert make_trajectory(value)["reference_audit"]["text_agreement"] is None
    value["reference_segments"] = []
    audit = make_trajectory(value)["reference_audit"]
    assert audit["status"] == "available" and audit["text_agreement"] == 0
    assert audit["reference_speakers"] == []


def test_empty_assistant_remains_invalid_with_reference_audit_default(tmp_path):
    source, replay, output = tmp_path / "source.jsonl", tmp_path / "replay.jsonl", tmp_path / "trace.jsonl"
    write_jsonl(source, [{"id": "bad", "audios": ["unused.wav"],
                          "messages": [{"role": "assistant", "content": "   "}]}])
    write_jsonl(replay, [])
    assert collect_main([str(source), "--output", str(output), "--replay-responses", str(replay)]) == 0
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["status"] == "error" and row["output"] == ""
    assert "assistant transcript" in row["error"]["message"]


def test_malformed_source_rows_do_not_abort_later_rows(tmp_path):
    source, replay = tmp_path / "source.jsonl", tmp_path / "replay.jsonl"
    output = tmp_path / "trace.jsonl"
    wav = tmp_path / "audio.wav"
    sf.write(wav, np.zeros(8000), 16000)
    value = {"id": "good", "audios": [str(wav)], "messages": [{"role": "assistant", "content": "<en><spk1>Hello"}]}
    source.write_text('{"id":NaN}\n{"id":[]}\nnot-json\n{"id":"overflow","value":1e999}\n' + json.dumps(value) + "\n", encoding="utf-8")
    write_jsonl(replay, [{"id": "good", "full": {"segments": [seg("Hello")]}, "prefixes": []}])
    args = [str(source), "--output", str(output), "--replay-responses", str(replay)]
    assert collect_main(args) == 0
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [r["status"] for r in rows] == ["error", "error", "error", "error", "ok"]
    assert "Non-finite JSON number" in rows[3]["error"]["message"]
    assert collect_main(args + ["--resume"]) == 0


def test_malformed_trace_id_does_not_abort_builder(tmp_path):
    source, output = tmp_path / "source.jsonl", tmp_path / "out.jsonl"
    good = trace([seg("cat")], [[seg("cat")]])
    write_jsonl(source, [{"id": []}, good])
    assert build_main([str(source), "--output", str(output)]) == 0
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [r["status"] for r in rows] == ["error", "ok"]


def test_truncated_resume_journal_is_not_edited(tmp_path):
    source, output = tmp_path / "source.jsonl", tmp_path / "out.jsonl"
    write_jsonl(source, [trace([seg("cat")], [[seg("cat")]])])
    output.write_bytes(b'{"unfinished":')
    assert build_main([str(source), "--output", str(output), "--resume"]) == 2
    assert output.read_bytes() == b'{"unfinished":'


@pytest.mark.parametrize("ends", [[500, 400], [500, 1001], [500, 999]])
def test_invalid_or_incomplete_prefix_time_axis(ends):
    value = trace([seg("cat")], [[seg("ca")], [seg("cat")]])
    for row, end in zip(value["prefixes"], ends):
        row["end_ms"] = end
    with pytest.raises(ValueError):
        make_trajectory(value)
