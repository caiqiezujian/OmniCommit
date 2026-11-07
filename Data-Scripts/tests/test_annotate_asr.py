"""Offline checks for the standalone cloud ASR annotation entry point.

No real model, audio, cloud mount, or network connection is used here.
"""
import copy
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest
import requests


SCRIPT = Path(__file__).resolve().parents[1] / "annotate_asr.py"
PROMPT_DIR = SCRIPT.parent / "prompts"
LANGUAGES = {"ar", "ja", "ko", "it", "fr", "pt", "es", "ru", "th", "en", "zh", "de"}
SOURCE = "<zh><spk1>请继续讨论。"


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("omnicommit_cloud_annotate_asr", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def forbid_real_requests(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A test attempted an unmocked HTTP request")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)


@pytest.fixture
def args():
    return SimpleNamespace(api_base="http://127.0.0.1:8080/v1/", concurrency=2)


def record(key="sample", source=SOURCE):
    return {
        "key": key,
        "audios": ["/data/unavailable_audio.wav"],
        "messages": [
            {"role": "system", "content": "SENTINEL_OLD_SYSTEM_DO_NOT_SEND"},
            {"role": "user", "content": "<audio>**Entity Reference:** []"},
            {"role": "assistant", "content": source},
        ],
        "metadata": {"keep": [1, "原字段"]},
    }


def annotation(source=SOURCE):
    return {
        "segments": [
            {"source_text": source, "action": "DIRECT"}
        ],
        "flush": False,
        "error": None,
    }


class FakeResponse:
    def __init__(self, content=None, *, status=200, finish_reason="stop", envelope=None):
        self.status_code = status
        self.text = "mock response"
        self.closed = False
        self._envelope = envelope
        if envelope is None:
            if content is None:
                content = json.dumps(annotation(), ensure_ascii=False)
            self._envelope = {
                "choices": [
                    {"message": {"content": content}, "finish_reason": finish_reason}
                ]
            }

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("mock HTTP failure")

    def json(self):
        if isinstance(self._envelope, Exception):
            raise self._envelope
        return self._envelope

    def close(self):
        self.closed = True


def write_rows(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_reads_original_assistant_content_exactly_without_audio_io(runner):
    source = "<zh><spk1>我，我同意。\n<en><spk1>All right. "
    original = record(source=source)
    before = copy.deepcopy(original)
    assert runner.get_source_text(original) == source
    assert original == before


@pytest.mark.parametrize(
    "item",
    [
        {},
        {"messages": "not a list"},
        {"messages": []},
        {"messages": [{"role": "user", "content": SOURCE}]},
        {"messages": [{"role": "assistant", "content": None}]},
        {"messages": [{"role": "assistant", "content": "   "}]},
        {
            "messages": [
                {"role": "assistant", "content": SOURCE},
                {"role": "assistant", "content": SOURCE},
            ]
        },
    ],
)
def test_bad_asr_source_is_a_record_error(runner, item):
    with pytest.raises((ValueError, TypeError)):
        runner.get_source_text(item)


def test_original_cloud_defaults_and_unknown_argument_compatibility(runner):
    actual = runner.parse_cmdline(["--unknown-cloud-argument", "value"])
    assert vars(actual) == {
        "data_path": "/home/ma-user/modelarts/inputs/input_data_0/",
        "index_data": "/home/ma-user/modelarts/inputs/data_url_1_1",
        "model_path": "/home/ma-user/modelarts/inputs/model_path_0/",
        "output_dir": "/home/ma-user/modelarts/outputs/out_path_0/",
        "resume_path": "/home/ma-user/modelarts/inputs/resume_path_2",
        "api_base": "http://127.0.0.1:8080/v1",
        "concurrency": 32,
    }
    assert runner.RESULT_FIELD == "output"
    assert runner.OUTPUT_SUFFIX == "_stream_output"


def test_full_language_prompts_are_embedded(runner):
    assert set(runner.PROMPTS) == LANGUAGES
    assert runner.PROMPT == runner.PROMPTS["zh"]
    for language, prompt in runner.PROMPTS.items():
        path = PROMPT_DIR / f"asr_segment_wait_direct_{language}.txt"
        assert prompt.strip() == path.read_text(encoding="utf-8").strip(), language


def test_all_prompt_examples_agree_with_runtime_validator(runner):
    pattern = (
        r"输入：\s*```json\s*(.*?)\s*```\s*"
        r"输出：\s*```json\s*(.*?)\s*```"
    )
    for language, prompt in runner.PROMPTS.items():
        examples = re.findall(pattern, prompt, re.S)
        assert examples, f"No machine-readable examples for {language}"
        valid = 0
        for example_index, (input_text, output_text) in enumerate(examples, 1):
            inp, value = json.loads(input_text), json.loads(output_text)
            if value["error"] is None:
                source = inp.get("transcript")
                if source is None:
                    source = runner.get_source_text(inp)
                source = runner.normalize_transcript(source)
                try:
                    assert runner.validate_annotation(value, source) == value
                except Exception as exc:
                    pytest.fail(f"{language} example {example_index}: {exc}")
                assert all(set(s) == {"source_text", "action"} for s in value["segments"])
                assert all(re.match(r"^<[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*><spk[1-9][0-9]*>", s["source_text"])
                           for s in value["segments"] if s["action"] == "DIRECT")
                valid += 1
            else:
                with pytest.raises(ValueError):
                    runner.validate_annotation(value, inp.get("transcript", ""))
        assert valid, language


def test_single_call_preserves_all_model_request_parameters(runner, args, monkeypatch):
    calls = []
    response = FakeResponse()

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return response

    monkeypatch.setattr(runner.requests, "post", post)
    output = runner.generate_one(record(), args)
    assert isinstance(output, str)
    assert json.loads(output) == annotation()
    assert len(calls) == 1
    url, request = calls[0]
    assert url == "http://127.0.0.1:8080/v1/chat/completions"
    assert set(request) == {"headers", "json", "timeout"}
    assert request["headers"] == {"Content-Type": "application/json"}
    assert request["timeout"] == 300
    payload = request["json"]
    assert set(payload) == {"model", "messages", "temperature", "top_p", "max_tokens"}
    assert {k: v for k, v in payload.items() if k != "messages"} == {
        "model": "dsv4", "temperature": 0.1, "top_p": 0.8, "max_tokens": 2048,
    }
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "user"
    content = payload["messages"][0]["content"]
    assert runner.PROMPT.strip() in content
    assert json.dumps({"transcript": SOURCE}, ensure_ascii=False) in content
    assert "SENTINEL_OLD_SYSTEM_DO_NOT_SEND" not in content
    assert response.closed


@pytest.mark.parametrize(
    "failure",
    ["http", "timeout", "envelope_json", "missing_choices", "null_content", "model_json", "length", "reported_error"],
)
def test_model_failures_are_detected_without_retry(runner, args, monkeypatch, failure):
    calls = []
    if failure == "http":
        response = FakeResponse(status=503)
    elif failure == "envelope_json":
        response = FakeResponse(envelope=ValueError("malformed response envelope"))
    elif failure == "missing_choices":
        response = FakeResponse(envelope={})
    elif failure == "null_content":
        response = FakeResponse(envelope={"choices": [{"message": {"content": None}}]})
    elif failure == "model_json":
        response = FakeResponse(content="this is not JSON")
    elif failure == "length":
        response = FakeResponse(finish_reason="length")
    elif failure == "reported_error":
        response = FakeResponse(content=json.dumps({"segments": [], "flush": False, "error": "CONFLICTING_TAGS"}))
    else:
        response = None

    def post(*a, **kw):
        calls.append(1)
        if failure == "timeout":
            raise requests.Timeout("mock timeout")
        return response

    monkeypatch.setattr(runner.requests, "post", post)
    with pytest.raises(Exception):
        runner.generate_one(record(), args)
    assert len(calls) == 1
    if response is not None:
        assert response.closed


def test_single_json_fence_is_accepted_without_rewriting_content(runner, args, monkeypatch):
    response = FakeResponse(content="```json\n" + json.dumps(annotation(), ensure_ascii=False) + "\n```")
    calls = []

    def post(*a, **kw):
        calls.append(1)
        return response

    monkeypatch.setattr(runner.requests, "post", post)
    assert json.loads(runner.generate_one(record(), args)) == annotation()
    assert len(calls) == 1
    assert response.closed


@pytest.mark.parametrize("damage", ["source_loss", "flex", "legacy_wait_reasons", "unknown_fields", "punctuation_only", "bad_flush", "empty", "non_object"])
def test_invalid_annotations_cannot_become_success_output(runner, damage):
    value = annotation()
    seg = value["segments"][0]
    if damage == "source_loss":
        seg["source_text"] = "<zh><spk1>请继续。"
    elif damage == "flex":
        seg["action"] = "FLEX"
    elif damage == "legacy_wait_reasons":
        seg["wait_reasons"] = ["ENTITY"]
    elif damage == "unknown_fields":
        seg["emit_text"] = SOURCE
    elif damage == "punctuation_only":
        seg["source_text"] = SOURCE[:-1]
        value["segments"].append({"source_text": "。", "action": "DIRECT"})
    elif damage == "bad_flush":
        value["flush"] = True
    elif damage == "empty":
        value["segments"] = []
    elif damage == "non_object":
        value = []
    with pytest.raises((ValueError, TypeError)):
        runner.validate_annotation(value, SOURCE)


def test_switch_wait_uses_read_state_then_releases_without_reasons(runner):
    pieces = [
        ("<zh><spk1>可以。", "DIRECT"),
        ("<en><spk2>Please check the file, ", "WAIT"),
        ("and send it back.", "DIRECT"),
    ]
    value = {
        "segments": [{"source_text": t, "action": a} for t, a in pieces],
        "flush": False, "error": None,
    }
    source = "".join(t for t, _ in pieces)
    runner.validate_annotation(value, source)
    wrong = copy.deepcopy(value)
    wrong["segments"][1]["action"] = "DIRECT"
    with pytest.raises(ValueError):
        runner.validate_annotation(wrong, source)


def test_complete_number_at_end_can_wait_and_flush(runner):
    value = {
        "segments": [
            {"source_text": "<de><spk1>Die Summe beträgt ", "action": "DIRECT"},
            {"source_text": "hundert Millionen Euro.", "action": "WAIT"},
        ],
        "flush": True, "error": None,
    }
    source = "".join(s["source_text"] for s in value["segments"])
    runner.validate_annotation(value, source)


def test_normalization_matches_user_11122211_run_compression(runner):
    identities = "11122211"
    words = "甲乙丙丁戊己庚辛"
    raw = "".join(f"<zh><spk{speaker}>{word}" for speaker, word in zip(identities, words))
    normalized = runner.normalize_transcript(raw)
    assert normalized == "<zh><spk1>甲乙丙<zh><spk2>丁戊己<zh><spk1>庚辛"
    assert runner.normalize_transcript(normalized) == normalized


def test_normalization_writes_complete_pair_when_either_attribute_changes(runner):
    raw = "<zh><spk1>甲<en>Hi<spk2>Yes<zh>乙"
    expected = "<zh><spk1>甲<en><spk1>Hi<en><spk2>Yes<zh><spk2>乙"
    assert runner.normalize_transcript(raw) == expected


def test_normalization_preserves_spaces_newlines_punctuation_and_words(runner):
    raw = "<zh><spk1> 甲 \n<zh><spk1> 乙\t<en><spk1> Hello <en><spk1> world.\n"
    expected = "<zh><spk1> 甲 \n 乙\t<en><spk1> Hello  world.\n"
    result = runner.normalize_transcript(raw)
    assert result == expected
    assert re.sub(r"<[^>]+>", "", result) == re.sub(r"<[^>]+>", "", raw)


def test_normalization_allows_same_value_redundant_headers(runner):
    raw = "<zh><zh><spk10><spk10>请继续。<zh><spk10>可以。"
    assert runner.normalize_transcript(raw) == "<zh><spk10>请继续。可以。"


@pytest.mark.parametrize(
    "raw",
    [
        "<zh><spk2><spk3>正文。",
        "<zh><spk1>甲。<spk2><spk3>乙。",
        "<zh><en><spk1>Text.",
        "<zh><spk1>甲。<spk2>",
        "<zh>缺少说话人。",
    ],
)
def test_normalization_never_guesses_conflicting_or_missing_attribution(runner, raw):
    with pytest.raises(ValueError):
        runner.normalize_transcript(raw)


def test_original_arabic_example_redundant_headers_are_removed(runner):
    raw = (
        "<ar><spk1>فجأةً على جانب الطريق لاحظ امرأة تقف وسط الظلام،"
        "<ar><spk1> لا يعرف من أين جاءت،"
        "<ar><spk1> لكن جمالها الأخاذ جعله يتوقف فوراً."
        "<ar><spk1> ابتسامة غامضة ارتسمت على وجهها،"
        "<ar><spk1> وعينان مضيئتان كأنهما زمرد في الظلام."
        "<ar><spk1> صرخت ببرود وقالت بصوت ناعم لكنه قوي."
    )
    normalized = runner.normalize_transcript(raw)
    assert normalized == "<ar><spk1>" + raw.replace("<ar><spk1>", "")
    assert normalized.count("<ar><spk1>") == 1


@pytest.mark.parametrize("language", sorted(LANGUAGES))
def test_routes_every_supported_data_language_component(runner, language):
    item = record()
    item["audios"] = [f"/data/data_process/zhh/multilingual_asr/data/{language}/1/audios/test.wav"]
    assert runner.detect_language(item, SOURCE) == language


def test_windows_path_language_routing(runner):
    item = record()
    item["audios"] = [r"D:\datasets\multilingual_asr\data\de\1\audios\sample.wav"]
    assert runner.detect_language(item, SOURCE) == "de"


def test_language_fallback_uses_first_tag_not_filename_substrings(runner):
    source = "<en><spk1>Hello.<zh><spk1>你好。"
    item = record(source=source)
    item["audios"] = ["/data/garbage/ar_fr_de_recording.wav"]
    assert runner.detect_language(item, source) == "en"
    item["audios"] = []
    assert runner.detect_language(item, source) == "en"


@pytest.mark.parametrize(
    "paths",
    [
        ["/data/ar/1.wav", "/data/en/2.wav"],
        ["/data/ar/archive/data/en/1.wav"],
    ],
)
def test_conflicting_path_language_candidates_are_not_guessed(runner, paths):
    item = record()
    item["audios"] = paths
    with pytest.raises(ValueError):
        runner.detect_language(item, SOURCE)


def test_unsupported_language_without_path_route_is_a_record_error(runner):
    with pytest.raises(ValueError):
        runner.detect_language({"audios": []}, "<fa><spk1>سلام")


def test_request_uses_path_prompt_and_normalized_mixed_source_without_rewriting_record(runner, args, monkeypatch):
    raw = "<zh><spk1>可以。<zh><spk1>继续。<en>All right.<en><spk1>Go on."
    normalized = "<zh><spk1>可以。继续。<en><spk1>All right.Go on."
    item = record(source=raw)
    item["audios"] = ["/data/data_process/zhh/multilingual_asr/data/ar/1/audios/example.wav"]
    before = copy.deepcopy(item)
    expected = {
        "segments": [
            {"source_text": "<zh><spk1>可以。", "action": "DIRECT"},
            {"source_text": "<zh><spk1>继续。", "action": "DIRECT"},
            {"source_text": "<en><spk1>All right.", "action": "WAIT"},
            {"source_text": "<en><spk1>Go on.", "action": "DIRECT"},
        ],
        "flush": False,
        "error": None,
    }
    response = FakeResponse(content=json.dumps(expected, ensure_ascii=False))
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return response

    monkeypatch.setattr(runner.requests, "post", post)
    assert json.loads(runner.generate_one(item, args)) == expected
    assert item == before
    assert len(calls) == 1 and response.closed
    content = calls[0][1]["json"]["messages"][0]["content"]
    assert content.startswith(runner.PROMPTS["ar"])
    sent = json.loads(content.rsplit("INPUT_JSON:", 1)[1].strip())
    assert sent == {"transcript": normalized}
    assert re.sub(r"<[^>]+>", "", sent["transcript"]) == re.sub(r"<[^>]+>", "", raw)


def test_repeated_correct_metadata_is_required_output_format_not_a_switch(runner):
    normalized = "<zh><spk1>请继续讨论。"
    value = {
        "segments": [
            {"source_text": "<zh><spk1>请继续", "action": "DIRECT"},
            {"source_text": "<zh><spk1>讨论。", "action": "DIRECT"},
        ],
        "flush": False, "error": None,
    }
    assert runner.validate_annotation(value, normalized) == value


def test_adjacent_ordinary_direct_segments_do_not_require_wait(runner):
    value = {
        "segments": [
            {"source_text": "<zh><spk1>请继续", "action": "DIRECT"},
            {"source_text": "讨论。", "action": "DIRECT"},
        ],
        "flush": False, "error": None,
    }
    result = runner.validate_annotation(value, SOURCE)
    assert result["segments"] == [
        {"source_text": "<zh><spk1>请继续", "action": "DIRECT"},
        {"source_text": "<zh><spk1>讨论。", "action": "DIRECT"},
    ]
    assert value["segments"][1]["source_text"] == "讨论。"  # no in-place mutation


def test_missing_direct_prefix_is_rendered_from_verified_inherited_state(runner, args, monkeypatch):
    source = "<zh><spk1>请联系张伟确认。<en><spk1>Please check the file.<en><spk2>I agree."
    value = {"segments": [
        {"source_text": "<zh><spk1>请联系", "action": "DIRECT"},
        {"source_text": "张伟", "action": "WAIT"},
        {"source_text": "确认。", "action": "DIRECT"},
        {"source_text": "<en><spk1>Please check ", "action": "WAIT"},
        {"source_text": "the file.", "action": "DIRECT"},
        {"source_text": "<en><spk2>I ", "action": "WAIT"},
        {"source_text": "agree.", "action": "DIRECT"},
    ], "flush": False, "error": None}
    calls = []
    def post(*a, **kw):
        calls.append(kw)
        return FakeResponse(content=json.dumps(value, ensure_ascii=False))
    monkeypatch.setattr(runner.requests, "post", post)
    result = json.loads(runner.generate_one(record(source=source), args))
    assert len(calls) == 1
    assert [s["source_text"] for s in result["segments"] if s["action"] == "DIRECT"] == [
        "<zh><spk1>请联系", "<zh><spk1>确认。", "<en><spk1>the file.", "<en><spk2>agree."]
    assert runner.validate_annotation(result, source) == result


@pytest.mark.parametrize("bad_prefix", ["<en><spk1>", "<zh><spk2>"])
def test_wrong_repeated_direct_prefix_cannot_be_silently_repaired(runner, bad_prefix):
    value = {"segments": [
        {"source_text": "<zh><spk1>请继续", "action": "DIRECT"},
        {"source_text": bad_prefix + "讨论。", "action": "DIRECT"},
    ], "flush": False, "error": None}
    with pytest.raises(ValueError):
        runner.validate_annotation(value, SOURCE)


def test_full_direct_header_preserves_initial_whitespace_and_roundtrips(runner):
    source = " \n<en><spk1>Please confirm. "
    value = {"segments": [{"source_text": source, "action": "DIRECT"}], "flush": False, "error": None}
    result = runner.validate_annotation(value, source)
    assert result["segments"][0]["source_text"] == "<en><spk1> \nPlease confirm. "
    assert runner.validate_annotation(result, source) == result


def test_long_complete_entity_is_not_rejected_by_old_translation_word_limit(runner):
    source = "<en><spk1>《The Small Observatory at the Edge of the Last Quiet Sea》"
    value = {
        "segments": [{"source_text": source, "action": "WAIT"}],
        "flush": True, "error": None,
    }
    runner.validate_annotation(value, source)


def test_input_parser_continues_after_json_object_and_utf8_errors(runner, tmp_path):
    source = tmp_path / "mixed.jsonl"
    good = json.dumps(record(), ensure_ascii=False).encode("utf-8")
    source.write_bytes(b"\xef\xbb\xbf" + good + b"\n\n{broken\n[]\n\xff\n" + good + b"\n")
    rows = list(runner.iter_jsonl(source))
    assert [n for n, _ in rows] == [1, 3, 4, 5, 6]
    assert rows[0][1] == rows[-1][1] == record()
    assert all(item.get("_input_error") for _, item in rows[1:-1])


def test_process_keeps_every_record_and_continues_after_bad_json(runner, args, monkeypatch, tmp_path):
    source, target = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    first, last = record("first"), record("last")
    source.write_bytes(
        json.dumps(first, ensure_ascii=False).encode("utf-8")
        + b"\n{broken\n[]\n\xff\n"
        + json.dumps(last, ensure_ascii=False).encode("utf-8") + b"\n"
    )
    calls = []

    def post(*a, **kw):
        calls.append(1)
        return FakeResponse()

    monkeypatch.setattr(runner.requests, "post", post)
    runner.process_file(source, target, args)
    rows = read_rows(target)
    assert len(rows) == 5 and len(calls) == 2
    assert rows[0] == dict(first, output=rows[0]["output"])
    assert rows[-1] == dict(last, output=rows[-1]["output"])
    assert json.loads(rows[0]["output"]) == annotation()
    assert json.loads(rows[-1]["output"]) == annotation()
    assert all(row["output"] == "" and row.get("_input_error") for row in rows[1:-1])


def test_model_failures_do_not_stop_later_records(runner, args, monkeypatch, tmp_path):
    args.concurrency = 1
    source, target = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    originals = [record(str(i)) for i in range(5)]
    write_rows(source, originals)
    replies = iter([
        FakeResponse(status=503), FakeResponse(content="not JSON"),
        FakeResponse(finish_reason="length"),
        FakeResponse(content=json.dumps({"segments": [], "flush": False, "error": "CONFLICTING_TAGS"})),
        FakeResponse(),
    ])
    calls = []

    def post(*a, **kw):
        calls.append(1)
        return next(replies)

    monkeypatch.setattr(runner.requests, "post", post)
    runner.process_file(source, target, args)
    rows = read_rows(target)
    assert [r["key"] for r in rows] == [str(i) for i in range(5)]
    assert [r["output"] for r in rows[:4]] == [""] * 4
    assert json.loads(rows[-1]["output"]) == annotation()
    assert len(calls) == 5
    assert all({k: v for k, v in row.items() if k != "output"} == original for row, original in zip(rows, originals))


def test_resume_skips_completed_successes_and_failures(runner, args, monkeypatch, tmp_path):
    source, target = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    originals = [record(str(i)) for i in range(3)]
    write_rows(source, originals)
    successful = json.dumps(annotation(), ensure_ascii=False)
    prior = [dict(originals[0], output=""), dict(originals[1], output=successful)]
    write_rows(target, prior)
    before = target.read_bytes()
    calls = []

    def post(*a, **kw):
        calls.append(1)
        return FakeResponse()

    monkeypatch.setattr(runner.requests, "post", post)
    stats = runner.process_file(source, target, args)
    assert len(calls) == 1
    assert stats["resumed"] == 2
    assert target.read_bytes().startswith(before)
    assert read_rows(target)[:2] == prior
    runner.process_file(source, target, args)
    assert len(calls) == 1
    assert len(read_rows(target)) == 3


def test_resume_does_not_mix_records_from_a_different_input(runner, tmp_path):
    source, target = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    write_rows(source, [record("new")])
    write_rows(target, [dict(record("old"), output="")])
    before = target.read_bytes()
    iterator = runner.iter_jsonl(source)
    try:
        with pytest.raises(ValueError):
            runner.resume_local_output(iterator, target)
    finally:
        iterator.close()
    assert target.read_bytes() == before


def test_bad_source_shape_is_empty_but_next_record_succeeds(runner, args, monkeypatch, tmp_path):
    source, target = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    invalid = {"key": "bad", "audios": ["/data/a.wav"], "messages": []}
    write_rows(source, [invalid, record("good")])
    calls = []

    def post(*a, **kw):
        calls.append(1)
        return FakeResponse()

    monkeypatch.setattr(runner.requests, "post", post)
    runner.process_file(source, target, args)
    rows = read_rows(target)
    assert rows[0] == dict(invalid, output="")
    assert json.loads(rows[1]["output"]) == annotation()
    assert len(calls) == 1


def test_normalization_and_route_failures_do_not_stop_following_record(runner, args, monkeypatch, tmp_path):
    source, target = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    conflict = record("conflicting-speaker", "<zh><spk1>甲。<spk2><spk3>乙。")
    route_conflict = record("conflicting-route")
    route_conflict["audios"] = ["/data/ar/one.wav", "/data/en/two.wav"]
    good = record("good")
    write_rows(source, [conflict, route_conflict, good])
    calls = []

    def post(*a, **kw):
        calls.append(1)
        return FakeResponse()

    monkeypatch.setattr(runner.requests, "post", post)
    runner.process_file(source, target, args)
    rows = read_rows(target)
    assert rows[0] == dict(conflict, output="")
    assert rows[1] == dict(route_conflict, output="")
    assert json.loads(rows[2]["output"]) == annotation()
    assert len(calls) == 1


def test_escaped_surrogate_in_bad_input_does_not_break_output_writing(runner, args, monkeypatch, tmp_path):
    source, target = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    bad = {"key": "bad", "audios": [], "messages": [], "note": "\ud800"}
    source.write_text(json.dumps(bad) + "\n" + json.dumps(record("good")) + "\n", encoding="utf-8")
    monkeypatch.setattr(runner.requests, "post", lambda *a, **kw: FakeResponse())
    runner.process_file(source, target, args)
    rows = read_rows(target)
    assert rows[0] == dict(bad, output="")
    assert json.loads(rows[1]["output"]) == annotation()


def test_main_preserves_damaged_history_and_still_processes_other_files(runner, args, monkeypatch, tmp_path):
    source_dir, target_dir = tmp_path / "inputs", tmp_path / "outputs"
    source_dir.mkdir()
    target_dir.mkdir()
    write_rows(source_dir / "a.jsonl", [record("a")])
    write_rows(source_dir / "b.jsonl", [record("b")])
    damaged = target_dir / "a_stream_output.jsonl"
    damaged.write_bytes(b"{broken historical output\n")
    before = damaged.read_bytes()
    args.data_path, args.output_dir = str(source_dir), str(target_dir)
    monkeypatch.setattr(runner, "parse_cmdline", lambda: args)
    calls = []

    def post(*a, **kw):
        calls.append(1)
        return FakeResponse()

    monkeypatch.setattr(runner.requests, "post", post)
    runner.main()
    assert damaged.read_bytes() == before
    completed = read_rows(target_dir / "b_stream_output.jsonl")
    assert len(completed) == 1
    assert completed[0]["key"] == "b"
    assert json.loads(completed[0]["output"]) == annotation()
    assert len(calls) == 1
