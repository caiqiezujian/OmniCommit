import copy

import pytest

from omnicommit.prefix_source import parse_source, plan_queries


def record(text, timestamps=None):
    value = {"audios": ["/data/audio.wav"], "messages": [
        {"role": "system", "content": "unused source instruction"},
        {"role": "user", "content": "<audio>"},
        {"role": "assistant", "content": text}]}
    if timestamps is not None:
        value["timestamps"] = timestamps
    return value


def test_same_attributes_semantic_headers_merge_losslessly():
    source = parse_source(record("<ar><spk1>أول،<ar><spk1>  ثان.<ar><spk1> آخر.\n"), 2100)
    assert source["text"] == "أول،  ثان. آخر.\n"
    assert len(source["spans"]) == len(source["language_runs"]) == 1
    assert source["events"] == []
    queries = plan_queries(source, 2100, 1000, 0.5)
    assert [q["end_sample"] for q in queries] == [500, 1000, 1500, 2000, 2100]
    assert queries[-1]["kind"] == "full"


def test_actual_arabic_schema_original_offsets_and_body():
    original = "<ar><spk1>الآن ما بقتش المسألة مجرد عهد أو ثمن غامض،<ar><spk1> الموضوع تحول إلى حب حقيقي من جنية قوية وحلم بالزواج بإنسان،<ar><spk1> لكن سعيد عارف أن الزواج بجنية مشي أمر سهل<ar><spk1> فيه مخاطر كبيرة وغموض أكثر.<ar><spk2>  جلس سعيد مقابلة للجنية<ar><spk2> والليل بدأ ينزل على الفيلا بحال ستار غامض."
    assert original.index("<spk2>") == 216
    value = record(original.replace("<spk2>", "[20440ms]<spk2>", 1), [{
        "index": 1, "part_index": 2, "message_index": 2, "changed": ["speaker"],
        "language": "ar", "speaker": "spk2", "previous_language": "ar", "previous_speaker": "spk1",
        "text": "  جلس سعيد مقابلة للجنية والليل بدأ ينزل على الفيلا بحال ستار غامض.",
        "tag_positions": [{"kind": "speaker", "message_index": 2, "char_start": 216,
                           "char_end": 222, "tag": "<spk2>"}], "time_ms": 20440, "backend": "mms"}])
    value["messages"][2]["content_original"] = original
    source = parse_source(value, 27510)
    assert len(source["spans"]) == 2
    assert source["events"][0]["char_offset"] == 172
    assert source["spans"][1]["text"].startswith("  جلس")
    assert len(source["language_runs"]) == 1
    queries = plan_queries(source, 27510, 1000, 0.5)
    anchor, = [q for q in queries if q["end_sample"] == 20440]
    assert anchor["kind"] == "switch_anchor" and anchor["start_sample"] == 0
    after, = [q for q in queries if q["end_sample"] == 20500]
    assert after["start_sample"] == 0
    wrong = copy.deepcopy(value)
    wrong["timestamps"][0]["tag_positions"][0].update(char_start=225, char_end=231)
    with pytest.raises(ValueError, match="content_original"):
        parse_source(wrong, 27510)


def test_mixed_language_reset_uses_actual_global_phase_and_full_anchor():
    source = parse_source(record("<zh><spk1>你好。<en><spk2>Please check.", [
        {"time_ms": 2300, "changed": ["speaker", "language"]}]), 4100)
    queries = plan_queries(source, 4100, 1000, 0.5)
    zh = [q for q in queries if q["run_index"] == 0]
    en = [q for q in queries if q["run_index"] == 1]
    assert zh[-1]["end_sample"] == 2300
    assert zh[-1]["kind"] == "full"
    assert [(q["start_sample"], q["end_sample"]) for q in en[:2]] == [(2300, 2500), (2300, 3000)]
    assert en[-1]["end_sample"] == 4100
    assert len({q["query_id"] for q in queries}) == len(queries)


def test_exact_chunk_boundary_short_runs_and_returning_language():
    source = parse_source(record("<zh><spk1>甲<en>B<zh>丙", [
        {"time_ms": 500}, {"time_ms": 600}]), 1000)
    queries = plan_queries(source, 1000, 1000, 0.5)
    assert [(q["start_sample"], q["end_sample"]) for q in queries] == [(0, 500), (500, 600), (600, 1000)]
    assert [run["language"] for run in source["language_runs"]] == ["zh", "en", "zh"]


@pytest.mark.parametrize("timestamps", [None, [], [{"time_ms": 500}, {"time_ms": 600}]])
def test_missing_or_extra_switch_times_rejected(timestamps):
    with pytest.raises(ValueError, match="exactly one timestamp"):
        parse_source(record("<en><spk1>Hello.<spk2>Goodbye.", timestamps), 1000)


@pytest.mark.parametrize("time", [0, -1, 1000, 1001, float("nan"), float("inf"), True, "500"])
def test_invalid_switch_times_rejected(time):
    with pytest.raises(ValueError):
        parse_source(record("<en><spk1>A<spk2>B", [{"time_ms": time}]), 1000)


@pytest.mark.parametrize("metadata", [
    {"changed": ["language"]}, {"changed": ["speaker", "speaker"]},
    {"speaker": "spk3"}, {"previous_language": "zh"},
    {"message_index": 1}, {"text": "wrong"},
])
def test_inconsistent_switch_metadata_rejected(metadata):
    with pytest.raises(ValueError):
        parse_source(record("<en><spk1>A<spk2>B", [{"time_ms": 500, **metadata}]), 1000)


@pytest.mark.parametrize("text", [
    "<en><spk1>A[500ms]<spk2>B",  # duplicate time differs from external evidence
    "<en><spk1>A<spk2>B[400ms]",  # marker is inside text, not at a header
    "<en><spk1>A[400ms][400ms]<spk2>B",  # duplicate marker at the same event
])
def test_unverified_inline_marker_rejected(text):
    with pytest.raises(ValueError, match="Inline timestamp"):
        parse_source(record(text, [{"time_ms": 400}]), 1000)


@pytest.mark.parametrize("text", ["unlabelled", "<en><spk1>", "<en><spk1><spk2>Hi", "<en><spk1>A<bad_tag>B"])
def test_malformed_transcript_rejected(text):
    with pytest.raises(ValueError):
        parse_source(record(text), 1000)


def test_source_grid_rounding_and_invalid_subsample_spans():
    source = parse_source(record("<en><spk1>A<spk2>B", [{"time_ms": 400.4}]), 1000)
    queries = plan_queries(source, 1000, 1000, .5)
    assert any(q["end_sample"] == 400 for q in queries)
    tiny = parse_source(record("<en><spk1>A<spk2>B", [{"time_ms": .1}]), 1000)
    with pytest.raises(ValueError, match="collapse"):
        plan_queries(tiny, 1000, 1000, .5)
    with pytest.raises(ValueError, match="duration"):
        plan_queries(source, 900, 1000, .5)


def test_no_timestamp_marker_allowed_without_real_switch():
    with pytest.raises(ValueError, match="Inline timestamp"):
        parse_source(record("<en><spk1>A<en>[400ms]<spk1>B"), 1000)
