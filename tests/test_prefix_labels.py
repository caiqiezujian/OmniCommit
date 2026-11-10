"""Causality and attribution regression cases for local-language supervision."""
from copy import deepcopy

import pytest

from omnicommit.prefix_labels import build_trajectory, comparison_units, teacher_body
from omnicommit.prefix_source import parse_source, plan_queries


def trace_for(spans, switches=(), hypotheses=None, duration=2000):
    """spans are (language,speaker,text); maps may use end or (run,end)."""
    record = {
        "audios": ["/data/example.wav"],
        "messages": [{"role": "assistant", "content": "".join(
            f"<{language}><{speaker}>{text}" for language, speaker, text in spans)}],
        "timestamps": [{"time_ms": value} for value in switches],
    }
    source = parse_source(record, duration)
    queries = plan_queries(source, duration, 1000, .5)
    hypotheses = hypotheses or {}
    for query in queries:
        run = source["language_runs"][query["run_index"]]
        value = hypotheses.get((run["index"], query["end_sample"]),
                               hypotheses.get(query["end_sample"], run["text"]))
        query.update(text=value, token_ids=[], error=None, prompt_tokens=10)
    return {
        "kind": "local_prefix_trace", "status": "ok", "id": "test",
        "source": source, "audio": "/data/example.wav",
        "audio_info": {"frames": duration, "sample_rate": 1000},
        "chunk_seconds": .5, "number_format": "normalized", "input_hash": "testhash",
        "queries": queries,
    }


def texts(trajectory, include_eos=False):
    turns = trajectory["turns"] if include_eos else trajectory["turns"][:-1]
    return ["".join(segment["text"] for segment in turn["segments"]) for turn in turns]


def test_wrong_tail_rejects_whole_hypothesis_not_merely_bad_suffix():
    trace = trace_for([("en", "spk1", "abcdef")], hypotheses={
        500: "ab", 1000: "abcdX", 1500: "abc", 2000: "abcdef"})
    trajectory = build_trajectory(trace, comparison="exact")
    assert texts(trajectory) == ["ab", "", "c", "def"]
    observations = trajectory["provenance"]["run_diagnostics"][0]["observations"]
    assert [o["stable_units"] for o in observations] == [2, 2, 3, 6]
    assert observations[1]["whole_hypothesis_matches"] is False


def test_historical_lsp_survives_a_shrinking_hypothesis():
    trace = trace_for([("en", "spk1", "abcdef")], hypotheses={
        500: "abc", 1000: "a", 1500: "abcd", 2000: "abcdef"})
    trajectory = build_trajectory(trace, comparison="exact")
    assert texts(trajectory) == ["abc", "", "d", "ef"]
    observed = trajectory["provenance"]["run_diagnostics"][0]["observations"]
    assert observed[1]["whole_hypothesis_matches"] is True
    assert observed[1]["hypothesis_equals_lsp"] is False
    assert observed[1]["stable_units"] == 3


def test_stable_interval_ends_on_growth_or_bad_hypothesis():
    trace = trace_for([("en", "spk1", "ABC")], duration=4000, hypotheses={
        500: "A", 1000: "A", 1500: "AX", 2000: "AB",
        2500: "A", 3000: "AB", 3500: "AB", 4000: "ABC"})
    trajectory = build_trajectory(trace, comparison="exact")
    intervals = trajectory["provenance"]["run_diagnostics"][0]["stable_intervals"]
    assert intervals == [
        {"start_sample": 500, "end_sample": 1000, "stable_units": 1},
        {"start_sample": 2000, "end_sample": 2000, "stable_units": 2},
        {"start_sample": 3000, "end_sample": 3500, "stable_units": 2},
        {"start_sample": 4000, "end_sample": 4000, "stable_units": 3},
    ]


def test_future_speaker_prediction_is_capped_before_and_during_switch_chunk():
    # Deliberately premature teacher content: the future attribution must still
    # never leak. This test does not claim within-speaker acoustic alignment.
    trace = trace_for([("en", "spk1", "ABC"), ("en", "spk2", "DEF")], [1250])
    trajectory = build_trajectory(trace, comparison="exact")
    assert texts(trajectory) == ["ABC", "", "", "DEF"]
    assert [t["provenance"]["switch_gate"] for t in trajectory["turns"][:-1]] == [
        False, False, True, False]
    assert trajectory["turns"][0]["provenance"]["blocked_new_text"] is True
    assert trajectory["turns"][3]["segments"] == [{
        "language": "en", "speaker": "spk2", "text": "DEF"}]


def test_switch_chunk_can_emit_old_confirmed_tail_instead_of_whole_round_wait():
    trace = trace_for([("en", "spk1", "ABC"), ("en", "spk2", "DEF")], [1250], {
        500: "A", 1000: "AB", 1250: "ABC", 1500: "ABCDE", 2000: "ABCDEF"})
    trajectory = build_trajectory(trace, comparison="exact")
    assert texts(trajectory) == ["A", "B", "C", "DEF"]
    mixed_turn = trajectory["turns"][2]
    assert mixed_turn["action"] == "DIRECT"
    assert mixed_turn["provenance"]["switch_gate"] is True
    assert mixed_turn["segments"][0]["speaker"] == "spk1"


def test_multiple_switches_in_one_chunk_use_earliest_gate_then_release_all():
    trace = trace_for([
        ("en", "spk1", "AAA"), ("en", "spk2", "BBB"), ("fr", "spk2", "CCC")],
        [600, 800])
    trajectory = build_trajectory(trace, comparison="exact")
    assert texts(trajectory) == ["AAA", "", "BBBCCC", ""]
    assert trajectory["turns"][1]["provenance"]["switch_times_ms"] == [600, 800]
    assert trajectory["turns"][2]["segments"] == [
        {"language": "en", "speaker": "spk2", "text": "BBB"},
        {"language": "fr", "speaker": "spk2", "text": "CCC"},
    ]


def test_switch_exactly_at_endpoint_belongs_to_next_audio_chunk():
    trace = trace_for([("en", "spk1", "AAA"), ("en", "spk2", "BBB")], [1000], {
        500: "AA", 1000: "AAA", 1500: "AAABB", 2000: "AAABBB"})
    trajectory = build_trajectory(trace, comparison="exact")
    assert texts(trajectory) == ["AA", "A", "", "BBB"]
    assert trajectory["turns"][1]["provenance"]["switch_gate"] is False
    assert trajectory["turns"][2]["provenance"]["switch_gate"] is True
    assert trajectory["turns"][3]["provenance"]["switch_gate"] is False


def test_final_chunk_switch_flushes_only_verified_tail_at_real_eos():
    trace = trace_for([("en", "spk1", "AAA"), ("en", "spk2", "BBB")], [1750], {
        500: "A", 1000: "AA", 1500: "AAA", 1750: "AAA", 2000: "AAABBB"})
    trajectory = build_trajectory(trace, comparison="exact")
    assert texts(trajectory, include_eos=True) == ["A", "A", "A", "", "BBB"]
    eos = trajectory["turns"][-1]
    assert eos["event"] == "end_of_audio"
    assert eos["start_ms"] == eos["end_ms"] == 2000
    assert eos["segments"] == [{"language": "en", "speaker": "spk2", "text": "BBB"}]
    assert trajectory["metrics"]["eos_characters"] == 3

    broken = deepcopy(trace)
    broken["queries"][-1]["text"] = "AAABX"
    with pytest.raises(ValueError, match="full teacher/reference mismatch"):
        build_trajectory(broken, reference_fallback="off")


def test_language_switch_resets_teacher_but_not_final_student_timeline():
    trace = trace_for([("zh", "spk1", "我们开始"), ("en", "spk1", "Please check")],
                      [1250], {(0, 500): "我们", (0, 1000): "我们开",
                               (0, 1250): "我们开始", (1, 1500): "Please",
                               (1, 2000): "Please check"})
    assert all(q["start_sample"] == 1250 for q in trace["queries"] if q["run_index"] == 1)
    trajectory = build_trajectory(trace)
    assert texts(trajectory) == ["我们", "开", "始", "Please check"]
    assert [(t["start_ms"], t["end_ms"]) for t in trajectory["turns"][:-1]] == [
        (0, 500), (500, 1000), (1000, 1500), (1500, 2000)]
    assert trajectory["turns"][-2]["segments"][0]["speaker"] == "spk1"


def test_partial_final_audio_chunk_is_preserved():
    trace = trace_for([("en", "spk1", "abc")], duration=1200,
                      hypotheses={500: "a", 1000: "ab", 1200: "abc"})
    trajectory = build_trajectory(trace)
    assert texts(trajectory) == ["a", "b", "c"]
    assert trajectory["turns"][-2]["end_ms"] == 1200


def test_full_teacher_reference_disagreement_never_becomes_reference_fill():
    trace = trace_for([("en", "spk1", "correct")], hypotheses={2000: "incorrect"})
    with pytest.raises(ValueError, match="full teacher/reference mismatch"):
        build_trajectory(trace)


@pytest.mark.parametrize("reference,teacher", [
    ("1.5", "15"), ("-5", "5"), ("don't", "dont"),
    ("1,500", "1500"), ("5%", "5"), ("Alice", "alice"),
    ("عَلَم", "علم"), ("a part", "apart"), ("1 5", "15"),
])
def test_surface_comparison_preserves_meaningful_numeric_and_lexical_differences(reference, teacher):
    assert comparison_units(reference)[0] != comparison_units(teacher)[0]
    trace = trace_for([("en", "spk1", reference)], hypotheses={2000: teacher})
    with pytest.raises(ValueError, match="full teacher/reference mismatch"):
        build_trajectory(trace)


def test_reference_spaces_and_punctuation_are_preserved_in_emitted_text():
    trace = trace_for([("en", "spk1", "Hello, world! ")], hypotheses={
        500: "Hello.", 1000: "Hello world", 1500: "Hello world", 2000: "Hello world"})
    trajectory = build_trajectory(trace)
    assert texts(trajectory) == ["Hello, ", "world! ", "", ""]
    assert "".join(texts(trajectory, include_eos=True)) == trace["source"]["text"]


def test_canonical_equivalence_does_not_rewrite_reference_spelling():
    trace = trace_for([("fr", "spk1", "éclair")], hypotheses={
        500: "e\u0301", 1000: "e\u0301cl", 1500: "e\u0301cla", 2000: "e\u0301clair"})
    trajectory = build_trajectory(trace)
    assert texts(trajectory) == ["é", "cl", "a", "ir"]


def test_partial_decomposed_character_cannot_release_unobserved_accent():
    trace = trace_for([("fr", "spk1", "éclair")], hypotheses={
        500: "e", 1000: "é", 1500: "écl", 2000: "éclair"})
    trajectory = build_trajectory(trace)
    assert texts(trajectory) == ["", "é", "cl", "air"]


def test_decomposed_reference_also_waits_for_complete_accented_character():
    trace = trace_for([("fr", "spk1", "e\u0301clair")], hypotheses={
        500: "e", 1000: "é", 1500: "écl", 2000: "éclair"})
    trajectory = build_trajectory(trace)
    assert texts(trajectory) == ["", "e\u0301", "cl", "air"]


@pytest.mark.parametrize("output", ["<en><spk1>", " <en><spk1> "])
def test_tag_only_teacher_result_is_not_synthetic_wait(output):
    trace = trace_for([("en", "spk1", "hello")], hypotheses={500: output})
    with pytest.raises(ValueError):
        build_trajectory(trace)


def test_actual_empty_teacher_hypothesis_can_produce_wait():
    trace = trace_for([("en", "spk1", "hello")], hypotheses={500: "", 1000: "", 1500: "he"})
    trajectory = build_trajectory(trace)
    assert texts(trajectory) == ["", "", "he", "llo"]
    assert [t["action"] for t in trajectory["turns"][:2]] == ["WAIT", "WAIT"]


@pytest.mark.parametrize("markup", ["<|WAIT|>", "<|DIRECT|>", "<think>guess</think>",
                                    "hello [500ms]", "<broken"])
def test_unknown_teacher_protocol_or_timestamp_is_rejected(markup):
    with pytest.raises(ValueError):
        teacher_body(markup)


def test_failed_teacher_hypothesis_is_rejected_not_converted_to_wait():
    trace = trace_for([("en", "spk1", "hello")])
    trace["queries"][0].update(text="", error="CUDA out of memory")
    with pytest.raises(ValueError, match="not a valid WAIT"):
        build_trajectory(trace)


def test_future_query_cannot_be_retroactively_used_for_earlier_rounds():
    trace = trace_for([("en", "spk1", "hello")], hypotheses={
        500: "X", 1000: "X", 1500: "he", 2000: "hello"})
    trajectory = build_trajectory(trace)
    assert texts(trajectory) == ["", "", "he", "llo"]


def test_complete_anchor_is_required_even_when_reference_is_known():
    trace = trace_for([("en", "spk1", "hello")])
    trace["queries"].pop()
    with pytest.raises(ValueError, match="Query plan|complete-language-run teacher anchor"):
        build_trajectory(trace)


@pytest.mark.parametrize("mutation", ["missing", "extra_run", "wrong_id", "wrong_time", "wrong_kind"])
def test_corrupted_or_incomplete_query_plan_cannot_manufacture_wait_labels(mutation):
    trace = trace_for([("en", "spk1", "hello")])
    if mutation == "missing":
        trace["queries"].pop(0)
    elif mutation == "extra_run":
        extra = dict(trace["queries"][0], run_index=9, query_id="run9:0:500")
        trace["queries"].append(extra)
    elif mutation == "wrong_id":
        trace["queries"][0]["query_id"] = "run0:0:999"
    elif mutation == "wrong_time":
        trace["queries"][0]["end_ms"] += 1
    elif mutation == "wrong_kind":
        trace["queries"][0]["kind"] = "full"
    with pytest.raises(ValueError):
        build_trajectory(trace)
