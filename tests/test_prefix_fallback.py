"""Trusted completed-span supervision, sample causality and local recovery."""
from copy import deepcopy

import pytest

from omnicommit.prefix_labels import build_trajectory
from omnicommit.prefix_source import parse_source, plan_queries, plan_span_queries


def trace_for(spans, switches, *, duration=2500, chunk=.5, primary=None, local=None):
    row = {"audios": ["/data/example.wav"], "messages": [{"role": "assistant", "content": "".join(
        f"<{language}><{speaker}>{text}" for language, speaker, text in spans)}],
        "timestamps": [{"time_ms": time} for time in switches]}
    source = parse_source(row, duration)
    queries = plan_queries(source, duration, 1000, chunk)
    for q in queries:
        q.update(text=(primary or {}).get((q["run_index"], q["end_sample"]), "WRONG"), error=None)
    trace = {"kind": "local_prefix_trace", "status": "ok", "source": source,
             "id": "test", "audio": "/data/example.wav", "input_hash": "test",
             "audio_info": {"frames": duration, "sample_rate": 1000},
             "chunk_seconds": chunk, "number_format": "normalized", "queries": queries}
    if local is not None:
        trace["local_queries"] = plan_span_queries(source, duration, 1000, chunk)
        for q in trace["local_queries"]:
            q.update(text=local.get((q["span_index"], q["end_sample"]), "WRONG"), error=None)
    return trace


def turn_texts(trajectory):
    return ["".join(s["text"] for s in t["segments"]) for t in trajectory["turns"]]


def test_short_language_b_released_with_c_after_shared_onset_chunk():
    trace = trace_for([("zh", "spk1", "AAA"), ("en", "spk1", "B"), ("fr", "spk1", "CCC")],
                      [1200, 1400], duration=2000, primary={
                          (0, 500): "A", (0, 1000): "AA", (0, 1200): "AAA",
                          (1, 1400): "", (2, 1500): "", (2, 2000): "CCC"})
    before = deepcopy(trace)
    result = build_trajectory(trace, comparison="exact")
    assert turn_texts(result) == ["A", "A", "A", "BCCC", ""]
    assert trace == before  # Do not repair or overwrite raw teacher evidence.
    assert result["turns"][3]["segments"] == [
        {"language": "en", "speaker": "spk1", "text": "B"},
        {"language": "fr", "speaker": "spk1", "text": "CCC"}]
    assert result["metrics"]["reference_fallback_spans"] == 1
    assert result["metrics"]["reference_fallback_characters"] == 1
    assert result["metrics"]["teacher_full_mismatch_runs"] == 1
    event, = result["provenance"]["reference_fallback_events"]
    assert event["source"] == "reference_completed_span"
    assert event["text"] == "B" and event["span_end_ms"] == 1400
    assert event["emitted_at_ms"] == 2000 and event["turn_index"] == 3


def test_exact_next_chunk_switch_holds_c_but_not_previous_b():
    result = build_trajectory(trace_for([
        ("zh", "spk1", "A"), ("en", "spk1", "B"), ("fr", "spk1", "C")],
        [1200, 1500], primary={(0, 500): "A", (0, 1000): "A", (0, 1200): "A",
                             (2, 2000): "C", (2, 2500): "C"}))
    assert turn_texts(result) == ["A", "", "", "B", "C", ""]


@pytest.mark.parametrize("chunk, expected_end", [(.5, 2000), (.8, 2400), (1., 3000)])
def test_chunk_length_controls_gate_in_samples_not_fixed_500ms(chunk, expected_end):
    trace = trace_for([("zh", "spk1", "A"), ("en", "spk1", "B"), ("fr", "spk1", "C")],
                      [1200, 1400], duration=4000, chunk=chunk)
    result = build_trajectory(trace)
    event = next(e for e in result["provenance"]["reference_fallback_events"] if e["text"] == "B")
    assert event["emitted_at_ms"] == expected_end
    assert event["emitted_at_ms"] >= event["span_end_ms"]
    assert result["policy"]["chunk_seconds"] == chunk


def test_partial_teacher_confirmed_text_is_not_repeated_or_counted_as_fallback():
    trace = trace_for([("zh", "spk1", "A"), ("en", "spk1", "BBBB"), ("fr", "spk1", "C")],
                      [600, 1800], duration=2500, primary={
                          (0, 500): "A", (0, 600): "A", (1, 1000): "B", (1, 1500): "BB",
                          (1, 1800): "wrong", (2, 2000): "C", (2, 2500): "C"})
    result = build_trajectory(trace)
    assert turn_texts(result) == ["A", "", "BB", "BB", "C", ""]
    event, = result["provenance"]["reference_fallback_events"]
    assert event["text"] == "BB" and event["char_start"] == 3 and event["char_end"] == 5
    assert result["metrics"]["reference_fallback_characters"] == 2


def test_later_switch_does_not_refreeze_older_completed_spans():
    trace = trace_for([("zh", "spk1", "A"), ("en", "spk1", "B"),
                       ("fr", "spk1", "C"), ("de", "spk1", "D")],
                      [1200, 1400, 1800], duration=2500)
    result = build_trajectory(trace)
    assert turn_texts(result) == ["", "", "A", "BC", "D", ""]
    assert result["turns"][3]["provenance"]["switch_gate"] is True


def test_many_short_spans_share_one_release_without_artificial_wait_queue():
    trace = trace_for([("zh", "spk1", "A"), ("en", "spk1", "B"),
                       ("fr", "spk1", "C"), ("de", "spk2", "D")],
                      [1100, 1200, 1300], duration=2000)
    result = build_trajectory(trace)
    assert turn_texts(result) == ["", "", "A", "BCD", ""]
    assert len(result["turns"][3]["segments"]) == 3


def test_same_language_local_queries_recover_after_missing_middle_speaker():
    trace = trace_for([("en", "spk1", "A"), ("en", "spk2", "B"), ("en", "spk1", "CCC")],
                      [600, 800], duration=2000,
                      primary={(0, 500): "A", (0, 600): "A", (0, 800): "A", (0, 1000): "AC",
                               (0, 1500): "ACC", (0, 2000): "ACCC"},
                      local={(1, 800): "", (2, 1000): "C", (2, 1500): "CC", (2, 2000): "CCC"})
    result = build_trajectory(trace)
    assert turn_texts(result) == ["A", "", "BCC", "C", ""]
    assert result["metrics"]["reference_fallback_characters"] == 1
    assert result["metrics"]["language_fallback_characters"] == {"en": 1}
    assert result["provenance"]["has_local_queries"] is True


def test_legacy_trace_without_local_queries_uses_completed_spans_with_warning():
    trace = trace_for([("en", "spk1", "A"), ("en", "spk2", "B"), ("en", "spk1", "CCC")],
                      [600, 800], duration=2000,
                      primary={(0, 500): "A", (0, 600): "A", (0, 800): "A", (0, 1000): "AC",
                               (0, 1500): "ACC", (0, 2000): "ACCC"})
    result = build_trajectory(trace)
    assert turn_texts(result) == ["A", "", "B", "CCC", ""]
    assert result["metrics"]["reference_fallback_characters"] == 4
    assert result["provenance"]["warnings"] == ["legacy_trace_without_local_speaker_queries"]


def test_a_b_a_has_separate_reference_progress_for_returning_language():
    trace = trace_for([("zh", "spk1", "甲"), ("en", "spk1", "B"), ("zh", "spk1", "乙丙")],
                      [600, 800], duration=2000)
    result = build_trajectory(trace)
    assert turn_texts(result) == ["", "甲", "B", "乙丙", ""]
    assert "".join(turn_texts(result)) == "甲B乙丙"


def test_end_of_audio_releases_closed_final_span_without_fake_audio():
    trace = trace_for([("zh", "spk1", "A"), ("en", "spk2", "B")], [1900], duration=2100)
    result = build_trajectory(trace)
    assert turn_texts(result) == ["", "", "", "A", "B", ""]
    # If the final span begins in the final partial audio chunk, its onset gate
    # expires only at explicit EOS; no extra 500 ms audio is invented.
    trace = trace_for([("zh", "spk1", "A"), ("en", "spk2", "B")], [2050], duration=2100)
    result = build_trajectory(trace)
    assert turn_texts(result) == ["", "", "", "", "A", "B"]
    event = result["provenance"]["reference_fallback_events"][-1]
    assert event["event"] == "end_of_audio" and event["emitted_at_ms"] == 2100


def test_unfinished_span_never_filled_from_future_reference():
    trace = trace_for([("zh", "spk1", "A"), ("en", "spk1", "long sentence")], [600], duration=4000)
    result = build_trajectory(trace)
    assert turn_texts(result) == ["", "A", "", "", "", "", "", "long sentence", ""]
    assert all(e["emitted_at_ms"] >= e["span_end_ms"] for e in result["provenance"]["reference_fallback_events"])


def test_off_preserves_strict_verification_and_single_span_stays_strict():
    mixed = trace_for([("zh", "spk1", "A"), ("en", "spk1", "B")], [600])
    with pytest.raises(ValueError, match="full teacher/reference mismatch"):
        build_trajectory(mixed, reference_fallback="off")
    single = trace_for([("zh", "spk1", "A")], [])
    with pytest.raises(ValueError, match="full teacher/reference mismatch"):
        build_trajectory(single)


@pytest.mark.parametrize("field", ["queries", "local_queries"])
def test_technical_failures_are_never_reference_fallback(field):
    trace = trace_for([("en", "spk1", "A"), ("en", "spk2", "B")], [600], local={})
    trace[field][0]["error"] = "generation reached max tokens"
    with pytest.raises(ValueError, match="Teacher query failed"):
        build_trajectory(trace)


@pytest.mark.parametrize("field", ["queries", "local_queries"])
def test_completed_known_tags_only_counts_as_empty_in_fallback_mode(field):
    trace = trace_for([("en", "spk1", "A"), ("en", "spk2", "B")], [600], local={})
    trace[field][0]["text"] = "<en><spk2>"
    raw = deepcopy(trace)
    result = build_trajectory(trace)
    assert "".join(turn_texts(result)) == "AB"
    assert result["metrics"]["reference_fallback_characters"] == 2
    assert trace == raw


def test_known_tags_only_short_language_falls_back_but_off_rejects():
    trace = trace_for([("zh", "spk1", "A"), ("en", "spk1", "B"), ("fr", "spk1", "C")],
                      [1200, 1400], duration=2000,
                      primary={(0, 500): "A", (0, 1000): "A", (0, 1200): "A",
                               (1, 1400): "<en><spk1>", (2, 1500): "", (2, 2000): "C"})
    result = build_trajectory(trace)
    assert turn_texts(result) == ["A", "", "", "BC", ""]
    assert result["metrics"]["reference_fallback_characters"] == 1
    with pytest.raises(ValueError, match="tags only"):
        build_trajectory(trace, reference_fallback="off")


@pytest.mark.parametrize("field", ["queries", "local_queries"])
def test_unknown_action_markup_remains_error_in_fallback_mode(field):
    trace = trace_for([("en", "spk1", "A"), ("en", "spk2", "B")], [600], local={})
    trace[field][0]["text"] = "<|WAIT|>"
    with pytest.raises(ValueError, match="Unexpected teacher markup"):
        build_trajectory(trace)


def test_incomplete_auxiliary_plan_cannot_silently_manufacture_fallback():
    trace = trace_for([("en", "spk1", "A"), ("en", "spk2", "B")], [600], local={})
    trace["local_queries"].pop()
    with pytest.raises(ValueError, match="Local span plan"):
        build_trajectory(trace)


def test_declared_new_collector_policy_cannot_masquerade_as_legacy_when_queries_missing():
    trace = trace_for([("en", "spk1", "A"), ("en", "spk2", "B")], [600])
    trace["teacher"] = {"local_query_policy": "later-speakers-in-language-run-v1"}
    with pytest.raises(ValueError, match="requires local_queries"):
        build_trajectory(trace)


def test_declared_no_local_queries_is_valid_when_no_auxiliary_plan_is_needed():
    trace = trace_for([("zh", "spk1", "A"), ("en", "spk1", "B")], [600], local={})
    assert trace["local_queries"] == []
    trace["teacher"] = {"local_query_policy": "later-speakers-in-language-run-v1"}
    result = build_trajectory(trace)
    assert result["provenance"]["has_local_queries"] is True


@pytest.mark.parametrize("mutation", ["end_time", "text", "offset"])
def test_tampered_source_geometry_cannot_enable_early_fallback(mutation):
    trace = trace_for([("zh", "spk1", "A"), ("en", "spk1", "B")], [1200])
    if mutation == "end_time":
        trace["source"]["spans"][1]["end_ms"] = 1400
    elif mutation == "text":
        trace["source"]["text"] = "AX"
    else:
        trace["source"]["events"][0]["char_offset"] = 0
    with pytest.raises(ValueError, match="geometry"):
        build_trajectory(trace)


def test_invalid_fallback_policy_is_rejected():
    with pytest.raises(ValueError, match="reference_fallback"):
        build_trajectory({}, reference_fallback="guess")
