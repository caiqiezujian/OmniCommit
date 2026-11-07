import pytest

from omnicommit.evaluation import evaluate_events, transcript_events


def events(texts):
    result = []
    for i, segments in enumerate(texts):
        result.append({"event": "audio", "start_seconds": i * .5, "end_seconds": (i + 1) * .5,
                       "action": "DIRECT" if segments else "WAIT", "segments": segments,
                       "generation_seconds": .1})
    result.append({"event": "end_of_audio", "start_seconds": len(texts) * .5,
                   "end_seconds": len(texts) * .5, "action": "WAIT", "segments": []})
    return result


def seg(text, lang="zh", speaker="spk1"):
    return {"text": text, "language": lang, "speaker": speaker}


def same_opcodes(a, b):
    assert a == b
    return [("equal", 0, len(a), 0, len(b))]


def test_global_identity_mismatch_is_not_remapped():
    reference = {"segments": [seg("你好"), seg("Hi", "en", "spk2"), seg("回来了")],
                 "speaker_return_probes": [{"name": "A returns", "start_char": 4, "end_char": 7}]}
    prediction = events([[seg("你好")], [seg("Hi", "en", "spk2")], [seg("回来了", speaker="spk3")]])
    report = evaluate_events(reference, prediction, opcodes=same_opcodes)
    assert report["character_error_rate"] == 0
    assert report["speaker_accuracy_on_text_matches"] == 4 / 7
    assert report["unexpected_speaker_ids"] == ["spk3"]
    assert report["speaker_return_probes"][0]["speaker_accuracy_on_matches"] == 0
    assert report["complete_eos"] is True


def test_language_switch_same_speaker():
    reference = {"segments": [seg("你好"), seg("hello", "en")]}
    report = evaluate_events(reference, events([[seg("你好")], [seg("hello", "zh")]]), opcodes=same_opcodes)
    assert report["speaker_accuracy_on_text_matches"] == 1
    assert report["language_accuracy_on_text_matches"] == 2 / 7


def test_deletions_reduce_coverage_and_no_speaker_score():
    reference = {"segments": [seg("hello", "en")]}
    report = evaluate_events(reference, events([[]]), opcodes=lambda a, b: [("delete", 0, 5, 0, 0)])
    assert report["character_error_rate"] == 1
    assert report["matched_character_coverage"] == 0
    assert report["speaker_accuracy_on_text_matches"] is None


def test_requires_causal_event_timeline_and_explicit_eos():
    reference = {"segments": [seg("你好")]}
    sequence = events([[seg("你好")]])
    report = evaluate_events(reference, sequence[:-1], opcodes=same_opcodes)
    assert not report["complete_eos"]
    sequence[0]["start_seconds"] = .2
    with pytest.raises(ValueError, match="contiguous"):
        evaluate_events(reference, sequence, opcodes=same_opcodes)


def test_full_token_normalization_does_not_hide_number_errors():
    reference = {"segments": [seg("100", "en")]}
    report = evaluate_events(reference, events([[seg("200", "en")]]),
                             opcodes=lambda a, b: [("replace", 0, 1, 0, 1), ("equal", 1, 3, 1, 3)])
    assert report["substitutions"] == 1
    assert report["character_error_rate"] == 1 / 3


def test_error_run_reports_incomplete_instead_of_hiding_failure():
    reference = {"segments": [seg("你好")]}
    sequence = events([[seg("你好")]])[:-1]
    sequence.append({"event": "error", "error": "context exhausted"})
    report = evaluate_events(reference, sequence, opcodes=same_opcodes)
    assert report["runtime_error"] == "context exhausted"
    assert report["complete_eos"] is False


def test_web_transport_is_filtered_but_bad_result_not_hidden():
    assert transcript_events([{"type": "ready"}, {"type": "control_error"},
                              {"type": "error", "error": "failed"}]) == [
        {"type": "error", "error": "failed", "event": "error"}]
    with pytest.raises(ValueError, match="Unknown"):
        transcript_events([{"type": "result", "text": "oops"}])


def test_actual_fast_edit_distance_library():
    pytest.importorskip("rapidfuzz")
    report = evaluate_events({"segments": [seg("100", "en")]}, events([[seg("200", "en")]]))
    assert report["substitutions"] == 1
    assert report["deletions"] == report["insertions"] == 0
