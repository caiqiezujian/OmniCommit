import json
import wave

import pytest

import filter_tail
from asr_gspo.tail_source import has_terminal_punctuation, inspect_source_tail


def source(tmp_path, end=1900):
    path = tmp_path / "audio.wav"
    if not path.exists():
        with wave.open(str(path), "wb") as f:
            f.setparams((1, 2, 16000, 0, "NONE", ""))
            f.writeframes(bytes(2 * 16000 * 3))
    return dict(audios=[str(path)], messages=[dict(role="assistant", content="<zh><spk1>你好。")],
                alignment=[dict(text="你好", start_ms=100, end_ms=end)])


@pytest.mark.parametrize("end,kept", [(1900, True), (2000, False), (2001, False), (1999.999, True)])
def test_strict_greater_than_one_second_without_rounding(tmp_path, end, kept):
    row = source(tmp_path, end)
    if kept:
        result = filter_tail.inspect_tail(row, tmp_path)
        assert result["tail_gap_ms"] > 1000
    else:
        with pytest.raises(filter_tail.RowRejected, match="tail_not_greater"):
            filter_tail.inspect_tail(row, tmp_path)


def test_separately_timed_period_is_not_the_last_speech_end(tmp_path):
    row = source(tmp_path)
    row["alignment"].append(dict(text="。", start_ms=2500, end_ms=2800))
    assert filter_tail.inspect_tail(row, tmp_path)["last_text_end_ms"] == 1900


def test_final_text_must_have_trusted_end_not_just_a_partial_prefix(tmp_path):
    row = source(tmp_path)
    for alignment in ([dict(text="你", start_ms=100, end_ms=300)],
                      [dict(text="你好", start_ms=100, end_ms=1900, trusted=False)],
                      [dict(text="你好", start_ms=100, end_ms=None)]):
        row["alignment"] = alignment
        with pytest.raises(filter_tail.RowRejected):
            filter_tail.inspect_tail(row, tmp_path)
    del row["alignment"]
    row["timestamps"] = [{"time_ms": 900, "language": "en", "speaker": "spk2"}]
    with pytest.raises(filter_tail.RowRejected, match="missing_trusted_final"):
        filter_tail.inspect_tail(row, tmp_path)


def test_only_final_timestamp_is_used_not_intermediate_timings(tmp_path):
    row = source(tmp_path)
    row["alignment"] = [dict(text="你", start_ms=100, end_ms=2300),
                        dict(text="好", start_ms=500, end_ms=1600)]
    # This gate follows the user's final-annotation rule, not full CTC replay.
    # Intermediate end times are not needed to rebuild a streaming target.
    assert filter_tail.inspect_tail(row, tmp_path)["tail_gap_ms"] == 1400


def test_merged_units_can_be_filtered_without_ctc_or_resegmentation(tmp_path):
    row = source(tmp_path)
    del row["alignment"]
    row["output"] = json.dumps({"segments": [{"source_text": "<zh><spk1>你好。", "action": "DIRECT"}]})
    row["timestamps"] = [dict(start_ms=100, end_ms=1900)]
    assert filter_tail.inspect_tail(row, tmp_path)["tail_gap_ms"] == 1100


def test_filter_preserves_kept_rows_and_skips_errors_without_audio_decode(tmp_path, monkeypatch):
    kept = source(tmp_path)
    dropped = source(tmp_path, 2000)
    missing = dict(kept, audios=[str(tmp_path / "missing.wav")])
    good_raw = (json.dumps(kept, ensure_ascii=False, indent=None) + "\r\n").encode("utf-8")
    path = tmp_path / "source.jsonl"
    raw = b"invalid\n" + good_raw + (json.dumps(dropped) + "\n" + json.dumps(missing) + "\n").encode()
    path.write_bytes(raw)
    output = tmp_path / "filtered.jsonl"
    monkeypatch.setattr(filter_tail.sf, "read", lambda *args, **kw: pytest.fail("Must not decode waveforms"))
    report = filter_tail.main([str(path), "--output", str(output)])
    assert report["completed"] and report["counts"] == {"lines": 4, "kept": 1, "rejected": 3}
    assert output.read_bytes() == good_raw and path.read_bytes() == raw
    rejected = [json.loads(s) for s in output.with_suffix(".rejected.jsonl").read_text().splitlines()]
    assert [r["line"] for r in rejected] == [1, 3, 4]
    assert all("messages" not in r for r in rejected)
    from asr_gspo.data import prepare_record
    task = json.loads(prepare_record(kept, tmp_path, tail_policy="keep")["asr_task"])
    assert task["duration_seconds"] == 3 and task["reference_text"] == "你好。"
    assert task["end_policy"] == "eof" and not task["terminal_period_removed"]


def test_filter_does_not_overwrite_source_or_existing_output(tmp_path):
    row = source(tmp_path)
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(SystemExit):
        filter_tail.main([str(path), "--output", str(path)])
    existing = tmp_path / "existing.jsonl"
    existing.write_text("keep me", encoding="utf-8")
    with pytest.raises(SystemExit):
        filter_tail.main([str(path), "--output", str(existing)])
    assert existing.read_text() == "keep me"


@pytest.mark.parametrize("text", ["你好", "你好”", "预算50%", "金额3.50", "Hello)", "Дальше"])
def test_unpunctuated_tail_kept_without_timestamps(tmp_path, text):
    row = source(tmp_path, 2900)
    row["messages"][-1]["content"] = "<zh><spk1>" + text
    row.pop("alignment")
    # No required end, no invented zero-gap/end-of-audio anchor.
    result = filter_tail.inspect_tail(row, tmp_path)
    assert result["tail_decision"] == "keep_unpunctuated"
    assert result["last_text_end_ms"] is None and result["tail_gap_ms"] is None


@pytest.mark.parametrize("text", ["你好。", "你好，", "Hello!", "Hello?", "Привет.", "هل؟", "你好。”"])
def test_terminal_punctuation_is_multilingual(text):
    assert has_terminal_punctuation(text)


def test_final_direct_without_punctuation_qualifies_even_if_old_reference_has_period(tmp_path):
    row = source(tmp_path, 2900)
    row["output"] = json.dumps({"segments": [{"source_text": "<zh><spk1>你好", "action": "DIRECT"}]})
    original = row["messages"][-1]["content"]
    result = filter_tail.inspect_tail(row, tmp_path)
    assert result["tail_decision"] == "keep_unpunctuated_segment"
    assert result["terminal_punctuation_trusted"] is False
    assert row["messages"][-1]["content"] == original


def test_trailing_wait_body_cannot_rewind_to_an_earlier_direct(tmp_path):
    row = source(tmp_path, 2900)
    row["output"] = {"segments": [{"source_text": "<zh><spk1>你", "action": "DIRECT"},
                                    {"source_text": "好。", "action": "WAIT"}]}
    with pytest.raises(filter_tail.RowRejected, match="tail_not_greater"):
        filter_tail.inspect_tail(row, tmp_path)


def test_early_invalid_timing_and_reflowed_spaces_do_not_reject_valid_tail(tmp_path):
    row = source(tmp_path)
    row["messages"][-1]["content"] = "<ru><spk1>Привет.<ru><spk2>  Как дела?<zh><spk3>现在继续讨论。"
    row.pop("alignment")
    row["output"] = {"segments": ["old malformed unused segment",
                                    {"source_text": "<zh><spk3>现在继续讨论。", "action": "DIRECT"}]}
    row["timestamps"] = [dict(start_ms="bad", end_ms=-5, text="ignored old text"),
                         dict(start_ms=1600, end_ms=1900, source_text="<zh><spk3>现在 继续讨论。")]
    result = filter_tail.inspect_tail(row, tmp_path)
    assert result["tail_gap_ms"] == 1100


def test_last_timestamp_cannot_be_for_an_earlier_direct_or_untrusted_text(tmp_path):
    row = source(tmp_path)
    row["messages"][-1]["content"] = "<zh><spk1>你好。最后一句。"
    row["timestamps"] = [dict(start_ms=100, end_ms=1900, source_text="你好。")]
    with pytest.raises(filter_tail.RowRejected, match="tail_timestamp_text_mismatch"):
        filter_tail.inspect_tail(row, tmp_path)
    row["timestamps"][-1].update(source_text="最后一句。", trusted=False)
    with pytest.raises(filter_tail.RowRejected, match="missing_trusted_final"):
        filter_tail.inspect_tail(row, tmp_path)


def test_metadata_only_mode_reports_gap_without_rewriting_or_interpolation():
    row = {"timestamps": [{"source_text": "最后一句。", "end_ms": 26080}]}
    result = inspect_source_tail(row, "前面。最后一句。", 27000, enforce=False)
    assert result["tail_decision"] == "reject_punctuated_short_tail" and result["tail_gap_ms"] == 920
    assert result["last_text_end_ms"] == 26080


def test_text_gate_does_not_require_any_timestamp_or_malformed_middle_segments():
    from asr_gspo.tail_source import tail_text_metadata
    row = {"output": {"segments": [None, {"source_text": "<zh><spk3>最后几个字", "action": "DIRECT"}]},
           "timestamps": "unused malformed old metadata"}
    info = tail_text_metadata(row, "完整转录，最后几个字。")
    assert info["terminal_punctuation_trusted"] is False
    assert info["tail_segment_index"] == 1


def test_wrong_numeric_tail_is_not_accepted_as_same_suffix():
    row = {"timestamps": [{"text": "350元。", "end_ms": 1900}]}
    with pytest.raises(filter_tail.RowRejected, match="tail_timestamp_text_mismatch"):
        inspect_source_tail(row, "金额3.50元。", 3000)


def test_unused_alignment_identity_does_not_reject_open_ending_or_timestamp_tail(tmp_path):
    row = source(tmp_path)
    row["id"] = "actual"
    row["alignment"] = {"status": "error", "id": "old", "audio": "/missing/old.wav", "duration_ms": 42}
    row["timestamps"] = [{"source_text": "你好。", "end_ms": 1900}]
    assert filter_tail.inspect_tail(row, tmp_path)["tail_decision"] == "keep_punctuated_with_gap"
    row.pop("timestamps")
    row["messages"][-1]["content"] = "<zh><spk1>你好"
    assert filter_tail.inspect_tail(row, tmp_path)["tail_decision"] == "keep_unpunctuated"


@pytest.mark.parametrize("where", ["alignment_container", "alignment_entry", "timestamp_entry"])
def test_identity_of_the_chosen_tail_timestamp_is_still_validated(tmp_path, where):
    row = source(tmp_path)
    row["id"] = "actual"
    end = {"text": "你好", "end_ms": 1900}
    if where == "alignment_container":
        row["alignment"] = {"audio": "/wrong/audio.wav", "words": [end]}
    elif where == "alignment_entry":
        row["alignment"] = [dict(end, id="wrong")]
    else:
        row["timestamps"] = [dict(end, audio="/wrong/audio.wav")]
    with pytest.raises(ValueError):
        filter_tail.inspect_tail(row, tmp_path)
