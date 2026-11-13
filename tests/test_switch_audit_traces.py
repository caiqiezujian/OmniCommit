import json

import pytest

from omnicommit.switch_audit_traces import load_trace, summarize_switch


SRC = "/reference/meeting.wav"
CLI = {"type": "ready", "input": SRC, "protocol": "first", "source_kind": "file", "chunk_ms": 500}
WEB = {"type": "ready", "protocol": "second", "chunk_seconds": .5,
       "file_start_seconds": 58.123, "file_end_seconds": 59.123}


def result(turn, start, end, *, event="audio", text="<|WAIT|>", raw=False, source_times=None):
    waiting = text == "<|WAIT|>"
    row = {"type": "result", "event": event, "turn": turn, "start_seconds": start,
           "end_seconds": end, "text": text, "action": "WAIT" if waiting else "DIRECT",
           "segments": [] if waiting else [{"language": "de", "speaker": "spk1", "text": "Hallo"}]}
    if source_times is not None:
        row["source_start_seconds"], row["source_end_seconds"] = source_times
    if raw:
        row["raw_model_text"] = " " + text
    return row


def dump(tmp_path, rows, filename="trace.jsonl"):
    path = tmp_path / filename
    content = json.dumps(rows) if path.suffix == ".json" else "\n".join(map(json.dumps, rows))
    path.write_text(content, encoding="utf-8")
    return path


def test_cli_results_errors_and_eos_are_distinct(tmp_path):
    de = "<|DIRECT|><de><spk1>Hallo"
    direct = result(2, .5, 1, text=de, raw=True)
    direct.update(generation_seconds=.25, service_seconds=.3, queue_wait_seconds=.01)
    rows = [CLI, result(1, 0, .5), direct, result(3, 1, 1, event="end_of_audio")]
    trace = load_trace(dump(tmp_path, rows, "cli.json"))
    assert trace.source == SRC
    assert trace.lookup(0, .5)["action"] == "WAIT"
    obs = trace.lookup(.5, 1)
    assert (obs["action"], obs["raw_model_text"], obs["turn"]) == ("DIRECT", " "+de, 2)
    assert obs["segments"][0]["language"] == "de" and obs["service_seconds"] == .3
    assert trace.lookup(.25, .75)["status"] == "grid_mismatch"
    assert trace.lookup(0, 1)["status"] == "grid_mismatch"
    assert trace.lookup(1, 1.5)["status"] == "missing"
    assert trace.lookup(1, 1, "end_of_audio")["status"] == "result"
    assert trace.terminal["status"] == "eos"


def test_web_range_source_time_and_result_gaps(tmp_path):
    rows = [WEB, result(1, 0, .5, source_times=(58.123, 58.623)),
            {"type": "file_wait"}, result(3, .8, 1, text="<de><spk1>Hallo", source_times=(58.923, 59.123)),
            result(4, 1, 1, event="end_of_audio", source_times=(59.123, 59.123)),
            {"type": "local_correction", "op": "rename_speaker"}]
    path = dump(tmp_path, {"segments": [], "events": rows}, "web.json")
    with pytest.raises(ValueError, match="supply the SFT recording path"):
        load_trace(path)
    trace = load_trace(path, source=SRC, offset_seconds=58.123)
    assert trace.lookup(58.123, 58.623)["turn"] == 1
    assert trace.lookup(58.623, 58.923)["status"] == "missing"
    assert trace.lookup(58.923, 59.123)["turn"] == 3
    with pytest.raises(ValueError, match="conflicts with"):
        load_trace(path, source=SRC, offset_seconds=58)


def test_cropped_cli_with_error_before_eos_and_partial_log(tmp_path):
    full_trace = [dict(CLI, input="/derived/clip.wav"), result(1, 0, .5)]
    failed = full_trace + [{"type": "error", "stage": "audio", "heard_seconds": .5,
                            "attempted_end_seconds": 1, "backend_called": False, "error": "budget"}]
    trace = load_trace(dump(tmp_path, failed), source=SRC, offset_seconds=58)
    assert trace.source == SRC and trace.header["input"] != SRC
    assert trace.lookup(58.5, 59)["status"] == "error"
    assert trace.terminal["source_heard_seconds"] == 58.5 and not trace.terminal["backend_called"]
    assert trace.lookup(59, 59, "end_of_audio")["status"] == "missing"
    path = dump(tmp_path, full_trace)
    assert load_trace(path).terminal == {"status": "incomplete"}
    with pytest.raises(ValueError, match="requires the original"):
        load_trace(path, offset_seconds=58)


def test_web_session_empty_or_failed_before_ready(tmp_path):
    empty = load_trace(dump(tmp_path, [WEB, {"type":"file_end_empty", "source_heard_seconds":58.123}]), source=SRC)
    assert empty.terminal == {"status": "empty", "source_heard_seconds": 58.123}
    assert empty.lookup(58.123, 58.623)["status"] == "missing"
    failed = load_trace(dump(tmp_path, [{"type":"error","stage":"load_model","error":"OOM"}]), source=SRC)
    assert failed.terminal["status"] == "error" and failed.terminal["error"] == "OOM"


def test_relative_web_file_error_after_one_turn(tmp_path):
    rows = [WEB, result(1, 0, .5), {"type":"error", "source_heard_seconds":58.623, "source_start_seconds":58.623,
                                    "source_end_seconds":59.123, "error":"budget"}]
    trace = load_trace(dump(tmp_path, rows), source=SRC)
    assert trace.lookup(58.123, 58.623)["action"] == "WAIT"
    assert trace.lookup(58.623, 59.123)["status"] == "error"


@pytest.mark.parametrize("bad", [
    [CLI, CLI], [CLI, result(1, 0, .5), result(2, .25, .75)],
    [CLI, result(1, 0, .5), result(1, .5, 1)],
    [CLI, result(1, 0, .5), result(2, .25, .25, event="end_of_audio")],
    [CLI, dict(result(1, 0, .5), source_start_seconds=58)], [dict(CLI, source_kind="microphone")],
])
def test_do_not_silently_combine_incompatible_observations(tmp_path, bad):
    with pytest.raises(ValueError):
        load_trace(dump(tmp_path, bad))


def test_switch_summary_uses_effective_language_and_bounds_not_turn_ids(tmp_path):
    # Simulate a web clip with its own turn numbering. An untagged continuation
    # can have a resolved German span; the German tag on an earlier result or
    # on EOS alone must not get credit for this switch.
    trace = load_trace(dump(tmp_path, [WEB, result(1, 0, .2, text="<de><spk1>Hallo"),
                                      result(2, .2, .45), result(3, .45, .8, text=" Hallo"),
                                      result(4, .8, 1), result(5,1,1,event="end_of_audio")]), source=SRC)
    turns = [{"turn": i+33, "event": "audio", "start_seconds": WEB["file_start_seconds"]+start,
              "end_seconds": WEB["file_start_seconds"]+end}
             for i,(start,end) in enumerate([(0,.2),(.2,.45),(.45,.8),(.8,1)])]
    anchor = turns[0]["end_seconds"]
    summary = summarize_switch(trace, turns, language="de", kind="both", anchor_seconds=anchor,
                               window_end_seconds=59.123, next_onset_seconds=58.95)
    assert (summary["observed_wait_turns"], summary["exact_audio_turns"]) == (1, 2)
    assert summary["first_target_tag_delay_seconds"] == pytest.approx(.6)
    assert summary["first_target_tag_turn"] == 3 and summary["first_target_tag_gold_turn"] == 35
    assert summary["direct_turns_by_language"] == {"de": 1}
    assert summary["longest_observed_wait_seconds"] == pytest.approx(.25)
    multiple = summarize_switch(trace, turns, language="de", kind="language", anchor_seconds=anchor,
                                window_end_seconds=59.123, next_onset_seconds=anchor+.15)
    assert multiple["status"] == "no_eligible_audio_turns" and multiple["first_target_tag_seconds"] is None
    unknown = summarize_switch(trace, turns, language="de", kind="language", anchor_seconds=None, window_end_seconds=59.123)
    assert unknown["exact_audio_turns"] == 4 and unknown["first_target_tag_delay_seconds"] is None
    speaker = summarize_switch(trace, turns, language="de", kind="speaker", anchor_seconds=anchor, window_end_seconds=59.123)
    assert speaker["first_target_tag_turn"] is None


def test_switch_summary_incomplete_data_and_target_attribution_missing(tmp_path):
    no_segments = result(2,.5,1,text="<de><spk1>Hallo")
    del no_segments["segments"]
    trace = load_trace(dump(tmp_path,[CLI, no_segments]))
    turns = [{"turn": i+1, "event":"audio", "start_seconds": i*.5, "end_seconds": (i+1)*.5} for i in range(3)]
    summary = summarize_switch(trace, turns, language="de", kind="language", anchor_seconds=.17, window_end_seconds=2)
    assert summary["status"] == "partial_coverage" and summary["coverage_counts"] == {"missing":2, "result":1}
    assert summary["observed_wait_turns"] == 0
    assert summary["first_target_tag_seconds"] is None and summary["direct_turns_without_segment_metadata"] == 1


def test_missing_or_regridded_audio_breaks_wait_chain(tmp_path):
    rows = [CLI, result(1,0,.5), result(3,1,1.5), result(4,1.5,2), result(5,2,2.8), result(6,3,3.5)]
    trace = load_trace(dump(tmp_path,rows))
    turns = [{"turn":i+1, "event":"audio", "start_seconds":i/2, "end_seconds":(i+1)/2} for i in range(7)]
    summary = summarize_switch(trace,turns,kind="language",language="ja",anchor_seconds=.4,window_end_seconds=3.5)
    assert summary["longest_observed_wait_seconds"] == pytest.approx(1.)
    assert summary["coverage_counts"] == {"result":4,"missing":1,"grid_mismatch":2}
