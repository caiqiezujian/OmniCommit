"""Fourth assigns emission times without changing the annotated transcript."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_asr.protocol import DIRECT, WAIT, SPK_CHANGE, get_protocol
from fourth_asr.recast import build_sft_row
from fourth_asr.semantic_preparation import prepare_units


def make_record(parts, marked=None):
    cells = [f"<{lang}><{speaker}>{text}" for _, _, lang, speaker, text in parts]
    return {"audios": ["test.wav"], "messages": [{"role": "assistant", "content": marked or "".join(cells)}],
            "timestamps": [{"index": i, "start_ms": a, "end_ms": b}
                           for i, (a, b, *_) in enumerate(parts, 1)],
            "output": json.dumps({"segments": [{"source_text": s, "action": "DIRECT"} for s in cells]})}


def build(record, duration, step=1000, **kwargs):
    units, stats = prepare_units(record, duration, chunk_ms=step, **kwargs)
    audit = {}
    row = build_sft_row(units, Path("test.wav"), duration*16, 16000, step, "normalized",
                        protocol=get_protocol("speaker"), audit=audit)
    replies = [m["content"] for m in row["messages"][2:-2:2]]
    emitted = "".join(r.removeprefix(DIRECT).replace(SPK_CHANGE, "") for r in replies if r != WAIT)
    assert emitted == "".join(u.text for u in units)
    assert row["messages"][-1]["content"] == WAIT
    return [None if r == WAIT else r.removeprefix(DIRECT) for r in replies], units, stats, audit


def monolingual(parts, tagged, lang="zh"):
    return make_record([(start, end, lang, "spk1", s) for start, end, s in parts],
                       "".join(f"<{lang}><spk1>{text}" for text in tagged))


def test_basketball_actual_intervals_and_faithful_cut():
    parts = [(10320,11120,"第二节最后，"),(11120,11360,"湖人以"),(11360,12240,"70比71"),
             (12240,12960,"小输一分，"),(12960,14160,"进入到下半场。")]
    record = monolingual(parts, ["第二节最后，","湖人以70比71小输一分，","进入到下半场。"])
    turns, units, stats, _ = build(record, 15600)
    assert turns[10:15] == [None,"第二节最后，","湖人以70比71小输一分，",None,"进入到下半场。"]
    assert stats["merged_units"] == 1
    assert [(u.start_ms, u.end_ms) for u in units][1] == (11120,12240)


@pytest.mark.parametrize("parts,phrases,lang,want,merges", [
    ([(320,1510,"湖人以70比71小输一分，"), (1580,1950,"进入下"), (1950,2540,"半场。")],
     ["湖人以70比71小输一分，","进入下半场。"], "zh", [None,"湖人以70比71小输一分，","进入下半场。"], 0),
    ([(70,970,"先看预算，"),(1030,1230,"再"),(1250,1450,"看"),(1470,1670,"第"),(1690,2040,"二项。")],
     ["先看预算，","再看第二项。"],"zh",["先看预算，",None,"再看第二项。"],3),
    ([(460,1390,"Genau,"),(1410,1660," 12,"),(1670,2090,"5 Prozent.")],
     ["Genau, 12,5 Prozent."],"de",[None,"Genau,"," 12,5 Prozent."],1),
    ([(430,1130,"届きました"),(1150,1300,"か？"),(1410,2640,"念のため確認します。")],
     ["届きましたか？","念のため確認します。"],"ja",[None,"届きましたか？","念のため確認します。"],1),
    ([(820,1320,"能，"),(1480,1760,"但是"),(2180,2690,"比较卡。")],
     ["能，但是比较卡。"],"zh",[None,"能，","但是比较卡。"],0),
    ([(760,1580,"对，对，我收到了。"),(1600,1940,"然后"),(2090,2710,"我发给你。")],
     ["对，对，我收到了。","然后我发给你。"],"zh",[None,"对，对，我收到了。","然后我发给你。"],0),
])
def test_fine_units_and_submission_are_separate(parts, phrases, lang, want, merges):
    turns, _, stats, _ = build(monolingual(parts, phrases, lang), 3500)
    assert turns[:3] == want
    assert stats["merged_units"] == merges


@pytest.mark.parametrize("parts,duration,want,switch_turns", [
    ([(420,1420,"zh","spk1","能听见吗？"), (1565,1720,"zh","spk2","能。")],
     3000, [None,"能听见吗？",SPK_CHANGE+"能。"], [2]),
    ([(480,1230,"zh","spk1","那我来。"),(1310,1480,"zh","spk2","好。"),(1580,2250,"zh","spk1","我共享屏幕。")],
     3000,[None,"那我来。",SPK_CHANGE+"好。"+SPK_CHANGE+"我共享屏幕。"],[2]),
    ([(350,1360,"zh","spk1","请打开"),(1420,1660,"en","spk1"," dark"),(1660,1880,"en","spk1"," mode")],
     3000,[None,"请打开"," dark mode"],[2]),
    ([(340,1250,"zh","spk1","先投票。"), (1320,1530,"en","spk1","OK."), (1620,2220,"zh","spk1","继续。"),
      (2350,2850,"en","spk1","All set.")],
     4000, [None,"先投票。","OK.继续。","All set."],[2,3]),
    ([(220,1450,"zh","spk1","还有谁要补充？"),(1530,1740,"de","spk2","Ja,"),(1780,2570,"de","spk2"," ich auch.")],
     3000,[None,"还有谁要补充？",SPK_CHANGE+"Ja, ich auch."],[2]),
    ([(200,920,"zh","spk1","下一位。"),(4230,4780,"en","spk2","I'm here.")],
     5400,["下一位。",None,None,None,SPK_CHANGE+"I'm here.",None],[]),
    ([(360,1330,"zh","spk1","先到这里。"),(1450,1730,"en","spk2","Bye.")],
     1840,[None,"先到这里。"+SPK_CHANGE+"Bye."],[]),
    ([(530,2210,"zh","spk1","这版我明天发。"),(1460,1740,"en","spk2","OK.")],
     3200,[None,None,"这版我明天发。"+SPK_CHANGE+"OK.",None],[]),
    ([(100,975,"zh","spk1","开始。"),(1000,1790,"en","spk2","Hi.")],
     3000,["开始。",SPK_CHANGE+"Hi.",None],[]),
])
def test_switch_only_when_both_states_share_grid(parts, duration, want, switch_turns):
    turns, _, _, audit = build(make_record(parts), duration)
    assert turns == want
    assert audit["switch_deferred_turns"] == switch_turns


def test_prepared_group_survives_timed_span_option():
    parts = [(400,1250,"已知，"),(1700,1850,"张"),(1860,2260,"伟确认。")]
    record = monolingual(parts, ["已知，","张伟确认。"])
    record["timestamps"][2]["spans"] = [
        {"start_ms":1860,"end_ms":1930,"text":"伟","language":"zh","speaker":"spk1"},
        {"start_ms":1930,"end_ms":2260,"text":"确认。","language":"zh","speaker":"spk1"}]
    turns, _, _, _ = build(record, 3100, release_on="timed-span")
    assert turns == [None,"已知，","张伟确认。",None]


@pytest.mark.parametrize("tail,following,expected_hold", [
    ("进入下", "半场。", True), ("但是", "比较卡。", True), ("我同意", "我们明天再谈。", False),
])
def test_repeated_source_headers_do_not_hide_a_dangling_tail(tail, following, expected_hold):
    record = make_record([(260,1420,"zh","spk1","先说预算，"),(1480,1870,"zh","spk1",tail),
                          (2150,2630,"zh","spk1",following)])
    turns, _, _, audit = build(record, 3500)
    assert turns[1] == ("先说预算，" if expected_hold else "先说预算，" + tail)
    assert audit["punctuation_deferred_turns"] == ([2] if expected_hold else [])


def test_mixed_parent_not_split_without_timing_evidence():
    head, tail = "<zh><spk1>我发了，","<en><spk2>Yes."
    record = {"audios":["test.wav"],"messages":[{"role":"assistant","content":head+tail}],
              "output":{"segments":[{"source_text":head+tail}]},
              "timestamps":[{"start_ms":120,"end_ms":1420}]}
    turns, _, stats, audit = build(record, 3100)
    assert turns == [None,"我发了，"+SPK_CHANGE+"Yes.",None,None]
    assert stats["mixed_units_kept"] == 1
    assert audit["switch_deferred_turns"] == []


def test_nested_overlap_in_supplied_child_spans():
    head, tail = "<zh><spk1>明天发。","<en><spk2>OK."
    record = {"audios":["test.wav"],"messages":[{"role":"assistant","content":head+tail}],
              "output":{"segments":[{"source_text":head+tail}]},
              "timestamps":[{"start_ms":500,"end_ms":2210,"spans":[
                  {"start_ms":530,"end_ms":2210,"language":"zh","speaker":"spk1","text":head},
                  {"start_ms":1460,"end_ms":1740,"language":"en","speaker":"spk2","text":tail}]}]}
    turns, _, stats, audit = build(record, 3200)
    assert turns == [None,None,"明天发。"+SPK_CHANGE+"OK.",None]
    assert stats["overlapping_switches"] == 1
    assert audit["switch_deferred_turns"] == []


def test_split_decimal_delimiter_preserved():
    pieces = [(330,1360,"Genau,"),(1390,1650," 12"),(1660,1690,","),(1700,2230,"5 Prozent.")]
    want = ["Genau, 12,5 Prozent."]
    turns, _, _, _ = build(monolingual(pieces, want, "de"), 3200)
    assert turns == [None,"Genau,"," 12,5 Prozent.",None]


def merged_row(record, duration, *, open_end=False):
    units, _ = prepare_units(record, duration, chunk_ms=1000, release_on="merged-unit")
    audit = {}
    row = build_sft_row(units, Path("test.wav"), duration * 16, 16000, 1000, "normalized",
                        protocol=get_protocol("speaker"), audit=audit,
                        release_completed=True, open_end=open_end)
    assert audit == {"switch_deferred_turns": [], "punctuation_deferred_turns": []}
    return row


def test_merged_group_commits_completed_interjection_before_next_language_finishes():
    record = make_record([(0, 440, "zh", "spk1", "听到了吗？"),
                          (530, 730, "zh", "spk2", "好。"),
                          (810, 1260, "en", "spk2", " Yes."),
                          (1320, 1500, "zh", "spk1", "谢谢。"),
                          (1550, 1870, "zh", "spk2", "客气。")])
    replies = [m["content"] for m in merged_row(record, 2550)["messages"] if m["role"] == "assistant"]
    assert replies == [DIRECT + "听到了吗？" + SPK_CHANGE + "好。",
                       DIRECT + " Yes." + SPK_CHANGE + "谢谢。" + SPK_CHANGE + "客气。",
                       WAIT, WAIT]


def test_merged_groups_preserve_adjacent_number_boundaries_and_open_recording():
    # Both numbers were kept separate by Step 1. The legacy number heuristic
    # could join even across a 0 ms gap with --short-ms 0.
    parts = [(130, 860, "12"), (860, 1430, "345")]
    record = monolingual(parts, ["12345"], "en")
    row = merged_row(record, 1430, open_end=True)
    assert [m["content"] for m in row["messages"][1:]] == [
        "<audio>", DIRECT + "12", "<audio>", DIRECT + "345"]
    assert len(row["audios"]) == 2


@pytest.mark.parametrize("timed,first,second", [
    (False, WAIT, DIRECT + "请。" + SPK_CHANGE + "OK."),
    (True, DIRECT + "请。", DIRECT + SPK_CHANGE + "OK."),
])
def test_mixed_merged_unit_uses_real_child_times_only(timed, first, second):
    head, tail = "<zh><spk1>请。", "<en><spk2>OK."
    record = {"audios": ["test.wav"], "messages": [{"role": "assistant", "content": head + tail}],
              "output": {"segments": [{"source_text": head + tail}]},
              "timestamps": [{"start_ms": 250, "end_ms": 1260}]}
    if timed:
        record["timestamps"][0]["spans"] = [
            {"start_ms": 250, "end_ms": 870, "language": "zh", "speaker": "spk1", "text": head},
            {"start_ms": 920, "end_ms": 1260, "language": "en", "speaker": "spk2", "text": tail}]
    row = merged_row(record, 2000)
    assert [m["content"] for m in row["messages"][2:-2:2]] == [first, second]
