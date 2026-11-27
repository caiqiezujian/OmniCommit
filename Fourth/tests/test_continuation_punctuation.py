"""Conservative punctuation edits operate only on the Fourth training view."""
import json
import sys
import wave
from collections import Counter
from pathlib import Path

import pytest

FOURTH = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(FOURTH), str(FOURTH / "Data-Scripts")]
from fourth_asr.continuation_punctuation import revise_continuation_periods
from fourth_asr.protocol import DIRECT, SPK_CHANGE, WAIT
from fourth_asr.sft_checks import validate_row
from fourth_asr.timestamped_source import Unit, synthesis_boundaries
from recast_1s import main


def segment(text, start, end, lang="zh", speaker="spk1"):
    return Unit(start, end, ((lang, speaker, text),))


@pytest.mark.parametrize("lang,left,right,expected", [
    ("zh", "从目录上。", "也能反映一些问题。", "从目录上"),
    ("zh", "我们把。", "会议往后挪。", "我们把"),
    ("zh", "如果明天下雨。", "我们就改线上。", "如果明天下雨，"),
    ("zh", "因为路远。", "所以今天出发。", "因为路远，"),
    ("zh", "我知道。", "这件事我来处理。", "我知道。"),
    ("zh", "如果下雨我们就取消。", "那就明天见。", "如果下雨我们就取消。"),
    ("zh", "对。", "我们接着说。", "对。"),
    ("zh", "我们把……", "门推开。", "我们把……"),
    ("zh", "“我们把。”", "门推开。", "“我们把。”"),
    ("en", "We need to.", "check the figures.", "We need to "),
    ("en", "We need to.", " check the figures.", "We need to"),
    ("en", "We need the. ", "figures.", "We need the "),
    ("en", "We need to.", "Tomorrow is fine.", "We need to."),
    ("en", "OK.", " let's go.", "OK."),
    ("en", "It is 3.", "14 exactly.", "It is 3."),
    ("en", "In the U.S.", " sales rose.", "In the U.S."),
    ("en", "Ask Dr.", " smith.", "Ask Dr."),
    ("en", "We need to...", " check the figures.", "We need to..."),
    ("ja", "如果明天下雨。", "我们就改线上。", "如果明天下雨。"),
])
def test_boundary_examples(lang, left, right, expected):
    source = [segment(left, 100, 800, lang), segment(right, 1100, 1400, lang)]
    result, stats = revise_continuation_periods(source, 300)
    assert [u.text for u in result] == [expected, right]
    assert source[0].text == left
    assert stats["internal_periods_removed"] + stats["internal_periods_to_comma"] == int(left != expected)


def test_no_edit_across_switch_overlap_long_gap_or_synthetic_record():
    left = segment("从目录上。", 200, 800)
    right = segment("也能反映一些问题。", 800, 1200)
    for source in (
        [left, segment(right.text, 800, 1200, speaker="spk2")],
        [left, segment(right.text, 800, 1200, lang="en")],
        [left, Unit(800, 1200, (("zh", "spk1", right.text), ("en", "spk2", " OK.")))],
        [left, segment(right.text, 799, 1200)],
        [left, segment(right.text, 1101, 1400)],
        [segment("插话", 0, 900, speaker="spk2"), left, right],
        [left, right, segment("插话", 1100, 1300, speaker="spk2")],
    ):
        assert revise_continuation_periods(source, 300)[0] is source
    assert revise_continuation_periods([left, right], 0)[0][0].text == "从目录上"
    assert revise_continuation_periods([left, right], 300, protected={1})[0][0] == left


def test_only_edit_interior_of_known_source_clip():
    units = [segment("我们把。", 0, 600), segment("时间改了，我们把。", 700, 1000), segment("地点也改了。", 1100, 1600)]
    meta = {"sample_rate": 16000, "clips": [
        {"start_frame": 0, "end_frame": 16800}, {"start_frame": 16800, "end_frame": 25600}
    ]}
    barriers = synthesis_boundaries({"synthesis": meta}, units, Counter())
    revised, _ = revise_continuation_periods(units, 300, protected=barriers)
    assert [u.text for u in revised] == ["我们把", "时间改了，我们把。", "地点也改了。"]
    unknown = synthesis_boundaries({"synthesis": {}}, units, Counter())
    assert revise_continuation_periods(units, 300, protected=unknown)[0] is units


def test_opt_in_recasts_once_preserving_audio_actions_and_validating_source(tmp_path):
    audio = tmp_path / "meeting.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", ""))
        stream.writeframes(bytes(7 * 16000 * 2))
    annotations = [
        ("zh", "spk1", "从目录上。", 100, 830),
        ("zh", "spk1", "也能看到。", 970, 1580),
        ("zh", "spk1", "如果明天下雨。", 1890, 2700),
        ("zh", "spk1", "我们就改线上。", 2990, 3400),
        ("en", "spk2", "We need to.", 4740, 5600),
        ("en", "spk2", "check the schedule.", 5700, 6600),
    ]
    texts, times = [], []
    for lang, speaker, text, start, end in annotations:
        texts.append(f"<{lang}><{speaker}>{text}")
        span = dict(language=lang, speaker=speaker, text_plain=text, start_ms=start, end_ms=end)
        times.append(dict(**span, spans=[span]))
    data = {"audios": [audio.name], "messages": [{"role": "assistant", "content": "".join(texts)}],
            "output": json.dumps({"segments": [{"source_text": t, "action": "DIRECT"} for t in texts]}),
            "timestamps": times}
    good = json.dumps(data, ensure_ascii=False)
    times[0]["text_plain"] = "本来就有错。"
    src = tmp_path / "input.jsonl"
    src.write_text(good + "\n" + json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
    original = src.read_bytes()

    def build(name, *extra):
        out = tmp_path / name
        assert main([str(src), "--output-dir", str(out), "--tail-policy", "short-open", *extra]) == 0
        summary = json.loads((out / "summary.json").read_bytes())
        assert (summary["kept"], summary["rejected"]) == (1, 1)
        return json.loads((out / "train.1000ms.jsonl").read_bytes()), summary

    old, _ = build("baseline")
    row, summary = build("edited", "--continuation-gap-ms", "300")
    assert src.read_bytes() == original
    assert old["audios"] == row["audios"]
    assert old["recast"]["events"] == row["recast"]["events"]
    assert old["messages"][2]["content"] == DIRECT + "从目录上。"
    assert [m["content"] for m in row["messages"] if m["role"] == "assistant"] == [
        DIRECT + "从目录上", DIRECT + "也能看到。", DIRECT + "如果明天下雨，", DIRECT + "我们就改线上。",
        WAIT, DIRECT + SPK_CHANGE + "We need to ", DIRECT + "check the schedule",
    ]
    stats = summary["by_chunk_ms"]["1000"]
    assert stats["internal_periods_removed"] == 2
    assert stats["internal_periods_to_comma"] == stats["terminal_periods_removed"] == 1
    validate_row(row, decode=True)
