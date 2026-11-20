import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Data-Scripts"))
from merge_timestamped_segments import merge_record, run
from fourth_asr.timestamped_source import parse_record


def record(parts):
    cells = [f"<{lang}><{speaker}>{text}" for _, _, lang, speaker, text in parts]
    return {"audios": ["audio.wav"], "messages": [{"role": "assistant", "content": "".join(cells)}],
            "output": {"segments": [{"source_text": s, "action": "WAIT"} for s in cells], "flush": True},
            "tags": ["<|WAIT|>"] * len(cells),
            "timestamps": [{"index": i, "start_ms": a, "end_ms": b}
                           for i, (a, b, *_) in enumerate(parts, 1)]}


def bodies(parts):
    r, _ = merge_record(record(parts))
    return [u.text for u in parse_record(r, max(p[1] for p in parts))]


@pytest.mark.parametrize("duration", [800, 1120, 1200])
def test_short_prefix_ends_at_complete_protected_unit(duration):
    parts = [(0, 560, "zh", "spk1", "湖人则是"), (560, 560+duration, "zh", "spk1", "用着双星的威吓，"),
             (560+duration, 720+duration, "zh", "spk1", "好。")]
    r = record(parts)
    r["timestamps"][1]["spans"] = [{"start_ms": 560, "end_ms": 560+duration, "text_plain": parts[1][4],
                                    "speaker": "spk1", "language": "zh", "alignment_text": "original"}]
    r["timestamps"].reverse()
    r["output"] = json.dumps(r["output"])
    out, counts = merge_record(r)
    assert out["messages"] == r["messages"] and out["audios"] == r["audios"]
    assert out["timestamps"][0]["text_plain"] == "湖人则是用着双星的威吓，"
    assert out["timestamps"][0]["spans"][1] == r["timestamps"][1]["spans"][0]
    assert counts["short_prefix_into_protected"] == 1
    assert "tags" not in out and "action" not in json.loads(out["output"])["segments"][0]


@pytest.mark.parametrize("parts, expected", [
    ([(0, 800, "en", "spk1", "We meet"), (800, 1800, "en", "spk1", " tomorrow.")],
     ["We meet", " tomorrow."]),
    ([(0, 300, "zh", "spk1", "可以。"), (300, 600, "zh", "spk1", "明天")], ["可以。", "明天"]),
    ([(0, 300, "zh", "spk1", "一"), (295, 620, "zh", "spk1", "起")], ["一", "起"]),
    ([(0, 300, "en", "spk1", "But"), (651, 900, "en", "spk1", " then")], ["But", " then"]),
    ([(0, 300, "zh", "spk1", "好"), (300, 600, "en", "spk1", " OK"), (600, 800, "en", "spk2", " yes")],
     ["好", " OK", " yes"]),
    ([(0, 300, "zh", "spk1", "你，"), (300, 590, "zh", "spk1", "你，"), (590, 920, "zh", "spk1", "你们。")],
     ["你，你，你们。"]),
    ([(0, 420, "de", "spk1", "12"), (420, 470, "de", "spk1", ","), (470, 1070, "de", "spk1", "5 Euro.")],
     ["12,5 Euro."]),
    ([(0, 900, "zh", "spk1", "可以了"), (900, 930, "zh", "spk1", "。"), (930, 1250, "zh", "spk1", "好的。")],
     ["可以了。", "好的。"]),
])
def test_boundaries(parts, expected):
    assert bodies(parts) == expected


def test_mixed_parent_and_long_run():
    mixed = [(0, 300, "en", "spk1", "No."), (300, 500, "fr", "spk2", "Oui.")]
    r = record(mixed)
    r["output"]["segments"] = [{"source_text": r["messages"][0]["content"]}]
    r["timestamps"] = [{"index": 1, "start_ms": 0, "end_ms": 500}]
    out, _ = merge_record(r)
    assert out["timestamps"] == r["timestamps"] and out["output"]["segments"] == r["output"]["segments"]
    parts = [(i*700, (i+1)*700, "en", "spk1", " go") for i in range(8)]
    out, _ = merge_record(record(parts))
    assert all(t["end_ms"] - t["start_ms"] <= 3000 for t in out["timestamps"])
    assert "".join(u.text for u in parse_record(out,5600)) == " go"*8


def test_jsonl_continues_on_error_without_audio(tmp_path):
    r = record([(0, 420, "en", "spk1", "Good"), (420, 720, "en", "spk1", " evening.")])
    source, dest = tmp_path / "source.jsonl", tmp_path / "new" / "merged.jsonl"
    data = b"invalid\n" + json.dumps(r).encode() + b"\n"
    source.write_bytes(data)
    counts = run(source, dest)
    assert counts["kept"] == counts["rejected"] == 1
    assert source.read_bytes() == data
    assert json.loads(dest.read_text(encoding="utf-8"))["audios"] == [str(tmp_path / "audio.wav")]
    assert json.loads(dest.with_suffix(".rejected.jsonl").read_text())["line"] == 1


@pytest.mark.parametrize("parts, expected", [
    ([(0, 370, "zh", "spk1", "我们用"), (370, 650, "en", "spk1", "Zoom"),
      (650, 1250, "zh", "spk1", "开会。")], ["我们用", "Zoom", "开会。"]),
    ([(0, 750, "zh", "spk1", "先说结果，"), (300, 500, "zh", "spk2", "嗯"),
      (770, 1050, "zh", "spk2", "我来。")], ["先说结果，", "嗯", "我来。"]),
    ([(0, 350, "de", "spk1", "Der Wert ist "), (350, 1270, "de", "spk1", "12,"),
      (1270, 1660, "de", "spk1", "5 Prozent.")], ["Der Wert ist 12,5 Prozent."]),
    ([(0, 300, "es", "spk1", "Sr."), (330, 800, "es", "spk1", " García")], ["Sr. García"]),
    ([(0, 350, "de", "spk1", "Art."), (350, 730, "de", "spk1", " 5 gilt.")], ["Art. 5 gilt."]),
    ([(0, 350, "en", "spk1", "Stop."), (350, 400, "en", "spk1", ' "'),
      (400, 800, "en", "spk1", 'Start here."')], ["Stop.", ' "Start here."']),
])
def test_multilingual_and_overlap_cases(parts, expected):
    assert bodies(parts) == expected


@pytest.mark.parametrize("known", [True, False])
def test_synthetic_splice_even_with_matching_tags(known):
    parts = [(0, 220, "zh", "spk1", "明天"), (220, 500, "zh", "spk1", "见"),
             (500, 900, "zh", "spk1", "太贵了。")]
    r = record(parts)
    r["synthesis"] = {"sample_rate": 16000, "clips": [
        {"start_frame": 0, "end_frame": 8000}, {"start_frame": 8000, "end_frame": 14400}
    ]} if known else {"scenario": "pair"}
    out, stats = merge_record(r)
    expected = ["明天见", "太贵了。"] if known else ["明天", "见", "太贵了。"]
    assert [u.text for u in parse_record(out, 900)] == expected
    assert stats["synthesis_boundaries_protected" if known else "synthesis_rows_without_usable_clip_times"] == 1


def test_protected_parent_bounds_survive_tighter_child_bounds():
    r = record([(0, 300, "zh", "spk1", "湖人"), (300, 1420, "zh", "spk1", "得分。")])
    r["timestamps"][1]["spans"] = [
        {"start_ms": 320, "end_ms": 800, "text": "得", "language": "zh", "speaker": "spk1"},
        {"start_ms": 850, "end_ms": 1380, "text": "分。", "language": "zh", "speaker": "spk1"}]
    merged, _ = merge_record(r)
    stamp, = merged["timestamps"]
    assert stamp["merge_source_units"][-1] == {"index": 2, "start_ms": 300, "end_ms": 1420, "span_count": 2}
    assert stamp["spans"][1:] == r["timestamps"][1]["spans"]


@pytest.mark.parametrize("fine", [True, False])
def test_existing_mixed_parent_independent_of_language_change_clock(fine):
    p = [(0, 370, "zh", "spk1", "我们用"), (370, 650, "en", "spk1", "Zoom"),
         (650, 1250, "zh", "spk1", "开会。")]
    r = record(p)
    r["output"]["segments"] = [{"source_text": r["messages"][0]["content"]}]
    stamp = {"index": 1, "start_ms": 0, "end_ms": 1250}
    if fine:
        stamp["spans"] = [dict(start_ms=a, end_ms=b, language=l, speaker=s, text=t) for a,b,l,s,t in p]
    r["timestamps"] = [stamp]
    out, counts = merge_record(r)
    assert out["timestamps"] == [stamp]
    assert counts["language_only_changes"] == 2
