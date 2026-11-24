"""The standalone 1s recast cleans transcript line breaks before strict validation."""
import json
from pathlib import Path
import subprocess
import sys
import wave

import pytest


FOURTH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FOURTH))
from fourth_asr.protocol import DIRECT, EOS, WAIT
sys.path.insert(0, str(FOURTH / "Data-Scripts"))
from build_sft import normalize_source_record
from fourth_asr.timestamped_source import parse_record


def annotated_record(parts, language="zh", separator="\n", states=None):
    segments, timestamps = [], []
    for index, (start, end, body) in enumerate(parts, 1):
        lang, speaker = states[index - 1] if states else (language, "spk1")
        text = f"<{lang}><{speaker}>{body}"
        segments.append({"source_text": text, "action": "DIRECT"})
        stamp = {"start_ms": start, "end_ms": end, "text": text, "text_plain": body,
                 "language": lang, "speaker": speaker}
        timestamps.append({**stamp, "index": index, "source_text": text, "languages": [lang],
                           "speakers": [speaker], "spans": [{**stamp, "alignment_text": body}],
                           "backend": "qwen"})
    return {
        "audios": ["clip.wav"],
        "messages": [
            {"role": "user", "content": "<audio>**Entity Reference:** []"},
            {"role": "assistant", "content": separator.join(s["source_text"] for s in segments)},
        ],
        "output": json.dumps({"segments": segments, "flush": False, "error": None}, ensure_ascii=False),
        "timestamps": timestamps,
        "tags": [DIRECT] * len(parts),
    }


def test_seven_timestamped_lines_recast_and_decode(tmp_path):
    parts = [
        (160, 2720, "我们在同样的样本数条件下"),
        (2960, 4400, "是损失了2dB，"),
        (4640, 7040, "这要注意不是这个数据量，"),
        (7520, 13760, "因为数据量等于采样速率和量化位数的乘积，"),
        (14160, 17920, "也就是说我们的数据量是会降低的啊，"),
        (18720, 21280, "但是呢，我们这个性能呢，"),
        (21360, 23360, "可能在这个条件下会有损失。"),
    ]
    with wave.open(str(tmp_path / "clip.wav"), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", ""))
        wav.writeframes(bytes(23360 * 16 * 2))
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(annotated_record(parts), ensure_ascii=False) + "\n", encoding="utf-8")
    original = source.read_bytes()
    out = tmp_path / "result"
    subprocess.run([sys.executable, str(FOURTH / "Data-Scripts/recast_1s.py"), str(source),
                    "--output-dir", str(out)], check=True)
    assert source.read_bytes() == original
    assert (out / "rejected.jsonl").read_text(encoding="utf-8") == ""
    train = out / "train.1000ms.jsonl"
    row = json.loads(train.read_bytes())
    assert len(row["audios"]) == 24
    assert row["recast"]["merged_units"] == 0
    expected = [WAIT] * 24
    for turn, part in zip((3, 5, 8, 14, 18, 22, 24), parts):
        expected[turn - 1] = DIRECT + part[2]
    assert [msg["content"] for msg in row["messages"] if msg["role"] == "assistant"] == [*expected, WAIT]
    assert row["messages"][-2]["content"] == EOS
    check = subprocess.run([sys.executable, str(FOURTH / "SFT-Scripts/check_sft.py"), str(train),
                            "--decode"], capture_output=True, check=True)
    assert json.loads(check.stdout)["checked_rows"] == 1


@pytest.mark.parametrize("separator", ["\n", "\r\n"])
@pytest.mark.parametrize(("language", "first", "second", "expected"), [
    ("zh", "保持\n正文", "完整。", ["保持正文", "完整。"]),
    ("en", "Hello", "world\nagain.", ["Hello", " world again."]),
    ("en", "Hello ", "world.", ["Hello", " world."]),
    ("en", "Hello", " world.", ["Hello", " world."]),
    ("en", "Hello  ", "  world.", ["Hello", " world."]),
    ("en", "Hello", ", world.", ["Hello", ", world."]),
    ("en", "At the café", "we met.", ["At the café", " we met."]),
    ("en", "Value: 3.", "14.", ["Value: 3.", "14."]),
])
def test_inline_and_header_linebreaks(language, first, second, expected, separator):
    record = annotated_record([(0, 900, first), (1000, 1700, second)], language, separator)
    units = parse_record(normalize_source_record(record), 2000)
    assert [unit.text for unit in units] == expected
    assert [(unit.start_ms, unit.end_ms) for unit in units] == [(0, 900), (1000, 1700)]


def test_mixed_language_and_speaker_attribution():
    record = annotated_record([(0, 900, "我们"), (950, 1600, "keep"), (1600, 2000, "好。")],
                              states=[("zh", "spk1"), ("en", "spk2"), ("zh", "spk2")])
    units = parse_record(normalize_source_record(record), 2000)
    assert [unit.spans for unit in units] == [
        (("zh", "spk1", "我们"),), (("en", "spk2", "keep"),), (("zh", "spk2", "好。"),),
    ]


def test_sorted_indices_and_child_span_boundaries():
    record = annotated_record([(0, 650, "Hello"), (700, 1000, "world."), (1300, 1900, "到了。")],
                              states=[("en", "spk1"), ("en", "spk2"), ("zh", "spk2")])
    first, second, last = record["timestamps"]
    both = first["text"] + "\n" + second["text"]
    record["output"] = {"segments": [{"source_text": both, "action": "DIRECT"},
                                     {"source_text": last["text"], "action": "DIRECT"}]}
    record["tags"] = [DIRECT, DIRECT]
    record["timestamps"] = [
        {**last, "index": 2},
        {"index": 1, "start_ms": 0, "end_ms": 1000,
         "source_text": both, "text": both, "text_plain": "Hello\nworld.",
         "languages": ["en"], "speakers": ["spk1", "spk2"], "spans": first["spans"] + second["spans"]},
    ]
    units = parse_record(normalize_source_record(record), 2000, release_on="timed-span")
    assert [(u.start_ms, u.end_ms, u.spans) for u in units] == [
        (0, 650, (("en", "spk1", "Hello"),)),
        (700, 1000, (("en", "spk2", " world."),)),
        (1300, 1900, (("zh", "spk2", "到了。"),)),
    ]


@pytest.mark.parametrize(("old", "new"), [
    ("the  sun", "the sun"),         # ordinary spaces away from the line break
    ("you.", "you!"),                # terminal punctuation changes
    ("you.", "them."),               # lexical substitution
    ("<en><spk1>See", "<en>See"),    # missing initial speaker
    ("<spk1>  you", "<spk2>  you"),  # reference speaker differs on segment 2
    ("<en><spk1>  you", "<de><spk1>  you"),
])
def test_formatting_never_hides_disagreement(old, new):
    record = annotated_record([(0, 900, "See the  sun"), (1000, 1700, "  you.")], "en")
    answer = record["messages"][-1]
    assert old in answer["content"]
    answer["content"] = answer["content"].replace(old, new)
    with pytest.raises(ValueError):
        parse_record(normalize_source_record(record), 2000)


def test_child_attribution_still_checked():
    record = annotated_record([(0, 900, "See"), (1000, 1700, " you.")], "en")
    record["timestamps"][1]["spans"][0]["speaker"] = "spk2"
    with pytest.raises(ValueError):
        parse_record(normalize_source_record(record), 2000)


def test_word_wrapped_in_only_one_source_copy_and_marked_original():
    record = annotated_record([(0, 900, "The international level is 2dB."),
                               (950, 1600, "Next.")], "en")
    answer = record["messages"][-1]
    answer["content_original"] = answer["content"].replace("\n", "\r\n")
    answer["content"] = answer["content"].replace("international", "inter\nnational").replace(
        "2dB", "2\ndB").replace("\n<en>", "\n[950ms]<en>")
    normalized = normalize_source_record(record)
    assert [u.text for u in parse_record(normalized, 2000)] == ["The international level is 2dB.", " Next."]
    answer["content_original"] = answer["content_original"].replace("2dB", "3dB")
    with pytest.raises(ValueError, match="content_original disagrees"):
        normalize_source_record(record)


def test_english_complete_replies(tmp_path):
    record = annotated_record([(0, 900, "Hello,"), (1000, 1800, "world\nagain.")], "en")
    with wave.open(str(tmp_path / "clip.wav"), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", ""))
        wav.writeframes(bytes(2 * 16000 * 2))
    from build_sft import prepare_record
    from fourth_asr.protocol import get_protocol
    outputs, _ = prepare_record(record, tmp_path, [1000], "normalized", "source-segment",
                                protocol=get_protocol("speaker"), short_ms=0)
    row = json.loads(outputs[1000][0])
    assert [m["content"] for m in row["messages"] if m["role"] == "assistant"] == [
        DIRECT + "Hello,", DIRECT + " world again.", WAIT,
    ]
