"""Complete source text is authoritative; old segmented actions are not targets."""
import json
import re
import wave

import pytest

from asr_gspo.data import prepare_record
from asr_gspo.protocol import DIRECT, SPK_CHANGE, WAIT
from asr_gspo.reward import score_trajectory
from prepare_data import main as prepare
from filter_tail import main as filter_source


def source(tmp_path):
    path = tmp_path / "source.wav"
    with wave.open(str(path), "wb") as f:
        f.setparams((1, 2, 16000, 0, "NONE", ""))
        f.writeframes(bytes(2 * 16000 * 3))
    bodies = [("ru", "spk1", "Ты готов?"), ("ru", "spk2", " Да, готов. "),
              ("ru", "spk2", "Начнём."), ("ru", "spk1", "Хорошо."),
              ("ru", "spk2", "Идём."), ("zh", "spk3", "我们开始，"),
              ("zh", "spk3", "明天继续。")]
    content = "".join(f"<{lang}><{speaker}>{text}" for lang, speaker, text in bodies)
    segments = [{"source_text": f"<{lang}><{speaker}>{text}", "action": "WAIT" if i in (1, 3, 5) else "DIRECT"}
                for i, (lang, speaker, text) in enumerate(bodies)]
    # This old intermediate annotation is stale. The valid final suffix alone
    # is used by the filter; neither becomes a per-round policy target.
    return dict(audios=[str(path)], messages=[{"role": "assistant", "content": content}],
                output=json.dumps({"segments": segments}), tags=["<|WAIT|>"] * len(segments),
                timestamps=[{"broken": "old unit"}, {"source_text": "<zh><spk3>明天继续。", "end_ms": 1600}])


def trajectory(task, text, commit):
    rounds = [dict(kind="audio", response=DIRECT + text if i == commit else WAIT,
                   audio_end_seconds=c["end_seconds"], finish_reason="stop")
              for i, c in enumerate(task["chunks"], 1)]
    return rounds + [dict(kind="eof", response=WAIT, audio_end_seconds=task["duration_seconds"], finish_reason="stop")]


def test_default_keeps_full_text_and_deduplicates_attribution_not_words(tmp_path):
    row = source(tmp_path)
    built = prepare_record(row, tmp_path)
    task = json.loads(built["asr_task"])
    expected = re.sub(r"<[^>]+>", "", row["messages"][0]["content"])
    assert task["reference_text"] == expected
    assert len(task["attribution_spans"]) == 5 and len(task["speaker_changes"]) == 4
    assert [s["lang"] for s in task["spans"]] == ["ru", "zh"]
    assert " Да, готов. " in expected and "Хорошо." in expected and "我们开始，" in expected
    assert task["timing_mode"] == "tail-only" and task["alignment"] == []
    assert "switches" not in task and "tags" not in task and "output" not in task
    assert all("action" not in s for s in task["attribution_spans"])
    assert task["duration_seconds"] == 3 and task["end_policy"] == "eof"
    assert len(built["audios"]) == 1
    assert not any("Ты готов" in m["content"] or "<ru>" in m["content"] for m in built["messages"])


def test_tail_only_has_no_invented_latency_signal(tmp_path):
    t = json.loads(prepare_record(source(tmp_path), tmp_path)["asr_task"])
    output = SPK_CHANGE.join(s["text"] for s in t["attribution_spans"])
    fast = score_trajectory(t, trajectory(t, output, 1))
    late = score_trajectory(t, trajectory(t, output, 3))
    assert fast["reward"] == late["reward"] == 1
    assert fast["timed_units"] == late["timed_units"] == 0
    assert fast["latency_penalty"] == 0 and fast["speaker_correct_changes"] == 4
    assert fast["switch_segments"] == 1


def test_untimed_source_does_not_depend_on_invalid_middle_segmentation(tmp_path):
    row = source(tmp_path)
    old = json.loads(row["output"])
    old["segments"][0] = {"malformed": "old data, not used"}
    row["output"] = json.dumps(old)
    row["alignment"] = {"status": "failed", "audio": "wrong old auxiliary file"}
    assert json.loads(prepare_record(row, tmp_path)["asr_task"])["alignment"] == []
    with pytest.raises(ValueError):
        prepare_record(row, tmp_path, timing_mode="aligned")


def test_final_unpunctuated_direct_masks_only_uncertain_terminal_stop(tmp_path):
    row = source(tmp_path)
    old = json.loads(row["output"])
    old["segments"][-1]["source_text"] = "<zh><spk3>明天继续"
    row["output"] = old
    t = json.loads(prepare_record(row, tmp_path)["asr_task"])
    assert t["reference_text"].endswith("。")  # never rewrite the full transcript
    assert not t["terminal_punctuation_trusted"] and t["punctuation_trusted"]
    reply = SPK_CHANGE.join(s["text"] for s in t["attribution_spans"])
    assert score_trajectory(t, trajectory(t, reply[:-1], 3))["reward"] == 1
    # Internal sentence punctuation remains supervised.
    damaged = reply.replace("我们开始，", "我们。开始，")
    assert score_trajectory(t, trajectory(t, damaged, 3))["extra_internal_stops"] == 1


def test_filter_then_default_prepare_uses_original_reference(tmp_path):
    row = source(tmp_path)
    src = tmp_path / "source.jsonl"
    src.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    filtered, tasks = tmp_path / "filtered.jsonl", tmp_path / "tasks.jsonl"
    report = filter_source([str(src), "--output", str(filtered)])
    assert report["counts"]["kept"] == 1
    prepare([str(filtered), "--output", str(tasks)])
    task = json.loads(json.loads(tasks.read_text(encoding="utf-8"))["asr_task"])
    assert task["reference_text"] == re.sub(r"<[^>]+>", "", row["messages"][0]["content"])
    assert task["alignment"] == [] and task["timing_mode"] == "tail-only"


def test_tail_only_cannot_silently_project_open_tail(tmp_path):
    with pytest.raises(ValueError, match="tail-only"):
        prepare_record(source(tmp_path), tmp_path, tail_policy="short-open")
