import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest

from asr_gspo.character_quality import cer_metrics
from asr_gspo.data import prepare_record
from asr_gspo.online import ASRStreamHooks, score_batch
from asr_gspo.protocol import DIRECT, SPK_CHANGE, WAIT, initial_messages, total_turns
from asr_gspo.reward import score_trajectory
from .test_asr_rewards import task, turns
import train_gspo


def source(tmp_path, seconds=3):
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as f:
        f.setparams((1, 2, 16000, 0, "NONE", ""))
        f.writeframes(bytes(2 * 16000 * seconds))
    return dict(audios=[str(path)], messages=[dict(role="assistant", content="<zh><spk1>开始。<en><spk2>OK.")],
                output={"segments": [{"source_text": "<zh><spk1>开始。"}, {"source_text": "<en><spk2>OK."}]},
                timestamps=[dict(start_ms=100, end_ms=1200), dict(start_ms=1300, end_ms=2300)])


def test_open_tail_matches_fourth_sft(tmp_path):
    import importlib.util
    location = Path(__file__).resolve().parents[2] / "Fourth" / "Data-Scripts" / "build_sft.py"
    spec = importlib.util.spec_from_file_location("fourth_builder", location)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    from fourth_asr.protocol import get_protocol
    from fourth_asr.chunk_refs import decode_chunk_ref
    row = source(tmp_path)
    prepared = prepare_record(row, tmp_path, tail_policy="short-open", timing_mode="aligned")
    t = json.loads(prepared["asr_task"])
    assert t["protocol"] == "fourth_speaker_v1"
    assert t["reference_text"] == "开始。OK" and t["terminal_period_removed"]
    assert t["end_policy"] == "annotation_end" and t["duration_seconds"] == 2.3
    assert t["speaker_changes"] == [3] and t["speaker_scorable"]
    built, _ = builder.prepare_record(row, tmp_path, [1000], "normalized", "merged-unit",
                                     protocol=get_protocol("speaker"), short_ms=0, tail_policy="short-open")
    sft = json.loads(built[1000][0])
    assert [decode_chunk_ref(a)["end"] for a in sft["audios"]] == [c["end_seconds"] for c in t["chunks"]]
    targets = [m["content"] for m in sft["messages"] if m["role"] == "assistant"]
    tr = [dict(kind="audio", response=r, audio_end_seconds=c["end_seconds"]) for r, c in zip(targets, t["chunks"])]
    result = score_trajectory(t, tr)
    assert result["reward"] == 1 and result["speaker_correct_changes"] == 1
    assert len(targets) == total_turns(t) == 3


def test_open_tail_requires_real_end_times(tmp_path):
    row = source(tmp_path)
    row["timestamps"] = [{"time_ms": 1300, "language": "en", "speaker": "spk2"}]
    with pytest.raises(ValueError, match="per-unit"):
        prepare_record(row, tmp_path, tail_policy="short-open", timing_mode="aligned")


def test_labeled_silence_keeps_full_audio_without_fake_tail(tmp_path):
    row = source(tmp_path)
    row["messages"][-1]["content"] = ""
    row.pop("timestamps")
    row.pop("output")
    result = json.loads(prepare_record(row, tmp_path, tail_policy="short-open", timing_mode="aligned")["asr_task"])
    assert result["duration_seconds"] == 3 and result["end_policy"] == "eof"
    assert result["bucket"] == "stability"


def test_conflicting_attribution_is_rejected_even_without_alignment(tmp_path):
    row = source(tmp_path)
    row["messages"][-1]["content"] = "<zh><spk1><spk2>你好"
    row.pop("output")
    row.pop("timestamps")
    with pytest.raises(ValueError, match="conflicting"):
        prepare_record(row, tmp_path)


def test_true_cer_is_not_latin_word_error():
    score = cer_metrics("Bonjour", "Bonjoir")
    assert score["character_errors"] == 1 and score["cer"] == pytest.approx(1/7)
    assert cer_metrics("３００元, Café", "300元 café")["cer"] == 0
    assert cer_metrics("Café", "cafe")["cer"] > 0
    assert cer_metrics("Bonjour", "你好")["cer"] == 1
    assert cer_metrics("3.50", "350")["cer"] > 0
    assert cer_metrics("50%", "50")["cer"] > 0
    t = task([("fr", "Bonjour")])
    assert score_trajectory(t, turns(t, {2: "Bonjoir"}))["quality"] == pytest.approx(6/7)
    assert score_trajectory(t, turns(t, {2: "Bonjoir"}))["reward"] > .8


def test_english_french_translation_and_missing_short_switch():
    t = task([("en", "The budget is ready. "), ("fr", "Merci beaucoup."), ("en", " Let's go.")],
             anchors=[("Merci", 1200, 1800), ("beaucoup", 1900, 2600)])
    good = score_trajectory(t, turns(t, {5: t["reference_text"]}))
    translated = score_trajectory(t, turns(t, {1: "The budget is ready. 谢谢。 Let's go."}))
    missing = score_trajectory(t, turns(t, {1: "The budget is ready. Let's go."}))
    assert good["reward"] > translated["reward"] and good["reward"] > missing["reward"]
    assert translated["switch_cer_worst"] >= 1 and missing["switch_cer_worst"] >= 1


def test_speaker_changes_correct_missing_extra_and_wrong_position():
    t = task([("zh", "开始。"), ("en", "OK. Next.")])
    t.update(speaker_scorable=True, speaker_changes=[3])
    correct = score_trajectory(t, turns(t, {2: "开始。" + SPK_CHANGE + "OK. Next."}))
    missing = score_trajectory(t, turns(t, {2: t["reference_text"]}))
    wrong = score_trajectory(t, turns(t, {2: "开始。OK. " + SPK_CHANGE + "Next."}))
    extra = score_trajectory(t, turns(t, {2: SPK_CHANGE + "开始。" + SPK_CHANGE + "OK. Next."}))
    assert correct["reward"] == 1 and correct["cer"] == 0
    assert missing["speaker_missed_changes"] == 1
    assert wrong["speaker_missed_changes"] == wrong["speaker_extra_changes"] == 1
    assert extra["speaker_extra_changes"] == 1
    assert all(s["reward"] < correct["reward"] for s in (missing, wrong, extra))
    t["speaker_changes"] = []  # same speaker, different language
    assert score_trajectory(t, turns(t, {2: t["reference_text"]}))["reward"] == 1


def test_punctuation_is_independent_of_turn_length_and_end_mask():
    t = task([("zh", "我知道你有问题")])
    good = score_trajectory(t, turns(t, {1: "我知道", 2: "你有问题"}))
    fragmented = score_trajectory(t, turns(t, {1: "我知道。", 2: "你有问题。"}))
    assert good["reward"] == 1 and fragmented["extra_internal_periods"] == 1
    assert fragmented["unsupported_short_breaks"] == 1 and fragmented["reward"] < 1
    t["punctuation_trusted"] = False
    assert score_trajectory(t, turns(t, {1: "我知道。", 2: "你有问题。"}))["punctuation_penalty"] == 0
    t = task([("zh", "好。可以。你说。")])
    assert score_trajectory(t, turns(t, {1: t["reference_text"]}))["reward"] == 1


def test_correct_late_beats_one_fast_character_error():
    t = task([("en", "a " * 100 + "budget")], seconds=40, anchors=[("budget", 100, 500)])
    correct = score_trajectory(t, turns(t, {40: t["reference_text"]}))
    wrong = score_trajectory(t, turns(t, {1: t["reference_text"][:-1] + "s"}))
    assert correct["reward"] > wrong["reward"]
    assert correct["latency_penalty"] < 1/correct["reference_characters"]


def test_online_annotation_end_never_sends_eof_or_reference(tmp_path):
    row = prepare_record(source(tmp_path), tmp_path, tail_policy="short-open", timing_mode="aligned")
    t = json.loads(row["asr_task"])
    scheduler = ASRStreamHooks()
    scheduler.max_turns = 65
    request = SimpleNamespace(messages=row["messages"], audios=row["audios"], images=[], videos=[], data_dict={"asr_task": row["asr_task"]})
    asyncio.run(scheduler.on_trajectory_start([request]))
    assert "reference_text" not in json.dumps(request.data_dict)
    replies = [WAIT, DIRECT + "开始。", DIRECT + SPK_CHANGE + "OK"]
    for i, reply in enumerate(replies, 1):
        choice = SimpleNamespace(token_ids=[7], logprobs={"content": [{"logprob": -.1}]},
                                 message=SimpleNamespace(content=reply), finish_reason="stop")
        request.messages.append(dict(role="assistant", content=reply))
        result = asyncio.run(scheduler.on_turn_end(request, choice, i))
        if i < 3:
            assert not result["done"]
            scheduler.step(request, choice, i)
    assert result["done"] and all(m["content"] != "<end_of_audio>" for m in request.messages)
    assert score_batch([replies[-1]], [row["asr_task"]], [result["rollout_infos"]], mode="quality_latency") == [1.]


def test_launch_eight_trajectories_and_checkpoint_defaults():
    a = train_gspo.parser().parse_args(["model", "tasks", "output", "--gpu-ids", "0,1,2,3"])
    cmd = train_gspo.build_command(a)
    for key, expected in {"num_generations": "8", "global_batch_size": "32", "train_iters": "2000", "save_steps": "500",
                          "save_total_limit": "6", "callbacks": "omnicommit_export", "async_save": "false", "report_to": "tensorboard"}.items():
        assert cmd[cmd.index("--" + key) + 1] == expected
    env = train_gspo.build_environment(a)
    assert env["ASR_GSPO_REWARD_MODE"] == "quality_latency"
    assert "Fourth" in env["PYTHONPATH"]


def test_component_metrics_use_deltas_not_lifetime_averages():
    from collections import Counter
    from asr_gspo.metrics import update_logs
    previous, logs = {}, {}
    update_logs(previous, {"rank0": Counter(trajectories=2, valid=2, reference_characters=100,
                                            character_errors=10, reward=-1)}, logs)
    assert logs["asr_cer"] == .1
    update_logs(previous, {"rank0": Counter(trajectories=4, valid=4, reference_characters=200,
                                            character_errors=12, reward=-3)}, logs)
    assert logs["asr_cer"] == .02
