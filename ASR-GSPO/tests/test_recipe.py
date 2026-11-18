import argparse
import json
from pathlib import Path
import sys
import wave

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluate
import prepare_data
import train_gspo
from finalize_export import verify_weights
from asr_gspo.protocol import DIRECT, WAIT
from split_data import split


def wav(path):
    with wave.open(str(path), "wb") as out:
        out.setparams((1, 2, 16000, 0, "NONE", ""))
        out.writeframes(bytes(16000 * 2 * 3))


def test_full_recording_to_rollout_scores(tmp_path):
    audio = tmp_path / "meeting.wav"
    wav(audio)
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({
        "id": "meeting", "audios": [audio.name],
        "messages": [{"role": "assistant", "content": "<zh><spk1>开始。<en><spk2>OK."}],
        "alignment": [{"text": "开始", "start_ms": 150, "end_ms": 800},
                      {"text": "OK", "start_ms": 1100, "end_ms": 1600}]
    }), encoding="utf-8")
    dataset = tmp_path / "task.jsonl"
    prepare_data.main([str(source), "--output", str(dataset)])
    row = json.loads(dataset.read_text(encoding="utf-8"))
    task = json.loads(row["asr_task"])
    assert task["reference_text"] == "开始。OK."

    class Backend:
        def __init__(self, replies):
            self.replies = iter(replies)
            self.inputs = []

        def generate(self, messages, audios):
            self.inputs.append((messages[-1]["content"], len(audios)))
            assert all(audio.shape == (16000,) for audio in audios)
            return {"text": next(self.replies)}

    traces = tmp_path / "traces.jsonl"
    with traces.open("w", encoding="utf-8") as trace:
        for variants in ([DIRECT + "开始。", DIRECT + "OK.", WAIT, WAIT],
                         [WAIT, WAIT, DIRECT + "开始。OK.", WAIT],
                         [DIRECT + "开始。", WAIT, WAIT, WAIT]):
            backend = Backend(variants)
            turns = evaluate.model_turns(backend, row, task)
            assert backend.inputs == [("<audio>", 1), ("<audio>", 2), ("<audio>", 3), ("<end_of_audio>", 3)]
            trace.write(json.dumps({"task_id": task["id"], "turns": turns}) + "\n")
    output = tmp_path / "results.jsonl"
    summary = evaluate.evaluate(argparse.Namespace(command="replay", dataset=dataset, traces=traces,
                                 limit=None, output=output, reward_mode="quality_latency"))
    scores = [line["score"] for line in evaluate.read_jsonl(output)]
    assert scores[0]["reward"] > scores[1]["reward"] > scores[2]["reward"]
    assert summary["all"]["switches"] == 3
    assert summary["all"]["short_switch_deletions"] == 1
    assert summary["all"]["p95_extra_wait_seconds"] == 2.
    assert summary["all"]["p95_switch_head_extra_wait_seconds"] == 1.


def test_split_keeps_transitive_donors_and_requested_counts(tmp_path):
    prepared = tmp_path / "input.jsonl"
    def row(i, sources, bucket):
        return json.dumps({"asr_task": json.dumps({"id": i, "source_ids": sources, "bucket": bucket})})
    special = [row("AB", ["a", "b"], "switch"), row("BC", ["b", "c"], "switch"),
               row("C", ["c"], "general")]
    lines = [row(i, [str(i)], ("general", "switch", "stability")[i % 3]) for i in range(120)]
    prepared.write_text("\n".join(special + lines), encoding="utf-8")
    full = tmp_path / "full"
    report = split(prepared, full, .25, 42)
    assert report["largest_group_rows"] == 3
    def members(path):
        return [json.loads(r["asr_task"]) for r in evaluate.read_jsonl(path)]
    train, dev = members(full / "train.jsonl"), members(full / "dev.jsonl")
    assert sum(map(len, (train, dev))) == 123
    assert not {s for r in train for s in r["source_ids"]} & {s for r in dev for s in r["source_ids"]}
    small = tmp_path / "small"
    report = split(prepared, small, .25, 42, train_size=10)
    assert report["written"]["train"] == {"general": 4, "switch": 4, "stability": 2}
    assert members(full / "dev.jsonl") == members(small / "dev.jsonl")
    with pytest.raises(ValueError, match="quotas"):
        split(prepared, tmp_path / "huge", .25, 42, train_size=10000)


def test_launch_modes_and_batch_layout():
    for n in (4, 8):
        a = train_gspo.parser().parse_args(["model", "tasks", "out", "--gpu-ids", ",".join(map(str, range(n))),
                                            "--reward-mode", "quality"])
        cmd = train_gspo.build_command(a)
        for key, value in {"global_batch_size": n * 8, "num_generations": 8, "importance_sampling_level": "sequence",
                           "kl_in_reward": "false", "multi_turn_scheduler": "asr_stream", "reward_funcs": "asr_online",
                           "freeze_vit": "true", "freeze_aligner": "true"}.items():
            assert cmd[cmd.index("--" + key) + 1] == str(value)
        assert train_gspo.build_environment(a)["ASR_GSPO_REWARD_MODE"] == "quality"


def test_export_catches_short_heads_and_missing_audio_encoder(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file
    config = {"model_type": "qwen3_omni_moe"}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    matrices = {key: np.zeros((6, 2), dtype=np.float32)
                for key in ("thinker.model.embed_tokens.weight", "thinker.lm_head.weight", "thinker.audio_tower.conv1.weight")}
    path = str(tmp_path / "model.safetensors")
    save_file(matrices, path)
    verify_weights(tmp_path, [4, 5])
    with pytest.raises(ValueError, match="token IDs"):
        verify_weights(tmp_path, [5, 6])
    del matrices["thinker.audio_tower.conv1.weight"]
    save_file(matrices, path)
    with pytest.raises(ValueError, match="audio encoder"):
        verify_weights(tmp_path, [4, 5])
