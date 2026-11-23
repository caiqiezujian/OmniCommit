"""Evaluate one JSONL against an independently located WAV tree."""
import json
from pathlib import Path
import sys
import wave

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import transcribe_wav_jsonl as runner


FIELD = "OmniCommit_1123"


def make_wav(path, frames=1600):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setparams((1, 2, 16000, 0, "NONE", ""))
        out.writeframes(bytes(frames * 2))
    return path


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf8")
    return path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()]


def test_main_streams_relocated_wav_and_eof_and_only_writes_named_field(tmp_path, monkeypatch):
    from fourth_asr.protocol import DIRECT, EOS, SPK_CHANGE, WAIT

    audio_root = tmp_path / "wavs"
    local = make_wav(audio_root / "relocated" / "meeting.part_01.wav", frames=40000)
    first = {"audio": "/old/en/" + local.name, "raw_output": "<en><spk1>RAW_REF",
             "text_output": "PLAIN_REF", FIELD: "OLD_PREDICTION", "scores": [8, 9]}
    duplicate = {**first, "audio": r"C:\old\meeting.part_01.wav", FIELD: ""}
    manifest = write_jsonl(tmp_path / "meetings.jsonl", [first, duplicate])
    replies = [DIRECT + "好。", WAIT, DIRECT + SPK_CHANGE + " Good", DIRECT + "!"]
    prompts = []
    loads = []

    class Replay:
        def generate(self, messages, audios):
            history = [m["content"] for m in messages if m["role"] == "assistant"]
            assert history == replies[:len(prompts)]
            assert all(s not in str(messages) for s in ("RAW_REF", "PLAIN_REF", "OLD_PREDICTION"))
            prompts.append(messages[-1]["content"])
            return {"text": replies[len(prompts) - 1]}

    def load_model(path, **options):
        loads.append((path, options))
        return Replay()

    checkpoint = tmp_path / "mock-model"
    for name, value in {"AUDIO_DIR": audio_root, "JSONL_FILE": manifest, "OUTPUT_FIELD": FIELD,
                        "MODEL": checkpoint, "GPU_IDS": "1", "BATCH_SIZE": 2, "CHUNK_MS": 1000,
                        "SAVE_EVERY": 1}.items():
        monkeypatch.setattr(runner, name, value)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(runner, "load_model", load_model)
    assert runner.main() == 0
    assert loads == [(checkpoint, {"gpu_ids": "1", "batch_size": 2, "chunk_ms": 1000,
                                   "window_seconds": runner.WINDOW_SECONDS})]
    assert prompts == ["<audio>", "<audio>", "<audio>", EOS]
    assert read_jsonl(manifest) == [{**row, FIELD: "好。 Good!"} for row in (first, duplicate)]


def test_missing_ambiguous_and_failed_audio_preserve_values_while_success_overwrites(tmp_path):
    audio_root = tmp_path / "wavs"
    paths = [make_wav(audio_root / name) for name in ("a/dup.wav", "b/dup.wav", "bad.wav", "ok.WAV")]
    foreign = make_wav(tmp_path / "outside" / "only-outside.wav")
    locators = ["dup.wav", "bad.wav", str(foreign), "missing.wav", "ok.wav", "ok.WAV"]
    rows = [{"audio": locator, FIELD: "previous", "text_output": "retain"} for locator in locators]
    manifest = write_jsonl(tmp_path / "test.jsonl", rows)
    visited = []

    def decode(path):
        visited.append(path)
        if path == paths[2]:
            raise RuntimeError("simulated decoder failure")
        assert path == paths[3]
        return ""

    assert runner.run(audio_root, manifest, FIELD, decode) == 5
    assert set(visited) == set(paths[2:]) and len(visited) == 2
    assert read_jsonl(manifest) == [*rows[:-1], {**rows[-1], FIELD: ""}]


def test_blank_and_malformed_lines_survive_and_next_successful_row_is_written(tmp_path):
    audio_root = tmp_path / "wavs"
    make_wav(audio_root / "ok.wav")
    manifest = write_jsonl(tmp_path / "test.jsonl", [{"audio": "ok.wav"}])
    prefix = b"\n{ broken }\n"
    manifest.write_bytes(prefix + manifest.read_bytes())
    assert runner.run(audio_root, manifest, FIELD, lambda _: "result") == 1
    lines = manifest.read_bytes().splitlines()
    assert lines[:2] == [b"", b"{ broken }"]
    assert json.loads(lines[2]) == {"audio": "ok.wav", FIELD: "result"}


def test_save_every_one_persists_first_wav_before_next_runs(tmp_path):
    audio_root = tmp_path / "wavs"
    first, second = [make_wav(audio_root / name) for name in ("first.wav", "second.wav")]
    rows = [{"audio": first.name}, {"audio": second.name}]
    manifest = write_jsonl(tmp_path / "test.jsonl", rows)

    def decode(path):
        if path == first:
            return "first result"
        assert path == second
        assert read_jsonl(manifest) == [{**rows[0], FIELD: "first result"}, rows[1]]
        return "second result"

    assert runner.run(audio_root, manifest, FIELD, decode, workers=1, save_every=1) == 0
    assert read_jsonl(manifest)[1] == {**rows[1], FIELD: "second result"}


def test_refuses_to_write_into_the_lookup_field(tmp_path):
    audio_root = tmp_path / "wavs"
    make_wav(audio_root / "ok.wav")
    manifest = write_jsonl(tmp_path / "test.jsonl", [{"audio": "ok.wav"}])
    before = manifest.read_bytes()
    with pytest.raises(ValueError):
        runner.run(audio_root, manifest, "audio", lambda _: pytest.fail("unexpected inference"))
    assert manifest.read_bytes() == before
