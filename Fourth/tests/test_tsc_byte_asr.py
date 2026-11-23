"""Verify the two result fields and audio-to-vid matching used for TSC and Byte."""
import json
from pathlib import Path
import sys
import wave

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import transcribe_tsc_byte as runner


PREFIX = "Fourth_2721"
ASR = PREFIX + "_asr"
TRANS = PREFIX + "_trans"


@pytest.fixture
def folders(tmp_path):
    audio, manifests = tmp_path / "audios", tmp_path / "jsonls"
    audio.mkdir()
    manifests.mkdir()
    return audio, manifests


def make_wav(directory, name, frames=1600):
    path = directory / name
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", ""))
        wav.writeframes(bytes(frames * 2))
    return path


def write_rows(directory, *rows, name="data.jsonl"):
    path = directory / name
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")
    return path


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def never_decode(*args):
    pytest.fail("This row should not call the ASR model")


@pytest.mark.parametrize("check_only", [True, False])
def test_wrong_dataset_pair_reports_no_match(folders, check_only):
    audio, manifests = folders
    make_wav(audio, "tsc.wav")
    manifest = write_rows(manifests, {"vid": "byte", "translation": "reference"})
    before = manifest.read_bytes()
    with pytest.raises(ValueError, match="没有匹配的 vid"):
        runner.run_dataset(audio, manifests, PREFIX, never_decode, check_only=check_only)
    assert manifest.read_bytes() == before


def test_audio_produces_asr_and_original_translation_supplies_trans(folders):
    audio, manifests = folders
    a = make_wav(audio, "中东教授访谈_001.WAV")
    b = make_wav(audio, "Dataflow_001.wav")
    make_wav(audio, "not_in_json.wav")
    original = {"vid": a.stem, "transcript": "参考原文，不能当作模型结果。",
                "translation": "  First line.\n第二行！\t ", "Flex_1022_trans": "已有译文",
                "Flex_1022_asr": "已有兼容字段", "source": {"clip": 42}}
    second = {"vid": b.name, "translation": "", "original_vids": [a.stem]}
    unmatched = {"vid": "absent", "translation": "different"}
    first_file = write_rows(manifests, unmatched, second, name="part-one.jsonl")
    second_file = write_rows(manifests, original, name="arbitrary.jsonl")
    answers = {a: "听到的内容。", b: "The speaker said this."}
    called = []

    def decode(path):
        called.append(path)
        return answers[path]

    assert runner.run_dataset(audio, manifests, PREFIX, decode) == 0
    assert set(called) == {a, b} and len(called) == 2
    assert read_rows(first_file) == [unmatched, {**second, ASR: answers[b], TRANS: ""}]
    assert read_rows(second_file) == [{**original, ASR: answers[a], TRANS: original["translation"]}]


def test_resume_keeps_even_empty_asr_but_refreshes_trans_then_overwrite_redecodes(folders):
    audio, manifests = folders
    a, b = make_wav(audio, "a.wav"), make_wav(audio, "b.wav")
    first = {"vid": "a", "translation": "new reference", ASR: "stored", TRANS: "stale"}
    second = {"vid": "b", "translation": "静音。", ASR: ""}
    manifest = write_rows(manifests, first, second)
    assert runner.run_dataset(audio, manifests, PREFIX, never_decode, overwrite=False) == 0
    saved = read_rows(manifest)
    assert saved == [{**first, TRANS: first["translation"]}, {**second, TRANS: second["translation"]}]
    answers = {a: "", b: "previously empty, now transcribed"}
    called = []

    def decode(path):
        called.append(path)
        return answers[path]

    assert runner.run_dataset(audio, manifests, PREFIX, decode) == 0
    assert len(called) == 2 and set(called) == {a, b}
    assert read_rows(manifest) == [{**saved[0], ASR: answers[a]}, {**saved[1], ASR: answers[b]}]


@pytest.mark.parametrize("bad_row", [
    {"vid": "b"}, {"vid": "b", "translation": None}, {"vid": "a.wav", "translation": "duplicate vid"},
])
def test_invalid_matched_reference_or_ambiguous_vid_aborts_before_any_prediction(folders, bad_row):
    audio, manifests = folders
    make_wav(audio, "a.wav")
    make_wav(audio, "b.wav")
    manifest = write_rows(manifests, {"vid": "a", "translation": "good"}, bad_row)
    before = manifest.read_bytes()
    with pytest.raises(ValueError):
        runner.run_dataset(audio, manifests, PREFIX, never_decode)
    assert manifest.read_bytes() == before


def test_failed_asr_keeps_original_pair_and_continues_with_next_audio(folders):
    audio, manifests = folders
    bad, good = make_wav(audio, "01.wav"), make_wav(audio, "02.wav")
    failed = {"vid": bad.stem, "translation": "new reference", ASR: "old answer", TRANS: "old reference"}
    untouched = {"vid": good.stem, "translation": "成功"}
    manifest = write_rows(manifests, failed, untouched)

    def decode(path):
        if path == bad:
            raise RuntimeError("scripted failure")
        assert path == good
        return "recognized"

    assert runner.run_dataset(audio, manifests, PREFIX, decode) == 1
    assert read_rows(manifest) == [failed, {**untouched, ASR: "recognized", TRANS: "成功"}]


@pytest.mark.parametrize("mode_args", [[], ["--overwrite"]], ids=["default", "explicit-overwrite"])
def test_both_presets_stream_wavs_and_save_plain_asr(folders, tmp_path, monkeypatch, mode_args):
    import transcribe_manifests as engine
    from fourth_asr.protocol import DIRECT, EOS, SPK_CHANGE, WAIT

    audio, manifests = folders
    path = make_wav(audio, "interview.wav", frames=40000)
    row = {"vid": path.stem, "transcript": "SOURCE_REFERENCE", "translation": "TARGET_REFERENCE",
           ASR: "", TRANS: "earlier translation"}
    manifest = write_rows(manifests, row)
    byte_audio, byte_jsonl = tmp_path / "byte-audio", tmp_path / "byte-jsonl"
    byte_audio.mkdir()
    byte_jsonl.mkdir()
    make_wav(byte_audio, path.name, frames=16000)
    byte_row = {**row, "translation": "BYTE_TARGET", ASR: "earlier ASR"}
    byte_manifest = write_rows(byte_jsonl, byte_row)
    monkeypatch.setattr(runner, "DATASETS", {"TSC": (audio, manifests), "Byte": (byte_audio, byte_jsonl)})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    replies = [DIRECT + "你好。", WAIT, DIRECT + SPK_CHANGE + " Hello", DIRECT + "!"]
    all_replies = replies + [DIRECT + "Another.", WAIT]
    turns = []
    model_loads = []

    class Model:
        def generate_batch(self, calls):
            results = []
            for messages, audios, prefix in calls:
                assert prefix == ""
                assert all(reference not in str(messages) for reference in ("SOURCE_REFERENCE", "TARGET_REFERENCE", "BYTE_TARGET"))
                start = 0 if len(turns) < len(replies) else len(replies)
                assert [m["content"] for m in messages if m["role"] == "assistant"] == all_replies[start:len(turns)]
                assert sum(m["content"].count("<audio>") for m in messages) == len(audios)
                turns.append(messages[-1]["content"])
                results.append({"text": all_replies[len(turns) - 1]})
            return results

    def load_model(model_path, **kwargs):
        assert kwargs["chunk_ms"] == 1000
        model_loads.append(model_path)
        return Model()

    monkeypatch.setattr(engine, "load_model", load_model)
    before = (manifest.read_bytes(), byte_manifest.read_bytes())
    assert runner.main(["--prefix", PREFIX, "--check-only", *mode_args]) == 0
    assert not model_loads
    assert (manifest.read_bytes(), byte_manifest.read_bytes()) == before
    assert runner.main(["--prefix", PREFIX, *mode_args]) == 0
    assert len(model_loads) == 1
    assert turns == ["<audio>", "<audio>", "<audio>", EOS, "<audio>", EOS]
    assert read_rows(manifest) == [{**row, ASR: "你好。 Hello!", TRANS: "TARGET_REFERENCE"}]
    assert read_rows(byte_manifest) == [{**byte_row, ASR: "Another.", TRANS: "BYTE_TARGET"}]
