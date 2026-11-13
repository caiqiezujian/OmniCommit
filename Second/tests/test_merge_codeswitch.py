import json
from pathlib import Path
import runpy
import wave

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "Data-Scripts/merge_codeswitch.py"
merge = runpy.run_path(str(SCRIPT))["merge"]


@pytest.fixture
def source_audio(tmp_path):
    source = tmp_path / "source.jsonl"
    offsets = {}
    with source.open("wb") as f:
        for name in "abcd":
            offsets[name] = f.tell()
            with wave.open(str(tmp_path / f"{name}.wav"), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", ""))
                wav.writeframes(bytes(200))
            f.write(json.dumps({"audios": [f"{name}.wav"]}).encode() + b"\r\n")
    return source, offsets


def clip(source, offsets, name, lo, hi, usage=None):
    return {"audio": str(source.parent / f"{name}.wav"), "source_offset": offsets[name],
            "source_start_frame": lo, "source_end_frame": hi,
            **({"source_frames": 100, "source_usage": usage} if usage else {})}


def write_supplement(path, clips_per_output):
    rows = [{"audios": [f"synth{i}.wav"], "synthesis": {"clips": clips}}
            for i, clips in enumerate(clips_per_output)]
    path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")


def test_full_inputs_replaced_short_middle_and_unused_original_retained(source_audio, tmp_path):
    source, offsets = source_audio
    before = source.read_bytes()
    full_a = clip(source, offsets, "a", 0, 100, "full")
    full_b = clip(source, offsets, "b", 0, 100, "full")
    short_c = clip(source, offsets, "c", 11, 37, "partial")
    supplement = tmp_path / "supplement.jsonl"
    write_supplement(supplement, [[full_a, short_c, full_b]])
    output = tmp_path / "new" / "combined.jsonl"
    counts = merge(source, supplement, output)
    assert (counts["removed_source_rows"], counts["partial_source_rows_retained"]) == (2, 1)
    assert [Path(json.loads(line)["audios"][0]).name for line in output.read_text().splitlines()] == [
        "c.wav", "d.wav", "synth0.wav"]
    assert source.read_bytes() == before and len(list(tmp_path.glob("*.wav"))) == 4
    with pytest.raises(FileExistsError):
        merge(source, supplement, output)


def test_legacy_only_count_whole_coverage_inside_one_synthesis(source_audio, tmp_path):
    source, offsets = source_audio
    supplement = tmp_path / "supplement.jsonl"
    # A's two separate crops wind up in different outputs, while B comes
    # back within the same new waveform and the two pieces span its whole WAV.
    write_supplement(supplement, [
        [clip(source, offsets, "a", 0, 45), clip(source, offsets, "b", 0, 45),
         clip(source, offsets, "c", 20, 46), clip(source, offsets, "b", 45, 100)],
        [clip(source, offsets, "a", 45, 100), clip(source, offsets, "d", 0, 100)]])
    output = tmp_path / "new" / "combined.jsonl"
    counts = merge(source, supplement, output)
    assert counts["removed_source_rows"] == 2
    assert [Path(json.loads(line)["audios"][0]).name for line in output.read_text().splitlines()] == [
        "a.wav", "c.wav", "synth0.wav", "synth1.wav"]


@pytest.mark.parametrize("error", ["incomplete_full", "mismatched_path", "wrong_offset"])
def test_invalid_provenance_does_not_leave_partial_combined_manifest(source_audio, tmp_path, error):
    source, offsets = source_audio
    correct = clip(source, offsets, "c", 0, 100, "full")
    bad = clip(source, offsets, "a", 0, 100, "full")
    if error == "incomplete_full":
        bad["source_end_frame"] = 99
    elif error == "mismatched_path":
        bad["audio"] = str(tmp_path / "d.wav")
    else:
        bad["source_offset"] += 1
    supplement = tmp_path / "supplement.jsonl"
    output = tmp_path / "new" / "combined.jsonl"
    write_supplement(supplement, [[correct, bad]])
    with pytest.raises(ValueError):
        merge(source, supplement, output)
    assert not output.exists()
