import json
import io
import subprocess
import sys
import wave
from pathlib import Path

import pytest

SECOND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SECOND))
from second_asr.protocol import EOS, WAIT, parse_reply
from second_asr.recast import build_sft_row
from second_asr.source import Unit
from omnicommit_audio import decode_chunk_ref, chunk_wav_bytes


def write_audio(path, frames=64000, rate=16000):
    with wave.open(str(path), "wb") as handle:
        handle.setparams((1, 2, rate, frames, "NONE", ""))
        handle.writeframes(bytes(2 * frames))


def example(path):
    units = [
        (50, 420, "<zh><spk1>请联系"),
        (430, 1080, "张伟，"),
        (1100, 1390, "<zh><spk1>预算是"),
        (1430, 2170, "300元。"),
        (2210, 2470, "<en><spk1>OK?"),
        (3060, 3410, "<en><spk2>Sure,"),
        (3540, 3910, " I will."),
    ]
    segments = [{"source_text": src, "action": "WAIT" if i in (1, 4) else "DIRECT"}
                for i, (_, _, src) in enumerate(units)]
    return {"audios": [path.name],
            "messages": [{"role": "assistant", "content": "".join(src for _, _, src in units)}],
            "output": json.dumps({"segments": segments, "flush": False, "error": None}, ensure_ascii=False),
            "timestamps": [{"index": i, "start_ms": start, "end_ms": end}
                           for i, (start, end, _) in enumerate(units, 1)]}


def run_cli(tmp_path, script, *args):
    return subprocess.run([sys.executable, str(SECOND / script), *map(str, args)], cwd=tmp_path,
                          capture_output=True, encoding="utf-8", errors="replace")


def replies(row):
    return [m["content"] for m in row["messages"] if m["role"] == "assistant"]


def test_build_and_audit_multiple_grids(tmp_path):
    audio = tmp_path / "conversation.wav"
    write_audio(audio)
    record = example(audio)
    bad = dict(record, timestamps=[{"time_ms": 2210, "changed": ["language"]}])
    rows = tmp_path / "input.jsonl"
    rows.write_text("\n".join([json.dumps(bad), "{", json.dumps(record)]), encoding="utf-8")
    out = tmp_path / "results"
    result = run_cli(tmp_path, "Data-Scripts/build_sft.py", rows, "--output-dir", out,
                     "--chunk-ms", "500", "700", "800", "1000")
    assert result.returncode == 0, result.stderr
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert (summary["kept"], summary["rejected"]) == (1, 2)
    assert summary["by_chunk_ms"]["500"] == {"audio_turns": 8, "wait_audio_turns": 3}
    assert summary["longest_unannotated_gap"] == {"line": 3, "start_ms": 2470, "end_ms": 3060, "duration_ms": 590}
    assert [json.loads(line)["line"] for line in (out / "rejected.jsonl").read_text().splitlines()] == [1, 2]
    expected = {
        500: ["<zh><spk1>请联系", WAIT, "张伟，预算是", WAIT, "300元。<en><spk1>OK?",
              WAIT, "<en><spk2>Sure,", " I will.", WAIT],
        700: ["<zh><spk1>请联系", "张伟，预算是", WAIT, "300元。<en><spk1>OK?",
              "<en><spk2>Sure,", " I will.", WAIT],
        800: ["<zh><spk1>请联系", "张伟，预算是", "300元。", "<en><spk1>OK?",
              "<en><spk2>Sure, I will.", WAIT],
        1000: ["<zh><spk1>请联系", "张伟，预算是", "300元。<en><spk1>OK?",
               "<en><spk2>Sure, I will.", WAIT],
    }
    for step, expected_replies in expected.items():
        dataset = out / f"train.{step}ms.jsonl"
        row = json.loads(dataset.read_text(encoding="utf-8"))
        assert set(row) == {"audios", "messages"}
        assert replies(row) == expected_replies
        assert row["messages"][-2] == {"role": "user", "content": EOS}
        spans, state = [], None
        for reply in replies(row):
            new, state = parse_reply(reply, state)
            spans.extend(new)
        assert "".join(s[2] for s in spans) == "请联系张伟，预算是300元。OK?Sure, I will."
        refs = [decode_chunk_ref(ref) for ref in row["audios"]]
        assert all(Path(ref["path"]) == audio for ref in refs)
        assert refs[0]["start"] == 0 and refs[-1]["end"] == 4
        assert all(a["end"] == b["start"] for a, b in zip(refs, refs[1:]))
        tokens = str(dataset) + ".tokens.txt"
        assert set(Path(tokens).read_text().splitlines()) == {WAIT, "<en>", "<zh>", "<spk1>", "<spk2>"}
        check = run_cli(tmp_path, "SFT-Scripts/check_sft.py", dataset, "--decode", "--tokens-file", tokens)
        assert check.returncode == 0, check.stderr
    rerun = run_cli(tmp_path, "Data-Scripts/build_sft.py", rows, "--output-dir", out)
    assert rerun.returncode != 0 and "already exists" in rerun.stderr


def test_whole_samples_and_real_tail(tmp_path):
    rate, frames = 22050, 22051
    audio = tmp_path / "one-extra-sample.wav"
    write_audio(audio, frames, rate)
    units = [Unit(100, 500, (("en", "spk7", "yes"),)),
             Unit(1000, frames * 1000 / rate, (("de", "spk7", "ja"),))]
    row = build_sft_row(units, audio, frames, rate, 500)
    assert replies(row) == ["<en><spk7>yes", WAIT, "<de><spk7>ja", WAIT]
    refs = [decode_chunk_ref(ref) for ref in row["audios"]]
    assert refs[-1]["start"] == 1 and refs[-1]["end"] == frames / rate
    assert len(chunk_wav_bytes(row["audios"][-1])) > 44


def test_crop_before_resampling_and_padding(tmp_path):
    import numpy as np
    import soundfile as sf

    # A loud impulse immediately after the grid boundary must be inaudible to
    # the first turn, even when resampling from 44.1 kHz.
    rate = 44100
    samples = np.zeros(rate, dtype=np.float32)
    samples[rate // 2] = 1
    path = tmp_path / "impulse.wav"
    sf.write(str(path), samples, rate, subtype="FLOAT")
    row = build_sft_row([Unit(500, 750, (("en", "spk1", "Jim"),))], path, rate, rate, 500)
    waves = [sf.read(io.BytesIO(chunk_wav_bytes(ref)))[0] for ref in row["audios"]]
    assert not waves[0].any() and waves[1].any()
    assert replies(row) == [WAIT, "<en><spk1>Jim", WAIT]


def test_short_switch_entirely_within_turn():
    units = [Unit(start, end, ((lang, spk, text),))
             for start, end, lang, spk, text in [
                 (730, 1200, "ar", "spk4", "نعم"),
                 (1200, 1400, "en", "spk4", "OK"),
                 (1400, 1510, "en", "spk2", "Sure"),
                 (1510, 1980, "ar", "spk4", "تمام"),
             ]]
    row = build_sft_row(units, Path.cwd() / "dummy.wav", 32000, 16000, 500)
    assert replies(row) == [WAIT, WAIT, "<ar><spk4>نعم<en><spk4>OK",
                            "<en><spk2>Sure<ar><spk4>تمام", WAIT]


@pytest.mark.parametrize("step", [0, -1, 8001])
def test_invalid_grid(step):
    with pytest.raises(ValueError):
        build_sft_row([], Path("ignored.wav"), 0, 0, step)
