import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from omnicommit.streaming import StreamOptions
from tools import diagnose_switch_segment as switch


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires FFmpeg")
@pytest.mark.parametrize("protocol", ["first", "second"])
def test_selected_clip_delivers_the_original_packets(tmp_path, protocol):
    source = tmp_path / "source.wav"
    output = tmp_path / "clip.wav"
    rate = 44100
    noise = np.random.default_rng(10).integers(-30000, 30000, (rate * 2, 2), dtype=np.int16)
    sf.write(source, noise, rate)
    switch.extract_packets(source, output, Decimal(".5"), Decimal("1.5"), 500, protocol)
    expected = list(switch.iter_media_chunks(source, .5, protocol))[1:3]
    actual = list(switch.iter_media_chunks(output, .5, protocol))
    assert len(actual) == 2 and actual[-1][1]
    for (before, _), (after, _) in zip(expected, actual):
        np.testing.assert_array_equal(before, after)


def test_off_grid_refused(tmp_path):
    with pytest.raises(ValueError, match="original 500 ms grid"):
        switch.extract_packets(None, tmp_path / "clip.wav", Decimal("58.1"), Decimal(96), 500, "first")


def test_58_to_96_uses_packets_117_to_192(tmp_path, monkeypatch):
    def numbered_packets(source, step, protocol):
        for i in range(1, 193):
            yield np.full(8000, i / 32768, dtype=np.float32), False
        pytest.fail("Should stop reading at original time 96.0")

    monkeypatch.setattr(switch, "iter_media_chunks", numbered_packets)
    output = tmp_path / "clip.wav"
    switch.extract_packets(None, output, Decimal(58), Decimal(96), 500, "first")
    data, rate = sf.read(output, dtype="int16")
    assert rate == 16000 and len(data) == 38 * rate
    assert data[::8000].tolist() == list(range(117, 193))


def test_truncated_clip_removed(tmp_path, monkeypatch):
    def short_packets(source, step, protocol):
        yield np.zeros(8000, dtype=np.float32), False
        yield np.zeros(5600, dtype=np.float32), True

    monkeypatch.setattr(switch, "iter_media_chunks", short_packets)
    output = tmp_path / "clip.wav"
    with pytest.raises(ValueError, match="extends past"):
        switch.extract_packets(None, output, Decimal(0), Decimal(1), 500, "first")
    assert not output.exists()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires FFmpeg")
def test_driver_runs_after_budget_error_and_compares_full_to_fresh(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(2 * 16000, dtype=np.int16), 16000)
    model = tmp_path / "model"
    model.mkdir()
    out = tmp_path / "results"
    prefix = tmp_path / "whole"
    live_cli = switch.ROOT / "tools" / "diagnose_stream.py"
    def header(name, wav):
        backend, cache = switch.RUNS[name]
        return {"type": "ready", "model": str(model), "input": str(wav), "source_kind": "file",
                "protocol": "first", "chunk_ms": 500, "requested_dtype": "bfloat16",
                "backend": backend, "enable_prefix_caching": cache, "mm_processor_cache_gb": 1 if cache is not None else None,
                "automatic_reference": "disabled", "max_context_tokens": 4096, "max_new_tokens": 256,
                **asdict(StreamOptions(context_policy="rolling", number_format="normalized"))}

    def trace(path, hdr, count, error=False):
        rows = [hdr]
        for i in range(1, count + 1):
            rows.append(dict(type="result", event="audio", action="WAIT", turn=i, segments=[],
                             start_seconds=(i - 1) / 2, end_seconds=i / 2, raw_model_text="<|WAIT|>"))
        if error:
            rows.append(dict(type="error", stage="audio", heard_seconds=count / 2, attempted_start_seconds=count / 2,
                             attempted_end_seconds=(count + 1) / 2, error_type="ContextBudgetError",
                             error="budget exceeded", backend_called=False))
        else:
            rows.append(dict(type="result", event="end_of_audio", turn=count + 1, action="WAIT", segments=[],
                             start_seconds=count / 2, end_seconds=count / 2, raw_model_text="<|WAIT|>"))
        path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")

    for name in switch.RUNS:
        trace(Path(f"{prefix}.{name}.jsonl"), header(name, source), 4)
    real_run = subprocess.run
    calls = []

    def run(args, **kwargs):
        if args[:2] != [sys.executable, str(live_cli)]:
            return real_run(args, **kwargs)
        name = Path(args[args.index("--output") + 1]).stem
        calls.append(name)
        assert args[2] == str(out / "clip.wav")
        for flag, expected in [("--chunk-ms", "500"), ("--dtype", "bfloat16"), ("--model", str(model))]:
            assert args[args.index(flag) + 1] == expected
        if name != "hf":
            assert ("--prefix-caching" if name == "vllm-on" else "--no-prefix-caching") in args
        failure = name == "vllm-on"
        trace(out / f"{name}.jsonl", header(name, out / "clip.wav"), 1 if failure else 2, failure)
        return subprocess.CompletedProcess(args, int(failure))

    monkeypatch.setattr(subprocess, "run", run)
    args = [str(prefix), "--start-seconds", ".5", "--end-seconds", "1.5", "--output-dir", str(out)]
    assert switch.main(args) == 1
    assert calls == list(switch.RUNS)
    assert "budget" in (out / "comparison.txt").read_text(encoding="utf-8")
    assert "Report:" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        switch.main(args)
    assert calls == list(switch.RUNS)
