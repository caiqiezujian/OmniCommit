"""Cross-boundary checks for the new annotated-text route, with no model calls."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("step", [.5, 1.0])
def test_annotation_timing_recast_sft_and_runtime_agree(tmp_path, step):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("text_pipeline_wiring_demo", root / "tools/run_text_pipeline_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "demo"
    report = module.build_demo(output, step)
    assert report["source_preserved"]
    assert report["audio_rounds"] == int(6 / step)
    assert report["eos_rounds"] == 1
    events = [json.loads(line) for line in (output / "replay-events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["event"] == "end_of_audio"
    assert events[-1]["start_seconds"] == events[-1]["end_seconds"] == 6
    # Digits/name units are never partially released by a physical boundary.
    spans = [part["text"] for event in events for part in event["segments"]]
    assert sum("张伟" in text for text in spans) == 1
    assert sum("100万美元" in text for text in spans) == 1
    assert "Please confirm the schedule." in "".join(spans)
    assert {s["language"] for e in events for s in e["segments"]} == {"zh", "en"}
    assert {s["speaker"] for e in events for s in e["segments"]} == {"spk1", "spk2"}
