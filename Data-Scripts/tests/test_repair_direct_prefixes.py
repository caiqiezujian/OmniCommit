import importlib.util
import json
from pathlib import Path

import pytest
import requests


def test_existing_annotations_repair_without_network_and_keep_bad_rows(tmp_path, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("Offline prefix rendering must never call a model")
    monkeypatch.setattr(requests, "post", forbidden)
    path = Path(__file__).resolve().parents[1] / "repair_direct_prefixes.py"
    spec = importlib.util.spec_from_file_location("repair_prefix_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    annotation = {"segments": [
        {"source_text": "<zh><spk1>请联系", "action": "DIRECT"},
        {"source_text": "张伟", "action": "WAIT"},
        {"source_text": "确认。", "action": "DIRECT"},
    ], "flush": False, "error": None}
    record = {"id": "good", "audios": ["/not-read.wav"], "messages": [
        {"role": "assistant", "content": "<zh><spk1>请联系张伟确认。"}],
        "output": json.dumps(annotation, ensure_ascii=False)}
    source, output = tmp_path / "old.jsonl", tmp_path / "fixed.jsonl"
    source.write_text(json.dumps(record, ensure_ascii=False) + '\n{broken\n' +
                      json.dumps(dict(record, id="bad-output", output="[]")) + '\n' +
                      json.dumps(dict(record, id="empty", output="")) + '\n' +
                      json.dumps(dict(record, id="last")) + '\n', encoding="utf-8")
    before = source.read_bytes()
    assert module.main([str(source), "--output", str(output)]) == 0
    rows = [json.loads(s) for s in output.read_text(encoding="utf-8").splitlines()]
    assert source.read_bytes() == before and len(rows) == 5
    assert all(r["output"] == "" for r in rows[1:4])
    for row in (rows[0], rows[-1]):
        assert json.loads(row["output"])["segments"][-1] == {
            "source_text": "<zh><spk1>确认。", "action": "DIRECT"}
        assert row["messages"] == record["messages"]
    with pytest.raises(SystemExit):
        module.main([str(source), "--output", str(source)])
