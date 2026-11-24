import base64
import codecs
import json
from pathlib import Path
from runpy import run_path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "Data-Scripts" / "convert_chosen_jsonl.py"
convert_file = run_path(str(SCRIPT))["convert_file"]


def test_partitions_all_records_and_preserves_original_filtered_rows(tmp_path, capsys):
    source = tmp_path / "input.jsonl"
    target = tmp_path / "chosen.direct.jsonl"
    good = json.dumps({"audio": "a.wav", "language": "Chinese", "chosen": "<spk1>你好。\n<spk2>早。"}).encode()
    filtered = [
        ('  {"audio":"b.wav","language":"English + Chinese","chosen":"<spk2>明天见",'
         '"metadata":{"score":2.5000},"source_index":%d}  ' % i).encode("utf-8")
        for i in range(24)
    ]
    raw = codecs.BOM_UTF8 + b"\r\n".join([filtered[0], good, b" ", *filtered[1:]])
    source.write_bytes(raw)
    convert_file(source, target)

    assert source.read_bytes() == raw
    annotation = json.loads(json.loads(target.read_bytes())["output"])
    assert annotation["segments"] == [
        {"source_text": "<zh><spk1>你好。", "action": "DIRECT"},
        {"source_text": "<zh><spk2>早。", "action": "DIRECT"},
    ]
    assert (tmp_path / "chosen.direct.rejected.jsonl").read_bytes() == b"\n".join(filtered) + b"\n"
    output = capsys.readouterr()
    assert output.err.count("[跳过]") == 20
    assert "成功 1，已过滤 24，空行 1" in output.out


def test_damaged_input_is_inspectable_and_rerun_replaces_both_files(tmp_path):
    source, target = tmp_path / "raw.jsonl", tmp_path / "out.jsonl"
    rejected = tmp_path / "out.rejected.jsonl"
    malformed = '{"audio":"未结束"'
    binary = b'\xff\xfe\xff\n'
    source.write_bytes(malformed.encode("utf-8") + b"\n" + binary + b"null\n")
    convert_file(source, target)
    syntax_error, encoding_error, original_null = [json.loads(line) for line in rejected.read_bytes().splitlines()]
    assert (syntax_error["line"], syntax_error["raw_line"]) == (1, malformed)
    assert encoding_error["line"] == 2 and base64.b64decode(encoding_error["raw_base64"]) == binary
    assert syntax_error["reason"] and encoding_error["reason"]
    assert original_null is None and target.read_bytes() == b""

    source.write_bytes(b'{"language":"English","audio":"en.wav","chosen":"We need\\nmore time."}')
    convert_file(source, target)
    assert json.loads(target.read_bytes())["messages"][-1]["content"] == "<en><spk1>We need<en><spk1> more time."
    assert rejected.read_bytes() == b""


def test_rejected_destination_cannot_overwrite_input(tmp_path):
    target = tmp_path / "out.jsonl"
    source = tmp_path / "out.rejected.jsonl"
    source.write_bytes(b"original")
    with pytest.raises(ValueError, match="不同路径"):
        convert_file(source, target)
    assert source.read_bytes() == b"original" and not target.exists()
