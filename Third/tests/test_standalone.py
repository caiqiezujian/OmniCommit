"""Exercise the published entry points with only a copy of Third installed."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import wave

import pytest


THIRD = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def deployed(tmp_path_factory):
    package = tmp_path_factory.mktemp("isolated") / "streaming asr"
    package.mkdir()
    for name in ("Data-Scripts", "SFT-Scripts", "third_asr", "tools", "web"):
        shutil.copytree(THIRD / name, package / name, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("requirements-data.txt", "requirements-web.txt", "start_server.sh"):
        shutil.copy(THIRD / name, package / name)
    return package


def python(*args, cwd, pythonpath=""):
    # Avoid inheriting the developer checkout on PYTHONPATH. Keep the normal
    # script-directory import behavior of "python tools/serve_web.py", etc.
    env = {**os.environ, "PYTHONPATH": str(pythonpath), "PYTHONUTF8": "1"}
    result = subprocess.run([sys.executable, "-s", *map(str, args)], cwd=cwd, env=env,
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.parametrize("registry,qwen_module", [("swift/llm/template", "template"), ("swift/template", "templates")])
def test_recast_then_decode_third_training_row(deployed, tmp_path, registry, qwen_module):
    audio = tmp_path / "switch.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", ""))
        wav.writeframes(bytes(2 * 16000 * 2))
    spans = ["<zh><spk1>你好。", "<en><spk2>Hello."]
    row = {
        "audios": [audio.name],
        "messages": [{"role": "assistant", "content": "".join(spans)}],
        "output": {"segments": [{"source_text": s} for s in spans]},
        "timestamps": [{"start_ms": 0, "end_ms": 1150}, {"start_ms": 1150, "end_ms": 2000}],
    }
    (tmp_path / "input.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    python(deployed / "Data-Scripts/build_sft.py", "input.jsonl", "--chunk-ms", 1000,
           "--output-dir", "data", cwd=tmp_path)
    train = tmp_path / "data/train.1000ms.jsonl"
    # Exercise Third's real --framework preflight without installing Megatron
    # or loading weights. Supply only the SWIFT registry API the plugin calls.
    swift = tmp_path / registry
    modules = swift / qwen_module
    modules.mkdir(parents=True)
    for package in [modules, *modules.parents]:
        if package == tmp_path:
            break
        (package / "__init__.py").touch()
    (swift / "register.py").write_text(
        "from types import SimpleNamespace\n"
        "TEMPLATE_MAPPING = {'qwen3_omni': SimpleNamespace()}\n"
        "def register_template(meta):\n    TEMPLATE_MAPPING[meta.template_type] = meta\n",
        encoding="utf-8")
    (modules / "qwen.py").write_text(
        "class Qwen3OmniTemplate:\n    def _preprocess_inputs(self, inputs):\n        return inputs\n",
        encoding="utf-8")
    checker = deployed / "SFT-Scripts/check_sft.py"
    result = json.loads(python(checker, train, "--decode", "--framework", cwd=tmp_path, pythonpath=tmp_path))
    assert result["checked_rows"] == 1 and result["decoded"] and result["framework_checked"]
    row = json.loads(train.read_text(encoding="utf-8"))
    assert [msg["content"] for msg in row["messages"] if msg["role"] == "assistant"] == [
        "<|WAIT|>", "<|DIRECT|>你好。Hello.", "<|WAIT|>",
    ]


@pytest.mark.parametrize("script,option", [
    ("audit_encoded_lengths.py", "--max-length"),
    ("merge_lora.py", "--check-only"),
])
def test_model_tool_entrypoints_import_without_parent_checkout(deployed, tmp_path, script, option):
    assert option in python(deployed / "SFT-Scripts" / script, "--help", cwd=tmp_path)


def test_web_from_standalone_cli(deployed, tmp_path):
    pytest.importorskip("httpx")
    replies = tmp_path / "replies.json"
    replies.write_text(json.dumps(["<|DIRECT|>Hello.", "<|WAIT|>"]), encoding="utf-8")
    probe = textwrap.dedent("""\
        import runpy
        import sys
        from unittest.mock import patch
        from fastapi.testclient import TestClient

        def visit(app, **kwargs):
            with TestClient(app) as client:
                assert client.get("/health").json()["action_protocol"] == "explicit"
                assert client.get("/").status_code == 200
                with client.websocket_connect("/stream") as ws:
                    ws.send_json({"type": "start", "sample_rate": 16000, "source_kind": "microphone"})
                    assert ws.receive_json()["protocol"] == "third"
                    ws.send_bytes(bytes(32000))
                    assert ws.receive_json()["text"] == "<|DIRECT|>Hello."
                    ws.send_json({"type": "end"})
                    assert ws.receive_json()["text"] == "<|WAIT|>"

        sys.argv = sys.argv[1:]
        with patch("uvicorn.run", side_effect=visit) as run:
            runpy.run_path(sys.argv[0], run_name="__main__")
            run.assert_called_once()
    """)
    python("-c", probe, deployed / "tools/serve_web.py", "--backend", "replay", "--replies", replies,
           "--no-recovery", "--chunk-ms", 1000, cwd=tmp_path)


def test_requirements_only_include_deployed_files(deployed):
    for requirements in deployed.rglob("requirements*.txt"):
        for line in requirements.read_text(encoding="utf-8").splitlines():
            if line.startswith("-r "):
                path = (requirements.parent / line[3:].strip()).resolve()
                assert deployed in path.parents and path.is_file(), (requirements, path)
