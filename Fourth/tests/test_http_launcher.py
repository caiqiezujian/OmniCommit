"""Verify the HTTP entrypoint passes the real checkpoint and model limits into the shared runtime."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("protocol, extra, expected_budgets", [
    ("speaker", [], (512, 4352)),
    ("explicit", ["--max-new-tokens", "768", "--max-context-tokens", "4608", "--response-timeout-ms", "900",
                  "--packet-period-to-comma"], (768, 4608)),
])
def test_http_launcher_uses_trained_protocol_and_sizes_model_for_batching(tmp_path, monkeypatch, protocol, extra, expected_budgets):
    uvicorn = pytest.importorskip("uvicorn")
    pytest.importorskip("fastapi")
    path = Path(__file__).resolve().parents[1] / "tools" / "serve_http.py"
    spec = importlib.util.spec_from_file_location("fourth_http_launcher", path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    from fourth_asr import http_service, recovery, service_backend
    from fourth_asr.runtime import vllm_backend
    from fourth_asr.protocol import get_protocol

    inventory = tmp_path / "token.txt"
    inventory.write_text("\n".join(get_protocol(protocol).tokens) + "\n", encoding="utf-8")
    tokens, metadata = service_backend.read_export_protocol(inventory)
    ids = {token: i for i, token in enumerate(tokens, start=80)}
    manifest = {**metadata, "tokens": tokens, "token_ids": ids, "tokenizer_length": 80 + len(tokens)}
    (tmp_path / "omnicommit_tokens.json").write_text(json.dumps(manifest), encoding="utf-8")
    engine = Mock(return_value=SimpleNamespace(generate_batch=lambda calls: [], max_batch_size=6))
    monkeypatch.setattr(vllm_backend, "VLLMBackend", engine)
    monkeypatch.setattr(recovery, "check_recovery_dependencies", lambda: None)
    from fourth_asr.http_logging import request_event

    run = Mock(side_effect=lambda *a, **kw: request_event("test_event"))
    monkeypatch.setattr(uvicorn, "run", run)

    def inspect(factory, options, **limits):
        assert options.protocol == protocol and options.enable_recovery
        assert options.chunk_seconds == 1 and options.context_policy == "window"
        assert (options.window_seconds, options.discard_seconds) == (60, 50)
        assert limits["max_in_flight"] == 8 and "response_timeout_ms" not in limits
        assert limits["packet_period_to_comma"] == ("--packet-period-to-comma" in extra)
        adapter = factory()
        try:
            assert adapter.max_batch_size == 6
        finally:
            adapter.close()
        return "app"

    monkeypatch.setattr(http_service, "create_app", inspect)
    request_log = tmp_path / "logs" / "http.jsonl"
    launcher.main(["--model", str(tmp_path), "--batch-size", "6", "--recovery",
                   "--request-log", str(request_log), *extra])
    assert (engine.call_args.kwargs["max_new_tokens"], engine.call_args.kwargs["max_context_tokens"]) == expected_budgets
    assert Path(engine.call_args.args[0]) == tmp_path
    assert engine.call_args.kwargs["manifest_validator"] is service_backend.validate_manifest
    assert engine.call_args.kwargs["max_audio_turns"] == 60
    assert engine.call_args.kwargs["max_audio_seconds"] == launcher.FourthStreamOptions.recovery_window_seconds
    assert engine.call_args.kwargs["max_batch_size"] == 6
    run.assert_called_once_with("app", host="127.0.0.1", port=18892, workers=1)
    assert json.loads(request_log.read_text(encoding="utf-8"))["event"] == "test_event"


@pytest.fixture
def shell_launcher(tmp_path):
    if os.name == "nt":
        git = shutil.which("git")
        bash = Path(git).resolve().parents[1] / "bin" / "bash.exe" if git else None
    else:
        bash = shutil.which("bash")
    if not bash or not Path(bash).is_file():
        pytest.skip("Requires Bash or Git Bash")

    fourth = Path(__file__).resolve().parents[1]
    recipe = tmp_path / "repo with spaces" / "Fourth"
    tools = recipe / "tools"
    tools.mkdir(parents=True)
    shutil.copy(fourth / "start_http.sh", recipe)
    (tools / "serve_http.py").write_text(
        "import json, os, sys\n"
        "print(json.dumps([os.getcwd(), os.getenv('CUDA_VISIBLE_DEVICES'), sys.argv[1:]]))\n",
        encoding="utf-8",
    )

    def run(*arguments):
        script = (recipe / "start_http.sh").relative_to(tmp_path).as_posix()
        proc = subprocess.run(
            [str(bash), script, *arguments], cwd=tmp_path, capture_output=True, text=True,
            timeout=30, env={**os.environ, "PYTHON": Path(sys.executable).as_posix(),
                             "CUDA_VISIBLE_DEVICES": "2", "MSYS2_ARG_CONV_EXCL": "*"},
        )
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout)

    return fourth, recipe, run


def test_no_arguments_load_the_shipped_http_config(shell_launcher):
    fourth, recipe, run = shell_launcher
    shutil.copy(fourth / "http_config.sh", recipe)
    cwd, gpu, args = run()
    assert Path(cwd) == recipe
    assert gpu == "6"
    assert args[args.index("--model") + 1].endswith("sft-checkpoint-2721")
    assert args[args.index("--port") + 1] == "18892"
    assert args[args.index("--chunk-ms") + 1] == "1000"
    assert args[args.index("--debug-dir") + 1] == "logs/http-debug"
    assert "--packet-period-to-comma" not in args


def test_enabling_packet_punctuation_in_the_shipped_http_config(shell_launcher):
    fourth, recipe, run = shell_launcher
    config = (fourth / "http_config.sh").read_bytes()
    (recipe / "http_config.sh").write_bytes(
        config.replace(b"PACKET_PERIOD_TO_COMMA=false", b"PACKET_PERIOD_TO_COMMA=true")
    )
    _, _, args = run()
    assert args.count("--packet-period-to-comma") == 1


def test_editing_http_config_preserves_arguments_containing_spaces(shell_launcher):
    _, recipe, run = shell_launcher
    (recipe / "http_config.sh").write_bytes(
        b"CUDA_VISIBLE_DEVICES=3\nASR_MODEL='models/checkpoint with spaces'\n"
        b"HTTP_ARGS=(--host 127.0.0.1 --port 18899 --debug-dir 'event logs')\n"
    )
    cwd, gpu, args = run()
    assert Path(cwd) == recipe
    assert gpu == "3"
    assert args == ["--model", "models/checkpoint with spaces", "--host", "127.0.0.1",
                    "--port", "18899", "--debug-dir", "event logs"]


def test_explicit_http_arguments_still_work_without_config(shell_launcher, tmp_path):
    _, _, run = shell_launcher
    cwd, gpu, args = run("models/original path", "--port", "18899")
    assert Path(cwd) == tmp_path
    assert gpu == "2"
    assert args == ["--model", "models/original path", "--host", "0.0.0.0", "--port", "18899"]
