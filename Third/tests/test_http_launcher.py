"""Verify the HTTP entrypoint passes the real checkpoint and model limits into the shared runtime."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("extra, expected_budgets", [
    ([], (512, 4352)),
    (["--max-new-tokens", "768", "--max-context-tokens", "4608"], (768, 4608)),
])
def test_http_launcher_uses_trained_protocol_and_sizes_model_for_batching(tmp_path, monkeypatch, extra, expected_budgets):
    uvicorn = pytest.importorskip("uvicorn")
    pytest.importorskip("fastapi")
    path = Path(__file__).resolve().parents[1] / "tools" / "serve_http.py"
    spec = importlib.util.spec_from_file_location("third_http_launcher", path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    from third_asr import http_service, recovery, service_backend
    from third_asr.runtime import vllm_backend
    from third_asr.protocol import DIRECT, WAIT

    manifest = service_backend._metadata("explicit")
    manifest["tokens"] = [WAIT, DIRECT]
    (tmp_path / "omnicommit_tokens.json").write_text(json.dumps(manifest), encoding="utf-8")
    engine = Mock(return_value=SimpleNamespace(generate_batch=lambda calls: [], max_batch_size=6))
    monkeypatch.setattr(vllm_backend, "VLLMBackend", engine)
    monkeypatch.setattr(recovery, "check_recovery_dependencies", lambda: None)
    run = Mock()
    monkeypatch.setattr(uvicorn, "run", run)

    def inspect(factory, options, **limits):
        assert options.protocol == "explicit" and options.enable_recovery
        assert options.chunk_seconds == 1 and options.context_policy == "window"
        assert (options.window_seconds, options.discard_seconds) == (60, 50)
        assert limits["max_in_flight"] == 8 and limits["response_timeout_ms"] == 900
        adapter = factory()
        try:
            assert adapter.max_batch_size == 6
        finally:
            adapter.close()
        return "app"

    monkeypatch.setattr(http_service, "create_app", inspect)
    launcher.main(["--model", str(tmp_path), "--chunk-ms", "1000", "--batch-size", "6", "--recovery", *extra])
    assert (engine.call_args.kwargs["max_new_tokens"], engine.call_args.kwargs["max_context_tokens"]) == expected_budgets
    assert engine.call_args.kwargs["max_audio_turns"] == 60
    assert engine.call_args.kwargs["max_audio_seconds"] == launcher.ThirdStreamOptions.recovery_window_seconds
    assert engine.call_args.kwargs["max_batch_size"] == 6
    run.assert_called_once_with("app", host="127.0.0.1", port=18892, workers=1)
