"""Check the actual Second entry point without binding a port or loading weights."""
from pathlib import Path
import runpy

import pytest
import uvicorn
from fastapi.testclient import TestClient

ENTRY = Path(__file__).resolve().parents[1] / "tools" / "serve_web.py"


@pytest.mark.parametrize(("step", "seconds"), [([], .5), (["--chunk-ms", "700"], .7),
                                               (["--chunk-ms", "80"], .08),
                                               (["--chunk-seconds", "8"], 8)])
def test_cli_uses_second_protocol_and_appropriate_ws_frame_size(monkeypatch, step, seconds):
    def serve(app, **settings):
        assert settings["port"] == 18890
        assert settings["ws_max_size"] >= round(seconds * 16000) * 2
        with TestClient(app) as client:
            health = client.get("/health").json()
            assert health["protocol"] == "second"
            assert health["chunk_seconds"] == seconds
            assert health["context_policy"] == "window"
            assert (health["window_seconds"], health["discard_seconds"]) == (60, 50)
            assert set(health["chunk_ms_choices"]) == {500, 700, 800, 1000, round(seconds * 1000)}
            assert health["chunk_ms_min"] == min(200, round(seconds * 1000))
            assert health["chunk_ms_max"] == max(5000, round(seconds * 1000))
            assert settings["ws_max_size"] >= health["chunk_ms_max"] * 32

    monkeypatch.setattr(uvicorn, "run", serve)
    runpy.run_path(str(ENTRY))["main"](["--backend", "replay", *step])


def test_incompatible_step_arguments_stop_before_model_load():
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(ENTRY))["main"](["--chunk-ms", "700", "--chunk-seconds", "0.5"])
    assert error.value.code == 2


def test_cli_window_defaults_propagate_to_engine_and_page(monkeypatch):
    main = runpy.run_path(str(ENTRY))["main"]
    monkeypatch.setattr("omnicommit.vllm_backend.VLLMBackend", lambda path, **kw: kw)

    def serve(parser, args, backend_factory, options, app_factory):
        assert (options.window_seconds, options.discard_seconds) == (90, 45)
        engine = backend_factory()
        assert engine["max_audio_turns"] >= 90 / .2 + options.max_speakers
        assert engine["enable_prefix_caching"] is True

    monkeypatch.setitem(main.__globals__, "serve_web", serve)
    main(["--model", "merged-hf", "--window-seconds", "90", "--discard-seconds", "45"])
