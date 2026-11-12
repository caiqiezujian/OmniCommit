"""Check the actual Second entry point without binding a port or loading weights."""
from pathlib import Path
import runpy

import pytest
import uvicorn
from fastapi.testclient import TestClient

ENTRY = Path(__file__).resolve().parents[1] / "tools" / "serve_web.py"


@pytest.mark.parametrize(("step", "seconds"), [([], .5), (["--chunk-ms", "700"], .7),
                                               (["--chunk-seconds", "8"], 8)])
def test_cli_uses_second_protocol_and_appropriate_ws_frame_size(monkeypatch, step, seconds):
    def serve(app, **settings):
        assert settings["port"] == 18890
        assert settings["ws_max_size"] >= round(seconds * 16000) * 2
        with TestClient(app) as client:
            health = client.get("/health").json()
            assert health["protocol"] == "second"
            assert health["chunk_seconds"] == seconds
            assert health["context_policy"] == "rolling"
            assert set(health["chunk_ms_choices"]) == {500, 700, 800, 1000, round(seconds * 1000)}
            assert settings["ws_max_size"] >= max(health["chunk_ms_choices"]) * 32

    monkeypatch.setattr(uvicorn, "run", serve)
    runpy.run_path(str(ENTRY))["main"](["--backend", "replay", *step])


def test_incompatible_step_arguments_stop_before_model_load():
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(ENTRY))["main"](["--chunk-ms", "700", "--chunk-seconds", "0.5"])
    assert error.value.code == 2
