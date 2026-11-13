"""Exercise the optional CPU decoder and its CLI without network or weights."""
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

SECOND = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SECOND), str(SECOND.parent)]
from second_asr.gap_backend import WhisperGapBackend, local_whisper_directory


def local_model(tmp_path):
    for name in ("model.bin", "config.json", "tokenizer.json"):
        (tmp_path / name).touch()
    return tmp_path


def test_clip_transcription_is_local_and_preserves_timed_word_text(tmp_path, monkeypatch):
    class Model:
        def __init__(self, path, **kwargs):
            assert path == str(tmp_path)
            assert kwargs == {"device": "cpu", "compute_type": "int8", "cpu_threads": 2,
                              "num_workers": 1, "local_files_only": True}
            self.model = SimpleNamespace(is_multilingual=True)

        def transcribe(self, pcm, **kwargs):
            assert kwargs["word_timestamps"] and kwargs["language"] is None
            assert kwargs["task"] == "transcribe" and not kwargs["condition_on_previous_text"]
            words = [SimpleNamespace(start=-.02, end=.4, word=" Guten", probability=.9),
                     SimpleNamespace(start=.45, end=.85, word=" Tag!", probability=.95),
                     SimpleNamespace(start=.85, end=.90, word=" bogus", probability=.7)]
            return iter([SimpleNamespace(words=words)]), SimpleNamespace(language="de", language_probability=.92)

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Model))
    backend = WhisperGapBackend(local_model(tmp_path))
    assert backend.transcribe(np.ones(12800, dtype=np.float32)) == [
        {"text": " Guten", "start": 0.0, "end": .4, "probability": .9,
         "language": "de", "language_probability": .92},
        {"text": " Tag!", "start": .45, "end": .8, "probability": .95,
         "language": "de", "language_probability": .92},
    ]


def test_missing_local_tokenizer_rejected_before_optional_import(tmp_path):
    local_model(tmp_path)
    (tmp_path / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match="tokenizer.json"):
        local_whisper_directory(tmp_path)


def test_english_only_gap_model_is_rejected(tmp_path, monkeypatch):
    constructor = lambda *args, **kwargs: SimpleNamespace(model=SimpleNamespace(is_multilingual=False))
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=constructor))
    with pytest.raises(ValueError, match="multilingual"):
        WhisperGapBackend(local_model(tmp_path))


def test_launcher_passes_lazy_local_cpu_factory(tmp_path, monkeypatch):
    import omnicommit.web_service
    import second_asr.service_web

    def serve(parser, args, backend_factory, opts, app_factory):
        assert opts.chunk_seconds == .8
        app_factory(backend_factory, opts, max_sessions=1)

    def app_factory(backend_factory, opts, *, max_sessions, gap_backend_factory):
        assert gap_backend_factory.func is WhisperGapBackend
        assert gap_backend_factory.args == (tmp_path,)

    monkeypatch.setattr(omnicommit.web_service, "serve_web", serve)
    monkeypatch.setattr(second_asr.service_web, "create_app", app_factory)
    runpy.run_path(str(SECOND / "tools/serve_web.py"))["main"](
        ["--backend", "replay", "--chunk-ms", "800", "--gap-asr-model", str(local_model(tmp_path))])
