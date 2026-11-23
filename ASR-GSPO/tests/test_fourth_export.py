import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import finalize_export as export
from asr_gspo.export_callback import complete_checkpoint
from asr_gspo.protocol import DIRECT, SPK_CHANGE, WAIT


def models(tmp_path, monkeypatch):
    from fourth_asr.service_backend import _metadata
    source, dest = tmp_path / "sft", tmp_path / "checkpoint-500-merged"
    source.mkdir()
    dest.mkdir()
    tokens = [WAIT, DIRECT, SPK_CHANGE]
    vocab = dict(zip(tokens, range(10, 13)))
    manifest = dict(_metadata("speaker"), tokens=tokens, token_ids=vocab, tokenizer_length=13)
    (source / "omnicommit_tokens.json").write_text(json.dumps(manifest), encoding="utf-8")
    tokenizer_json = {"model": {"vocab": vocab}, "added_tokens": [
        {"content": token, "id": token_id, "special": True} for token, token_id in vocab.items()]}
    assets = {
        "tokenizer.json": tokenizer_json,
        "tokenizer_config.json": {"additional_special_tokens": tokens},
        "preprocessor_config.json": {"sampling_rate": 16000},
        "video_preprocessor_config.json": {"video_processor_type": "Qwen2VLVideoProcessor"},
    }
    for name, value in assets.items():
        (source / name).write_text(json.dumps(value), encoding="utf-8")
    (source / "chat_template.jinja").write_text("fourth-template", encoding="utf-8")
    (dest / "config.json").write_text('{"actual_rl_config":true}', encoding="utf-8")
    (dest / "model.safetensors").write_bytes(b"rl-weights-untouched")

    class Tokenizer:
        def __init__(self, path):
            self.path = Path(path)
            config = json.loads((self.path / "tokenizer_config.json").read_bytes())
            self.all_special_tokens = config.get("additional_special_tokens", [])
            self.audio_token = config.get("audio_token")
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs == {"local_files_only": True}
            config = json.loads((Path(path) / "tokenizer_config.json").read_text())
            # Transformers 4.57 expects extra_special_tokens to be a named mapping.
            config.get("extra_special_tokens", {}).keys()
            return cls(path)
        def get_vocab(self):
            return json.loads((self.path / "tokenizer.json").read_text())["model"]["vocab"]
        def convert_tokens_to_ids(self, token):
            return self.get_vocab()[token]

    class Processor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            path = Path(path)
            feature = json.loads((path / "preprocessor_config.json").read_text())
            return SimpleNamespace(tokenizer=Tokenizer(path), chat_template=(path / "chat_template.jinja").read_text(),
                                   feature_extractor=SimpleNamespace(to_dict=lambda: feature))

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=Tokenizer, Qwen3OmniMoeProcessor=Processor))
    def validate(tokenizer, m):
        for token, token_id in m["token_ids"].items():
            assert tokenizer.convert_tokens_to_ids(token) == token_id
    monkeypatch.setattr(export, "validate_manifest", validate)
    monkeypatch.setattr(export, "verify_weights", lambda *args: None)
    return source, dest


def test_finalize_repairs_missing_assets_but_preserves_rl_weights(tmp_path, monkeypatch):
    source, dest = models(tmp_path, monkeypatch)
    status = export.finalize(source, dest)
    assert status["ready"] and status["protocol"] == "speaker"
    assert (dest / "model.safetensors").read_bytes() == b"rl-weights-untouched"
    assert json.loads((dest / "config.json").read_text()) == {"actual_rl_config": True}
    assert (dest / "omnicommit_tokens.json").read_bytes() == (source / "omnicommit_tokens.json").read_bytes()
    assert (dest / "video_preprocessor_config.json").read_bytes() == (source / "video_preprocessor_config.json").read_bytes()
    assert export.finalize(source, dest)["copied"] == []


def test_existing_wrong_tokenizer_is_not_overwritten(tmp_path, monkeypatch):
    source, dest = models(tmp_path, monkeypatch)
    wrong = {"model": {"vocab": {"wrong": 123}}}
    (dest / "tokenizer.json").write_text(json.dumps(wrong), encoding="utf-8")
    (dest / "tokenizer_config.json").write_text('{}', encoding="utf-8")
    with pytest.raises(ValueError, match="tokenizer differs"):
        export.finalize(source, dest)
    assert json.loads((dest / "tokenizer.json").read_text()) == wrong
    assert not (dest / "omnicommit_tokens.json").exists()


def test_existing_wrong_audio_processor_is_not_overwritten(tmp_path, monkeypatch):
    source, dest = models(tmp_path, monkeypatch)
    (dest / "preprocessor_config.json").write_text('{"sampling_rate":24000}', encoding="utf-8")
    with pytest.raises(ValueError, match="feature extractor"):
        export.finalize(source, dest)
    assert json.loads((dest / "preprocessor_config.json").read_text())["sampling_rate"] == 24000


def test_equal_vocab_but_wrong_audio_placeholder_is_rejected(tmp_path, monkeypatch):
    source, dest = models(tmp_path, monkeypatch)
    path = source / "tokenizer.json"
    tokenizer = json.loads(path.read_bytes())
    tokenizer["model"]["vocab"].update({"<audio_pad>": 1, "<image_pad>": 2})
    path.write_text(json.dumps(tokenizer), encoding="utf-8")
    config = json.loads((source / "tokenizer_config.json").read_bytes())
    config["audio_token"] = "<audio_pad>"
    (source / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
    config["audio_token"] = "<image_pad>"
    wrong = json.dumps(config).encode()
    (dest / "tokenizer_config.json").write_bytes(wrong)
    with pytest.raises(ValueError, match="audio_token differs"):
        export.finalize(source, dest)
    assert (dest / "tokenizer_config.json").read_bytes() == wrong
    assert not (dest / "omnicommit_tokens.json").exists()


def test_callback_success_and_failure_status_keep_training_alive(tmp_path, monkeypatch):
    source, dest = models(tmp_path, monkeypatch)
    monkeypatch.setenv("ASR_GSPO_SFT_MODEL", str(source))
    native = tmp_path / "checkpoint-500"
    native.mkdir()
    (native / "optimizer.pt").write_bytes(b"native")
    assert complete_checkpoint(native)["ready"]
    (dest / "tokenizer.json").write_text('{"model":{"vocab":{"wrong":123}}}', encoding="utf-8")
    result = complete_checkpoint(native)
    assert not result["ready"]
    assert not json.loads((dest / "omnicommit_export_status.json").read_text())["ready"]
    assert (native / "optimizer.pt").read_bytes() == b"native"


def test_finalize_normalizes_swift_save_before_reload_and_leaves_parent_untouched(tmp_path, monkeypatch):
    source, dest = models(tmp_path, monkeypatch)
    config = "tokenizer_config.json"
    original = (source / config).read_bytes()
    (dest / config).write_text(json.dumps({"extra_special_tokens": [WAIT, DIRECT, SPK_CHANGE]}), encoding="utf-8")
    status = export.finalize(source, dest)
    assert status["ready"] and config in status["normalized"]
    assert (source / config).read_bytes() == original
    assert (dest / "model.safetensors").read_bytes() == b"rl-weights-untouched"
    assert export.finalize(source, dest)["normalized"] == []
