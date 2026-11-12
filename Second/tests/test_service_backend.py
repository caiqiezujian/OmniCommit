"""Second checkpoint export and serving handshake, exercised without a GPU."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

SECOND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SECOND))
sys.path.insert(0, str(SECOND.parent))
from second_asr.protocol import WAIT
from second_asr.service_backend import build_backend, read_export_protocol, validate_manifest
from omnicommit.inference_backend import validate_token_manifest
from omnicommit.stream_cli import add_runtime_arguments
from omnicommit.vllm_backend import VLLMBackend

TOKENS = [WAIT, "<zh>", "<en>", "<spk1>"]  # Preserve checkpoint order, even for an explicit token file.


class Tokenizer:
    unk_token_id = 0

    def __init__(self, trained=True):
        self.all_special_tokens = list(TOKENS) if trained else []

    def __len__(self):
        return 8 + len(self.all_special_tokens)

    def convert_tokens_to_ids(self, token):
        return 8 + self.all_special_tokens.index(token) if token in self.all_special_tokens else 0

    def encode(self, token, **kwargs):
        return [self.convert_tokens_to_ids(token)]

    def add_special_tokens(self, value):
        self.all_special_tokens = list(value["additional_special_tokens"])


def test_inventory_and_protocol_are_checked_before_weight_loading(tmp_path):
    token_file = tmp_path / "train.jsonl.tokens.txt"
    token_file.write_text("\n".join(TOKENS), encoding="utf-8")
    tokens, metadata = read_export_protocol(token_file)
    assert tokens == TOKENS
    tokenizer = Tokenizer()
    manifest = {**metadata, "tokens": tokens, "token_ids": {t: tokenizer.convert_tokens_to_ids(t) for t in tokens},
                "tokenizer_length": len(tokenizer)}
    assert validate_manifest(tokenizer, manifest) == set(tokens)
    with pytest.raises(ValueError, match="first-generation"):
        validate_token_manifest(tokenizer, manifest)
    with pytest.raises(ValueError, match="Expected a Second"):
        validate_manifest(tokenizer, {**manifest, "schema_version": 1})
    with pytest.raises(ValueError, match="ID mismatch"):
        validate_manifest(tokenizer, {**manifest, "token_ids": {**manifest["token_ids"], WAIT: 0}})


@pytest.mark.parametrize("tokens", [[WAIT, "<|DIRECT|>", "<en>", "<spk1>"],
                                    [WAIT, "<en>"], [WAIT, "<spk1>"], [WAIT, "<en>", "<spk1>", "<en>"]])
def test_wrong_inventory_rejected_at_export(tmp_path, tokens):
    path = tmp_path / "input.tokens.txt"
    path.write_text("\n".join(tokens), encoding="utf-8")
    with pytest.raises(ValueError):
        read_export_protocol(path)


@pytest.mark.parametrize(("engine", "constructor"), [("hf", "omnicommit.inference_backend.HFBackend"),
                                                    ("vllm", "omnicommit.vllm_backend.VLLMBackend")])
def test_real_backend_constructors_receive_second_validator(engine, constructor, monkeypatch):
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    args = parser.parse_args(["--backend", engine, "--model", "/models/merged"])
    monkeypatch.setattr(constructor, lambda path, **kw: kw)
    assert build_backend(args)["manifest_validator"] is validate_manifest


@pytest.mark.parametrize(("default_seconds", "minimum", "maximum"), [
    (1.0, .2, 1.0), (.7, .2, 1.0), (.08, .08, 1.0), (2.0, .2, 2.0)])
def test_vllm_reserves_audio_capacity_for_every_page_choice(default_seconds, minimum, maximum, monkeypatch):
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    args = parser.parse_args(["--backend", "vllm", "--model", "/models/merged",
                              "--max-reference-seconds", "0.5"])
    args.chunk_seconds = default_seconds
    monkeypatch.setattr("omnicommit.vllm_backend.VLLMBackend", lambda path, **kw: kw)
    settings = build_backend(args)
    assert settings["max_audio_turns"] == args.max_context_seconds / minimum + args.max_speakers + 2
    assert settings["max_audio_seconds"] == maximum


@pytest.mark.parametrize("step_ms", [500, 1001])
def test_vllm_decodes_sparse_reply_from_token_ids_preserving_initial_space(step_ms):
    engine = VLLMBackend.__new__(VLLMBackend)
    decoded = []

    def decode(ids, **options):
        decoded.append((ids, options))
        return " I will."

    engine.tokenizer = SimpleNamespace(encode=lambda *a, **k: [7],
                                       apply_chat_template=lambda *a, **k: "prompt", decode=decode)
    engine.max_context_tokens, engine.max_new_tokens = 32, 8
    engine.max_audio_turns, engine.max_audio_seconds = 4, step_ms / 1000
    engine.stop_ids, engine.sampling, engine.TokensPrompt = [99], object(), dict
    engine.audio_counter = SimpleNamespace(count=lambda *args: (0, False))
    completion = SimpleNamespace(token_ids=[11, 12, 99], text="I will.", finish_reason="stop")
    output = SimpleNamespace(prompt_token_ids=[7], outputs=[completion])
    engine.llm = SimpleNamespace(generate=lambda *a, **kw: [output])
    reply = engine.generate([{"role": "user", "content": "<audio>"}], [np.zeros(step_ms * 16, dtype=np.float32)])
    assert reply["text"] == " I will."
    assert decoded == [([11, 12], {"skip_special_tokens": False, "clean_up_tokenization_spaces": False})]


def test_second_export_writes_a_loadable_manifest_and_checks_training_order(tmp_path, monkeypatch):
    # Real tiny safetensors and row checks. Stub the large-model SWIFT process and tokenizer loader.
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    base, adapter, output = (tmp_path / n for n in ("hf", "checkpoint-1000", "merged"))
    base.mkdir()
    adapter.mkdir()
    config = {"model_type": "qwen3_omni_moe", "thinker_config": {"text_config": {"vocab_size": 16}}}
    keys = ["thinker.model.embed_tokens.weight", "thinker.lm_head.weight"]
    (base / "config.json").write_text(json.dumps(config))
    (adapter / "adapter_config.json").write_text('{"peft_type":"LORA"}')
    token_file = tmp_path / "tokens.txt"
    token_file.write_text("\n".join(TOKENS))
    save_file({k: torch.zeros(16, 4) for k in keys}, str(base / "model.safetensors"))
    saved = {"base_model.model." + k: torch.ones(16, 4) for k in keys}
    saved.update({"layer.lora_A.weight": torch.ones(1, 1), "layer.lora_B.weight": torch.ones(1, 1)})
    save_file(saved, str(adapter / "adapter_model.safetensors"))

    def run_swift(command, **kwargs):
        assert command[command.index("--new_special_tokens") + 1] == str(token_file)
        exported = Path(command[command.index("--output_dir") + 1])
        exported.mkdir()
        (exported / "config.json").write_text(json.dumps(config))
        save_file({k: torch.ones(16, 4) for k in keys}, str(exported / "model.safetensors"))

    loader = SimpleNamespace(from_pretrained=lambda path, **kw: Tokenizer(trained=Path(path) != base))
    processor = SimpleNamespace(from_pretrained=lambda *a, **kw: SimpleNamespace(tokenizer=Tokenizer()))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=loader, AutoProcessor=processor))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "4.1.3")
    monkeypatch.setattr(subprocess, "run", run_swift)
    # Export is normally run in a separate process. Isolate its offline/CUDA env here as well.
    import os
    monkeypatch.setattr(os, "environ", dict(os.environ))
    merge = runpy.run_path(str(SECOND / "SFT-Scripts" / "merge_lora.py"))["main"]
    args = [str(base), str(adapter), str(output), "--device-map", "cpu", "--special-tokens", str(token_file)]
    (adapter / "args.json").write_text(json.dumps({"new_special_tokens": TOKENS[::-1]}))
    assert merge(args) == 1 and not output.exists()
    (adapter / "args.json").write_text(json.dumps({"new_special_tokens": TOKENS}))
    assert merge(args) == 0
    manifest = json.loads((output / "omnicommit_tokens.json").read_text())
    assert manifest["tokens"] == TOKENS and validate_manifest(Tokenizer(), manifest) == set(TOKENS)
    report = json.loads((output / "omnicommit_merge_report.json").read_text())
    assert report["trained_protocol_rows_preserved_in"] == keys
