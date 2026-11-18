import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

THIRD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIRD))
from third_asr.action_rows_manifest import FILENAME, read_action_rows_manifest, write_action_rows_manifest
from third_asr.protocol import DIRECT, WAIT

spec = importlib.util.spec_from_file_location("merge_lora_third", THIRD / "SFT-Scripts" / "merge_lora.py")
merge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge)
IDS = {WAIT: 3, DIRECT: 6}


def test_marker_is_optional_for_legacy_checkpoints_and_strict_for_row_mode(tmp_path):
    assert read_action_rows_manifest(tmp_path, IDS) is None
    with pytest.raises(ValueError, match="Missing"):
        read_action_rows_manifest(tmp_path, IDS, required=True)
    write_action_rows_manifest(tmp_path, {WAIT: 3})
    assert read_action_rows_manifest(tmp_path, {WAIT: 3}, required=True)["token_ids"] == {WAIT: 3}
    with pytest.raises(ValueError, match="disagree"):
        read_action_rows_manifest(tmp_path, IDS)
    write_action_rows_manifest(tmp_path, IDS)
    assert read_action_rows_manifest(tmp_path, IDS)["token_ids"] == IDS
    with pytest.raises(ValueError, match="disagree"):
        read_action_rows_manifest(tmp_path, {WAIT: 6, DIRECT: 3})
    with pytest.raises(ValueError, match="distinct"):
        write_action_rows_manifest(tmp_path, {WAIT: 6, DIRECT: 6})
    (tmp_path / FILENAME).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        read_action_rows_manifest(tmp_path, IDS)


class Tokenizer:
    unk_token_id = 0
    all_special_tokens = list(IDS)

    def __len__(self):
        return 8

    def add_special_tokens(self, tokens):
        assert tokens == {"additional_special_tokens": list(IDS)}

    def convert_tokens_to_ids(self, token):
        return IDS[token]

    def encode(self, token, add_special_tokens=False):
        return [IDS[token]]


def test_saved_adapter_and_merged_model_keep_exact_vocab_rows(tmp_path, monkeypatch):
    tokenizer = Tokenizer()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer),
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: SimpleNamespace(tokenizer=tokenizer))))
    base, adapter, output = (tmp_path / name for name in ("base", "adapter", "output"))
    for folder in (base, adapter, output):
        folder.mkdir()
    cfg = {"model_type": "qwen3_omni_moe", "thinker_config": {"text_config": {"vocab_size": 8}}}
    for folder in (base, output):
        (folder / "config.json").write_text(json.dumps(cfg))
    (adapter / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
    (adapter / "args.json").write_text(json.dumps({"new_special_tokens": list(IDS)}))
    token_file = tmp_path / "tokens.txt"
    token_file.write_text("\n".join(IDS))
    write_action_rows_manifest(adapter, IDS)
    base_weights = {key: torch.arange(24, dtype=torch.float32).reshape(8, 3) / (9 + i)
                    for i, key in enumerate(merge.VOCAB_KEYS)}
    save_file(base_weights, base / "model.safetensors")
    merged_weights = {key: weight.to(torch.bfloat16) for key, weight in base_weights.items()}
    for i, weight in enumerate(merged_weights.values()):
        weight[IDS[WAIT]] = 30 + i
        weight[IDS[DIRECT]] = 40 + i
    adapter_weights = {f"base_model.model.{key}": value for key, value in merged_weights.items()}
    for suffix in ("lora_A", "lora_B"):
        adapter_weights[f"layer.{suffix}.weight"] = torch.ones((1, 1))
    save_file(adapter_weights, adapter / "adapter_model.safetensors")
    save_file(merged_weights, output / "model.safetensors")
    expected = merge.preflight(base, adapter, token_file, require_action_rows=True)
    report = merge.verify_export(output, expected)
    assert report["frozen_base_rows_verified"]
    assert report["action_rows_changed_from_base"] == {key: {WAIT: True, DIRECT: True} for key in merge.VOCAB_KEYS}

    key = merge.VOCAB_KEYS[0]
    merged_weights[key][5, 1] = -4
    save_file(merged_weights, output / "model.safetensors")
    with pytest.raises(ValueError, match="Vocabulary rows differ"):
        merge.verify_export(output, expected)

    damaged = load_file(adapter / "adapter_model.safetensors")
    damaged = {key: tensor.clone() for key, tensor in damaged.items()}
    damaged[f"base_model.model.{key}"][7, 1] = -5
    save_file(damaged, adapter / "adapter_model.safetensors")
    with pytest.raises(ValueError, match="Vocabulary rows differ"):
        merge.preflight(base, adapter, token_file)
    (adapter / FILENAME).unlink()
    assert merge.preflight(base, adapter, token_file)["action_rows"] is None
