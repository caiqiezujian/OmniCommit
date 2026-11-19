import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

FOURTH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FOURTH))
from fourth_asr.protocol import DIRECT, SPK_CHANGE, WAIT

spec = importlib.util.spec_from_file_location("merge_lora_fourth", FOURTH / "SFT-Scripts" / "merge_lora.py")
merge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge)
IDS = {WAIT: 3, DIRECT: 6, SPK_CHANGE: 4}


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

    def get_vocab(self):
        return {**{str(i): i for i in range(8) if i not in IDS.values()}, **IDS}


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
    base_weights = {key: torch.arange(24, dtype=torch.float32).reshape(8, 3) / (9 + i)
                    for i, key in enumerate(merge.VOCAB_KEYS)}
    save_file(base_weights, base / "model.safetensors")
    # All ordinary rows change too. The adapter is FP32 and the HF export BF16.
    trained = {key: weight + 2.5 for key, weight in base_weights.items()}
    for i, weight in enumerate(trained.values()):
        weight[IDS[WAIT]] = 30 + i
        weight[IDS[DIRECT]] = 40 + i
        weight[IDS[SPK_CHANGE]] = 50 + i
    merged_weights = {key: value.to(torch.bfloat16) for key, value in trained.items()}
    adapter_weights = {f"base_model.model.{key}": value for key, value in trained.items()}
    for suffix in ("lora_A", "lora_B"):
        adapter_weights[f"layer.{suffix}.weight"] = torch.ones((1, 1))
    save_file(adapter_weights, adapter / "adapter_model.safetensors")
    save_file(merged_weights, output / "model.safetensors")
    expected = merge.preflight(base, adapter, token_file)
    report = merge.verify_export(output, expected)
    assert report["full_vocabulary_verified_in"] == list(merge.VOCAB_KEYS)
    for key in merge.VOCAB_KEYS:
        damaged = {**merged_weights, key: merged_weights[key].clone()}
        damaged[key][-1, 1] = -4
        save_file(damaged, output / "model.safetensors")
        with pytest.raises(ValueError, match="Vocabulary rows differ"):
            merge.verify_export(output, expected)

    marker = adapter / "omnicommit_action_rows.json"
    marker.write_text(json.dumps({"schema_version": 1, "token_ids": IDS}))
    assert merge.preflight(base, adapter, token_file)["protocol_token_ids"] == IDS
    marker.write_text(json.dumps({"schema_version": 1, "token_ids": {**IDS, WAIT: 2}}))
    with pytest.raises(ValueError, match="Legacy checkpoint token IDs"):
        merge.preflight(base, adapter, token_file)

    # An ordinary ID swap is just as damaging as a misregistered action token.
    monkeypatch.setattr(tokenizer, "get_vocab", lambda: {**expected["vocabulary"], "1": 2, "2": 1})
    with pytest.raises(ValueError, match="Exported tokenizer differs"):
        merge.verify_export(output, expected)
