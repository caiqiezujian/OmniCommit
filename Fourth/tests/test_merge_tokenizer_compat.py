"""Exercise the merge CLI boundaries without launching the GPU exporter."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

FOURTH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FOURTH))
from fourth_asr.protocol import DIRECT, SPK_CHANGE, WAIT

spec = importlib.util.spec_from_file_location("merge_tokenizer_compat", FOURTH / "SFT-Scripts/merge_lora.py")
merge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge)


@pytest.mark.parametrize("version,model_type,tuning_flag", [
    ("3.12.1", "qwen3_omni", "--train_type"),
    ("3.14.2", "qwen3_omni", "--train_type"),
    ("4.1.3", "qwen3_omni_moe", "--tuner_type"),
])
def test_merge_command_uses_installed_swift_cli(version, model_type, tuning_flag):
    args = SimpleNamespace(base="hf-base", adapter="checkpoint-1", special_tokens="tokens.txt", device_map="auto")
    command = merge.export_command(args, "merged", version)
    assert command[:4] == [sys.executable, "-u", "-m", "swift.cli.export"]
    assert dict(zip(command[4::2], command[5::2])) == {
        "--model": "hf-base", "--adapters": "checkpoint-1", "--model_type": model_type,
        "--template": "qwen3_omni", "--load_args": "false", tuning_flag: "lora",
        "--new_special_tokens": "tokens.txt", "--merge_lora": "true",
        "--torch_dtype": "bfloat16", "--device_map": "auto", "--safe_serialization": "true",
        "--max_shard_size": "5GB", "--output_dir": "merged",
    }


def test_training_protocol_appends_to_existing_base_registry_in_both_hf_apis():
    class V4:
        def add_special_tokens(self, mapping, replace_additional_special_tokens=True):
            assert mapping == {"additional_special_tokens": [WAIT, DIRECT]}
            assert replace_additional_special_tokens is False

    class V5:
        def add_special_tokens(self, mapping, replace_extra_special_tokens=True):
            assert mapping == {"extra_special_tokens": [WAIT, DIRECT]}
            assert replace_extra_special_tokens is False

    for tokenizer in (V4(), V5()):
        merge.add_training_tokens(tokenizer, [WAIT, DIRECT])


def test_conflicting_audio_settings_between_hf_versions_cannot_be_published(tmp_path):
    base, adapter, out = [tmp_path / name for name in ("base", "adapter", "output")]
    for directory in (base, adapter, out):
        directory.mkdir()
    (base / "preprocessor_config.json").write_text('{"sampling_rate":16000}')
    (out / "processor_config.json").write_text('{"feature_extractor":{"sampling_rate":48000}}')
    with pytest.raises(ValueError, match="sampling_rate"):
        merge.complete_processor_assets(out, adapter, base)


@pytest.mark.parametrize("damaged", ["tokenizer", "processor"])
def test_verification_detects_swapped_modality_roles_with_identical_vocab(tmp_path, monkeypatch, damaged):
    ids = {WAIT: 1, DIRECT: 2, "<audio>": 3, "<image>": 4}

    class Tokenizer:
        unk_token_id = 0
        audio_token = "<audio>"
        image_token = "<image>"
        all_special_tokens = list(ids)

        def get_vocab(self):
            return ids

        def convert_tokens_to_ids(self, token):
            return ids[token]

        def encode(self, token, add_special_tokens=False):
            return [ids[token]]

    expected = {"tokens": [WAIT, DIRECT], "protocol_token_ids": {WAIT: 1, DIRECT: 2},
                "vocabulary": ids, "special_token_contract": merge.special_token_contract(Tokenizer())}
    tokenizer, processor_tokenizer = Tokenizer(), Tokenizer()
    broken = tokenizer if damaged == "tokenizer" else processor_tokenizer
    broken.audio_token, broken.image_token = broken.image_token, broken.audio_token
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer),
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: SimpleNamespace(tokenizer=processor_tokenizer))))
    with pytest.raises(ValueError, match="audio_token|image_token"):
        merge.verify_export(tmp_path, expected)


def test_normalize_before_merge_and_publication_without_rewriting_inputs(tmp_path, monkeypatch):
    for name in ("CUDA_VISIBLE_DEVICES", "ENABLE_AUDIO_OUTPUT", "PYTHONUNBUFFERED", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    ids = {WAIT: 1, DIRECT: 2, SPK_CHANGE: 3}
    tokens = list(ids)
    token_file = tmp_path / "tokens.txt"
    token_file.write_text("\n".join(tokens), encoding="utf-8")
    base, adapter, output = (tmp_path / name for name in ("base", "adapter", "merged"))
    raw = json.dumps({
        "extra_special_tokens": tokens,
        "added_tokens_decoder": {i: {"content": t, "special": True} for t, i in ids.items()},
    })
    for source in (base, adapter):
        source.mkdir()
        (source / "tokenizer_config.json").write_text(raw, encoding="utf-8")
        (source / "weights.safetensors").write_bytes(b"unchanged weights")
    (base / "preprocessor_config.json").write_text('{"sampling_rate":16000}')
    (adapter / "video_preprocessor_config.json").write_text('{"patch_size":16}')

    def assert_portable(directory):
        config = merge.read_json(directory / "tokenizer_config.json")
        assert not isinstance(config.get("extra_special_tokens"), list)
        assert set(tokens) <= {v["content"] if isinstance(v, dict) else v for v in config["additional_special_tokens"]}

    inputs = []

    def preflight(view_base, view_adapter, token_path):
        inputs.extend([view_base, view_adapter])
        for view, source in zip(inputs, (base, adapter)):
            assert view != source
            assert_portable(view)
            assert os.path.samefile(view / "weights.safetensors", source / "weights.safetensors")
        return dict(protocol_token_ids=ids, expected_vocab_size=4, tokens=tokens,
                    training_base_recorded=str(base), manifest_metadata={})

    def swift_export(command, **kwargs):
        assert command[command.index("--model") + 1] == str(inputs[0])
        assert command[command.index("--adapters") + 1] == str(inputs[1])
        staging = Path(command[command.index("--output_dir") + 1])
        staging.mkdir()
        # The SWIFT serialization step is allowed to reintroduce its own HF schema.
        (staging / "tokenizer_config.json").write_text(raw, encoding="utf-8")
        (staging / "config.json").write_text(json.dumps({"_name_or_path": str(inputs[0])}), encoding="utf-8")

    def verify(staging, expected):
        assert_portable(staging)
        assert merge.read_json(staging / "config.json")["_name_or_path"] == str(base)
        assert merge.read_json(staging / "preprocessor_config.json") == {"sampling_rate": 16000}
        assert merge.read_json(staging / "video_preprocessor_config.json") == {"patch_size": 16}
        return dict(protocol_token_ids=ids, tokenizer_length=4)

    monkeypatch.setattr(merge.importlib.metadata, "version", lambda dist: "4.1.3")
    monkeypatch.setattr(merge, "preflight", preflight)
    monkeypatch.setattr(merge.subprocess, "run", swift_export)
    monkeypatch.setattr(merge, "verify_export", verify)
    assert merge.main([str(base), str(adapter), str(output), "--special-tokens", str(token_file)]) == 0
    assert_portable(output)
    assert merge.read_json(output / "omnicommit_tokens.json")["token_ids"] == ids
    assert merge.read_json(output / "omnicommit_merge_report.json")["tokenizer_metadata_normalized"] is True
    assert all(not directory.exists() for directory in inputs)
    for source in (base, adapter):
        assert (source / "tokenizer_config.json").read_text(encoding="utf-8") == raw
