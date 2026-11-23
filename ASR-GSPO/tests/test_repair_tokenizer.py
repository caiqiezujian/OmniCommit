"""Real fast-tokenizer roundtrips; tiny header-readable weights, no GPU."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import repair_tokenizer as repair
from fourth_asr.service_backend import _metadata


@pytest.fixture
def model(tmp_path, monkeypatch):
    try:
        from transformers import PreTrainedTokenizerFast
        from tokenizers import AddedToken, Tokenizer
        from tokenizers.models import WordLevel
        import numpy as np
        from safetensors.numpy import save_file
    except ImportError as exc:
        pytest.skip(f"Real tokenizer test requires a compatible Transformers environment: {exc}")
    directory = tmp_path / "model"
    directory.mkdir()
    tokens = ["<|WAIT|>", "<|DIRECT|>", "<|SPK_CHANGE|>"]
    backend = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("<audio>", lstrip=True, normalized=False, special=True)]})
    tokenizer.add_tokens([AddedToken(token, normalized=False) for token in tokens])
    tokenizer.chat_template = "{{ messages }}"
    tokenizer.save_pretrained(directory, save_jinja_files=False)
    manifest = dict(_metadata("speaker"), tokens=tokens,
                    token_ids={token: tokenizer.get_vocab()[token] for token in tokens},
                    tokenizer_length=len(tokenizer))
    (directory / "omnicommit_tokens.json").write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "config.json").write_text('{"model_type":"qwen3_omni_moe"}', encoding="utf-8")
    save_file({"thinker.model.embed_tokens.weight": np.zeros((len(tokenizer), 2), dtype=np.float32),
               "thinker.lm_head.weight": np.zeros((len(tokenizer), 2), dtype=np.float32),
               "thinker.audio_tower.test.weight": np.zeros((1, 1), dtype=np.float32)},
              str(directory / "model.safetensors"))
    (directory / "preprocessor_config.json").write_text('{"sampling_rate":16000}', encoding="utf-8")
    # Exercise the processor-tokenizer validation path with the same real
    # serialized tokenizer; constructing the multimodal processor is tested by
    # the production CLI, and would require unrelated image/audio dependencies.
    monkeypatch.setattr(repair, "load_processor", lambda p: SimpleNamespace(tokenizer=repair.load_tokenizer(p)))
    return directory


def files(path):
    return {p.relative_to(path).as_posix(): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def test_diagnose_readonly_and_real_roundtrip_preserves_every_id_and_asset(model):
    before = files(model)
    tokenizer = repair.load_tokenizer(model)
    vocab = tokenizer.get_vocab()
    result = repair.repair(model)
    assert result["status"] == "registration_repair_needed"
    assert result["missing_special_registration"] == ["<|WAIT|>", "<|DIRECT|>", "<|SPK_CHANGE|>"]
    assert files(model) == before
    result = repair.repair(model, apply=True)
    assert result["status"] == "repaired"
    assert all(item["backend"]["special"] for item in result["tokens"])
    fixed = repair.load_tokenizer(model)
    assert fixed.get_vocab() == vocab and len(fixed) == len(tokenizer)
    assert set(tokenizer.all_special_tokens).issubset(fixed.all_special_tokens)
    assert fixed.added_tokens_decoder[vocab["<audio>"]].lstrip
    assert fixed.chat_template == tokenizer.chat_template
    for name, data in before.items():
        if name not in repair.TOKENIZER_FILES:
            assert (model / name).read_bytes() == data
    assert set(result["changed_files"]).issubset(repair.TOKENIZER_FILES)
    assert not list(model.parent.glob(".tokenizer-registration-*"))
    now = files(model)
    assert repair.repair(model, apply=True)["status"] == "already_valid"
    assert files(model) == now


@pytest.mark.parametrize("kind", ["length", "id", "missing", "third"])
def test_refuses_identity_errors_without_writes(model, kind):
    path = model / "omnicommit_tokens.json"
    manifest = json.loads(path.read_bytes())
    if kind == "length":
        manifest["tokenizer_length"] += 1
    elif kind == "id":
        manifest["token_ids"]["<|WAIT|>"] = 1
    elif kind == "missing":
        manifest["tokens"][0] = "<|NEW_WAIT|>"
        manifest["token_ids"]["<|NEW_WAIT|>"] = manifest["token_ids"].pop("<|WAIT|>")
    else:
        manifest.update(_metadata("explicit"))
        manifest["tokens"].remove("<|SPK_CHANGE|>")
        manifest["token_ids"].pop("<|SPK_CHANGE|>")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    before = files(model)
    with pytest.raises(ValueError):
        repair.repair(model, apply=True)
    assert files(model) == before


def test_weight_header_failure_precedes_registration(model, monkeypatch):
    before = files(model)
    monkeypatch.setattr(repair, "verify_weights", lambda *a: (_ for _ in ()).throw(ValueError("weights too small")))
    monkeypatch.setattr(repair, "_register", lambda *a: pytest.fail("must not mutate before weight check"))
    with pytest.raises(ValueError, match="weights too small"):
        repair.repair(model, apply=True)
    assert files(model) == before


def test_registration_cannot_grow_any_vocabulary_rows(model, monkeypatch):
    before = files(model)
    register = repair._register
    def bad_register(tokenizer, manifest):
        register(tokenizer, manifest)
        tokenizer.add_tokens(["new_unrequested_token"])
    monkeypatch.setattr(repair, "_register", bad_register)
    with pytest.raises(ValueError, match="length differs|vocabulary"):
        repair.repair(model, apply=True)
    assert files(model) == before


def test_non_atomic_existing_token_is_refused(model, monkeypatch):
    before = files(model)
    loader = repair.load_tokenizer
    def load(path):
        tokenizer = loader(path)
        encode = tokenizer.encode
        tokenizer.encode = lambda text, **kwargs: [0, 1] if text == "<|WAIT|>" else encode(text, **kwargs)
        return tokenizer
    monkeypatch.setattr(repair, "load_tokenizer", load)
    with pytest.raises(ValueError, match="non-atomic"):
        repair.repair(model, apply=True)
    assert files(model) == before


def test_publish_failure_rolls_back_bytes(model, monkeypatch):
    before = files(model)
    writer = repair.atomic_write
    count = 0
    def fail_once(path, data):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("test publication failure")
        writer(path, data)
    monkeypatch.setattr(repair, "atomic_write", fail_once)
    with pytest.raises(RuntimeError, match="original tokenizer files restored"):
        repair.repair(model, apply=True)
    assert files(model) == before


def test_final_processor_reload_failure_rolls_back(model, monkeypatch):
    before = files(model)
    def processor(path):
        if Path(path).resolve() == model.resolve():
            raise ValueError("processor reload rejected")
        return SimpleNamespace(tokenizer=repair.load_tokenizer(path))
    monkeypatch.setattr(repair, "load_processor", processor)
    with pytest.raises(RuntimeError, match="original tokenizer files restored"):
        repair.repair(model, apply=True)
    assert files(model) == before


def test_backend_only_native_special_is_retained(model):
    config_path = model / "tokenizer_config.json"
    config = json.loads(config_path.read_bytes())
    config["additional_special_tokens"] = []
    config.pop("extra_special_tokens", None)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    mapping_path = model / "special_tokens_map.json"
    if mapping_path.is_file():
        mapping = json.loads(mapping_path.read_bytes())
        mapping.pop("additional_special_tokens", None)
        mapping.pop("extra_special_tokens", None)
        mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    original = repair.load_tokenizer(model)
    assert "<audio>" not in original.all_special_tokens
    assert original.added_tokens_decoder[original.get_vocab()["<audio>"]].special
    repair.repair(model, apply=True)
    fixed = repair.load_tokenizer(model)
    assert "<audio>" in fixed.all_special_tokens
    assert fixed.added_tokens_decoder[fixed.get_vocab()["<audio>"]].lstrip


def _use_list_style_config(model, values):
    config_path = model / "tokenizer_config.json"
    config = json.loads(config_path.read_bytes())
    config.pop("additional_special_tokens", None)
    config["extra_special_tokens"] = values
    config_path.write_text(json.dumps(config), encoding="utf-8")


def test_format_only_repair_precedes_every_hf_load_and_is_idempotent(model, monkeypatch):
    repair.repair(model, apply=True)
    original_vocab = repair.load_tokenizer(model).get_vocab()
    manifest = json.loads((model / "omnicommit_tokens.json").read_bytes())
    _use_list_style_config(model, ["<audio>", *manifest["tokens"]])
    before = files(model)
    loader = repair.load_tokenizer

    def strict_loader(path):
        config = json.loads((path / "tokenizer_config.json").read_bytes())
        assert not isinstance(config.get("extra_special_tokens"), list)
        return loader(path)

    monkeypatch.setattr(repair, "load_tokenizer", strict_loader)
    diagnose = repair.repair(model)
    assert diagnose["status"] == "format_repair_needed" and files(model) == before
    report = repair.repair(model, apply=True)
    assert report["status"] == "repaired" and report["format_repair_needed_on_input"]
    assert "tokenizer_config.json" in report["changed_files"]
    fixed = repair.load_tokenizer(model)
    assert fixed.get_vocab() == original_vocab
    for token in manifest["tokens"]:
        assert token in fixed.all_special_tokens
    for name in ("model.safetensors", "omnicommit_tokens.json", "tokenizer.json"):
        assert (model / name).read_bytes() == before[name]
    after = files(model)
    assert repair.repair(model, apply=True)["status"] == "already_valid"
    assert files(model) == after


def test_schema_and_registration_repaired_together_and_bad_weights_leave_originals(model, monkeypatch):
    _use_list_style_config(model, ["<audio>"])
    original = files(model)
    with monkeypatch.context() as scoped:
        def bad_weights(*args):
            raise ValueError("bad weight headers")

        scoped.setattr(repair, "verify_weights", bad_weights)
        with pytest.raises(ValueError, match="bad weight headers"):
            repair.repair(model, apply=True)
        assert files(model) == original
    result = repair.repair(model, apply=True)
    assert result["status"] == "repaired" and result["format_repair_needed_on_input"]
    assert all(item["registered_special"] for item in result["tokens"])
    assert (model / "model.safetensors").read_bytes() == original["model.safetensors"]
    assert repair.repair(model, apply=True)["status"] == "already_valid"


def test_legacy_registration_with_named_modality_and_anonymous_actions(model):
    path = model / "tokenizer_config.json"
    config = json.loads(path.read_bytes())
    config["additional_special_tokens"] = ["<audio>"]
    config["extra_special_tokens"] = {"audio_token": "<audio>"}
    path.write_text(json.dumps(config), encoding="utf-8")
    manifest = json.loads((model / "omnicommit_tokens.json").read_bytes())
    originals = files(model)
    result = repair.repair(model, apply=True)
    assert result["status"] == "repaired"
    tokenizer = repair.load_tokenizer(model)
    assert str(tokenizer.audio_token) == "<audio>"
    assert tokenizer.audio_token_id == tokenizer.get_vocab()["<audio>"]
    for token, idx in manifest["token_ids"].items():
        assert token in tokenizer.all_special_tokens and tokenizer.encode(token, add_special_tokens=False) == [idx]
    assert (model / "model.safetensors").read_bytes() == originals["model.safetensors"]
    assert repair.repair(model, apply=True)["status"] == "already_valid"


def test_extra_special_tokens_api_compatibility(monkeypatch):
    # Transformers 5 renamed the list and replacement argument. Exercise the
    # API dispatch independently of the installed Transformers 4 runtime.
    pytest.importorskip("tokenizers")
    calls = []
    class Tokenizer:
        SPECIAL_TOKENS_ATTRIBUTES = ["extra_special_tokens"]
        all_special_tokens = ["<audio>"]
        added_tokens_decoder = {}
        def add_special_tokens(self, data, replace_extra_special_tokens=True):
            calls.append((data, replace_extra_special_tokens))
    repair._register(Tokenizer(), {"tokens": ["<|WAIT|>"], "token_ids": {"<|WAIT|>": 1}})
    assert calls[0][1] is True
    assert [str(t) for t in calls[0][0]["extra_special_tokens"]] == ["<audio>", "<|WAIT|>"]


def test_v5_snapshot_without_removed_extended_properties():
    class Tokenizer:
        added_tokens_decoder = {}
        all_special_tokens = ["<audio>"]
        special_tokens_map = {"audio_token": "<audio>", "extra_special_tokens": ["<audio>"]}
        def __len__(self):
            return 1
        def get_vocab(self):
            return {"<audio>": 0}
    state = repair._snapshot(Tokenizer())
    assert state["roles"] == {"audio_token": {"content": "<audio>"}}
