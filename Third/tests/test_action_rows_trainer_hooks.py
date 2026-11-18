import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from third_asr import mcore_action_rows as integration
from third_asr.action_rows_manifest import write_action_rows_manifest
from third_asr.protocol import DIRECT, WAIT


class Tokenizer:
    ids = {WAIT: 23, DIRECT: 24}

    def convert_tokens_to_ids(self, token):
        return self.ids[token]

    def encode(self, token, *, add_special_tokens):
        assert not add_special_tokens
        return [self.ids[token]]

    def convert_ids_to_tokens(self, value):
        return {i: t for t, i in self.ids.items()}[value]


@pytest.fixture
def trainer(tmp_path, monkeypatch):
    path = tmp_path / "tokens"
    path.write_text(f"{WAIT}\n{DIRECT}\n", encoding="utf-8")
    monkeypatch.setenv("OMNICOMMIT_ACTION_TOKENS_FILE", str(path))
    return SimpleNamespace(
        args=SimpleNamespace(no_save_optim=True, save_safetensors=True, merge_lora=False,
                             train_type="lora", new_special_tokens=[WAIT, DIRECT]),
        template=SimpleNamespace(tokenizer=Tokenizer()),
        bridge=SimpleNamespace(save_weights=lambda *args, **kwargs: None),
    )


def test_v3_inserts_row_params_after_peft_before_optimizer(trainer, monkeypatch):
    events = []

    class LegacyTrainer:
        def setup_model_and_optimizer(self, *, fail=False):
            model = object()
            assert namespace.prepare_mcore_model(model) == "peft"
            events.append("optimizer")
            if fail:
                raise RuntimeError("setup failed")
            return model

    def native_prepare(model):
        events.append("peft")
        return "peft"

    monkeypatch.setattr(integration, "attach_action_rows", lambda *args: events.append("rows"))
    namespace = SimpleNamespace(BaseMegatronTrainer=LegacyTrainer, prepare_mcore_model=native_prepare)
    integration._install_trainer_hooks(namespace)
    instance = LegacyTrainer()
    instance.__dict__.update(vars(trainer))
    for failure in (False, True):
        if failure:
            with pytest.raises(RuntimeError, match="setup failed"):
                instance.setup_model_and_optimizer(fail=True)
        else:
            instance.setup_model_and_optimizer()
        assert namespace.prepare_mcore_model is native_prepare
        assert events == ["peft", "rows", "optimizer"]
        events.clear()


def test_v4_inserts_rows_after_peft_and_before_wrapping(trainer, monkeypatch):
    events = []

    class NativeTrainer:
        def _prepare_peft_model(self, models):
            events.append("peft")
            return ["native-adapter"]

    def attach(model, token_ids):
        assert token_ids == Tokenizer.ids
        events.append(model)

    monkeypatch.setattr(integration, "attach_action_rows", attach)
    namespace = SimpleNamespace(BaseMegatronTrainer=NativeTrainer)
    integration._install_trainer_hooks(namespace)
    instance = NativeTrainer()
    instance.__dict__.update(vars(trainer))
    assert instance._prepare_peft_model(["model"]) == ["native-adapter"]
    assert events == ["peft", "model"]


def test_action_rows_restore_requires_matching_hf_adapter(trainer, tmp_path):
    trainer.args.adapters = [str(tmp_path)]
    with pytest.raises(ValueError, match="manifest"):
        integration._resolve_ids(trainer)
    write_action_rows_manifest(tmp_path, {WAIT: 34, DIRECT: 35})
    with pytest.raises(ValueError, match="IDs"):
        integration._resolve_ids(trainer)
    write_action_rows_manifest(tmp_path, Tokenizer.ids)
    assert integration._resolve_ids(trainer) == Tokenizer.ids
    trainer.args.load = "distributed-base"
    with pytest.raises(ValueError, match="HF base"):
        integration._resolve_ids(trainer)


@pytest.mark.parametrize("field, value", [
    ("tensor_model_parallel_size", 2),
    ("context_parallel_size", 2),
    ("pipeline_model_parallel_size", 2),
    ("no_save_optim", False),
    ("save_safetensors", False),
    ("merge_lora", True),
    ("adapter_load", "native-checkpoint"),
    ("new_special_tokens", [DIRECT, WAIT]),
])
def test_rejects_unsupported_variants(trainer, field, value):
    setattr(trainer.args, field, value)
    with pytest.raises(ValueError):
        integration._resolve_ids(trainer)
