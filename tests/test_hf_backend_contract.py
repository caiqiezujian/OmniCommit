"""HF boundary contract checks with stubs; no model loading or GPU claims."""
from contextlib import nullcontext
import json
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from omnicommit.inference_backend import HFBackend
from omnicommit.streaming import ContextBudgetError


TOKENS = ["<|WAIT|>", "<|DIRECT|>", "<en>", "<spk1>", "<|im_end|>", "<|endoftext|>"]


class Tokenizer:
    all_special_tokens = TOKENS

    def __len__(self):
        return 100

    def encode(self, value, **kwargs):
        return [TOKENS.index(value) + 10]

    def decode(self, values, **kwargs):
        assert kwargs["skip_special_tokens"] is False
        return "".join(TOKENS[value - 10] if 10 <= value < 16 else "text" for value in values)


class Inputs(dict):
    def to(self, device):
        return self


@pytest.fixture
def backend(tmp_path, monkeypatch):
    state = SimpleNamespace(generated=[10], prompt_length=12)
    tokenizer = Tokenizer()

    class Processor:
        def __init__(self):
            self.tokenizer = tokenizer

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs["local_files_only"] is True
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            state.last_messages = messages
            return "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n"

        def __call__(self, **kwargs):
            state.processor_kwargs = kwargs
            return Inputs(input_ids=np.zeros((1, state.prompt_length), dtype=np.int64))

    class Model:
        device, dtype = "cpu", "float32"

        def __init__(self):
            self.thinker = SimpleNamespace(
                get_input_embeddings=lambda: SimpleNamespace(weight=SimpleNamespace(shape=(128, 4))),
                get_output_embeddings=lambda: SimpleNamespace(weight=SimpleNamespace(shape=(128, 4))))

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            state.load_kwargs = kwargs
            return cls()

        def eval(self):
            return self

        def generate(self, **kwargs):
            state.generation_kwargs = kwargs
            sequence = [0] * state.prompt_length + state.generated
            return np.array([sequence]), None

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        Qwen3OmniMoeForConditionalGeneration=Model, Qwen3OmniMoeProcessor=Processor))
    manifest = {"schema_version": 1, "tokens": TOKENS[:4],
                "token_ids": {token: i + 10 for i, token in enumerate(TOKENS[:4])}, "tokenizer_length": 100}
    (tmp_path / "omnicommit_tokens.json").write_text(json.dumps(manifest), encoding="utf-8")
    return HFBackend(tmp_path, max_context_tokens=32, max_new_tokens=8), state


def test_only_wait_stops_generation_and_survives_decoding(backend):
    model, state = backend
    output = model.generate([{"role": "user", "content": "<audio>"}], [np.zeros(8000)])
    assert output["text"] == "<|WAIT|>"
    assert state.load_kwargs["enable_audio_output"] is False
    assert state.generation_kwargs["return_audio"] is False
    assert state.generation_kwargs["thinker_do_sample"] is False
    assert state.generation_kwargs["thinker_return_dict_in_generate"] is False
    assert 10 in state.generation_kwargs["thinker_eos_token_id"]
    assert 11 not in state.generation_kwargs["thinker_eos_token_id"]
    assert state.processor_kwargs["sampling_rate"] == 16000


def test_direct_keeps_tags_and_removes_only_chat_eos(backend):
    model, state = backend
    state.generated = [11, 12, 13, 40, 14]
    output = model.generate([{"role": "user", "content": "<audio>"}], [np.zeros(8000)])
    assert output["text"] == "<|DIRECT|><en><spk1>text"


def test_context_counts_expanded_processor_tokens_and_never_truncates(backend):
    model, state = backend
    state.prompt_length = 25
    with pytest.raises(ContextBudgetError):
        model.generate([{"role": "user", "content": "<audio>"}], [np.zeros(8000)])
    assert not hasattr(state, "generation_kwargs")


def test_max_generation_partial_output_is_rejected(backend):
    model, state = backend
    state.generated = [11, 12, 13, 40, 40, 40, 40, 40]
    with pytest.raises(RuntimeError, match="max_new_tokens"):
        model.generate([{"role": "user", "content": "<audio>"}], [np.zeros(8000)])


def test_text_only_eos_does_not_add_audio(backend):
    model, state = backend
    model.generate([{"role": "user", "content": "<audio>"},
                    {"role": "assistant", "content": "<|WAIT|>"},
                    {"role": "user", "content": "<end_of_audio>"}], [np.zeros(8000)])
    assert len(state.processor_kwargs["audio"]) == 1
    assert state.last_messages[-1]["content"] == [{"type": "text", "text": "<end_of_audio>"}]


def test_audio_count_mismatch_rejected_before_processor(backend):
    model, state = backend
    with pytest.raises(ValueError, match="placeholders"):
        model.generate([{"role": "user", "content": "<audio><audio>"}], [np.zeros(8000)])
