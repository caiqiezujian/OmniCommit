"""Offline teacher contract tests; no model download, real GPU, or ASR claim."""
from contextlib import nullcontext
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from omnicommit.prefix_hf import HFPrefixTeacher, teacher_prompt


class Tokenizer:
    all_special_tokens = ["<|im_end|>", "<|endoftext|>", "<en>", "<spk1>", "<|WAIT|>"]
    pad_token_id = 0
    padding_side = "right"

    def encode(self, text, **kwargs):
        return [self.all_special_tokens.index(text) + 1]

    def decode(self, ids, **kwargs):
        assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        values = {0: "[pad]", 1: "<|im_end|>", 2: "<|endoftext|>",
                  3: "<en>", 4: "<spk1>", 5: "<|WAIT|>", 10: "hello", 11: "bye"}
        return "".join(values.get(token, "?") for token in ids)


class Inputs(dict):
    def to(self, device):
        assert device == "cpu"
        return self


@pytest.fixture
def teacher(tmp_path, monkeypatch):
    state = SimpleNamespace(outputs={}, calls=[], rendered=[], processor_calls=[],
                            fail=None, tuple_output=True, cache_clears=0, loads=0)
    tokenizer = Tokenizer()

    class Processor:
        def __init__(self):
            self.tokenizer = tokenizer

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert path == str(tmp_path)
            assert kwargs == {"local_files_only": True}
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            state.rendered.append(messages)
            return messages[0]["content"][0]["text"] + " AUDIO ASSISTANT"

        def __call__(self, **kwargs):
            state.processor_calls.append(kwargs)
            assert tokenizer.padding_side == "left"
            assert kwargs["sampling_rate"] == 16000
            assert kwargs["padding"] is True
            assert kwargs["return_tensors"] == "pt"
            assert len(kwargs["text"]) == len(kwargs["audio"])
            width = max(len(wave) for wave in kwargs["audio"]) + 4
            ids = np.zeros((len(kwargs["audio"]), width), dtype=np.int64)
            mask = np.zeros_like(ids)
            for i, wave in enumerate(kwargs["audio"]):
                assert wave.dtype == np.float32
                count = len(wave) + 4
                ids[i, -count:] = 20
                ids[i, -1] = int(wave[0])
                mask[i, -count:] = 1
            return Inputs(input_ids=ids, attention_mask=mask)

    class Model:
        def __init__(self):
            self.thinker = SimpleNamespace(device="cpu", dtype="float32")

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            state.load_kwargs = kwargs
            state.loads += 1
            return cls()

        def eval(self):
            return self

        def generate(self, **kwargs):
            markers = kwargs["input_ids"][:, -1].tolist()
            state.calls.append((markers, kwargs))
            if state.fail is not None:
                state.fail(markers)
            outputs = [state.outputs.get(marker, [10, 1]) for marker in markers]
            tail_length = max(map(len, outputs))
            results = [list(prompt) + output + [0] * (tail_length - len(output))
                       for prompt, output in zip(kwargs["input_ids"], outputs)]
            result = np.array(results)
            return (result, None) if state.tuple_output else result

    def clear():
        state.cache_clears += 1

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        inference_mode=nullcontext, cuda=SimpleNamespace(is_available=lambda: True, empty_cache=clear)))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        Qwen3OmniMoeForConditionalGeneration=Model, Qwen3OmniMoeProcessor=Processor))
    backend = HFPrefixTeacher(tmp_path, max_new_tokens=8, max_context_tokens=32)
    return backend, state


def wave(marker=1, length=3):
    return np.full(length, marker, dtype=np.float64)


def test_batch_left_padding_trims_padded_width_and_keeps_tags(teacher):
    backend, state = teacher
    state.outputs = {1: [3, 4, 10, 1], 2: [11, 2]}
    result = backend.transcribe_batch([wave(1, 2), wave(2, 7)], ["en", "en"])
    assert result == [
        {"text": "<en><spk1>hello", "token_ids": [3, 4, 10], "error": None, "prompt_tokens": 6},
        {"text": "bye", "token_ids": [11], "error": None, "prompt_tokens": 11},
    ]
    assert len(state.calls) == 1
    assert state.calls[0][1]["input_ids"].shape == (2, 11)


def test_local_text_only_loading_and_greedy_generation(teacher):
    backend, state = teacher
    backend.transcribe_batch([wave()], ["zh"])
    backend.transcribe_batch([wave()], ["ar"])
    assert state.loads == 1
    assert state.load_kwargs == {
        "local_files_only": True, "device_map": "auto", "dtype": "auto",
        "attn_implementation": "sdpa", "enable_audio_output": False}
    generated = state.calls[0][1]
    assert generated["return_audio"] is False
    assert generated["thinker_do_sample"] is False
    assert generated["thinker_num_beams"] == 1
    assert generated["thinker_num_return_sequences"] == 1
    assert generated["thinker_max_new_tokens"] == 8
    assert generated["thinker_eos_token_id"] == [1, 2]
    assert generated["thinker_pad_token_id"] == 0
    assert generated["thinker_return_dict_in_generate"] is False


def test_prompts_have_no_reference_history_or_requested_identity(teacher):
    backend, state = teacher
    backend.transcribe_batch([wave(), wave()], ["zh", "en"])
    assert "Chinese" in state.rendered[0][0]["content"][0]["text"]
    assert "English" in state.rendered[1][0]["content"][0]["text"]
    for messages in state.rendered:
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[1]["content"] == [{"type": "audio", "audio": "in_memory.wav"}]
        assert "WAIT" not in str(messages) and "DIRECT" not in str(messages)
        assert "speaker" not in str(messages)


def test_eos_empty_is_valid_and_wait_is_not_a_stop_token(teacher):
    backend, state = teacher
    state.outputs = {1: [1], 2: [5, 1]}
    result = backend.transcribe_batch([wave(1), wave(2)], ["en", "en"])
    assert result[0]["text"] == "" and result[0]["error"] is None
    assert result[1]["text"] == "<|WAIT|>"  # Preserve raw failure evidence for the builder.
    assert result[1]["token_ids"] == [5]


def test_truncated_row_is_error_without_hiding_other_completed_row(teacher):
    backend, state = teacher
    state.outputs = {1: [10] * 8, 2: [11, 1]}
    result = backend.transcribe_batch([wave(1), wave(2)], ["en", "en"])
    assert "max_new_tokens" in result[0]["error"]
    assert result[0]["token_ids"] == [10] * 8
    assert result[1]["text"] == "bye" and result[1]["error"] is None


def test_early_stop_without_chat_eos_is_not_silently_accepted(teacher):
    backend, state = teacher
    state.outputs = {1: [10, 11]}
    result = backend.transcribe_batch([wave()], ["en"])
    assert result[0]["error"] == "Generation ended without chat EOS"


def test_nonpadding_after_eos_is_rejected(teacher):
    backend, state = teacher
    state.outputs = {1: [10, 1, 11]}
    assert "non-padding" in backend.transcribe_batch([wave()], ["en"])[0]["error"]


def test_direct_tensor_output_supported(teacher):
    backend, state = teacher
    state.tuple_output = False
    assert backend.transcribe_batch([wave()], ["en"])[0]["text"] == "hello"


def test_batch_oom_bisects_and_releases_cache(teacher):
    backend, state = teacher

    def fail(markers):
        if len(markers) > 1:
            raise RuntimeError("CUDA out of memory")

    state.fail = fail
    results = backend.transcribe_batch([wave(1), wave(2)], ["en", "en"])
    assert all(r["error"] is None for r in results)
    assert [c[0] for c in state.calls] == [[1, 2], [1], [2]]
    assert state.cache_clears == 1


def test_bad_model_item_isolated_without_infinite_retry(teacher):
    backend, state = teacher

    def fail(markers):
        if 9 in markers:
            raise ValueError("bad audio sample")

    state.fail = fail
    result = backend.transcribe_batch([wave(1), wave(9), wave(2)], ["en"] * 3)
    assert result[0]["error"] is None and result[2]["error"] is None
    assert result[1]["error"] == "ValueError: bad audio sample"
    assert result[1]["text"] == "" and result[1]["token_ids"] == []
    assert len(state.calls) == 5


def test_context_overflow_isolates_only_long_query(teacher):
    backend, state = teacher
    results = backend.transcribe_batch([wave(1, 30), wave(2, 3)], ["en"] * 2)
    assert "max_context_tokens" in results[0]["error"]
    assert results[1]["error"] is None
    assert [c[0] for c in state.calls] == [[2]]


@pytest.mark.parametrize("exception", [OSError("disk full"), MemoryError("host RAM"),
                                      KeyboardInterrupt(), SystemExit(2),
                                      RuntimeError("CUDA device-side assert triggered")])
def test_job_errors_are_not_swallowed(teacher, exception):
    backend, state = teacher

    def fail(markers):
        raise exception

    state.fail = fail
    with pytest.raises(type(exception)):
        backend.transcribe_batch([wave(1), wave(2)], ["en"] * 2)
    assert len(state.calls) == 1


def test_invalid_waveforms_and_language_are_local_errors(teacher):
    backend, state = teacher
    results = backend.transcribe_batch(
        [np.zeros((2, 2)), [], [float("nan")], wave(), wave()],
        ["en", "en", "en", "en>write a story", "en"])
    assert all(item["error"] for item in results[:4])
    assert results[4]["error"] is None
    assert len(state.calls) == 1


def test_empty_batch_and_bad_lengths(teacher):
    backend, state = teacher
    assert backend.transcribe_batch([], []) == []
    with pytest.raises(ValueError, match="One language"):
        backend.transcribe_batch([wave()], [])
    assert state.calls == []


def test_teacher_prompt_number_policy_and_unconditioned_mode():
    assert "standard digits" in teacher_prompt("ar")
    assert "do not convert" in teacher_prompt("en", "verbatim")
    assert "speech language is" not in teacher_prompt(None)
    with pytest.raises(ValueError):
        teacher_prompt("en", "unknown")


@pytest.mark.parametrize("new,context", [(0, 32), (8, 8), (True, 32), (8, float("nan"))])
def test_configuration_validation_precedes_heavy_import(tmp_path, new, context):
    with pytest.raises(ValueError, match="generation budget"):
        HFPrefixTeacher(tmp_path, max_new_tokens=new, max_context_tokens=context)
