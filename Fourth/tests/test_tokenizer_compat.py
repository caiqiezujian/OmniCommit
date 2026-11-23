"""Run this same fixture with real Transformers 4.x and 5.x, without a model or a GPU."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_asr.tokenizer_compat import normalize_tokenizer_directory


@pytest.mark.parametrize("old_style", [False, True])
def test_qwen_registry_and_multimodal_roles_roundtrip(tmp_path, old_style):
    from tokenizers import AddedToken, Tokenizer
    from tokenizers.models import BPE
    from transformers import AutoTokenizer

    roles = {"audio_token": "<|audio_pad|>", "image_token": "<|image_pad|>"}
    extra = ["<|WAIT|>", "<|DIRECT|>", "<|SPK_CHANGE|>"]
    base = Tokenizer(BPE({"x": 0}, []))
    for word in ["<|endoftext|>", *roles.values(), *extra, "<native>"]:
        # Match the flags of the released Qwen3-Omni specials.
        base.add_special_tokens([AddedToken(word, normalized=False, special=True)])
    base.save(str(tmp_path / "tokenizer.json"))
    native_bytes = (tmp_path / "tokenizer.json").read_bytes()
    path = tmp_path / "tokenizer_config.json"
    template = "<|im_start|>{{ messages[0].role }}\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n{% if add_generation_prompt %}<|im_start|>assistant\n{{ '' }}{% endif %}"
    config = {"tokenizer_class": "Qwen2Tokenizer", "eos_token": "<|endoftext|>", **roles,
              "extra_special_tokens": extra}
    legacy = tmp_path / "chat_template.json"
    if old_style:
        config.update(extra_special_tokens=roles, additional_special_tokens=extra)
        legacy.write_text(json.dumps({"chat_template": template}), encoding="utf-8")
    else:
        config["chat_template"] = template
        legacy.write_text('{"chat_template":"an older processor template"}', encoding="utf-8")
    original_legacy = legacy.read_bytes()
    path.write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    before = path.read_bytes()
    assert normalize_tokenizer_directory(tmp_path, check_only=True)
    assert path.read_bytes() == before
    assert normalize_tokenizer_directory(tmp_path)
    normalized = path.read_bytes()
    assert not normalize_tokenizer_directory(tmp_path)
    assert path.read_bytes() == normalized
    assert legacy.read_bytes() == original_legacy
    assert (tmp_path / "tokenizer.json").read_bytes() == native_bytes

    def verify(directory):
        tok = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        assert tok.get_vocab() == base.get_vocab()
        assert tok.chat_template == template
        messages = [{"role": "user", "content": [{"type": "audio", "audio": "in_memory.wav"}]}]
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        assert rendered == "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n"
        for key, token in roles.items():
            assert getattr(tok, key) == token
            assert getattr(tok, key + "_id") == base.token_to_id(token)
        for token in [*roles.values(), *extra, "<native>"]:
            assert token in tok.all_special_tokens
            assert tok.encode(token, add_special_tokens=False) == [base.token_to_id(token)]
        assert tok.added_tokens_decoder[base.token_to_id("<native>")].special
        return tok

    tok = verify(tmp_path)
    saved = tmp_path / "resaved"
    tok.save_pretrained(saved)
    (saved / "config.json").write_bytes((tmp_path / "config.json").read_bytes())
    normalize_tokenizer_directory(saved)
    verify(saved)


def test_conflicting_roles_refused_before_any_write(tmp_path):
    config = tmp_path / "tokenizer_config.json"
    original = b'{"audio_token":"a","extra_special_tokens":{"audio_token":"b"}}'
    config.write_bytes(original)
    with pytest.raises(ValueError, match="audio_token"):
        normalize_tokenizer_directory(tmp_path)
    assert config.read_bytes() == original
