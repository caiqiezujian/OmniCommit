"""Local Qwen3-Omni backend; reuses exported tokenizer, never adds new tokens.

This correctness baseline re-encodes bounded multimodal context on each call.
It is not native encoder streaming and does not claim KV/feature-cache reuse.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from .streaming import ContextBudgetError, WAIT


def validate_token_manifest(tokenizer, manifest):
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported OmniCommit tokenizer manifest")
    tokens = manifest.get("tokens")
    ids = manifest.get("token_ids")
    if not isinstance(tokens, list) or len(tokens) != len(set(tokens)) or not isinstance(ids, dict):
        raise ValueError("Malformed token manifest")
    if not {"<|WAIT|>", "<|DIRECT|>"}.issubset(tokens) or set(ids) != set(tokens):
        raise ValueError("Manifest missing action tokens or IDs")
    if manifest.get("tokenizer_length") != len(tokenizer):
        raise ValueError("Exported tokenizer length differs from manifest")
    seen = set()
    for token in tokens:
        actual = tokenizer.encode(token, add_special_tokens=False)
        if actual != [ids[token]] or ids[token] in seen:
            raise ValueError("Exported tokenizer ID mismatch for " + token)
        if token not in tokenizer.all_special_tokens:
            raise ValueError("Protocol token is not registered special: " + token)
        seen.add(ids[token])
    return set(tokens)


def multimodal_messages(messages):
    """Preserve trained wrappers; replace only explicit audio placeholders."""
    result = []
    for message in messages:
        parts = message["content"].split("<audio>")
        content = []
        for index, part in enumerate(parts):
            if index:
                # tokenize=False below, so this sentinel path is never opened.
                content.append({"type": "audio", "audio": "in_memory.wav"})
            if part:
                content.append({"type": "text", "text": part})
        result.append({"role": message["role"], "content": content})
    return result


class HFBackend:
    def __init__(self, model_path, *, max_context_tokens=4096, max_new_tokens=256,
                 device_map="auto", dtype="auto", attn_implementation="sdpa"):
        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        path = Path(model_path).resolve(strict=True)
        manifest = json.loads((path / "omnicommit_tokens.json").read_text(encoding="utf-8"))
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(str(path), local_files_only=True)
        self.tokenizer = self.processor.tokenizer
        self.allowed_tokens = validate_token_manifest(self.tokenizer, manifest)
        probe = self.processor.apply_chat_template(multimodal_messages([
            {"role": "user", "content": "<audio>"}]), tokenize=False, add_generation_prompt=True)
        audio_body = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", probe, re.S)
        if (not audio_body or audio_body.group(1) != "<|audio_start|><|audio_pad|><|audio_end|>"
                or not probe.endswith("<|im_start|>assistant\n")):
            raise ValueError("Exported processor chat audio wrapper differs from inspected FLEX/SWIFT SFT template")
        if max_new_tokens <= 0 or max_context_tokens <= max_new_tokens:
            raise ValueError("Need positive generation budget smaller than context limit")
        self.max_context_tokens, self.max_new_tokens = max_context_tokens, max_new_tokens
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            str(path), local_files_only=True, device_map=device_map, dtype=dtype,
            attn_implementation=attn_implementation, enable_audio_output=False).eval()
        self.torch = torch
        input_rows = self.model.thinker.get_input_embeddings().weight.shape[0]
        output_rows = self.model.thinker.get_output_embeddings().weight.shape[0]
        if max(manifest["token_ids"].values()) >= min(input_rows, output_rows):
            raise ValueError("Model embedding/output weights do not cover the trained token IDs")
        self.wait_id = manifest["token_ids"][WAIT]
        self.stop_ids = []
        for name in ("<|im_end|>", "<|endoftext|>"):
            encoded = self.tokenizer.encode(name, add_special_tokens=False)
            if len(encoded) == 1 and name in self.tokenizer.all_special_tokens:
                self.stop_ids.extend(encoded)
        if not self.stop_ids:
            raise ValueError("Missing trained chat EOS tokens")

    def generate(self, messages, audios):
        if sum(m["content"].count("<audio>") for m in messages) != len(audios):
            raise ValueError("Audio placeholders and waveforms disagree")
        prompt = self.processor.apply_chat_template(multimodal_messages(messages),
                    tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=prompt, audio=audios or None, sampling_rate=16000,
                                return_tensors="pt", padding=True)
        count = int(inputs["input_ids"].shape[-1])
        if count + self.max_new_tokens > self.max_context_tokens:
            raise ContextBudgetError("Encoded context plus generation reserve exceeds token budget; no truncation performed")
        inputs = inputs.to(self.model.device)
        # Float32 waveform features must follow model dtype; IDs stay integer.
        for key, value in inputs.items():
            if hasattr(value, "is_floating_point") and value.is_floating_point():
                inputs[key] = value.to(self.model.dtype)
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, return_audio=False,
                thinker_do_sample=False, thinker_max_new_tokens=self.max_new_tokens,
                thinker_eos_token_id=self.stop_ids + [self.wait_id],
                thinker_return_dict_in_generate=False)
        if isinstance(ids, tuple):
            ids = ids[0]
        generated = ids[0, count:].tolist()
        # A max-length partial transcript must not be committed as complete.
        if len(generated) >= self.max_new_tokens and generated[-1] not in self.stop_ids + [self.wait_id]:
            raise RuntimeError("Generation hit max_new_tokens before completing a reply")
        while generated and generated[-1] in self.stop_ids:
            generated.pop()
        text = self.tokenizer.decode(generated, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)
        return {"text": text, "prompt_tokens": count}
