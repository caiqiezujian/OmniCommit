"""Optional FLEX-derived vLLM 0.11.1 inference with native bounded caches.

Never graft old KV tensors onto a cropped prompt. Native caching decides which
prefixes are reusable; stable audio content IDs allow processor-cache hits.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .audio_token_counter import AudioTokenCounter
from .inference_backend import multimodal_messages, validate_token_manifest
from .streaming import ContextBudgetError, WAIT

AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"


class VLLMBackend:
    def __init__(self, model_path, *, max_context_tokens=4096, max_new_tokens=256,
                 tensor_parallel_size=1, gpu_memory_utilization=.85,
                 max_audio_turns=80, max_audio_seconds=5.0,
                 mm_processor_cache_gb=1.0, image_slots=1, dtype="auto",
                 manifest_validator=validate_token_manifest):
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        import vllm
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        from vllm.inputs.preprocess import InputPreprocessor
        from transformers import AutoTokenizer
        if vllm.__version__ != "0.11.1":
            raise RuntimeError("This optional backend targets vLLM 0.11.1; use the existing FLEX inference environment")
        if max_new_tokens <= 0 or max_context_tokens <= max_new_tokens:
            raise ValueError("Invalid context/generation budgets")
        path = Path(model_path).resolve(strict=True)
        manifest = json.loads((path/"omnicommit_tokens.json").read_text(encoding="utf-8"))
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.allowed_tokens = manifest_validator(self.tokenizer, manifest)
        self.TokensPrompt = TokensPrompt
        self.max_context_tokens = max_context_tokens
        self.max_new_tokens = max_new_tokens
        self.max_audio_turns = max_audio_turns
        self.max_audio_seconds = max_audio_seconds
        self.wait_id = manifest["token_ids"][WAIT]
        self.stop_ids = []
        for token in ("<|im_end|>", "<|endoftext|>"):
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(ids) != 1 or token not in self.tokenizer.all_special_tokens:
                raise ValueError("Exported tokenizer lacks trained chat EOS tokens")
            self.stop_ids.extend(ids)
        self.audio_turn = self._encode("<|im_start|>user\n"+AUDIO_PLACEHOLDER+"<|im_end|>\n")
        rendered = self.tokenizer.apply_chat_template(multimodal_messages([
            {"role":"user","content":"<audio>"}]), tokenize=False, add_generation_prompt=True)
        if ("<|im_start|>user\n"+AUDIO_PLACEHOLDER+"<|im_end|>" not in rendered
                or not rendered.endswith("<|im_start|>assistant\n")):
            raise ValueError("Tokenizer audio wrapper differs from FLEX/SWIFT SFT layout")
        self.llm = LLM(model=str(path), dtype=dtype, tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_context_tokens,
            max_num_seqs=1, enforce_eager=True, enable_prefix_caching=True,
            limit_mm_per_prompt={"audio":{"count":max_audio_turns,
                "length":round(max_audio_seconds*16000)},"image":image_slots,"video":0},
            mm_processor_cache_gb=mm_processor_cache_gb, seed=0)
        # Counting must not mutate the live engine's mirrored MM sender cache.
        self.counter = InputPreprocessor(self.llm.llm_engine.model_config,
                                          self.tokenizer, mm_processor_cache=None)
        self.audio_counter = AudioTokenCounter.for_preprocessor(self.counter,
            vllm_version=vllm.__version__, max_entries=64)
        self.sampling = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1,
            max_tokens=max_new_tokens, stop_token_ids=self.stop_ids+[self.wait_id],
            skip_special_tokens=False, include_stop_str_in_output=True)

    def _encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _exact_audio_expansion(self, audio):
        probe = self.TokensPrompt(prompt_token_ids=self.audio_turn,
                                 multi_modal_data={"audio":[audio]})
        expanded = self.counter.preprocess(probe)["prompt_token_ids"]
        pad = self.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        if self.audio_turn.count(pad) != 1 or expanded.count(pad) < 1:
            raise RuntimeError("Unexpected audio placeholder expansion")
        if [i for i in expanded if i != pad] != [i for i in self.audio_turn if i != pad]:
            raise RuntimeError("Audio preprocessor modified non-audio tokens")
        return len(expanded)-len(self.audio_turn)

    def generate(self, messages, audios):
        if sum(m["content"].count("<audio>") for m in messages) != len(audios):
            raise ValueError("Audio placeholders and waveform count disagree")
        if len(audios)>self.max_audio_turns or any(len(a)>self.max_audio_seconds*16000 for a in audios):
            raise ContextBudgetError("Audio count/length exceeds configured vLLM capacity")
        prompt = self.tokenizer.apply_chat_template(multimodal_messages(messages),
                            tokenize=False, add_generation_prompt=True)
        ids = self._encode(prompt)
        extra = sum(self.audio_counter.count(a,self._exact_audio_expansion)[0] for a in audios)
        expected = len(ids)+extra
        if expected+self.max_new_tokens>self.max_context_tokens:
            raise ContextBudgetError("Expanded prompt plus generation reserve exceeds context budget")
        uuids = ["omnicommit-16k-"+hashlib.sha256(np.asarray(a,dtype="<f4").tobytes()).hexdigest()
                 for a in audios]
        request = self.TokensPrompt(prompt_token_ids=ids, multi_modal_data={"audio":audios},
                                   multi_modal_uuids={"audio":uuids})
        outputs = self.llm.generate([request],self.sampling,use_tqdm=False)
        if len(outputs)!=1 or len(outputs[0].outputs)!=1:
            raise RuntimeError("Unexpected vLLM output count")
        output=outputs[0];completion=output.outputs[0]
        actual_ids=getattr(output,"prompt_token_ids",None)
        if actual_ids is None or len(actual_ids)!=expected:
            self.audio_counter.disable("actual expansion mismatch")
            raise RuntimeError("Actual multimodal prompt expansion differs from audited counter")
        tokens=list(completion.token_ids)
        if completion.finish_reason=="length":
            raise RuntimeError("vLLM generation truncated; no partial output committed")
        while tokens and tokens[-1] in self.stop_ids:
            tokens.pop()
        text=self.tokenizer.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        return {"text":text,"prompt_tokens":expected,
                "num_cached_tokens":getattr(output, "num_cached_tokens", None)}
