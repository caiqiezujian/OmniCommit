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

from . import DEFAULT_MAX_CONTEXT_TOKENS, DEFAULT_MAX_NEW_TOKENS
from .audio_token_counter import AudioTokenCounter
from .inference_backend import AssistantPrefixDecoder, multimodal_messages
from ..service_backend import validate_manifest
from .streaming import ContextBudgetError, GenerationTruncated, WAIT

AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"


class VLLMBackend(AssistantPrefixDecoder):
    def __init__(self, model_path, *, max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS,
                 max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                 tensor_parallel_size=1, gpu_memory_utilization=.85,
                 max_audio_turns=80, max_audio_seconds=5.0,
                 enable_prefix_caching=True, mm_processor_cache_gb=1.0, image_slots=1, dtype="auto",
                 manifest_validator=validate_manifest, max_batch_size=1):
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
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        path = Path(model_path).resolve(strict=True)
        manifest = json.loads((path/"omnicommit_tokens.json").read_text(encoding="utf-8"))
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.allowed_tokens = manifest_validator(self.tokenizer, manifest)
        self.TokensPrompt = TokensPrompt
        self.max_context_tokens = max_context_tokens
        self.max_new_tokens = max_new_tokens
        self.max_audio_turns = max_audio_turns
        self.max_audio_seconds = max_audio_seconds
        self.max_batch_size = max_batch_size
        self.last_batch_size = 0
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
            max_num_seqs=max_batch_size, enforce_eager=True, enable_prefix_caching=enable_prefix_caching,
            limit_mm_per_prompt={"audio":{"count":max_audio_turns,
                "length":round(max_audio_seconds*16000)},"image":image_slots,"video":0},
            mm_processor_cache_gb=mm_processor_cache_gb, seed=0)
        config = self.llm.llm_engine.vllm_config
        self.enable_prefix_caching = config.cache_config.enable_prefix_caching
        self.mm_processor_cache_gb = config.model_config.multimodal_config.mm_processor_cache_gb
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

    def _prepare_request(self, messages, audios, prefix):
        if sum(m["content"].count("<audio>") for m in messages) != len(audios):
            raise ValueError("Audio placeholders and waveform count disagree")
        max_frames = round(self.max_audio_seconds * 16000)
        if len(audios)>self.max_audio_turns or any(len(a)>max_frames for a in audios):
            raise ContextBudgetError("Audio count/length exceeds configured vLLM capacity")
        prompt = self.tokenizer.apply_chat_template(multimodal_messages(messages),
                            tokenize=False, add_generation_prompt=True)
        ids = self._encode(prompt + prefix)
        extra = sum(self.audio_counter.count(a,self._exact_audio_expansion)[0] for a in audios)
        expected = len(ids)+extra
        if expected+self.max_new_tokens>self.max_context_tokens:
            raise ContextBudgetError("Expanded prompt plus generation reserve exceeds context budget")
        uuids = ["omnicommit-16k-"+hashlib.sha256(np.asarray(a,dtype="<f4").tobytes()).hexdigest()
                 for a in audios]
        request = self.TokensPrompt(prompt_token_ids=ids, multi_modal_data={"audio":audios},
                                   multi_modal_uuids={"audio":uuids})
        return request, expected

    def _run_requests(self, requests):
        outputs = self.llm.generate(requests, self.sampling, use_tqdm=False)
        if len(outputs) != len(requests):
            raise RuntimeError("Unexpected vLLM output count")
        self.last_batch_size = len(requests)
        return outputs

    def _decode_reply(self, output, expected):
        if len(output.outputs) != 1:
            raise RuntimeError("Unexpected vLLM completion count")
        completion = output.outputs[0]
        if completion.finish_reason not in ("stop", "length"):
            raise RuntimeError(f"Unexpected vLLM finish reason {completion.finish_reason!r}; no partial output committed")
        actual_ids = getattr(output, "prompt_token_ids", None)
        if actual_ids is None or len(actual_ids)!=expected:
            self.audio_counter.disable("actual expansion mismatch")
            raise RuntimeError("Actual multimodal prompt expansion differs from audited counter")
        if completion.finish_reason == "length":
            raise GenerationTruncated("vLLM", len(completion.token_ids), len(actual_ids), self.max_new_tokens,
                                      self.max_context_tokens)
        tokens=list(completion.token_ids)
        while tokens and tokens[-1] in self.stop_ids:
            tokens.pop()
        text=self.tokenizer.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        return {"text":text,"prompt_tokens":expected,
                "num_cached_tokens":getattr(output, "num_cached_tokens", None)}

    def _generate(self, messages, audios, prefix):
        request, expected = self._prepare_request(messages, audios, prefix)
        output, = self._run_requests([request])
        return self._decode_reply(output, expected)

    def generate_batch(self, calls):
        """One offline vLLM generate call for independent Third meetings.

        Return one full reply or exception in input order. An invalid/overlong
        prompt or a malformed completion belongs to that meeting alone.
        """
        requests = []
        pending = []
        replies = [None] * len(calls)
        self.last_batch_size = 0
        for index, (messages, audios, prefix) in enumerate(calls):
            try:
                if prefix and (prefix != "<|DIRECT|>" or prefix not in self.allowed_tokens):
                    raise ValueError("Forced decoding requires a trained DIRECT token")
                request, expected = self._prepare_request(messages, audios, prefix)
            except Exception as exc:
                replies[index] = exc
                continue
            requests.append(request)
            pending.append((index, expected, prefix))
        if requests:
            # vLLM 0.11.1 LLM.generate returns outputs in input prompt order.
            outputs = self._run_requests(requests)
            for output, (index, expected, prefix) in zip(outputs, pending):
                try:
                    decoded = self._decode_reply(output, expected)
                    decoded["text"] = prefix + decoded["text"]
                    replies[index] = decoded
                except Exception as exc:
                    replies[index] = exc
        return replies
