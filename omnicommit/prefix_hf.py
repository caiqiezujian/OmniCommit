"""Batched offline HF teacher for audio-prefix annotation.

This is deliberately separate from the trained streaming service backend: an
offline teacher needs no OmniCommit action-token manifest. Each query is an
independent, supplied-audio-only transcription, with no reference transcript or
previous recognition in its prompt. Callers must crop before resampling to 16 kHz.

The loading/generation contract follows Transformers 4.57.1 Qwen3-Omni. No model
downloads, tokenizer edits, KV-cache reuse, or native audio streaming are implied.
"""
from __future__ import annotations

import gc
from pathlib import Path
import re

import numpy as np


LANGUAGE_CODE = re.compile(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})*\Z")
LANGUAGE_NAMES = {
    "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
    "de": "German", "ru": "Russian", "it": "Italian", "fr": "French",
    "es": "Spanish", "pt": "Portuguese", "ar": "Arabic", "th": "Thai",
}


def teacher_prompt(language, number_format="normalized"):
    """Known language is annotation metadata, never a transcription answer."""
    if language is not None and (
            not isinstance(language, str) or not LANGUAGE_CODE.fullmatch(language)):
        raise ValueError("Teacher language must be a lowercase language code or None")
    if number_format == "normalized":
        numbers = "Convert numerical expressions to standard digits."
    elif number_format == "verbatim":
        numbers = "Preserve spoken number words; do not convert them to digits."
    else:
        raise ValueError("number_format must be normalized or verbatim")
    language_hint = (" The speech language is " + LANGUAGE_NAMES.get(language, language) + "."
                     if language is not None else "")
    return ("Transcribe only the speech present in the supplied audio in its original language."
            + language_hint + " Do not translate or complete missing speech. " + numbers
            + " Output only the transcription, without explanations or labels.")


def _failure(error, *, prompt_tokens=None, text="", token_ids=None):
    return {"text": text, "token_ids": token_ids or [], "error": error,
            "prompt_tokens": prompt_tokens}


class HFPrefixTeacher:
    """Load one local Qwen3-Omni and transcribe independent queries in batches.

    ``transcribe_batch(waveforms, languages)`` returns one ordered dictionary per
    input, containing ``text``, ``token_ids``, ``error`` and ``prompt_tokens``.
    Successful empty speech output has ``error=None``; failed decoding never
    becomes WAIT. Failed batches are bisected to isolate bad samples or reduce
    GPU memory use. A single failing sample is recorded without retrying forever.
    Filesystem/host-memory failures and fatal CUDA errors remain job-level errors.
    """

    def __init__(self, model_path, *, max_new_tokens=512, max_context_tokens=32768,
                 device_map="auto", dtype="auto", attn_implementation="sdpa",
                 number_format="normalized"):
        if (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int)
                or max_new_tokens <= 0 or isinstance(max_context_tokens, bool)
                or not isinstance(max_context_tokens, int)
                or max_context_tokens <= max_new_tokens):
            raise ValueError("Need positive generation budget smaller than context limit")
        teacher_prompt(None, number_format)
        path = Path(model_path).resolve(strict=True)
        if not path.is_dir():
            raise ValueError("model_path must be a local HF model directory")

        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        self.torch = torch
        self.number_format = number_format
        self.max_new_tokens = max_new_tokens
        self.max_context_tokens = max_context_tokens
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            str(path), local_files_only=True)
        self.tokenizer = self.processor.tokenizer
        # Decoder-only batch generation must slice after the padded prompt width,
        # not after each sample's non-pad length.
        self.tokenizer.padding_side = "left"
        self.stop_ids = []
        for name in ("<|im_end|>", "<|endoftext|>"):
            encoded = self.tokenizer.encode(name, add_special_tokens=False)
            if len(encoded) == 1 and name in self.tokenizer.all_special_tokens:
                self.stop_ids.append(int(encoded[0]))
        self.stop_ids = sorted(set(self.stop_ids))
        if not self.stop_ids:
            raise ValueError("Teacher tokenizer lacks recognized chat EOS tokens")
        if self.tokenizer.pad_token_id is None:
            # This aliases an existing token; it never inserts/reorders vocabulary.
            self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(self.stop_ids[0])
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            raise ValueError("Teacher tokenizer could not supply a padding token")
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            str(path), local_files_only=True, device_map=device_map, dtype=dtype,
            attn_implementation=attn_implementation, enable_audio_output=False).eval()

    def transcribe_batch(self, waveforms, languages):
        waveforms, languages = list(waveforms), list(languages)
        if len(waveforms) != len(languages):
            raise ValueError("One language entry is required for each waveform")
        results = [None] * len(waveforms)
        valid, prompts, positions = [], [], []
        for index, (waveform, language) in enumerate(zip(waveforms, languages)):
            try:
                waveform = np.asarray(waveform, dtype=np.float32)
                if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
                    raise ValueError("Audio must be a finite, nonempty 1-D 16 kHz waveform")
                prompt = teacher_prompt(language, self.number_format)
            except (TypeError, ValueError) as exc:
                results[index] = _failure(type(exc).__name__ + ": " + str(exc))
                continue
            valid.append(np.ascontiguousarray(waveform))
            prompts.append(prompt)
            positions.append(index)
        if positions:
            for index, value in zip(positions, self._isolated_batch(valid, prompts)):
                results[index] = value
        return results

    def _isolated_batch(self, waveforms, prompts):
        # Recurse only after leaving the except block: its traceback otherwise
        # retains failed CUDA inputs and can defeat OOM recovery.
        try:
            return self._generate_batch(waveforms, prompts)
        except (OSError, MemoryError):
            raise
        except Exception as exc:
            message = str(exc)
            if any(term in message.lower() for term in (
                    "device-side assert", "illegal memory access", "cuda driver error")):
                raise
            error = type(exc).__name__ + ": " + message[:2000]
            oom = "out of memory" in message.lower()
        if oom:
            gc.collect()
            cuda = getattr(self.torch, "cuda", None)
            if cuda is not None and cuda.is_available():
                cuda.empty_cache()
        if len(waveforms) == 1:
            return [_failure(error)]
        middle = len(waveforms) // 2
        return (self._isolated_batch(waveforms[:middle], prompts[:middle])
                + self._isolated_batch(waveforms[middle:], prompts[middle:]))

    def _generate_batch(self, waveforms, prompts):
        rendered = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": prompt}]},
                {"role": "user", "content": [{"type": "audio", "audio": "in_memory.wav"}]},
            ]
            # tokenize=False means the sentinel filename is never opened.
            rendered.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))
        inputs = self.processor(text=rendered, audio=waveforms, sampling_rate=16000,
                                return_tensors="pt", padding=True)
        width = int(inputs["input_ids"].shape[-1])
        if width + self.max_new_tokens > self.max_context_tokens:
            raise ValueError("Encoded context plus generation budget exceeds max_context_tokens; "
                             "audio and text were not truncated")
        if int(inputs["input_ids"].shape[0]) != len(waveforms):
            raise ValueError("Processor returned a different batch size")
        counts = ([int(row.sum().item()) for row in inputs["attention_mask"]]
                  if "attention_mask" in inputs else [width] * len(waveforms))
        # Accelerate dispatches subsequent layers; place inputs on Thinker's
        # device rather than a Talker device of a multimodal wrapper.
        device = self.model.thinker.device
        inputs = inputs.to(device)
        for key, value in inputs.items():
            if hasattr(value, "is_floating_point") and value.is_floating_point():
                inputs[key] = value.to(self.model.thinker.dtype)
        with self.torch.inference_mode():
            sequences = self.model.generate(
                **inputs, return_audio=False, thinker_do_sample=False,
                thinker_num_beams=1, thinker_num_return_sequences=1,
                thinker_max_new_tokens=self.max_new_tokens,
                thinker_eos_token_id=self.stop_ids, thinker_pad_token_id=self.pad_id,
                thinker_return_dict_in_generate=False)
        # 4.57.1 returns (sequences, None) in text-only mode; newer releases may
        # return sequences directly. Neither case justifies skipping special tags.
        if isinstance(sequences, tuple):
            sequences = sequences[0]
        if hasattr(sequences, "sequences"):
            sequences = sequences.sequences
        if (getattr(sequences, "ndim", None) != 2
                or int(sequences.shape[0]) != len(waveforms)
                or int(sequences.shape[1]) < width):
            raise ValueError("Unexpected teacher generation shape")
        generated_batch = sequences[:, width:]
        if hasattr(generated_batch, "detach"):
            generated_batch = generated_batch.detach().cpu()
        results = []
        for row, count in zip(generated_batch.tolist(), counts):
            stop = next((i for i, token in enumerate(row) if token in self.stop_ids), None)
            body = row if stop is None else row[:stop]
            text = self.tokenizer.decode(body, skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False)
            if stop is None:
                reason = ("Generation reached max_new_tokens without chat EOS"
                          if len(row) >= self.max_new_tokens
                          else "Generation ended without chat EOS")
                results.append(_failure(reason, prompt_tokens=count, text=text, token_ids=body))
            elif any(token != self.pad_id for token in row[stop + 1:]):
                results.append(_failure("Unexpected non-padding tokens after chat EOS",
                                        prompt_tokens=count, text=text, token_ids=body))
            else:
                results.append({"text": text, "token_ids": body,
                                "error": None, "prompt_tokens": count})
        return results
