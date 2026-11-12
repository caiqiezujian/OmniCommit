#!/usr/bin/env python3
"""Encode real audio through the installed SWIFT training template, without weights.

Run inside the SFT environment before training. Unlike character/token estimates,
this includes audio feature expansion, chat wrappers and assistant targets.
"""
import argparse
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys

from check_sft import check_row, read_tokens


def audit_encoded(encoded, max_length):
    if not isinstance(encoded, dict) or "input_ids" not in encoded or "labels" not in encoded:
        raise ValueError("Training template did not return input_ids and labels")
    ids, labels = encoded["input_ids"], encoded["labels"]
    if len(ids) != len(labels):
        raise ValueError("Template input_ids/labels length mismatch")
    supervised = sum(int(value) != -100 for value in labels)
    if not supervised:
        raise ValueError("No supervised target tokens after template encoding")
    if len(ids) > max_length:
        raise ValueError(f"Encoded length {len(ids)} exceeds {max_length}; rebuild windows, do not truncate")
    return {"length": len(ids), "supervised_tokens": supervised}


def load_template(model_dir, tokens, *, packing):
    import omnicommit_sft_plugin  # register the same template used by the launcher
    from merge_lora import protocol_ids
    major = int(importlib.metadata.version("ms-swift").split(".")[0])
    options = dict(max_length=None, truncation_strategy="raise", loss_scale="default")
    if major == 4:
        from swift.model import get_processor
        from swift.template import get_template
        processor = get_processor(str(model_dir), model_type="qwen3_omni_moe", new_special_tokens=tokens)
        template = get_template(processor, template_type="omnicommit_qwen3_omni", **options)
    elif major == 3:
        from swift.llm import get_model_tokenizer, get_template
        model, processor = get_model_tokenizer(str(model_dir), model_type="qwen3_omni", load_model=False,
                                               new_special_tokens=tokens)
        if model is not None:
            raise RuntimeError("Processor-only audit unexpectedly loaded a model")
        template = get_template("omnicommit_qwen3_omni", processor, **options)
    else:
        raise ValueError(f"No supported training template loader for ms-swift major version {major}")
    tokenizer = getattr(processor, "tokenizer", processor)
    ids = protocol_ids(tokenizer, tokens)
    template.set_mode("train")
    if packing:
        template.use_megatron = template.packing = template.padding_free = True
    return template, tokenizer, ids


def check_encoded_protocol(row, encoded, token_ids, tokenizer):
    replies = [message["content"] for message in row["messages"] if message["role"] == "assistant"]
    expected = [token_ids[token] for reply in replies for token in re.findall(r"<[^<>]+>", reply)]
    protocol = set(token_ids.values())
    labels = [int(value) for value in encoded["labels"]]
    actual = [value for value in labels if value in protocol]
    if actual != expected:
        raise ValueError("Supervised actions/language/speaker tokens differ from the original assistant turns")
    mask = encoded["feature_attention_mask"]
    if encoded["input_features"].shape[0] != len(row["audios"]) or mask.shape[0] != len(row["audios"]):
        raise ValueError("Encoded audio features/masks do not match the input audio chunk count")

    # Locate actual assistant payloads so a masked response cannot be confused
    # with identically worded reference speech in the user input.
    ids = [int(value) for value in encoded["input_ids"]]
    marker = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    stop = tokenizer.convert_tokens_to_ids("<|im_end|>")
    starts = [i + len(marker) for i, value in enumerate(ids)
              if value == marker[0] and ids[i:i + len(marker)] == marker]
    if len(starts) != len(replies):
        raise ValueError(f"Expected {len(replies)} encoded assistant turns, found {len(starts)}")
    for turn, (start, expected) in enumerate(zip(starts, replies), 1):
        end = ids.index(stop, start)
        payload = ids[start:end]
        if labels[start:end] != payload:
            raise ValueError(f"Assistant turn {turn} has masked or changed target IDs")
        decoded = tokenizer.decode(payload, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if decoded != expected:
            raise ValueError(f"Assistant turn {turn} differs from the JSONL transcript after tokenization")


def check_packed_batch(template, examples, max_length):
    """Collate a real Megatron microbatch. Keep individual packs within the requested pack length."""
    import torch
    lengths = [len(example["input_ids"]) for example in examples]
    if len(lengths) != 2:
        raise ValueError("Packing audit needs two valid JSONL records")
    packs = [examples] if sum(lengths) <= max_length else [[example] for example in examples]
    expected_tokens = [token for ex in examples for token in ex["input_ids"]]
    expected_labels = [token for ex in examples for token in ex["labels"]]
    packed = template.data_collator(packs)
    if packed["input_ids"].tolist() != [expected_tokens] or packed["labels"].tolist() != [expected_labels]:
        raise ValueError("Packed token order or supervision differs from the independently encoded records")
    positions = list(range(lengths[0])) + list(range(lengths[1]))
    if packed["text_position_ids"].tolist() != [positions]:
        raise ValueError("Packed text positions must restart at zero at each original record boundary")
    if tuple(packed["position_ids"].shape) != (3, 1, sum(lengths)):
        raise ValueError("Missing Qwen3-Omni packed multimodal rotary positions")
    boundaries = [0, lengths[0], sum(lengths)]
    for key in ("cu_seq_lens_q", "cu_seq_lens_k"):
        if key in packed and packed[key].tolist() != boundaries:
            raise ValueError(f"Unexpected packed attention boundaries in {key}")
    cursor = 0
    for example in examples:
        count = example["input_features"].shape[0]
        for key in ("input_features", "feature_attention_mask"):
            src = example[key]
            part = packed[key][cursor:cursor + count]
            if not torch.equal(part[..., :src.shape[-1]], src) or torch.count_nonzero(part[..., src.shape[-1]:]):
                raise ValueError(f"Packed {key} lost, reordered or changed the audio chunks")
        cursor += count
    if any(packed[key].shape[0] != cursor for key in ("input_features", "feature_attention_mask")):
        raise ValueError("Unexpected packed audio chunk count")
    return {"record_lengths": lengths, "records_per_pack": [len(pack) for pack in packs],
            "sequence_boundaries": boundaries, "audio_chunks": cursor}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", type=Path, required=True, help="Original HF base with processor files, not MCore")
    parser.add_argument("--tokens-file", type=Path)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=0, help="0 checks the entire file")
    parser.add_argument("--check-packing", action="store_true",
                        help="Verify supervision and collate the first two valid rows through the Megatron template")
    args = parser.parse_args(argv)
    if args.max_length < 1 or args.rows < 0 or args.report.exists():
        parser.error("Positive max length, nonnegative row limit, new report path required")
    if args.check_packing and args.rows == 1:
        parser.error("--check-packing requires at least two rows")
    os.environ["ENABLE_AUDIO_OUTPUT"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    tokens_path = args.tokens_file or Path(str(args.dataset) + ".tokens.txt")
    inventory = read_tokens(tokens_path)
    tokens = tokens_path.read_text(encoding="utf-8-sig").splitlines()
    template, tokenizer, token_ids = load_template(args.model.resolve(strict=True), tokens, packing=args.check_packing)
    examples = []
    count, valid, maximum, errors = 0, 0, 0, []
    with args.dataset.open(encoding="utf-8-sig") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            count += 1
            try:
                row = json.loads(line)
                check_row(row, decode=False, token_inventory=inventory)
                encoded = template.encode({"messages": deepcopy(row["messages"]), "audios": list(row["audios"])},
                                          return_length=args.check_packing)
                metrics = audit_encoded(encoded, args.max_length)
                if args.check_packing:
                    check_encoded_protocol(row, encoded, token_ids, tokenizer)
                    if len(examples) < 2:
                        examples.append(encoded)
                maximum = max(maximum, metrics["length"])
                valid += 1
            except Exception as exc:
                if len(errors) < 100:
                    errors.append({"line": line_no, "error": str(exc)})
            if count % 100 == 0:
                print(f"[encode] rows={count} valid={valid} max_length={maximum}", flush=True)
            if args.rows and count >= args.rows:
                break
    report = {"rows": count, "valid": valid, "invalid": count - valid,
              "max_valid_encoded_length": maximum, "limit": args.max_length,
              "full_dataset_checked": args.rows == 0, "real_template_encoding_checked": True,
              "model_weights_loaded": False, "gpu_training_verified": False,
              "ms_swift": importlib.metadata.version("ms-swift"), "errors": errors}
    if args.check_packing:
        try:
            report["packed_batch"] = check_packed_batch(template, examples, args.max_length)
        except Exception as exc:
            report["packing_error"] = str(exc)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "errors"}))
    return 0 if count and count == valid and "packing_error" not in report else 1


if __name__ == "__main__":
    sys.exit(main())
