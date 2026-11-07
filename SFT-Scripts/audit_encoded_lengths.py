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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", type=Path, required=True, help="Original HF base with processor files, not MCore")
    parser.add_argument("--tokens-file", type=Path)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=0, help="0 checks the entire file")
    args = parser.parse_args(argv)
    if args.max_length < 1 or args.rows < 0 or args.report.exists():
        parser.error("Positive max length, nonnegative row limit, new report path required")
    os.environ["ENABLE_AUDIO_OUTPUT"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    tokens_path = args.tokens_file or Path(str(args.dataset) + ".tokens.txt")
    inventory = read_tokens(tokens_path)
    tokens = tokens_path.read_text(encoding="utf-8-sig").splitlines()
    # Source-audited interface: ms-swift v3.12.1 swift/llm/model/register.py.
    # Processor metadata attached by get_model_tokenizer is required by template.
    import omnicommit_sft_plugin  # registers virtual audio materialization
    from swift.llm import get_model_tokenizer, get_template
    from merge_lora import protocol_ids
    model, processor = get_model_tokenizer(str(args.model.resolve(strict=True)),
                                          model_type="qwen3_omni", load_model=False,
                                          new_special_tokens=tokens)
    if model is not None:
        raise RuntimeError("Processor-only audit unexpectedly loaded a model")
    tokenizer = getattr(processor, "tokenizer", processor)
    protocol_ids(tokenizer, tokens)
    template = get_template("omnicommit_qwen3_omni", processor, max_length=None,
                            truncation_strategy="raise", loss_scale="default")
    template.set_mode("train")
    count, valid, maximum, errors = 0, 0, 0, []
    with args.dataset.open(encoding="utf-8-sig") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            count += 1
            try:
                row = json.loads(line)
                check_row(row, decode=False, token_inventory=inventory)
                encoded = template.encode({"messages": deepcopy(row["messages"]), "audios": list(row["audios"])})
                metrics = audit_encoded(encoded, args.max_length)
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
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "errors"}))
    return 0 if count and count == valid else 1


if __name__ == "__main__":
    sys.exit(main())
