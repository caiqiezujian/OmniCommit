#!/usr/bin/env python3
"""Export one SFT PEFT adapter plus its original HF base as a standalone model.

Uses the ms-swift 3.x HF export interface for 3.12.1 and the project's 3.14.2
image, or the 4.x interface for 4.1.3. Preserves inputs and publishes the final
directory only after validating the merged HF export.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import faulthandler
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_asr.service_backend import read_export_protocol
from fourth_asr.tokenizer_compat import (
    check_special_token_contract, normalize_tokenizer_directory, runtime_versions, special_token_contract,
)

VOCAB_KEYS = ("thinker.model.embed_tokens.weight", "thinker.lm_head.weight")


@contextmanager
def diagnostic_stage(name, stack_after_seconds=60):
    """Log a local stage and dump its stack if slow; never abort or skip checks.

    Stages must not be nested: faulthandler has one process-wide timer.
    This diagnoses this Python process, not the separate SWIFT export process.
    """
    started = time.monotonic()
    print(f"[merge] BEGIN: {name} (stack dump every {stack_after_seconds}s if still running)", flush=True)
    armed = False
    try:
        try:
            faulthandler.dump_traceback_later(stack_after_seconds, repeat=True, file=sys.stderr, exit=False)
            armed = True
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"[merge] NOTE: stack watchdog unavailable: {exc}", flush=True)
        yield
    except BaseException:
        print(f"[merge] FAILED: {name} elapsed={time.monotonic() - started:.1f}s", flush=True)
        raise
    else:
        print(f"[merge] DONE: {name} elapsed={time.monotonic() - started:.1f}s", flush=True)
    finally:
        if armed:
            faulthandler.cancel_dump_traceback_later()


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def compatible_merge_input(source, stack):
    """Use a private tokenizer config when a read-only merge input uses another HF schema."""
    if not (source / "tokenizer_config.json").is_file() or not normalize_tokenizer_directory(source, check_only=True):
        return source
    view = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="omnicommit-merge-input-")))
    for item in source.iterdir():
        destination = view / item.name
        if item.name in {"tokenizer_config.json", "special_tokens_map.json", "tokenizer.json"}:
            shutil.copyfile(item, destination)
        elif os.name == "nt" and item.is_file() and source.drive == view.drive:
            os.link(item, destination)
        else:
            destination.symlink_to(item, target_is_directory=item.is_dir())
    normalize_tokenizer_directory(view)
    return view


def restore_input_paths(output, inputs):
    """HF may serialize a temporary input directory in model/tokenizer metadata."""
    aliases = {str(view): str(source) for view, source in inputs if view != source}
    if not aliases:
        return

    def restore(value):
        if isinstance(value, str):
            return aliases.get(value, value)
        if isinstance(value, list):
            return [restore(v) for v in value]
        if isinstance(value, dict):
            return {k: restore(v) for k, v in value.items()}
        return value

    for name in ("config.json", "tokenizer_config.json", "processor_config.json", "preprocessor_config.json",
                 "generation_config.json", "args.json"):
        path = output / name
        if path.is_file():
            original = read_json(path)
            restored = restore(original)
            if restored != original:
                path.write_text(json.dumps(restored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def complete_processor_assets(output, adapter, base):
    # Older HF readers look up processor components in separate files.
    # Prefer the exact SFT checkpoint when it supplied a component.
    for name in ("preprocessor_config.json", "feature_extractor_config.json", "video_preprocessor_config.json"):
        if (output / name).exists():
            continue
        for source in (adapter, base):
            if (source / name).is_file():
                shutil.copyfile(source / name, output / name)
                break
    combined = output / "processor_config.json"
    legacy = output / "preprocessor_config.json"
    if combined.is_file() and legacy.is_file():
        extractor = read_json(combined).get("feature_extractor") or {}
        previous = read_json(legacy)
        for key in ("sampling_rate", "feature_size", "hop_length", "n_fft", "dither", "padding_value"):
            if key in extractor and key in previous and extractor[key] != previous[key]:
                raise ValueError(f"Inconsistent audio feature extractor setting {key} between processor configs")


def weight_inventory(directory, prefix):
    """Read safetensors shapes and check sharded indexes without loading weights."""
    with diagnostic_stage(f"Read {prefix} safetensors headers: {directory}"):
        return _weight_inventory(directory, prefix)


def _weight_inventory(directory, prefix):
    from safetensors import safe_open

    directory = Path(directory)
    index_path = directory / f"{prefix}.safetensors.index.json"
    weight_map = None
    if index_path.is_file():
        weight_map = read_json(index_path).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid weight_map: {index_path}")
        names = sorted(set(weight_map.values()))
        if any(not isinstance(name, str) or Path(name).name != name for name in names):
            raise ValueError(f"Shard names must be local filenames: {index_path}")
        files = [directory / name for name in names]
    elif (directory / f"{prefix}.safetensors").is_file():
        files = [directory / f"{prefix}.safetensors"]
    else:
        raise ValueError(f"No {prefix}.safetensors or its shard index in {directory}")
    inventory = {}
    for number, path in enumerate(files, 1):
        started = time.monotonic()
        print(f"[merge] Header {number}/{len(files)}: {path}", flush=True)
        if not path.is_file():
            raise FileNotFoundError(f"Missing safetensors shard: {path}")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in inventory:
                    raise ValueError(f"Duplicate tensor across shards: {key}")
                inventory[key] = {"file": str(path), "shape": list(handle.get_slice(key).get_shape())}
                if weight_map is not None and weight_map.get(key) != path.name:
                    raise ValueError(f"Tensor/index disagreement: {key}")
        print(f"[merge] Header {number}/{len(files)} complete elapsed={time.monotonic() - started:.1f}s", flush=True)
    if weight_map is not None and set(weight_map) != set(inventory):
        raise ValueError(f"Index references missing tensors: {index_path}")
    if not inventory:
        raise ValueError(f"Empty model weights: {directory}")
    return inventory


def find_adapter_module(inventory, target):
    matches = []
    for key in inventory:
        normalized = key.replace(".modules_to_save.default", "").replace(".modules_to_save", "")
        if normalized.endswith(target):
            matches.append(key)
    if len(matches) != 1:
        raise ValueError(f"Expected one saved full module for {target}; found {matches}. "
                         "Use the PEFT checkpoint saved with modules_to_save word_embeddings output_layer.")
    return matches[0]


def protocol_ids(tokenizer, tokens):
    ids = {token: tokenizer.convert_tokens_to_ids(token) for token in tokens}
    for token, token_id in ids.items():
        if not isinstance(token_id, int) or token_id == tokenizer.unk_token_id:
            raise ValueError(f"Missing protocol token: {token}")
        if tokenizer.encode(token, add_special_tokens=False) != [token_id]:
            raise ValueError(f"Protocol token is split into multiple tokens: {token}")
        if token not in tokenizer.all_special_tokens:
            raise ValueError(f"Protocol token is not registered as special: {token}")
    if len(set(ids.values())) != len(tokens):
        raise ValueError("Protocol token IDs must be distinct")
    return ids


def add_training_tokens(tokenizer, tokens):
    # HF changed both the registry name and the append argument in Transformers 5.
    parameters = inspect.signature(tokenizer.add_special_tokens).parameters
    if "replace_extra_special_tokens" in parameters:
        tokenizer.add_special_tokens({"extra_special_tokens": tokens}, replace_extra_special_tokens=False)
    elif "replace_additional_special_tokens" in parameters:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens}, replace_additional_special_tokens=False)
    else:
        # Older/custom tokenizer with only the token dictionary argument.
        existing = list(getattr(tokenizer, "additional_special_tokens", ()) or ())
        tokenizer.add_special_tokens({"additional_special_tokens": existing + [t for t in tokens if t not in existing]})


def compare_vocab_rows(reference, actual, target, *, rows_per_read=2048):
    """Check every trained row after export's dtype conversion, using bounded CPU slices."""
    import torch
    from safetensors import safe_open

    if actual["shape"] != reference["shape"]:
        raise ValueError(f"Vocabulary shape mismatch for {target}")
    row_count = reference["shape"][0]
    with safe_open(reference["file"], framework="pt", device="cpu") as src, \
            safe_open(actual["file"], framework="pt", device="cpu") as dst:
        original = src.get_slice(reference["key"])
        result = dst.get_slice(actual["key"])
        for start in range(0, row_count, rows_per_read):
            end = min(start + rows_per_read, row_count)
            values = result[start:end]
            if not values.is_floating_point() or not torch.isfinite(values).all():
                raise ValueError(f"Invalid vocabulary weights for {target} in [{start}, {end})")
            if not torch.equal(original[start:end].to(values.dtype), values):
                raise ValueError(f"Vocabulary rows differ for {target} in [{start}, {end})")


def preflight(base, adapter, tokens_file):
    with diagnostic_stage("Import transformers.AutoTokenizer (may initialize PyTorch)"):
        from transformers import AutoTokenizer

    with diagnostic_stage("Read local model config, adapter config and training args"):
        config = read_json(base / "config.json")
        if config.get("model_type") != "qwen3_omni_moe":
            raise ValueError("--base must be the original Hugging Face Qwen3-Omni model directory, not MCore")
        if config.get("quantization_config") or config.get("thinker_config", {}).get("quantization_config"):
            raise ValueError("Use the original unquantized HF base for this merge")
        tokens, manifest_metadata = read_export_protocol(tokens_file)
        adapter_config = read_json(adapter / "adapter_config.json")
        if str(adapter_config.get("peft_type", "")).upper() != "LORA":
            raise ValueError("Expected a PEFT LoRA adapter_config.json, not a merged/MCore checkpoint")
        training_args = read_json(adapter / "args.json")
        trained_tokens = training_args.get("new_special_tokens")
        if trained_tokens != tokens:
            raise ValueError("Checkpoint args.json new_special_tokens does not match the protocol token list. "
                             "Check the actual training token order; do not guess or edit it just to bypass this check.")
    # Explicitly rebuild the same token additions from the original base.
    with diagnostic_stage(f"Load local tokenizer: {base}"):
        tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)
    with diagnostic_stage("Reconstruct and validate the full training token inventory"):
        add_training_tokens(tokenizer, tokens)
        ids = protocol_ids(tokenizer, tokens)
        # Older Fourth checkpoints also contain both dense tables. If their token-ID
        # sidecar is available it must identify the same tokenizer rows.
        legacy_manifest = adapter / "omnicommit_action_rows.json"
        if legacy_manifest.is_file():
            legacy = read_json(legacy_manifest)
            if legacy.get("schema_version") != 1 or legacy.get("token_ids") != ids:
                raise ValueError("Legacy checkpoint token IDs disagree with the training inventory")
        vocabulary = tokenizer.get_vocab()
        contract = special_token_contract(tokenizer)
        if any((adapter / name).is_file() for name in ("tokenizer.json", "vocab.json")):
            training_tokenizer = AutoTokenizer.from_pretrained(str(adapter), local_files_only=True, trust_remote_code=False)
            if training_tokenizer.get_vocab() != vocabulary or protocol_ids(training_tokenizer, tokens) != ids:
                raise ValueError("Checkpoint tokenizer differs from the base plus training tokens")
            check_special_token_contract(contract, training_tokenizer)
    base_weights = weight_inventory(base, "model")
    adapter_weights = weight_inventory(adapter, "adapter_model")
    if not any("lora_A" in key for key in adapter_weights) or not any("lora_B" in key for key in adapter_weights):
        raise ValueError("Adapter has no LoRA A/B weights")
    expected_vocab = config["thinker_config"]["text_config"]["vocab_size"]
    if expected_vocab < len(tokenizer):
        expected_vocab = ((len(tokenizer) + 127) // 128) * 128
    modules = {}
    for target in VOCAB_KEYS:
        if target not in base_weights:
            raise ValueError(f"Original base is missing {target}")
        key = find_adapter_module(adapter_weights, target)
        shape = adapter_weights[key]["shape"]
        base_shape = base_weights[target]["shape"]
        if len(shape) != 2 or shape != [expected_vocab, base_shape[1]]:
            raise ValueError(f"Saved module {key} has shape {shape}; HF loader expects "
                             f"{[expected_vocab, base_shape[1]]}. Do not discard trained vocabulary rows.")
        if max(ids.values()) >= shape[0]:
            raise ValueError(f"Protocol IDs do not fit saved module {key}")
        modules[target] = {"key": key, **adapter_weights[key]}
    return {"tokens": tokens, "manifest_metadata": manifest_metadata,
            "protocol_token_ids": ids, "tokenizer_length": len(tokenizer),
            "expected_vocab_size": expected_vocab, "saved_modules": modules,
            "vocabulary": vocabulary, "special_token_contract": contract,
            "base_tensor_shapes": {key: value["shape"] for key, value in base_weights.items()},
            "training_base_recorded": training_args.get("model"),
            "training_args_tokens_verified": True}


def verify_export(output, expected):
    with diagnostic_stage("Import export verification dependencies"):
        from transformers import AutoProcessor, AutoTokenizer

    with diagnostic_stage(f"Load exported tokenizer: {output}"):
        tokenizer = AutoTokenizer.from_pretrained(str(output), local_files_only=True, trust_remote_code=False)
    if (protocol_ids(tokenizer, expected["tokens"]) != expected["protocol_token_ids"]
            or tokenizer.get_vocab() != expected["vocabulary"]):
        raise ValueError("Exported tokenizer differs from the expected training protocol tokenizer")
    check_special_token_contract(expected["special_token_contract"], tokenizer)
    with diagnostic_stage(f"Load exported processor: {output}"):
        processor = AutoProcessor.from_pretrained(str(output), local_files_only=True, trust_remote_code=False)
    if (protocol_ids(processor.tokenizer, expected["tokens"]) != expected["protocol_token_ids"]
            or processor.tokenizer.get_vocab() != expected["vocabulary"]):
        raise ValueError("Exported processor tokenizer vocabulary mismatch")
    check_special_token_contract(expected["special_token_contract"], processor.tokenizer)
    config = read_json(output / "config.json")
    if config.get("model_type") != "qwen3_omni_moe":
        raise ValueError("Unexpected exported model_type")
    if config["thinker_config"]["text_config"]["vocab_size"] != expected["expected_vocab_size"]:
        raise ValueError("Exported config vocabulary size does not match trained modules")
    inventory = weight_inventory(output, "model")
    if any("lora_A" in key or "lora_B" in key or "modules_to_save" in key for key in inventory):
        raise ValueError("Export still contains unmerged adapter tensor names")
    # ENABLE_AUDIO_OUTPUT=0 exports the Thinker for speech transcription.
    # Its text, audio and vision modules must all still be present.
    for key, shape in expected["base_tensor_shapes"].items():
        if not key.startswith("thinker."):
            continue
        wanted = expected["saved_modules"][key]["shape"] if key in VOCAB_KEYS else shape
        if key not in inventory or inventory[key]["shape"] != wanted:
            raise ValueError(f"Export missing/mismatched Thinker tensor: {key}")
    checked = []
    for target, saved in expected["saved_modules"].items():
        exported = inventory[target]
        with diagnostic_stage(f"Compare all trained input/output vocabulary rows: {target}"):
            compare_vocab_rows(saved, {"key": target, **exported}, target)
        checked.append(target)
    return {"protocol_token_ids": expected["protocol_token_ids"], "tokenizer_length": len(tokenizer),
            "vocab_size": expected["expected_vocab_size"], "tensor_count": len(inventory),
            "full_vocabulary_verified_in": checked, "processor_load_verified": True,
            "inference_verified": False}


def export_command(args, staging, swift_version):
    swift4 = swift_version.split(".", 1)[0] == "4"
    return [sys.executable, "-u", "-m", "swift.cli.export",
            "--model", str(args.base), "--adapters", str(args.adapter),
            "--model_type", "qwen3_omni_moe" if swift4 else "qwen3_omni", "--template", "qwen3_omni",
            "--load_args", "false", "--tuner_type" if swift4 else "--train_type", "lora",
            "--new_special_tokens", str(args.special_tokens), "--merge_lora", "true",
            "--torch_dtype", "bfloat16", "--device_map", args.device_map,
            "--safe_serialization", "true", "--max_shard_size", "5GB",
            "--output_dir", str(staging)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path, help="Original unquantized HF Qwen3-Omni directory (NOT MCore)")
    parser.add_argument("adapter", type=Path, help="One checkpoint-N with adapter_config.json and adapter weights")
    parser.add_argument("output", type=Path, help="New directory for the merged full model")
    parser.add_argument("--gpu-ids", default="0", help="Free GPU IDs for this single-process export, e.g. 0 or 0,1")
    parser.add_argument("--device-map", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--special-tokens", type=Path, required=True, help="Exact dataset-generated inventory used in SFT; preserve its order")
    parser.add_argument("--check-only", action="store_true", help="Check inputs, tokens and shapes without loading full weights")
    args = parser.parse_args(argv)
    staging = None
    views = ExitStack()
    try:
        for name in ("base", "adapter", "special_tokens"):
            setattr(args, name, getattr(args, name).resolve(strict=True))
        args.output = args.output.resolve()
        if args.output.exists():
            raise FileExistsError(f"Output must not already exist: {args.output}")
        for root in (args.base, args.adapter):
            if root == args.output or root in args.output.parents:
                raise ValueError("Output must not be inside the original base/adapter directory")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids if args.device_map == "auto" else ""
        os.environ["ENABLE_AUDIO_OUTPUT"] = "0"
        os.environ["PYTHONUNBUFFERED"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        version = importlib.metadata.version("ms-swift")
        print(f"[merge] Python: {sys.executable}\n[merge] ms-swift: {version}", flush=True)
        if version.split("+", 1)[0] == "3.14.2":
            print("[merge] Using the project's 3.14.2 image with the 3.x export CLI.", flush=True)
        elif version.split("+", 1)[0] not in {"3.12.1", "4.1.3"}:
            print("[merge] NOTE: upstream CLI source checked for 3.12.1 and 4.1.3; "
                  "the project's 3.14.2 image uses the 3.x CLI.", flush=True)
        print("[merge] Checking local HF base, adapter tensors and protocol tokens...", flush=True)
        export_args = argparse.Namespace(**vars(args))
        export_args.base = compatible_merge_input(args.base, views)
        export_args.adapter = compatible_merge_input(args.adapter, views)
        expected = preflight(export_args.base, export_args.adapter, args.special_tokens)
        print(json.dumps({"input_checks_passed": True, "protocol_token_ids": expected["protocol_token_ids"],
                          "vocab_size": expected["expected_vocab_size"],
                          "training_base_recorded": expected["training_base_recorded"]}, ensure_ascii=False), flush=True)
        if args.check_only:
            print("[merge] Check-only complete; no model was merged or written.", flush=True)
            return 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        staging = args.output.with_name(args.output.name + ".incomplete-" + uuid.uuid4().hex[:8])
        command = export_command(export_args, staging, version)
        print("[merge] " + shlex.join(command), flush=True)
        started = time.monotonic()
        # Run as one process; device_map can distribute HF weights across visible GPUs.
        child_env = dict(os.environ)
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "NPROC_PER_NODE", "NNODES"):
            child_env.pop(key, None)
        subprocess.run(command, env=child_env, check=True)
        complete_processor_assets(staging, args.adapter, args.base)
        restore_input_paths(staging, [(export_args.base, args.base), (export_args.adapter, args.adapter)])
        with diagnostic_stage("Finalize tokenizer configuration for serving and GSPO"):
            normalised = normalize_tokenizer_directory(staging)
        print("[merge] Export finished; verifying the full trained input and output vocabulary tables...", flush=True)
        verified = verify_export(staging, expected)
        report = {"created_utc": datetime.now(timezone.utc).isoformat(), "base": str(args.base),
                  "adapter": str(args.adapter), "output": str(args.output), "ms_swift": version,
                  "runtime_versions": runtime_versions(),
                  "tokenizer_metadata_normalized": normalised,
                  "elapsed_seconds": round(time.monotonic() - started, 2), **verified}
        (staging / "omnicommit_merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest = {**expected["manifest_metadata"], "tokens": expected["tokens"],
                    "token_ids": verified["protocol_token_ids"], "tokenizer_length": verified["tokenizer_length"]}
        (staging / "omnicommit_tokens.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (staging / "omnicommit_tokens.txt").write_text("\n".join(expected["tokens"]) + "\n", encoding="utf-8")
        if args.output.exists():
            raise FileExistsError(f"Output appeared during export; leaving staging intact: {args.output}")
        staging.rename(args.output)
        print(f"[merge] SUCCESS: {args.output}", flush=True)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    except KeyboardInterrupt:
        print(f"[merge] Interrupted. Partial output, if any: {staging}", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        print(f"[merge] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if staging is not None:
            print(f"[merge] Partial/unverified export retained at: {staging}; do not use it for inference.", file=sys.stderr, flush=True)
        return 1
    finally:
        views.close()


if __name__ == "__main__":
    sys.exit(main())
