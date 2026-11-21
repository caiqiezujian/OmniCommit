"""Complete and verify a Fourth merged HF export using its exact SFT parent."""
import argparse
import json
import os
from pathlib import Path
import tempfile

import asr_gspo  # expose the sibling Fourth package
from fourth_asr.service_backend import detect_model_protocol, validate_manifest


ASSET_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
               "vocab.json", "merges.txt", "tokenizer.model", "preprocessor_config.json", "processor_config.json",
               "feature_extractor_config.json", "chat_template.json", "chat_template.jinja", "generation_config.json")


def atomic_write(path, data):
    fd, staging = tempfile.mkstemp(prefix=".export-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
        os.replace(staging, path)
    finally:
        Path(staging).unlink(missing_ok=True)


def restore_missing_assets(source, merged):
    """Do not overwrite existing assets or copy model config/weights from SFT."""
    copied = []
    candidates = [(source / name, merged / name) for name in ASSET_FILES]
    if (source / "chat_templates").is_dir():
        candidates.extend((p, merged / "chat_templates" / p.name) for p in (source / "chat_templates").glob("*.jinja"))
    for src, dest in candidates:
        if src.is_file() and not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(dest, src.read_bytes())
            copied.append(dest.relative_to(merged).as_posix())
    return copied


def verify_weights(directory, token_ids):
    from safetensors import safe_open
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        shards = set(weight_map.values())
    else:
        weight_map = None
        shards = {"model.safetensors"}
    if not shards or any(not (directory / name).is_file() or not (directory / name).stat().st_size for name in shards):
        raise ValueError(f"Missing merged HF weights at {directory}")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_omni_moe":
        raise ValueError(f"Expected a Qwen3-Omni HF model at {directory}")
    if weight_map is None:
        with safe_open(str(directory / "model.safetensors"), framework="np") as handle:
            weight_map = dict.fromkeys(handle.keys(), "model.safetensors")
    if not any(key.startswith("thinker.audio_tower.") for key in weight_map):
        raise ValueError("No Thinker audio encoder weights in the HF export")
    tied = config.get("thinker_config", {}).get("text_config", {}).get("tie_word_embeddings", False)
    for key in ("thinker.model.embed_tokens.weight", "thinker.lm_head.weight"):
        if tied and key.endswith("lm_head.weight") and key not in weight_map:
            continue
        if key not in weight_map:
            raise ValueError(f"Missing {key} in HF weights")
        with safe_open(str(directory / weight_map[key]), framework="np") as handle:
            shape = handle.get_slice(key).get_shape()
            if len(shape) != 2 or max(token_ids) >= shape[0]:
                raise ValueError(f"{key} shape {shape} does not cover trained protocol token IDs")


def finalize(source, merged):
    from transformers import AutoTokenizer, Qwen3OmniMoeProcessor

    source, merged = Path(source), Path(merged)
    if source.resolve() == merged.resolve() or detect_model_protocol(source) != "speaker":
        raise ValueError("Supply the exact Fourth SFT parent and a separate merged GSPO model")
    if not merged.is_dir():
        raise ValueError(f"Merged checkpoint directory missing: {merged}")
    # Invalidate any old success marker before checking a potentially changed
    # export. A failed manual recheck must not leave a stale ready:true file.
    atomic_write(merged / "omnicommit_export_status.json", b'{"ready":false,"status":"checking"}\n')
    name = "omnicommit_tokens.json"
    source_bytes = (source / name).read_bytes()
    manifest = json.loads(source_bytes)
    reference_tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    validate_manifest(reference_tokenizer, manifest)
    verify_weights(merged, manifest["token_ids"].values())
    # Verify an existing tokenizer before filling absent files; a mismatched
    # checkpoint must never be made to appear compatible by replacing its vocab.
    if (merged / "tokenizer_config.json").is_file() and (merged / "tokenizer.json").is_file():
        existing = AutoTokenizer.from_pretrained(merged, local_files_only=True)
        if existing.get_vocab() != reference_tokenizer.get_vocab():
            raise ValueError("Existing export tokenizer differs from the exact SFT parent")
    copied = restore_missing_assets(source, merged)
    reference_processor = Qwen3OmniMoeProcessor.from_pretrained(source, local_files_only=True)
    processor = Qwen3OmniMoeProcessor.from_pretrained(merged, local_files_only=True)
    tokenizer = processor.tokenizer
    validate_manifest(tokenizer, manifest)
    if reference_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise ValueError("Merged tokenizer vocabulary differs from the reference checkpoint")
    if reference_processor.chat_template != processor.chat_template:
        raise ValueError("Merged processor chat template differs from its SFT parent")
    if reference_processor.feature_extractor.to_dict() != processor.feature_extractor.to_dict():
        raise ValueError("Merged audio feature extractor differs from its SFT parent")
    dest = merged / name
    if dest.exists() and json.loads(dest.read_bytes()) != manifest:
        raise ValueError(f"Conflicting protocol manifest in {merged}")
    if not dest.exists():
        atomic_write(dest, source_bytes)
        copied.append(name)
    tokens = merged / "omnicommit_tokens.txt"
    if not tokens.exists():
        atomic_write(tokens, ("\n".join(manifest["tokens"]) + "\n").encode("utf-8"))
    status = dict(ready=True, protocol=detect_model_protocol(merged), sft_parent=str(source.resolve()), copied=copied)
    atomic_write(merged / "omnicommit_export_status.json", (json.dumps(status, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(f"[export] Verified Fourth HF checkpoint: {merged}; restored={copied}", flush=True)
    return status


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sft_model", type=Path)
    p.add_argument("merged_rl_model", type=Path)
    args = p.parse_args()
    finalize(args.sft_model, args.merged_rl_model)
