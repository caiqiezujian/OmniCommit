"""Make a merged Swift HF checkpoint usable with the Third HTTP/Web loader."""
import argparse
import json
import os
from pathlib import Path
import tempfile

import asr_gspo  # expose the sibling Third package
from third_asr.service_backend import detect_model_protocol, validate_manifest


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

    if source.resolve() == merged.resolve() or detect_model_protocol(source) != "explicit":
        raise ValueError("Supply a Third explicit SFT export and a separate merged GSPO model")
    name = "omnicommit_tokens.json"
    source_bytes = (source / name).read_bytes()
    manifest = json.loads(source_bytes)
    reference_tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    tokenizer = Qwen3OmniMoeProcessor.from_pretrained(merged, local_files_only=True).tokenizer
    validate_manifest(reference_tokenizer, manifest)
    validate_manifest(tokenizer, manifest)
    verify_weights(merged, manifest["token_ids"].values())
    if reference_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise ValueError("Merged tokenizer vocabulary differs from the reference checkpoint")
    dest = merged / name
    if dest.exists() and json.loads(dest.read_bytes()) != manifest:
        raise ValueError(f"Conflicting protocol manifest in {merged}")
    if not dest.exists():
        fd, staging = tempfile.mkstemp(prefix=".omnicommit_tokens-", dir=merged)
        try:
            with os.fdopen(fd, "wb") as out:
                out.write(source_bytes)
            os.replace(staging, dest)
        finally:
            Path(staging).unlink(missing_ok=True)
    print(f"Verified Third {detect_model_protocol(merged)} HF export: {merged}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sft_model", type=Path)
    p.add_argument("merged_rl_model", type=Path)
    args = p.parse_args()
    finalize(args.sft_model, args.merged_rl_model)
