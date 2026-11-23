"""Complete and verify a Fourth merged HF export using its exact SFT parent."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import asr_gspo  # expose the sibling Fourth package
from fourth_asr.service_backend import detect_model_protocol, validate_manifest
from fourth_asr.tokenizer_compat import check_special_token_contract, normalize_tokenizer_directory, special_token_contract


ASSET_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
               "vocab.json", "merges.txt", "tokenizer.model", "preprocessor_config.json", "video_preprocessor_config.json", "processor_config.json",
               "feature_extractor_config.json", "chat_template.json", "chat_template.jinja", "generation_config.json")


def atomic_write(path, data, *, new_file_mode=None):
    mode = path.stat().st_mode if path.exists() else new_file_mode
    fd, staging = tempfile.mkstemp(prefix=".export-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
        if mode is not None:
            os.chmod(staging, mode)
        os.replace(staging, path)
    finally:
        Path(staging).unlink(missing_ok=True)


def stage_assets(source, merged, stage):
    """Prefer the RL export's files, filling holes from its exact SFT parent."""
    names = set(ASSET_FILES)
    for directory in (source, merged):
        names.update(p.relative_to(directory).as_posix() for p in (directory / "chat_templates").glob("*.jinja"))
    for name in sorted(names):
        path = merged / name if (merged / name).is_file() else source / name
        if path.is_file():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    # Processors may look up their type from the model config. Always use the RL one.
    shutil.copyfile(merged / "config.json", stage / "config.json")


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
    dest = merged / name
    if dest.exists() and json.loads(dest.read_bytes()) != manifest:
        raise ValueError(f"Conflicting protocol manifest in {merged}")
    tokens = merged / "omnicommit_tokens.txt"
    if tokens.exists() and tokens.read_text(encoding="utf-8-sig").splitlines() != manifest["tokens"]:
        raise ValueError(f"Conflicting protocol token inventory in {merged}")
    reference_processor = Qwen3OmniMoeProcessor.from_pretrained(source, local_files_only=True)
    contract = special_token_contract(reference_tokenizer)
    check_special_token_contract(contract, reference_processor.tokenizer)
    copied, normalized = [], []
    # Swift may reserialize the tokenizer when it saves RL weights. Check the
    # complete portable artifact before publishing even a repaired metadata file.
    with tempfile.TemporaryDirectory(prefix=".export-assets-", dir=merged) as tmp:
        stage = Path(tmp)
        stage_assets(source, merged, stage)
        normalize_tokenizer_directory(stage)
        exported_tokenizer = AutoTokenizer.from_pretrained(stage, local_files_only=True)
        processor = Qwen3OmniMoeProcessor.from_pretrained(stage, local_files_only=True)
        vocab = reference_tokenizer.get_vocab()
        for tokenizer in (exported_tokenizer, processor.tokenizer):
            if tokenizer.get_vocab() != vocab:
                raise ValueError("Existing export tokenizer differs from the exact SFT parent")
            validate_manifest(tokenizer, manifest)
            check_special_token_contract(contract, tokenizer)
        if reference_processor.chat_template != processor.chat_template:
            raise ValueError("Merged processor chat template differs from its SFT parent")
        if reference_processor.feature_extractor.to_dict() != processor.feature_extractor.to_dict():
            raise ValueError("Merged audio feature extractor differs from its SFT parent")
        for path in sorted(stage.rglob("*")):
            if not path.is_file() or path == stage / "config.json":
                continue
            relative = path.relative_to(stage).as_posix()
            target = merged / relative
            data = path.read_bytes()
            if target.exists() and target.read_bytes() == data:
                continue
            (normalized if target.exists() else copied).append(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(target, data, new_file_mode=(merged / "config.json").stat().st_mode)
    if not dest.exists():
        atomic_write(dest, source_bytes, new_file_mode=(source / name).stat().st_mode)
        copied.append(name)
    if not tokens.exists():
        atomic_write(tokens, ("\n".join(manifest["tokens"]) + "\n").encode("utf-8"),
                     new_file_mode=dest.stat().st_mode)
    status = dict(ready=True, protocol=detect_model_protocol(merged), sft_parent=str(source.resolve()),
                  copied=copied, normalized=normalized)
    atomic_write(merged / "omnicommit_export_status.json", (json.dumps(status, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(f"[export] Verified Fourth HF checkpoint: {merged}; restored={copied}; normalized={normalized}", flush=True)
    return status


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sft_model", type=Path)
    p.add_argument("merged_rl_model", type=Path)
    args = p.parse_args()
    finalize(args.sft_model, args.merged_rl_model)
