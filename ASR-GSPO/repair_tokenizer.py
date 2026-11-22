"""Diagnose or explicitly repair existing Fourth protocol-token registration.

No token is created, moved, or assigned a new vocabulary row. Run this with
the same Python environment used to launch GSPO; stop other writers first.
"""
import argparse
import inspect
import json
import platform
from importlib import metadata
from pathlib import Path
import shutil
import tempfile

import asr_gspo  # exposes the sibling Fourth package
from fourth_asr.service_backend import _manifest_mode, validate_manifest
from finalize_export import atomic_write, verify_weights


TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "merges.txt", "tokenizer.model",
    "spiece.model", "vocab.txt",
)
SUPPORT_FILES = (
    "config.json", "preprocessor_config.json", "processor_config.json",
    "feature_extractor_config.json", "chat_template.json", "chat_template.jinja",
)
FLAGS = ("single_word", "lstrip", "rstrip", "normalized", "special")


def versions():
    result = {"python": platform.python_version()}
    for name in ("transformers", "tokenizers", "huggingface-hub", "safetensors", "ms-swift"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def load_tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def load_processor(path):
    from transformers import Qwen3OmniMoeProcessor
    return Qwen3OmniMoeProcessor.from_pretrained(path, local_files_only=True)


def _token_state(token):
    return {"content": str(token), **{key: getattr(token, key) for key in FLAGS
                                     if type(getattr(token, key, None)) is bool}}


def _snapshot(tokenizer):
    decoder = tokenizer.added_tokens_decoder
    by_text = {str(value): value for value in decoder.values()}
    special_map = getattr(tokenizer, "special_tokens_map_extended", None)
    if special_map is None:
        special_map = tokenizer.special_tokens_map
    return {
        "vocab": dict(tokenizer.get_vocab()), "length": len(tokenizer),
        "special": set(tokenizer.all_special_tokens),
        "added": {int(k): _token_state(v) for k, v in decoder.items()},
        "roles": {k: _token_state(by_text.get(str(v), v)) for k, v in special_map.items()
                  if not isinstance(v, (list, tuple, dict))},
        "chat_template": getattr(tokenizer, "chat_template", None),
    }


def _check_ids(tokenizer, manifest):
    if _manifest_mode(manifest) != "speaker":
        raise ValueError("Only the exact Fourth speaker-protocol manifest is supported")
    tokens, ids = manifest.get("tokens"), manifest.get("token_ids")
    if not isinstance(ids, dict) or set(ids) != set(tokens) or len(tokens) != len(set(tokens)):
        raise ValueError("Invalid Fourth protocol token/ID mapping")
    if manifest.get("tokenizer_length") != len(tokenizer):
        raise ValueError("Tokenizer length differs from the trained manifest; registration repair refused")
    vocab = tokenizer.get_vocab()
    used = set()
    for token in tokens:
        idx = ids[token]
        if (type(idx) is not int or idx < 0 or idx >= len(tokenizer) or idx in used
                or vocab.get(token) != idx or tokenizer.encode(token, add_special_tokens=False) != [idx]):
            raise ValueError(f"Missing, remapped, or non-atomic protocol token {token!r}; registration repair refused")
        used.add(idx)


def _check_preserved(tokenizer, manifest, original):
    _check_ids(tokenizer, manifest)
    validate_manifest(tokenizer, manifest)
    if len(tokenizer) != original["length"] or tokenizer.get_vocab() != original["vocab"]:
        raise ValueError("Repair changed the vocabulary or token IDs")
    if not original["special"].issubset(tokenizer.all_special_tokens):
        raise ValueError("Repair removed an existing special token")
    current = _snapshot(tokenizer)
    if current["roles"] != original["roles"]:
        raise ValueError("Repair changed a named special-token role or its AddedToken flags")
    if current["chat_template"] != original["chat_template"]:
        raise ValueError("Repair changed the tokenizer chat template")
    for idx, previous in original["added"].items():
        expected = dict(previous)
        if expected["content"] in manifest["tokens"]:
            expected["special"] = True
        if current["added"].get(idx) != expected:
            raise ValueError(f"Repair changed existing AddedToken flags at vocabulary row {idx}")
        if previous.get("special") and previous["content"] not in tokenizer.all_special_tokens:
            raise ValueError(f"Backend special token lost from registry: {previous['content']}")


def _register(tokenizer, manifest):
    from tokenizers import AddedToken
    # Include backend-only specials: they may have disappeared from the wrapper
    # registry when moving between Transformers releases.
    extended = getattr(tokenizer, "all_special_tokens_extended", None)
    existing = list(extended if extended is not None else tokenizer.all_special_tokens)
    existing.extend(v for v in tokenizer.added_tokens_decoder.values() if getattr(v, "special", False))
    by_text = {str(v): v for v in existing}
    for token in manifest["tokens"]:
        old = tokenizer.added_tokens_decoder.get(manifest["token_ids"][token])
        flags = {k: getattr(old, k) for k in FLAGS
                 if k != "special" and type(getattr(old, k, None)) is bool}
        by_text[token] = AddedToken(token, **flags, special=True)
    # Resolve strings to their original AddedToken objects to preserve flags.
    decoder_by_text = {str(v): v for v in tokenizer.added_tokens_decoder.values()}
    full = [decoder_by_text.get(text, value) if text not in manifest["tokens"] else value
            for text, value in by_text.items()]
    attrs = getattr(tokenizer, "SPECIAL_TOKENS_ATTRIBUTES", ())
    params = inspect.signature(tokenizer.add_special_tokens).parameters
    if "additional_special_tokens" in attrs:
        key, replacement = "additional_special_tokens", "replace_additional_special_tokens"
    elif "extra_special_tokens" in attrs or "replace_extra_special_tokens" in params:
        key, replacement = "extra_special_tokens", "replace_extra_special_tokens"
    else:
        raise ValueError("Unsupported tokenizer special-token API; no files changed")
    kwargs = {replacement: True} if replacement in params else {}
    tokenizer.add_special_tokens({key: full}, **kwargs)


def _file_details(path, tokens):
    result = {}
    for name in ("tokenizer_config.json", "special_tokens_map.json", "tokenizer.json"):
        source = path / name
        if not source.is_file():
            result[name] = {"present": False}
            continue
        data = json.loads(source.read_text(encoding="utf-8"))
        result[name] = {k: data[k] for k in ("tokenizer_class", "additional_special_tokens", "extra_special_tokens") if k in data}
        added = data.get("added_tokens", data.get("added_tokens_decoder", {}))
        values = added.values() if isinstance(added, dict) else added
        result[name]["protocol_added_tokens"] = [v for v in values if isinstance(v, dict) and v.get("content") in tokens]
    return result


def _check_processor(path, manifest, original):
    processor = load_processor(path)
    _check_preserved(processor.tokenizer, manifest, original)


def _provenance(model):
    result = {}
    config_path = model / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_bytes())
        result["config_transformers_version"] = config.get("transformers_version")
        result["thinker_transformers_version"] = config.get("thinker_config", {}).get("transformers_version")
    report_path = model / "omnicommit_merge_report.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_bytes())
            result["merge_ms_swift"] = report.get("ms_swift")
        except (ValueError, AttributeError):
            result["merge_report_error"] = "Merge report is not a JSON object"
    return result


def repair(model, apply=False):
    model = Path(model).resolve()
    manifest_bytes = (model / "omnicommit_tokens.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    tokenizer = load_tokenizer(model)
    _check_ids(tokenizer, manifest)
    original = _snapshot(tokenizer)
    missing = [t for t in manifest["tokens"] if t not in tokenizer.all_special_tokens]
    result = {
        "model": str(model), "versions": versions(), "export_versions": _provenance(model),
        "mode": "apply" if apply else "diagnose",
        "tokenizer_length": len(tokenizer), "missing_special_registration": missing,
        "tokens": [{"token": t, "manifest_id": manifest["token_ids"][t],
                    "encoded_ids": tokenizer.encode(t, add_special_tokens=False),
                    "registered_special": t in tokenizer.all_special_tokens,
                    "backend": _token_state(tokenizer.added_tokens_decoder[manifest["token_ids"][t]])
                    if manifest["token_ids"][t] in tokenizer.added_tokens_decoder else None}
                   for t in manifest["tokens"]],
        "files": _file_details(model, manifest["tokens"]), "changed_files": [],
    }
    if not apply:
        result["status"] = "registration_repair_needed" if missing else "registered"
        result["note"] = "Read-only. --apply also checks HF weight headers and processor/tokenizer reloads."
        return result
    verify_weights(model, manifest["token_ids"].values())
    if not missing:
        _check_preserved(tokenizer, manifest, original)
        _check_processor(model, manifest, original)
        result["status"] = "already_valid"
        return result

    originals = {name: (model / name).read_bytes() if (model / name).is_file() else None for name in TOKENIZER_FILES}
    support = {name: (model / name).read_bytes() for name in SUPPORT_FILES if (model / name).is_file()}
    _register(tokenizer, manifest)
    _check_preserved(tokenizer, manifest, original)
    with tempfile.TemporaryDirectory(prefix=".tokenizer-registration-", dir=model.parent) as directory:
        stage = Path(directory)
        for name, data in {**originals, **support}.items():
            if data is not None:
                (stage / name).write_bytes(data)
        if (model / "chat_templates").is_dir():
            shutil.copytree(model / "chat_templates", stage / "chat_templates")
        tokenizer.save_pretrained(stage, save_jinja_files=False)
        # Never migrate or remove an original embedded template during repair.
        config_path = stage / "tokenizer_config.json"
        old_config = json.loads(originals.get("tokenizer_config.json") or b"{}")
        new_config = json.loads(config_path.read_bytes())
        if "chat_template" in old_config:
            new_config["chat_template"] = old_config["chat_template"]
            config_path.write_text(json.dumps(new_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _check_preserved(load_tokenizer(stage), manifest, original)
        _check_processor(stage, manifest, original)
        candidates = {name: (stage / name).read_bytes() for name in TOKENIZER_FILES if (stage / name).is_file()}
        changed = {name: data for name, data in candidates.items() if data != originals[name]}
        for name, before in originals.items():
            now = (model / name).read_bytes() if (model / name).is_file() else None
            if now != before:
                raise ValueError(f"Concurrent tokenizer modification detected: {name}; no files changed")
        if (model / "omnicommit_tokens.json").read_bytes() != manifest_bytes:
            raise ValueError("Concurrent manifest modification detected; no files changed")
        for name, before in support.items():
            if (model / name).read_bytes() != before:
                raise ValueError(f"Concurrent model metadata modification detected: {name}; no files changed")
        touched = []
        try:
            for name, data in changed.items():
                touched.append(name)
                atomic_write(model / name, data)
            verified_tokenizer = load_tokenizer(model)
            _check_preserved(verified_tokenizer, manifest, original)
            _check_processor(model, manifest, original)
        except Exception as exc:
            rollback_errors = []
            for name in reversed(touched):
                try:
                    if originals[name] is None:
                        (model / name).unlink(missing_ok=True)
                    else:
                        atomic_write(model / name, originals[name])
                except Exception as rollback_exc:
                    rollback_errors.append(f"{name}: {rollback_exc}")
            if rollback_errors:
                raise RuntimeError(f"Repair failed ({exc}); rollback incomplete: {rollback_errors}") from exc
            raise RuntimeError(f"Repair failed; original tokenizer files restored: {exc}") from exc
        result["changed_files"] = list(changed)
    result["status"] = "repaired"
    result["missing_special_registration"] = []
    for token in result["tokens"]:
        token["registered_special"] = True
        backend = verified_tokenizer.added_tokens_decoder.get(token["manifest_id"])
        token["backend"] = _token_state(backend) if backend is not None else None
    result["files"] = _file_details(model, manifest["tokens"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Existing merged Fourth HF model directory")
    parser.add_argument("--apply", action="store_true", help="Repair registration only after all identity and weight checks")
    args = parser.parse_args(argv)
    try:
        report = repair(args.model, apply=args.apply)
    except Exception as exc:
        print(json.dumps({"status": "refused", "versions": versions(), "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
