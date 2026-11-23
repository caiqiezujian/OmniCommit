"""Write one Fourth tokenizer config readable by Transformers 4.x and 5.x."""
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import tempfile


STANDARD_ROLES = {"bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token"}
ALIAS_PREFIX = "omnicommit_registered_"
MODALITY_ROLES = ("audio_token", "audio_bos_token", "audio_eos_token", "image_token",
                  "video_token", "vision_bos_token", "vision_eos_token")


def runtime_versions():
    """Identify the active export/consumer environment without loading CUDA."""
    versions = {"python": platform.python_version()}
    for package in ("ms-swift", "vllm", "transformers", "tokenizers", "torch", "huggingface-hub", "safetensors"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def special_token_contract(tokenizer):
    """Capture the reference inventory and Qwen's acoustic/visual placeholder rows."""
    roles = {}
    for key in MODALITY_ROLES:
        token = getattr(tokenizer, key, None)
        if token is not None:
            roles[key] = (str(token), tokenizer.convert_tokens_to_ids(str(token)))
    return set(tokenizer.all_special_tokens), roles


def check_special_token_contract(expected, tokenizer):
    specials, roles = special_token_contract(tokenizer)
    missing = expected[0] - specials
    if missing:
        raise ValueError(f"Exported tokenizer lost special tokens: {sorted(missing)}")
    for name, value in expected[1].items():
        if roles.get(name) != value:
            raise ValueError(f"Exported modality role {name} differs from the reference: {roles.get(name)} vs {value}")


def _content(value):
    text = value.get("content") if isinstance(value, dict) else value
    if not isinstance(text, str) or not text:
        raise ValueError(f"Invalid special-token declaration: {value!r}")
    return text


def normalize_tokenizer_directory(path, *, check_only=False):
    """Return whether the JSON needs changing. Never save a tokenizer or allocate IDs."""
    directory = Path(path)
    config_path = directory / "tokenizer_config.json"
    original = config_path.read_bytes()
    config = json.loads(original)
    mapping_path = directory / "special_tokens_map.json"
    mapping = json.loads(mapping_path.read_bytes()) if mapping_path.is_file() else {}
    generic = {}
    named = {}
    standard = {}

    def add_role(key, value):
        text = _content(value)
        roles = standard if key in STANDARD_ROLES else named
        if key in roles and roles[key] != text:
            raise ValueError(f"Conflicting special-token role {key}: {roles[key]!r} vs {text!r}")
        roles[key] = text

    for data in (config, mapping):
        for key, value in data.items():
            if key.endswith("_token") and isinstance(value, (str, dict)):
                add_role(key, value)
        for key in ("additional_special_tokens", "extra_special_tokens", "model_specific_special_tokens"):
            value = data.get(key)
            if value is None:
                continue
            if isinstance(value, dict) and key != "additional_special_tokens":
                for role, item in value.items():
                    add_role(role, item)
            elif isinstance(value, list) and key != "model_specific_special_tokens":
                for item in value:
                    generic.setdefault(_content(item), item)
            else:
                raise ValueError(f"Invalid {key} in tokenizer metadata")

    # A v5 save can turn the compatibility aliases back into named roles and
    # omit the generic list. Reconstruct that list before generating the map.
    aliases = {role: named.pop(role) for role in list(named) if role.startswith(ALIAS_PREFIX)}
    for text in aliases.values():
        generic.setdefault(text, text)

    # Collect backend-only specials as well. Keep the serialized token flags and IDs;
    # they control whitespace and normalization as well as decode(skip_special_tokens).
    decoder = config.get("added_tokens_decoder", {})
    backend_path = directory / "tokenizer.json"
    rows = json.loads(backend_path.read_bytes()).get("added_tokens", []) if backend_path.is_file() else []
    backend = {str(row["id"]): {k: v for k, v in row.items() if k != "id"} for row in rows}
    by_text = {}
    for table in (decoder, backend):
        for idx, row in table.items():
            text = _content(row)
            if text in by_text and by_text[text] != idx:
                raise ValueError(f"Conflicting vocabulary IDs for {text!r}")
            if idx in decoder and idx in backend and decoder[idx]["content"] != backend[idx]["content"]:
                raise ValueError(f"Conflicting token content at vocabulary ID {idx}")
            by_text[text] = idx
            if row.get("special") and text not in standard.values() and text not in named.values():
                generic.setdefault(text, text)

    def serialized(item):
        text = _content(item)
        idx = by_text.get(text)
        row = backend.get(idx) or decoder.get(idx) or {}
        # Re-registering a plain string resets these nondefault AddedToken flags
        # in some HF backends, changing how adjacent text is tokenized.
        if any(row.get(flag) for flag in ("lstrip", "rstrip", "single_word", "normalized")):
            return dict(row, __type="AddedToken")
        return dict(item, __type="AddedToken") if isinstance(item, dict) else text

    # Some v5 loaders discard the v4 generic list if a named modality map is
    # present (transformers#47838). They also read top-level *_token attributes.
    # Give otherwise anonymous specials a stable alias to the *same* existing
    # string. v4 reads the list and its native modality map. No vocabulary row is
    # created by an alias. On a v5 re-save these aliases may be named roles.
    if named:
        bound = set(standard.values()) | set(named.values()) | set(aliases.values())
        for token in generic.keys() - bound:
            suffix = by_text.get(token, "utf8_" + token.encode("utf-8").hex())
            role = f"{ALIAS_PREFIX}{suffix}_token"
            if role in aliases and aliases[role] != token:
                raise ValueError(f"Conflicting special-token alias {role}")
            aliases[role] = token
    result = dict(config)
    # Qwen's legacy processor reads chat_template.json. AutoTokenizer, used
    # directly by the Web backend, reads a Jinja file or an embedded template.
    # Expose the same template to both, without replacing a trained override.
    legacy_template = directory / "chat_template.json"
    if (not result.get("chat_template") and not (directory / "chat_template.jinja").is_file()
            and not any((directory / "chat_templates").glob("*.jinja")) and legacy_template.is_file()):
        legacy = json.loads(legacy_template.read_bytes())
        template = legacy.get("chat_template") if isinstance(legacy, dict) else None
        if not isinstance(template, str) or not template.strip():
            raise ValueError(f"Missing default chat template in {legacy_template}")
        result["chat_template"] = template
    result.pop("model_specific_special_tokens", None)
    if named:
        result["extra_special_tokens"] = {role: serialized(text) for role, text in named.items()}
    else:
        result.pop("extra_special_tokens", None)
    result.update({role: serialized(text) for role, text in named.items()})
    result.update({role: serialized(text) for role, text in sorted(aliases.items())})
    for role, text in standard.items():
        if result.get(role) is None:
            result[role] = serialized(text)
    if generic or "additional_special_tokens" in result:
        result["additional_special_tokens"] = [serialized(item) for item in generic.values()]
    # v4/v5 otherwise import the legacy special_tokens_map again, which may
    # contain the very list-instead-of-dict we just resolved. tokenizer.json
    # provides the original AddedToken IDs/flags when a v5 writer omitted them.
    if mapping and "added_tokens_decoder" not in config and backend:
        result["added_tokens_decoder"] = backend
    if result == config:
        return False
    if not check_only:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".tokenizer-config-", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            tmp_path.write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
            if config_path.read_bytes() != original:
                raise ValueError(f"Concurrent change to {config_path}")
            # Preserve the export's read permissions when publishing via replace.
            os.chmod(tmp_path, config_path.stat().st_mode)
            os.replace(tmp_path, config_path)
        finally:
            tmp_path.unlink(missing_ok=True)
    return True
