"""Training-time identity of action vocabulary rows in a Fourth PEFT checkpoint."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from .protocol import DIRECT, WAIT

FILENAME = "omnicommit_action_rows.json"


def _validate_ids(token_ids):
    if not isinstance(token_ids, dict) or set(token_ids) not in ({WAIT}, {WAIT, DIRECT}):
        raise ValueError(f"Action-row training requires {WAIT} with optional {DIRECT}")
    if any(type(value) is not int or value < 0 for value in token_ids.values()) or len(set(token_ids.values())) != len(token_ids):
        raise ValueError("Action-row IDs must be distinct nonnegative integers")


def write_action_rows_manifest(directory, token_ids):
    """Call on the writer rank after the effective weights have been saved."""
    _validate_ids(token_ids)
    destination = Path(directory) / FILENAME
    payload = {"schema_version": 1, "token_ids": token_ids}
    handle = NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent, delete=False)
    path = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(path, destination)
    finally:
        path.unlink(missing_ok=True)


def read_action_rows_manifest(directory, token_ids, *, required=False):
    path = Path(directory) / FILENAME
    if not path.is_file():
        if required:
            raise ValueError(f"Missing action-row checkpoint manifest: {path}")
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError(f"Unsupported action-row checkpoint manifest: {path}")
    _validate_ids(payload.get("token_ids"))
    if payload["token_ids"] != token_ids:
        raise ValueError(f"Tokenizer IDs disagree with the training-time action rows: {path}")
    return payload
