"""Streaming JSONL I/O with explicit, validated append-only resume."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def source_rows(path):
    with open(path, encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                def reject_constant(value):
                    raise ValueError("Non-finite JSON constant: " + value)
                def finite_float(value):
                    number = float(value)
                    if not math.isfinite(number):
                        raise ValueError("Non-finite JSON number: " + value)
                    return number
                value = json.loads(line, parse_constant=reject_constant, parse_float=finite_float)
                if not isinstance(value, dict):
                    raise ValueError("JSONL row must be an object")
                yield number, value, None
            except (ValueError, TypeError) as error:
                yield number, {"_malformed_line_sha256": hashlib.sha256(line.encode()).hexdigest()}, str(error)


def row_id(value, source, line):
    result = value.get("id", value.get("key"))
    if result is None:
        result = "row-" + fingerprint({"file": str(Path(source).resolve()), "line": line})[:20]
    if not isinstance(result, (str, int)) or isinstance(result, bool):
        raise ValueError("id/key must be a string or integer")
    return str(result)


def write_row(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    stream.flush()


class Journal:
    """Keep record order. Existing failures are completed records, not retries.

    Resume refuses mismatching input/config and an incomplete final line instead
    of repairing or backing up user files. No request is made for resumed rows.
    """
    def __init__(self, output, config, resume=False):
        self.path = Path(output)
        self.config_hash = fingerprint(config)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists()
        if exists and not resume:
            raise ValueError("Output exists; use --resume or a new output path")
        if exists and self.path.stat().st_size:
            with self.path.open("rb") as check:
                check.seek(-1, 2)
                if check.read(1) != b"\n":
                    raise ValueError("Output has an incomplete final line; preserve it and use a new output path")
        self.existing = self.path.open(encoding="utf-8") if exists else None
        self.writer = None

    def previous(self, input_hash, identifier):
        if self.existing is None:
            return None
        line = self.existing.readline()
        if not line:
            self.existing.close()
            self.existing = None
            return None
        value = json.loads(line)
        if (value.get("input_hash"), value.get("config_hash"), value.get("id")) != (
                input_hash, self.config_hash, identifier):
            raise ValueError("Resume input/config/order mismatch; choose a new output path")
        return value

    def append(self, value):
        if self.existing is not None:
            raise ValueError("Cannot append before consuming existing output")
        if self.writer is None:
            self.writer = self.path.open("a", encoding="utf-8", newline="\n")
        value["config_hash"] = self.config_hash
        write_row(self.writer, value)

    def close(self, check_excess=True):
        excess = bool(self.existing and self.existing.readline()) if check_excess else False
        if self.existing:
            self.existing.close()
        if self.writer:
            self.writer.close()
        if excess:
            raise ValueError("Output contains more records than current input")
