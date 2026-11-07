#!/usr/bin/env python3
"""Render full DIRECT prefixes in existing annotations, without any model call."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotate_asr import (get_source_text, iter_jsonl, log, normalize_transcript,
                         parse_model_json, serialize_result, validate_annotation)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve() or args.output.exists():
        parser.error("Choose a new output path; source records are not overwritten")
    if not args.input.is_file():
        parser.error("Input JSONL file does not exist")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"written": 0, "valid": 0, "changed": 0, "existing_empty": 0, "invalid": 0}
    with args.output.open("x", encoding="utf-8") as out:
        for line_number, row in iter_jsonl(args.input):
            result = ""
            try:
                source = normalize_transcript(get_source_text(row))
                old = row.get("output")
                if old == "":
                    counts["existing_empty"] += 1
                else:
                    parsed = parse_model_json(old)
                    repaired = validate_annotation(parsed, source)
                    result = json.dumps(repaired, ensure_ascii=False, separators=(",", ":"))
                    counts["valid"] += 1
                    counts["changed"] += repaired != parsed
            except Exception as exc:
                counts["invalid"] += 1
                log(f"[ERROR] line={line_number}: {exc}; output remains empty")
            serialized, _, _ = serialize_result(row, result)
            out.write(serialized)
            out.flush()
            counts["written"] += 1
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
