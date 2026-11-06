#!/usr/bin/env python3
"""Audit every prepared SFT row without modifying it; --decode checks audio samples."""
import argparse
import json
from pathlib import Path
import sys

from check_sft import check_row, read_tokens


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report", type=Path, help="Write summary and invalid row locations to a new JSON file")
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--tokens-file", type=Path)
    parser.add_argument("--max-details", type=int, default=100, help="Bound retained error details; counts still cover all rows")
    args = parser.parse_args(argv)
    if args.report and args.report.exists():
        parser.error("Report already exists; choose another path")
    if args.max_details < 0:
        parser.error("--max-details must be nonnegative")
    count, passed, invalid_count, invalid = 0, 0, 0, []
    try:
        inventory = read_tokens(args.tokens_file or Path(str(args.dataset) + ".tokens.txt"))
        with args.dataset.open(encoding="utf-8-sig") as source:
            for line_no, line in enumerate(source, 1):
                if not line.strip():
                    continue
                count += 1
                try:
                    check_row(json.loads(line), decode=args.decode, token_inventory=inventory)
                    passed += 1
                except Exception as exc:
                    invalid_count += 1
                    if len(invalid) < args.max_details:
                        invalid.append({"line": line_no, "error": str(exc)})
                if count % 1000 == 0:
                    print(f"[audit] rows={count} passed={passed} invalid={invalid_count}", flush=True)
        report = {"total": count, "passed": passed, "invalid": invalid_count,
                  "full_dataset_checked": True, "decoded": args.decode,
                  "gpu_training_verified": False, "errors": invalid,
                  "error_details_truncated": invalid_count > len(invalid)}
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "errors"}), flush=True)
        return 0 if count and not invalid_count else 1
    except Exception as exc:
        print(f"[audit] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
