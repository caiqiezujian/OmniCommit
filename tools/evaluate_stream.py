#!/usr/bin/env python3
"""Compare one meeting's streaming result with its globally attributed reference."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omnicommit.evaluation import evaluate_events, transcript_events


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--id", help="Reference record ID/key; required if file has multiple records")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.resolve() in {args.reference.resolve(), args.predictions.resolve()}:
        parser.error("Choose a new output report path")
    refs = [json.loads(line) for line in args.reference.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if args.id is not None:
        refs = [r for r in refs if str(r.get("id", r.get("key"))) == args.id]
    if len(refs) != 1:
        parser.error("Reference selection must match exactly one meeting")
    events = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    report = evaluate_events(refs[0], transcript_events(events))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(report, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0 if report["complete_eos"] else 1


if __name__ == "__main__":
    sys.exit(main())
