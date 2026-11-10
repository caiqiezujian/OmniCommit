#!/usr/bin/env python3
"""Reference-check local HF prefixes and export directly trainable compact SFT."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SFT-Scripts"))
from omnicommit.io import source_rows, write_row
from omnicommit.prefix_labels import ALGORITHM, build_trajectory
from prepare_sft import ACTION_TOKENS, prepare_row, sort_tokens
from check_sft import check_row


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="collect_hf_prefixes.py trace JSONL")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--comparison", choices=("surface", "exact"), default="surface")
    p.add_argument("--max-wait-ratio", type=float, default=1., help="Optional filter; default reports only")
    p.add_argument("--max-eos-text-ratio", type=float, default=1., help="Optional filter; default reports only")
    return p


def run(args):
    for name in ("max_wait_ratio", "max_eos_text_ratio"):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(name + " must be between 0 and 1")
    source = args.input.resolve(strict=True)
    output = args.output.resolve()
    paths = {"train": output, **{name: Path(str(output) + suffix) for name, suffix in (
        ("tokens", ".tokens.txt"), ("rejected", ".rejected.jsonl"),
        ("review", ".review.jsonl"), ("trajectory", ".trajectory.jsonl"), ("summary", ".summary.json"))}}
    if source in paths.values() or any(p.exists() for p in paths.values()):
        raise ValueError("Input/output must differ; output and sidecars must be new paths")
    output.parent.mkdir(parents=True, exist_ok=True)
    inventory = set(ACTION_TOKENS)
    counts = Counter()
    language_rows, language_chars, reject_reasons = Counter(), Counter(), Counter()
    from contextlib import ExitStack
    with ExitStack() as stack:
        handles = {key: stack.enter_context(paths[key].open("x", encoding="utf-8", newline="\n"))
                   for key in ("train", "rejected", "review", "trajectory")}
        for line, trace, error in source_rows(source):
            counts["input_rows"] += 1
            try:
                if error:
                    raise ValueError(error)
                trajectory = build_trajectory(trace, args.comparison)
                m = trajectory["metrics"]
                wait_ratio = m["wait_turns"] / m["audio_turns"]
                eos_ratio = m["eos_characters"] / m["total_characters"]
                if wait_ratio > args.max_wait_ratio:
                    raise ValueError("WAIT ratio exceeds configured maximum")
                if eos_ratio > args.max_eos_text_ratio:
                    raise ValueError("EOS text ratio exceeds configured maximum")
                prepared, tokens = prepare_row(trajectory)
                compact = {"audios": prepared["audios"], "messages": prepared["messages"]}
                check_row(compact, decode=False, token_inventory=tokens)
                # Serialize before writing, so a malformed row cannot leave a
                # partially written training line. File-system failures propagate.
                serialized = json.dumps(compact, ensure_ascii=False, allow_nan=False)
                serialized.encode("utf-8")
            except (MemoryError, ImportError):
                raise
            except Exception as exc:
                counts["rejected_rows"] += 1
                reason = str(exc)
                reject_reasons[reason.split(":", 1)[-1].strip()] += 1
                write_row(handles["rejected"], {"line": line, "id": trace.get("id"),
                                               "audio": trace.get("audio"), "error": reason})
                print(f"[reject] line={line}: {reason}", file=sys.stderr, flush=True)
                continue
            handles["train"].write(serialized + "\n")
            handles["train"].flush()
            write_row(handles["trajectory"], trajectory)
            counts["kept_rows"] += 1
            counts["audio_turns"] += m["audio_turns"]
            counts["wait_turns"] += m["wait_turns"]
            counts["eos_characters"] += m["eos_characters"]
            counts["total_characters"] += m["total_characters"]
            language_rows.update(m["languages"])
            for turn in trajectory["turns"]:
                for span in turn["segments"]:
                    language_chars[span["language"]] += len(span["text"])
            inventory.update(tokens)
            warnings = []
            if wait_ratio >= .8:
                warnings.append("at_least_80_percent_audio_turns_wait")
            if eos_ratio >= .5:
                warnings.append("at_least_half_text_released_at_eos")
            if warnings:
                counts["review_rows"] += 1
                write_row(handles["review"], {"line": line, "id": trace.get("id"),
                    "audio": trace["audio"], "warnings": warnings,
                    "wait_ratio": wait_ratio, "eos_text_ratio": eos_ratio})
    paths["tokens"].write_text("\n".join(sort_tokens(inventory)) + "\n", encoding="utf-8")
    summary = {"algorithm": ALGORITHM, "comparison": args.comparison, **dict(counts),
               "language_rows": dict(language_rows), "language_characters": dict(language_chars),
               "reject_reasons": dict(reject_reasons), "paths": {k: str(v) for k, v in paths.items()},
               "full_structural_check": True, "waveform_decode_checked": False,
               "real_template_length_checked": False, "acoustic_quality_verified": False,
               "note": "Synthetic/local teacher supervision; validate switch timing and recognition by audio replay."}
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main(argv=None):
    try:
        result = run(parser().parse_args(argv))
        return 0 if result.get("kept_rows", 0) else 1
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
