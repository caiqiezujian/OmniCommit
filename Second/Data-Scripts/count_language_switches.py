"""Count adjacent, directed language switches in annotated OmniCommit JSONL."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from second_asr.protocol import LANGUAGE, SPEAKER, WAIT
from second_asr.source import HEADER_TIME, TAG

LANG = re.compile(LANGUAGE)
SPK = re.compile(SPEAKER)


def language_runs(record):
    """Count language states only when the assistant actually emits text."""
    messages = record.get("messages") if isinstance(record, dict) else None
    if not isinstance(messages, list):
        raise ValueError("missing messages array")
    runs = []
    current = None
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("assistant content is not text")
        if content.strip() == WAIT:
            continue
        content = HEADER_TIME.sub("", content.removeprefix("<|DIRECT|>"))
        pos = 0
        for match in TAG.finditer(content):
            append_text_language(content[pos:match.start()], current, runs)
            tag = match[1]
            if LANG.fullmatch(tag):
                current = tag
            elif not SPK.fullmatch(tag):
                raise ValueError(f"unknown transcript tag <{tag}>")
            pos = match.end()
        append_text_language(content[pos:], current, runs)
    if not runs:
        raise ValueError("no language-tagged transcript text")
    return runs


def append_text_language(text, current, runs):
    if not text.strip():
        return
    if "<" in text or ">" in text:
        raise ValueError("malformed transcript tag")
    if current is None:
        raise ValueError("text before initial language tag")
    if not runs or runs[-1] != current:
        runs.append(current)


def scan(path, languages):
    totals = dict(lines=0, blank=0, invalid=0, valid=0, monolingual=0, multilingual=0)
    by_language, by_group = Counter(), Counter()
    events, records = Counter(), Counter()
    pair_events, pair_records = Counter(), Counter()
    with path.open("rb") as source:
        for number, raw in enumerate(source, 1):
            totals["lines"] = number
            if not raw.strip():
                totals["blank"] += 1
                continue
            try:
                runs = language_runs(json.loads(raw))
            except ValueError as exc:
                totals["invalid"] += 1
                if totals["invalid"] <= 5:
                    print(f"Skipped line {number}: {exc}", file=sys.stderr)
                continue
            group = tuple(sorted(set(runs)))
            by_group[group] += 1
            by_language.update(group)
            totals["valid"] += 1
            if len(group) == 1:
                totals["monolingual"] += 1
                continue
            totals["multilingual"] += 1
            edges = Counter(zip(runs, runs[1:]))
            events.update(edges)
            records.update(edges.keys())
            if len(group) == 2:
                pair_events.update(edges)
                pair_records.update(edges.keys())

    codes = sorted(set(languages) | set(by_language))
    directions = []
    for a, b in combinations(codes, 2):
        for src, dst in ((a, b), (b, a)):
            edge = (src, dst)
            directions.append(dict(source=src, target=dst, events=events[edge],
                                   records=records[edge], pair_only_events=pair_events[edge],
                                   pair_only_records=pair_records[edge]))
    return dict(totals=totals,
                languages=[dict(language=code, records=by_language[code],
                                monolingual=by_group[(code,)],
                                multilingual=by_language[code] - by_group[(code,)])
                           for code in codes],
                exact_groups={"+".join(group): count for group, count in sorted(by_group.items())},
                directions=directions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Timestamped, augmented, or Second SFT JSONL")
    parser.add_argument("--report", type=Path, help="Also write a JSON report with exact language groups")
    parser.add_argument("--languages", nargs="+", default=[], metavar="CODE",
                        help="Include these languages even when absent; observed languages are always included")
    args = parser.parse_args()
    if any(not LANG.fullmatch(code) for code in args.languages):
        parser.error("invalid code in --languages")
    if args.report and (args.input.resolve() == args.report.resolve()
                        or (args.report.exists() and args.input.samefile(args.report))):
        parser.error("input and report must be different files")

    report = scan(args.input, args.languages)
    print("JSONL", json.dumps(report["totals"]))
    print("\n语种\t含该语种的记录\t单语记录\t多语记录")
    for item in report["languages"]:
        print("{language}\t{records}\t{monolingual}\t{multilingual}".format(**item))
    print("\n方向\t全库切换次数\t全库覆盖记录\t纯双语切换次数\t纯双语覆盖记录")
    for item in report["directions"]:
        print("{source}->{target}\t{events}\t{records}\t{pair_only_events}\t{pair_only_records}".format(**item))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
