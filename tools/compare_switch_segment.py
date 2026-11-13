#!/usr/bin/env python3
"""Compare original-session and fresh-session streaming replies on the original audio clock."""
import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.compare_stream_traces import records


BACKENDS = ("hf", "vllm-on", "vllm-off")


@dataclass
class Trace:
    header: dict = field(default_factory=dict)
    audio: dict = field(default_factory=dict)  # Keyed by turn number within the selected interval.
    terminal: str = "incomplete: no EOS or error record"
    complete: bool = False


def grid_index(seconds, step):
    turn = round(seconds / step)
    if not math.isclose(turn * step, seconds, rel_tol=0, abs_tol=1e-7):
        raise ValueError(f"{seconds:g}s does not lie on the {step:g}s grid")
    return turn


def read_interval(path, start, end, fresh):
    trace = Trace()
    offset = start if fresh else 0
    try:
        last_turn = 0
        last_end = 0.0
        finished = False
        for idx, row in enumerate(records(path)):
            if not isinstance(row, dict) or finished:
                raise ValueError(f"unexpected record {idx + 1}")
            if row.get("type") == "error":
                msg = f"ERROR {row.get('stage', '?')}: {row.get('error', '?')}"
                attempted = row.get("attempted_end_seconds", row.get("heard_seconds"))
                if attempted is not None:
                    msg += f" (original {offset + attempted:g}s)"
                if row.get("raw_model_text") is not None:
                    msg += f" raw={json.dumps(row['raw_model_text'], ensure_ascii=False)}"
                trace.terminal = msg
                finished = True
                continue
            if idx == 0:
                if row.get("type") != "ready" or type(row.get("chunk_ms")) is not int or row["chunk_ms"] <= 0:
                    raise ValueError("missing ready header or invalid chunk_ms")
                trace.header = row
                step = row["chunk_ms"] / 1000
                first = grid_index(start - offset, step)
                limit = grid_index(end - offset, step)
                continue
            if row.get("type") != "result" or row.get("turn") != last_turn + 1:
                raise ValueError(f"missing or invalid result at turn {last_turn + 1}")
            last_turn = row["turn"]
            raw = row.get("raw_model_text")
            if not isinstance(raw, str) or row.get("action") not in ("WAIT", "DIRECT"):
                raise ValueError(f"invalid model result at turn {last_turn}")
            left, right = row.get("start_seconds"), row.get("end_seconds")
            if not all(type(x) in (int, float) and math.isfinite(x) for x in (left, right)):
                raise ValueError(f"invalid audio timestamps at turn {last_turn}")
            if not math.isclose(left, last_end, abs_tol=1e-7, rel_tol=0):
                raise ValueError(f"noncontiguous interval at turn {last_turn}")
            if row.get("event") == "end_of_audio" and left == right:
                trace.terminal = f"EOS at original {offset + right:g}s: {json.dumps(raw, ensure_ascii=False)}"
                if right + offset < end - 1e-7:
                    trace.terminal += " (ended before requested interval was covered)"
                else:
                    trace.complete = True
                finished = True
                continue
            if row.get("event") != "audio" or right <= left:
                raise ValueError(f"invalid audio event at turn {last_turn}")
            if fresh and right + offset > end + 1e-7:
                raise ValueError("fresh trace exceeds the selected interval")
            if first < last_turn <= limit:
                if not (grid_index(left, step) == last_turn - 1 and grid_index(right, step) == last_turn):
                    raise ValueError(f"turn {last_turn} is off the expected grid")
                trace.audio[last_turn - first] = (row["action"], raw)
            last_end = right
    except (OSError, ValueError) as exc:
        trace.terminal = f"unavailable or incomplete: {exc}"
        trace.complete = False
    return trace


def print_report(groups, start, end, language):
    traces = [trace for group in groups.values() for trace in group.values()]
    steps = {trace.header["chunk_ms"] for trace in traces if trace.header}
    if len(steps) > 1:
        raise ValueError(f"incompatible chunk_ms settings: {sorted(steps)}")
    if steps:
        step = next(iter(steps)) / 1000
        first = grid_index(start, step)
        count = grid_index(end - start, step)
        tagged_text = re.compile(rf"<{re.escape(language)}><spk[1-9][0-9]*>([^<>]*)")
    for group, runs in groups.items():
        print(f"\n[{group}]")
        if steps:
            print("\t".join(("clip_turn", "original_turn", "clip_end_s", "original_end_s", *runs)))
            for turn in range(1, count + 1):
                replies = [json.dumps(run.audio[turn][1], ensure_ascii=False) if turn in run.audio
                           else "(no result)" for run in runs.values()]
                print("\t".join((str(turn), str(first + turn), f"{turn * step:g}",
                                 f"{start + turn * step:g}", *replies)))
        for name, trace in runs.items():
            first_direct = first_tag = None
            waits = 0
            for turn, (action, raw) in trace.audio.items():
                if action == "DIRECT":
                    if first_direct is None:
                        first_direct = turn
                    if first_tag is None and any(text.strip() for text in tagged_text.findall(raw)):
                        first_tag = turn
                elif first_direct is None:
                    waits += 1

            def when(turn):
                if turn is None:
                    return "unobserved"
                return f"{start + turn * step:g}s (+{turn * step:g}s)"

            print(f"{name}: initial observed WAITs={waits}, first text={when(first_direct)}, "
                  f"first explicit <{language}> with text={when(first_tag)}")
            print(f"  terminal: {trace.terminal}")
    print("\nTimes refer to completed input turns. Audio-turn metrics exclude EOS. "
          "The language metric checks the emitted tag only.")
    return 0 if all(trace.complete for trace in traces) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in BACKENDS:
        parser.add_argument(f"--{name}", type=Path, required=True, metavar="FRESH_TRACE")
        parser.add_argument(f"--full-{name}", type=Path, metavar="ORIGINAL_TRACE")
    parser.add_argument("--start-seconds", type=float, required=True)
    parser.add_argument("--end-seconds", type=float, required=True)
    parser.add_argument("--language", default="de", help="Explicit language tag to summarize, e.g. de")
    args = parser.parse_args(argv)
    start, end = args.start_seconds, args.end_seconds
    if not (0 <= start < end and math.isfinite(end)):
        parser.error("Expected finite original times 0 <= start < end")
    if not re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", args.language):
        parser.error("Expected a language code such as de")
    groups = {}
    for fresh, title in ((True, "fresh session"), (False, "original session")):
        runs = {}
        for name in BACKENDS:
            key = name.replace("-", "_")
            path = getattr(args, key if fresh else "full_" + key)
            if path is not None:
                runs[name] = read_interval(path, start, end, fresh)
        if runs:
            groups[title] = runs
    try:
        return print_report(groups, start, end, args.language)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
