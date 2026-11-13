"""Audit language/speaker switches in the final training JSONL on the original audio clock."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from contextlib import ExitStack
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omnicommit.switch_audit import enrich_with_alignment, find_switches, parse_sft_row
from omnicommit.switch_audit_report import render_review
from omnicommit.switch_audit_sidecars import Index, match_trajectory, pathkey
from omnicommit.switch_audit_traces import load_trace, summarize_switch


def dump_line(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sft", type=Path, help="Final messages/audios JSONL actually passed to SFT")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-jsonl", type=Path, help="Trusted attributed transcript with switch or segment timestamps")
    parser.add_argument("--trajectory-jsonl", type=Path, help="Optional recast trajectory, auto-detects SFT.jsonl.trajectory.jsonl")
    parser.add_argument("--trace", metavar="ARG", nargs="+", action="append", default=[],
                        help="LOG [ORIGINAL_AUDIO [OFFSET_SECONDS]], repeat for each model run")
    parser.add_argument("--audio", help="Only this recording (full source identity, never a basename search)")
    parser.add_argument("--audio-root", type=Path, help="Base directory for relative recording paths")
    parser.add_argument("--protocol", choices=["auto", "first", "second"], default="auto")
    parser.add_argument("--kind", choices=["all", "language", "speaker", "both"], default="all",
                        help="Exact change type; language/speaker alone exclude simultaneous changes")
    parser.add_argument("--from-language")
    parser.add_argument("--to-language")
    parser.add_argument("--sample-size", type=int, default=30, help="Number of cases in the listening report")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--before-seconds", type=float, default=2)
    parser.add_argument("--after-seconds", type=float, default=4)
    parser.add_argument("--write-clips", action="store_true", help="Write source WAV excerpts for sampled cases")
    args = parser.parse_args(argv)
    if args.sample_size < 0:
        parser.error("--sample-size must be nonnegative")
    if any(not math.isfinite(x) or x < 0 for x in (args.before_seconds, args.after_seconds)):
        parser.error("Window durations must be nonnegative finite seconds")
    if any(len(entry) > 3 for entry in args.trace):
        parser.error("--trace takes LOG [ORIGINAL_AUDIO [OFFSET_SECONDS]]")
    return args


def load_traces(args):
    traces = defaultdict(dict)
    names = set()
    for spec in args.trace:
        name = spec[0]
        if name in names:
            raise ValueError(f"Repeated trace path: {name}")
        names.add(name)
        source = spec[1] if len(spec) >= 2 else None
        trace = load_trace(name, source=source, offset_seconds=float(spec[2]) if len(spec) == 3 else None)
        traces[pathkey(trace.source, args.audio_root)][name] = trace
    return traces


def select_events(parsed, source, trajectory, issue):
    audio = parsed["audio"]
    provenance, aligned = {}, None
    if trajectory is not None:
        provenance, aligned, warning = match_trajectory(trajectory.get(audio), parsed)
        if warning:
            issue("trajectory_join", warning)
    candidates = []
    if source is not None:
        rows = source.get(audio)
        if len(rows) == 1:
            candidates.append(("source", rows[0]))
        else:
            issue("source_join", f"Expected one source record for this recording, found {len(rows)}")
    if aligned is not None:
        candidates.append(("trajectory_alignment", aligned))
    for origin, reference in candidates:
        try:
            events = find_switches(parsed, reference)
        except (TypeError, ValueError, KeyError) as exc:
            issue("reference_rejected", f"{origin}: {exc}")
        else:
            if origin == "source" and aligned is not None:
                events, warnings = enrich_with_alignment(events, parsed, aligned)
                for warning in warnings:
                    issue("secondary_alignment", warning)
            return events, provenance, origin
    return find_switches(parsed), provenance, "sft_only"


def add_turn_profile(profiles, parsed, events, origin, after):
    # Compare like-for-like within switching recordings; single-state files get their own baseline.
    if origin == "sft_only" or not parsed["complete"]:
        return
    onsets = [e["onset_seconds"] for e in events if e["event_basis"] == "reference"]
    if any(t is None for t in onsets):
        return
    for turn in parsed["turns"]:
        if turn["event"] != "audio":
            continue
        start, end = turn["start_seconds"], turn["end_seconds"]
        i = bisect_left(onsets, end)
        if not onsets:
            group = "single_attribution_recordings"
        elif i and start < onsets[i - 1] + after:
            group = "at_or_after_switch"
        elif i < len(onsets) and end > onsets[i] - after:
            continue  # Leave a guard before the switch, so it doesn't enter the ordinary baseline.
        else:
            group = "away_from_switch_same_recordings"
        p = profiles[group]
        p["audio_turns"] += 1
        p["audio_seconds"] += end - start
        p["wait_turns"] += int(turn["action"] == "WAIT")
        p["direct_turns"] += int(turn["action"] == "DIRECT")
        p["direct_characters"] += sum(len(s["text"]) for s in turn["segments"])


def count_commits(parsed, profiles):
    """Describe what the training targets emit even if there is no acoustic annotation."""
    state = parsed["initial_state"]
    waits = 0
    for turn in parsed["turns"]:
        if turn["event"] != "audio":
            continue
        if turn["action"] == "WAIT":
            waits += 1
            continue
        initial = state is None
        changed = False
        for span in turn["segments"]:
            current = (span["language"], span["speaker"])
            if state is not None and current != state:
                changed = True
            state = current
        group = "switch_commit" if changed else ("first_attribution" if initial else "continuing_attribution")
        p = profiles[group]
        p["commits"] += 1
        p["preceding_wait_turns"] += waits
        p["max_preceding_wait_turns"] = max(p["max_preceding_wait_turns"], waits)
        p["characters"] += sum(len(s["text"]) for s in turn["segments"])
        waits = 0


def switch_case(parsed, event, line, number, origin, args, traces, waits, intervals):
    index = event["gold_turn"] - 1
    turns = parsed["turns"]
    commit = event["gold_seconds"]
    onset = event["onset_seconds"]
    anchor = commit if onset is None else onset
    # Include the gold turn even when an entire new span was released late or at EOS.
    lo = max(parsed["start_seconds"], min(anchor, turns[index]["start_seconds"]) - args.before_seconds)
    expected_time = event.get("gold_expected_tag_seconds") if event["event_basis"] == "reference" else None
    hi = min(parsed["end_seconds"], max(anchor, commit, expected_time or 0) + args.after_seconds)
    starts, ends = intervals
    window = turns[bisect_right(ends, lo):bisect_left(starts, hi)]
    if turns[-1]["event"] == "end_of_audio" and lo <= turns[-1]["end_seconds"] <= hi:
        window = [*window, turns[-1]]
    timing_valid = onset is not None and event.get("onset_in_window") is not False
    warnings = []
    if onset is None:
        warnings.append("缺少可对齐的参考切换时刻，仅能按 SFT 提交轮查看。")
    elif not timing_valid:
        warnings.append("原始切换点在当前训练窗口之外，本窗口只能展示部分上下文。")
    if event.get("gold_transition_matches") is False:
        warnings.append("此处 SFT 的标签变化与独立参考不一致，请检查原始正文及其归属。")
    if onset is not None and commit < onset:
        warnings.append("SFT 在参考切换点之前已经提交了该处正文，请核对标注时间。")
    case = dict(case_id=f"L{line}-S{number}", sft_line=line, audio=parsed["audio"], protocol=parsed["protocol"],
                kind=event["kind"], before=dict(language=event["previous_language"], speaker=event["previous_speaker"]),
                after=dict(language=event["language"], speaker=event["speaker"]), time_basis=event["anchor_source"],
                reference_origin=origin, event_basis=event["event_basis"], anchor_seconds=onset,
                evidence_seconds=event["first_unit_end_seconds"], gold_commit_seconds=commit,
                evidence_text=event.get("evidence_text"), evidence_source=event.get("evidence_source"),
                trajectory_onset_seconds=event.get("trajectory_onset_seconds"),
                onset_in_window=event.get("onset_in_window"),
                gold_delay_seconds=round(commit - onset, 6) if timing_valid else None,
                gold_turn=event["gold_turn"], gold_event=turns[index]["event"],
                gold_expected_tag_seconds=expected_time,
                gold_expected_tag_turn=event.get("gold_expected_tag_turn") if event["event_basis"] == "reference" else None,
                wait_turns_before_commit=waits[index],
                new_span_characters=len(event["gold_text"]),
                first_commit_characters=(event["first_commit_characters"] if event["event_basis"] == "reference" else None),
                turn_characters=sum(len(s["text"]) for s in turns[index]["segments"]),
                gold_language=event["gold_language"], gold_speaker=event["gold_speaker"],
                gold_attribution_matches=(event["gold_attribution_matches"] if event["event_basis"] == "reference" else None),
                gold_transition_matches=event.get("gold_transition_matches"),
                reference_text=event["reference_text"], gold_text=event["gold_text"],
                reference_char_offset=event["reference_char_offset"], char_offset=event["char_offset"],
                window_start_seconds=lo, window_end_seconds=hi, warnings=warnings, models={})
    for name, trace in traces.items():
        case["models"][name] = summarize_switch(trace, window, language=event["language"], kind=event["kind"],
                                                anchor_seconds=onset if timing_valid else None, window_end_seconds=hi,
                                                next_onset_seconds=event["next_onset_seconds"])
        protocol = trace.header.get("protocol")
        if protocol in ("first", "second") and protocol != parsed["protocol"]:
            warnings.append(f"{name} 的输出协议为 {protocol}，训练样本为 {parsed['protocol']}。")
    return case, window


def detail(window, provenance, traces):
    return [dict(turn=t["turn"], event=t["event"], start_seconds=t["start_seconds"], end_seconds=t["end_seconds"],
                 gold=t["raw"], gold_segments=t["segments"], provenance=provenance.get(t["turn"]),
                 models={name: trace.lookup(t["start_seconds"], t["end_seconds"], t["event"])
                         for name, trace in traces.items()}) for t in window]


def extract_clip(case, output_dir, audio_root):
    import soundfile as sf

    audio = Path(case["audio"])
    if not audio.is_absolute() and audio_root:
        audio = audio_root / audio
    with sf.SoundFile(str(audio)) as source:
        start = round(case["window_start_seconds"] * source.samplerate)
        end = round(case["window_end_seconds"] * source.samplerate)
        if not 0 <= start < end <= len(source):
            raise ValueError("Requested window falls outside the original recording")
        source.seek(start)
        samples = source.read(end - start, dtype="float32", always_2d=True)
        if len(samples) != end - start:
            raise ValueError("Short read from source recording")
        filename = Path("clips") / (case["case_id"] + ".wav")
        sf.write(str(output_dir / filename), samples, source.samplerate, subtype="FLOAT")
    return filename.as_posix()


def run(args):
    traces = load_traces(args)
    trajectory_path = args.trajectory_jsonl
    if trajectory_path is None:
        adjacent = Path(str(args.sft) + ".trajectory.jsonl")
        trajectory_path = adjacent if adjacent.is_file() else None
    filter_audio = pathkey(args.audio, args.audio_root) if args.audio else None
    counts = Counter(rows_read=0, rows_parsed=0, rows_selected=0, switches_found=0, switches_selected=0)
    issues, kinds, joined, directions, source_joins = Counter(), Counter(), Counter(), Counter(), Counter()
    profiles, commits = defaultdict(Counter), defaultdict(Counter)
    rng = random.Random(args.seed)
    cases = []
    out = args.output_dir
    with ExitStack() as stack:
        data = stack.enter_context(args.sft.open("rb"))
        source = stack.enter_context(Index(args.source_jsonl, args.audio_root)) if args.source_jsonl else None
        trajectory = stack.enter_context(Index(trajectory_path, args.audio_root)) if trajectory_path else None
        out.mkdir(parents=True, exist_ok=True)
        outputs = ["summary.json", "switches.jsonl", "cases.jsonl", "issues.jsonl", "review.md"]
        if args.write_clips:
            outputs.append("clips")
        if any((out / f).exists() for f in outputs):
            raise ValueError(f"Audit outputs already exist in {out}. Choose another --output-dir.")
        switches = stack.enter_context((out / "switches.jsonl").open("x", encoding="utf-8"))
        problems = stack.enter_context((out / "issues.jsonl").open("x", encoding="utf-8"))

        def log_issue(kind, **fields):
            issues[kind] += 1
            dump_line(problems, dict(kind=kind, **fields))

        for label, sidecar in (("source_index", source), ("trajectory_index", trajectory)):
            if sidecar is not None:
                for problem in sidecar.issues:
                    log_issue(label, diagnostic=problem)
        for line, raw in enumerate(data, 1):
            counts["rows_read"] += 1
            try:
                parsed = parse_sft_row(json.loads(raw), args.protocol)
                key = pathkey(parsed["audio"], args.audio_root)
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                log_issue("invalid_sft", sft_line=line, error=str(exc))
                continue
            counts["rows_parsed"] += 1
            if filter_audio is not None and key != filter_audio:
                continue
            counts["rows_selected"] += 1
            row_traces = traces.get(key, {})
            joined.update(row_traces.keys())
            def row_issue(kind, error):
                log_issue(kind, sft_line=line, audio=parsed["audio"], error=error)
            events, provenance, origin = select_events(parsed, source, trajectory, row_issue)
            source_joins[origin] += 1
            add_turn_profile(profiles, parsed, events, origin, args.after_seconds)
            count_commits(parsed, commits)
            waits, consecutive = [], 0
            intervals = ([t["start_seconds"] for t in parsed["turns"]], [t["end_seconds"] for t in parsed["turns"]])
            for turn in parsed["turns"]:
                waits.append(consecutive)
                consecutive = consecutive + 1 if turn["action"] == "WAIT" else 0
            for number, event in enumerate(events, 1):
                counts["switches_found"] += 1
                if (args.kind not in ("all", event["kind"])
                        or args.from_language and event["previous_language"] != args.from_language
                        or args.to_language and event["language"] != args.to_language):
                    continue
                counts["switches_selected"] += 1
                case, window = switch_case(parsed, event, line, number, origin, args, row_traces, waits, intervals)
                kinds[case["kind"]] += 1
                directions[event["previous_language"] + "->" + event["language"]] += 1
                counts["with_reference_onset"] += int(case["anchor_seconds"] is not None)
                counts["sft_wrong_attribution_at_reference_change"] += int(case["gold_attribution_matches"] is False)
                counts["sft_change_disagrees_with_reference"] += int(case["gold_transition_matches"] is False)
                dump_line(switches, case)
                slot = len(cases) if len(cases) < args.sample_size else rng.randrange(counts["switches_selected"])
                if slot < args.sample_size:
                    case["turns"] = detail(window, provenance, row_traces)
                    if slot == len(cases):
                        cases.append(case)
                    else:
                        cases[slot] = case
        for group in traces.values():
            for name in group:
                if not joined[name]:
                    log_issue("trace_source_not_in_selected_sft", trace=name, source=group[name].source)
        cases.sort(key=lambda case: (case["sft_line"], case["char_offset"]))
        if args.write_clips:
            (out / "clips").mkdir()
            for case in cases:
                try:
                    case["clip"] = extract_clip(case, out, args.audio_root)
                except (OSError, ValueError, RuntimeError, ImportError) as exc:
                    case["warnings"].append(f"试听片段生成失败：{exc}")
                    log_issue("clip_error", case_id=case["case_id"], audio=case["audio"], error=str(exc))
        for group in profiles.values():
            group["wait_fraction"] = round(group["wait_turns"] / group["audio_turns"], 4)
            group["characters_per_direct"] = (round(group["direct_characters"] / group["direct_turns"], 2)
                                               if group["direct_turns"] else None)
            group["audio_seconds"] = round(group["audio_seconds"], 6)
        for group in commits.values():
            group["characters_per_commit"] = round(group["characters"] / group["commits"], 2)
            group["mean_preceding_wait_turns"] = round(group["preceding_wait_turns"] / group["commits"], 2)
        summary = dict(sft=str(args.sft), source=str(args.source_jsonl) if args.source_jsonl else None,
                       trajectory=str(trajectory_path) if trajectory_path else None,
                       counts=dict(counts), by_kind=dict(kinds), by_language_pair=dict(directions),
                       row_reference_origins=dict(source_joins), issues=dict(issues), sampled=len(cases), seed=args.seed,
                       gold_audio_turn_profiles=dict(profiles), sft_commit_profiles=dict(commits),
                       profile_after_seconds=args.after_seconds,
                       traces={name: dict(source=trace.source, joined_sft_rows=joined[name], terminal=trace.terminal)
                               for group in traces.values() for name, trace in group.items()})
        with (out / "cases.jsonl").open("x", encoding="utf-8") as rows:
            for case in cases:
                dump_line(rows, case)
        (out / "review.md").write_text(render_review(summary, cases), encoding="utf-8")
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    args = get_args(argv)
    try:
        summary = run(args)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(dict(counts=summary["counts"], issues=summary["issues"], review=str(args.output_dir / "review.md")),
                     ensure_ascii=False))
    return 0 if summary["counts"]["rows_selected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
