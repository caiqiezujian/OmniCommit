#!/usr/bin/env python3
"""Filter finished Fourth SFT recordings without opening WAVs or changing training targets."""
import argparse
from bisect import bisect_right
from collections import Counter
from itertools import accumulate
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_asr.annotation_tail import simple_period_offset
from fourth_asr.chunk_refs import decode_chunk_ref
from fourth_asr.continuation_punctuation import continuation_replacement
from fourth_asr.protocol import PROTOCOLS, SPK_CHANGE, WAIT, protocol_for_tokens
from fourth_asr.semantic_preparation import STANDALONE
from fourth_asr.sft_checks import validate_row


STOPS = re.compile(r"[。．.!?！？…]+")
HAN = re.compile(r"[\u3400-\u9fff]+")
SHORT_CLAUSE = re.compile(r"[\u3400-\u9fff]{1,3}|[A-Za-z]+(?:['’][A-Za-z]+)?(?:\s+[A-Za-z]+)?")
ACKNOWLEDGMENTS = STANDALONE | {"行", "嗯嗯", "是的", "没错", "明白", "收到", "对的",
                                 "好的", "okay", "sure", "right", "thanks", "thank you"}
BY_PROMPT = {p.system_prompt(mode): p for p in PROTOCOLS.values() for mode in ("normalized", "verbatim")}
TRAIN_FILE = "train.filtered.jsonl"
ROW_ERRORS = (ValueError, TypeError, IndexError, KeyError, OverflowError, RecursionError)


def speaker_runs(row, profile):
    """Only compare text inside a reply or in consecutive DIRECTs with known speaker continuity."""
    run = []
    for turn in range(1, len(row["audios"]) + 1):
        reply = row["messages"][2 * turn]["content"]
        if reply == WAIT or not profile.speaker_changes:
            if run:
                yield run
            run = []
        if reply != WAIT:
            for index, part in enumerate(reply[len(profile.prefix):].split(SPK_CHANGE)):
                if index and run:
                    yield run
                    run = []
                if part:
                    run.append((turn, part))
    if run:
        yield run


def run_findings(run):
    text = "".join(part for _, part in run)
    if not any(mark in text for mark in (".", "。", "．")):
        return
    ends = list(accumulate(len(part) for _, part in run))
    clause_start = short_run = short_nonanswers = 0
    dense_reported = False
    for stop in STOPS.finditer(text):
        pos, end = stop.span()
        clause = text[clause_start:pos].strip()
        clause_start = end
        left = text[max(0, pos - 100):pos]
        if len(stop[0]) != 1 or stop[0] not in ".。．" or simple_period_offset(left + stop[0]) is None:
            short_run = short_nonanswers = 0
            dense_reported = False
            continue
        if SHORT_CLAUSE.fullmatch(clause):
            short_run += 1
            short_nonanswers += clause.casefold() not in ACKNOWLEDGMENTS
        else:
            short_run = short_nonanswers = 0
            dense_reported = False
        reasons = []
        if short_run >= 4 and short_nonanswers >= 2 and not dense_reported:
            reasons.append("four_short_sentences")
            dense_reported = True
        window = text[end:end + 100]
        right = window.lstrip()
        turn = run[bisect_right(ends, pos)][0]
        right_turn = run[bisect_right(ends, end + len(window) - len(right))][0] if right else None
        if right:
            if HAN.fullmatch(left[-1]) and HAN.fullmatch(right[0]):
                if continuation_replacement("zh", left, right) is not None:
                    reasons.append("suspect_period")
                for size in range(min(8, len(left), len(right)), 1, -1):
                    if (left[-size:] == right[:size] and HAN.fullmatch(right[:size])
                            and right[:size] not in ACKNOWLEDGMENTS):
                        reasons.append("repeated_boundary_text")
                        break
            elif left[-1].isascii() and left[-1].isalpha() and "a" <= right[0] <= "z":
                # Protect tokens such as domains inside a turn. A new DIRECT can lack a word space.
                if (window[0].isspace() or turn != right_turn) and continuation_replacement("en", left, right) is not None:
                    reasons.append("suspect_period")
        for reason in reasons:
            yield reason, {"turn": turn, "next_text_turn": right_turn,
                           "left": left[-45:] + stop[0], "right": right[:45]}


def punctuation_evidence(row, profile):
    counts, examples, max_per_run = Counter(), {}, 0
    for run in speaker_runs(row, profile):
        hits = 0
        for reason, sample in run_findings(run):
            counts[reason] += 1
            hits += reason == "suspect_period"
            if counts[reason] <= 3:
                examples.setdefault(reason, []).append(sample)
        max_per_run = max(max_per_run, hits)
    return max_per_run, counts, examples


def write_json(stream, value, original_record=None):
    data = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    encoded = data.encode("utf-8", "backslashreplace")
    if original_record is None:
        stream.write(encoded + b"\n")
    else:
        # This input already passed JSON and structural validation. Embed its actual bytes.
        stream.write(encoded[:-1] + b',"record":')
        stream.write(original_record.strip())
        stream.write(b"}\n")


def invalid_number(value):
    raise ValueError(f"Non-JSON number {value}")


def only_wait_targets(row):
    """Identify declared WAIT-only supervision, including in otherwise malformed rows for the audit."""
    if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
        return False
    has_reply = False
    for message in row["messages"]:
        if isinstance(message, dict) and message.get("role") == "assistant":
            if message.get("content") != WAIT:
                return False
            has_reply = True
    return has_reply


def punctuation_exclusion(maximum, signals, policy):
    if maximum >= 2:
        return "suspect_periods"
    if signals and policy == "strict":
        return "quarantined_punctuation_candidate"
    return None


def profile_after_leading_silence(stream, fallback, policy):
    """With no sidecar, use the first surviving speech profile, then resume at the original position."""
    position = stream.tell()
    try:
        for raw in stream:
            try:
                row = json.loads(raw.decode("utf-8"), parse_constant=invalid_number)
                if only_wait_targets(row):
                    continue
                validate_row(row, decode=False)
            except ROW_ERRORS:
                continue
            actual = BY_PROMPT[row["messages"][0]["content"]]
            maximum, signals, _ = punctuation_evidence(row, actual)
            if punctuation_exclusion(maximum, signals, policy) is None:
                return actual
    finally:
        stream.seek(position)
    return fallback


def scan(source, staging, profile, policy):
    stats = Counter(rows=0, kept=0, rejected=0, kept_for_review=0,
                    invalid_all_wait_rows=0, all_wait_system_prompts_adapted=0)
    exclusions, flagged_rows = Counter(), Counter()
    before = Counter(rows=0, audio_turns=0, direct_turns=0, wait_turns=0, silence_only_rows=0, speaker_change_rows=0)
    after = before.copy()
    with source.open("rb") as src, (staging / TRAIN_FILE).open("xb") as train, \
            (staging / "rejected.jsonl").open("xb") as rejected, (staging / "review.jsonl").open("xb") as review:
        for line, raw in enumerate(src, 1):
            if line == 1:
                raw = raw.removeprefix(b"\xef\xbb\xbf")
            stats["rows"] += 1
            reason = None
            row = None
            details = {"line": line}
            try:
                row = json.loads(raw.decode("utf-8"), parse_constant=invalid_number)
                used = validate_row(row, decode=False)
            except ROW_ERRORS as exc:
                reason = "invalid_record"
                details["error"] = str(exc)
                if only_wait_targets(row):
                    details["all_wait_targets"] = True
                    stats["invalid_all_wait_rows"] += 1
            else:
                actual = BY_PROMPT[row["messages"][0]["content"]]
                audio_turns = len(row["audios"])
                waits = sum(row["messages"][2 * t]["content"] == WAIT for t in range(1, audio_turns + 1))
                totals = dict(rows=1, audio_turns=audio_turns, direct_turns=audio_turns - waits, wait_turns=waits,
                              silence_only_rows=int(waits == audio_turns), speaker_change_rows=int(SPK_CHANGE in used))
                before.update(totals)
                if waits == audio_turns:
                    if profile is None:
                        profile = profile_after_leading_silence(src, actual, policy)
                    if actual != profile:
                        # Recognized Fourth protocols share the same standalone WAIT and audio events.
                        prompt = row["messages"][0]
                        number_format = "normalized" if prompt["content"] == actual.system_prompt("normalized") else "verbatim"
                        prompt["content"] = profile.system_prompt(number_format)
                        raw = (json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n").encode("ascii")
                        stats["all_wait_system_prompts_adapted"] += 1
                    after.update(totals)
                elif profile is not None and actual != profile:
                    reason = "protocol_mismatch"
                    details.update(expected_protocol=profile.name, actual_protocol=actual.name)
                else:
                    maximum, signals, examples = punctuation_evidence(row, actual)
                    if signals:
                        flagged_rows.update(signals.keys())
                        path = row.get("recast", {}).get("source_audio") or decode_chunk_ref(row["audios"][0])["path"]
                        details.update(source_audio=path,
                                       max_suspect_periods_in_run=maximum, signals=dict(signals), examples=examples)
                    reason = punctuation_exclusion(maximum, signals, policy)
                    if reason is None:
                        if profile is None:
                            profile = actual
                        after.update(totals)
                        if signals:
                            write_json(review, details)
                            stats["kept_for_review"] += 1
            if reason:
                details["reason"] = reason
                if reason == "invalid_record":
                    details["raw_line"] = raw.rstrip(b"\r\n").decode("utf-8", "replace")
                    write_json(rejected, details)
                else:
                    write_json(rejected, details, original_record=raw)
                exclusions[reason] += 1
                stats["rejected"] += 1
            else:
                train.write(raw)
                if not raw.endswith(b"\n"):
                    train.write(b"\n")
                stats["kept"] += 1
            if line % 10000 == 0:
                print(f"Rows {line} | kept {stats['kept']} | rejected {stats['rejected']} | "
                      f"kept for review {stats['kept_for_review']}", file=sys.stderr, flush=True)
    if profile is not None:
        tokens = ("\n".join(profile.tokens) + "\n").encode("utf-8")
        for name in ("token.txt", TRAIN_FILE + ".tokens.txt"):
            (staging / name).write_bytes(tokens)
    return {**stats, "input": str(source), "protocol": profile.name if profile else None, "policy": policy,
            "rejected_by_reason": dict(exclusions), "rows_with_signal": dict(flagged_rows),
            "validated_before": dict(before), "kept_after": dict(after)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Final streaming train JSONL")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory for the filtered dataset and audit")
    parser.add_argument("--tokens-file", type=Path,
                        help="Default: INPUT.tokens.txt, else first kept speech profile (first valid WAIT-only row if no speech)")
    parser.add_argument("--policy", choices=("conservative", "strict"), default="conservative",
                        help="conservative excludes 2+ lexical hits in one run; strict also excludes all review candidates")
    args = parser.parse_args(argv)
    source, output = args.input.resolve(), args.output_dir.resolve()
    if not source.is_file() or output.exists():
        parser.error("Select an existing input file and a new output directory")
    tokens = args.tokens_file or Path(str(source) + ".tokens.txt")
    try:
        profile = None
        if args.tokens_file is not None or tokens.exists():
            profile = protocol_for_tokens(tokens.read_text(encoding="utf-8-sig").splitlines())
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temp:
            staging = Path(temp) / "result"
            staging.mkdir()
            report = scan(source, staging, profile, args.policy)
            (staging / "summary.json").write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
            if output.exists():
                raise FileExistsError(f"Another run created {output}")
            staging.rename(output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Filter failed: {exc}\n")
    print(f"Rows {report['rows']} | kept {report['kept']} | rejected {report['rejected']} | "
          f"kept for review {report['kept_for_review']} | {output}", flush=True)
    print(f"Valid all-WAIT recordings: kept {report['kept_after']['silence_only_rows']}/"
          f"{report['validated_before']['silence_only_rows']} | "
          f"malformed all-WAIT rows: {report['invalid_all_wait_rows']}", flush=True)
    return 0 if report["kept"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
