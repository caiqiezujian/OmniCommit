"""CER-led trajectory rewards for Fourth streaming ASR."""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from math import ceil, isclose
import unicodedata

from .protocol import PROTOCOL, SPK_CHANGE, WAIT, total_turns
from .character_quality import cer_metrics, local_switch_cer
from .speaker_quality import speaker_metrics
from .timing_quality import timing_metrics

from .sentence_boundaries import sentence_boundary_errors
from .switch_quality import disjoint_script_metrics, switch_quality_metrics

SWITCH_HEAD = 3
REWARD_VERSION = "fourth_boundary_v2"


@dataclass(frozen=True)
class Unit:
    value: str
    start: int
    end: int


def _character_script(c):
    n = ord(c)
    return (0x3400 <= n <= 0x9fff or 0xf900 <= n <= 0xfaff or 0x20000 <= n <= 0x323af
            or 0x3040 <= n <= 0x30ff or 0x31f0 <= n <= 0x31ff or 0xff66 <= n <= 0xff9d
            or 0xac00 <= n <= 0xd7a3 or 0xe01 <= n <= 0xe5b)


def _units(text, characters=False):
    """Tokenize both sides identically and retain original Python offsets for timing."""
    result = []
    i = 0
    while i < len(text):
        c = text[i]
        if not c.isalnum():
            i += 1
            continue
        start = i
        i += 1
        single = characters or (_character_script(c) and not c.isdecimal())
        while i < len(text):
            following = text[i]
            if unicodedata.category(following).startswith("M"):
                i += 1
            elif not single and following.isalnum() and (not _character_script(following) or following.isdecimal()):
                i += 1
            else:
                break
        form = unicodedata.normalize("NFKC", text[start:i]).casefold()
        form = "".join(str(unicodedata.decimal(ch)) if ch.isdecimal() else ch for ch in form)
        result.append(Unit(form, start, i))
    return result


def _edits(ref, hyp, commit_times, onsets, offsets, duration):
    """Minimize edits, then maximize matches. Time only breaks remaining ties.

    For repeated words prefer fewer pre-onset matches, then smaller acoustic
    time differences. Remaining ties map to earlier occurrences. Each integer
    weight dominates all lower-priority terms combined over the entire path.
    """
    m = len(hyp)
    max_matches = min(len(ref), m)
    max_ms = ceil(duration * 1000)
    early_weight = max_matches * max_ms + 1
    match_weight = max_matches * (early_weight + max_ms) + 1
    edit_weight = max_matches * match_weight + 1
    hyp_ms = [round(t * 1000) for t in commit_times]
    previous = [j * edit_weight for j in range(m + 1)]
    path = [bytearray([2]) * (m + 1)]
    for i, unit in enumerate(ref, 1):
        directions = bytearray(m + 1)
        directions[0] = 1
        current = [i * edit_weight] + [0] * m
        onset = round(1000 * onsets[i - 1]) if i - 1 in onsets else None
        target = round(1000 * offsets[i - 1]) if i - 1 in offsets else onset
        for j, guess in enumerate(hyp, 1):
            if unit.value == guess.value:
                observed = hyp_ms[j - 1]
                diag = previous[j - 1] - match_weight
                if onset is not None and observed < onset:
                    diag += early_weight
                if target is not None:
                    diag += abs(observed - target)
            else:
                diag = previous[j - 1] + edit_weight
            delete = previous[j] + edit_weight
            insert = current[j - 1] + edit_weight
            score = min(diag, delete, insert)
            current[j] = score
            if score == delete:
                directions[j] = 1
            elif score == insert:
                directions[j] = 2
        path.append(directions)
        previous = current
    operations = []
    i, j = len(ref), m
    while i or j:
        direction = path[i][j]
        if direction == 0:
            i -= 1
            j -= 1
            operations.append(("match" if ref[i].value == hyp[j].value else "sub", i, j))
        elif direction == 1:
            i -= 1
            operations.append(("del", i, None))
        else:
            j -= 1
            # Attach insertions in a gap to its following reference unit, or last at EOF.
            operations.append(("ins", min(i, len(ref) - 1), j))
    return operations[::-1]


def _changed_spans(spans, reference_text):
    """Use language changes only; suppress repeated tags and speaker-only transitions."""
    if not spans:
        return []
    if "".join(s["text"] for s in spans) != reference_text:
        raise ValueError("reference spans do not reconstruct reference_text")
    runs = []
    start = 0
    for s in spans:
        end = start + len(s["text"])
        lang = s.get("lang")
        if end != start:
            if runs and runs[-1][0] == lang:
                runs[-1] = (lang, runs[-1][1], end)
            else:
                runs.append((lang, start, end))
        start = end
    return [(lo, hi) for previous, (lang, lo, hi) in zip(runs, runs[1:])
            if lang and previous[0] and "und" not in (lang, previous[0])]


def _switches(changes, ref):
    windows = []
    starts = [unit.start for unit in ref]
    for lo, hi in changes:
        first, stop = bisect_left(starts, lo), bisect_left(starts, hi)
        if first < stop:
            windows.append((first, min(first + SWITCH_HEAD, stop), stop - first <= SWITCH_HEAD))
    return windows


def _anchors(task, ref):
    """Only exact outer endpoints supply timing, never interpolate inside a coarse span."""
    text = task["reference_text"]
    starts = {unit.start: n for n, unit in enumerate(ref)}
    ends = {unit.end: n for n, unit in enumerate(ref)}
    onsets, offsets = {}, {}
    duration_ms = task["duration_seconds"] * 1000
    # A switch anchors the earliest possible speech of the new span. Without
    # word-level timing, it cannot tell us when a word in that span completed.
    switches = task.get("switches") or []
    unit_starts = list(starts)
    for index, switch in enumerate(switches):
        lo, begin = switch["char_start"], switch["time_ms"]
        hi = switches[index + 1]["char_start"] if index + 1 < len(switches) else len(text)
        if not (0 <= lo < hi <= len(text) and 0 <= begin <= duration_ms + .001):
            raise ValueError("invalid switch onset or offsets")
        for n in range(bisect_left(unit_starts, lo), bisect_left(unit_starts, hi)):
            onsets[n] = begin / 1000
    for anchor in task.get("alignment") or []:
        lo, hi = anchor["char_start"], anchor["char_end"]
        if not (0 <= lo < hi <= len(text)) or anchor["text"] != text[lo:hi]:
            raise ValueError("alignment offsets do not match reference")
        begin, end = anchor.get("start_ms"), anchor.get("end_ms")
        if (any(t is not None and not 0 <= t <= duration_ms + .001 for t in (begin, end))
                or (begin is not None and end is not None and begin >= end)):
            raise ValueError("alignment times outside audio")
        parts = _units(text[lo:hi])
        if not parts:
            continue
        first, last = starts.get(lo + parts[0].start), ends.get(lo + parts[-1].end)
        if first is not None and begin is not None:
            onsets[first] = max(onsets.get(first, 0.), begin / 1000)
        if last is not None and end is not None:
            offsets[last] = end / 1000
    return onsets, offsets


def _parse_turns(task, turns):
    chunks = task["chunks"]
    if len(turns) != total_turns(task):
        return "incomplete_trajectory", [], [], []
    pieces, char_ends, times = [], [], []
    length = 0
    for i, turn in enumerate(turns):
        eof = i == len(chunks)
        clock = task["duration_seconds"] if eof else chunks[i]["end_seconds"]
        if turn.get("kind") != ("eof" if eof else "audio") or not isclose(
                turn["audio_end_seconds"], clock, abs_tol=1e-6, rel_tol=0):
            raise ValueError("trajectory observations disagree with task audio schedule")
        if turn.get("finish_reason") not in (None, "stop", "eos"):
            return "truncated_or_invalid_finish", [], [], []
        try:
            delta = PROTOCOL.parse_reply(turn.get("response"))
        except ValueError:
            return "invalid_action", [], [], []
        if eof and turn["response"] != WAIT:
            return "text_at_eof", [], [], []
        if delta:
            length += len(delta)
            pieces.append(delta)
            char_ends.append(length)
            times.append(clock)
    return None, pieces, char_ends, times


def score_trajectory(task, turns, mode="quality_latency", *, switch_grace_turns=1):
    """Score one candidate independently, before the trainer's group comparison."""
    if mode not in ("quality", "quality_latency"):
        raise ValueError("expected quality or quality_latency reward mode")
    if type(switch_grace_turns) is not int or not 0 <= switch_grace_turns <= 4:
        raise ValueError("switch_grace_turns must be an integer in 0..4")
    reason, pieces, char_ends, commit_times = _parse_turns(task, turns)
    if reason:
        return {"valid": False, "reason": reason, "reward": -3.0,
                "reward_version": REWARD_VERSION, "switch_grace_turns": switch_grace_turns}
    text = task["reference_text"]
    ref = _units(text)
    changes = _changed_spans(task.get("spans"), text)
    starts = [unit.start for unit in ref]
    # Some synthesized spans have no separator (e.g. <en>Hello<de>Guten).
    # Choose the scoring granularity from the reference alone and apply it to
    # both sides, so omission/addition of a space cannot hide the switch head.
    characters = False
    for start in [lo for lo, _ in changes] + task.get("speaker_changes", []):
        previous = bisect_left(starts, start) - 1
        if previous >= 0 and start < ref[previous].end:
            characters = True
            ref = _units(text, characters=True)
            break
    speaker_offsets, clean_pieces, clean_ends, cursor = [], [], [], 0
    for piece in pieces:
        sections = piece.split(SPK_CHANGE)
        for i, section in enumerate(sections):
            if i:
                speaker_offsets.append(cursor)
            cursor += len(section)
        clean_pieces.append("".join(sections))
        clean_ends.append(cursor)
    hypothesis = "".join(clean_pieces)
    char_ends = clean_ends
    hyp = _units(hypothesis, characters=characters)
    onsets, offsets = _anchors(task, ref)
    emitted_at = [commit_times[bisect_left(char_ends, unit.end)] for unit in hyp]
    timed = mode == "quality_latency"
    operations = _edits(ref, hyp, emitted_at, onsets if timed else {}, offsets if timed else {},
                        task["duration_seconds"])
    costs = [0] * len(ref)
    deleted = set()
    matches = {}
    subs = inserts = 0
    for op, i, j in operations:
        if op == "match":
            matches[i] = emitted_at[j]
        else:
            if i >= 0:
                costs[i] += 1
            if op == "del":
                deleted.add(i)
            elif op == "sub":
                subs += 1
            else:
                inserts += 1
    errors = len(deleted) + subs + inserts
    error_rate = errors / max(1, len(ref))
    cer = cer_metrics(text, hypothesis)
    quality = 1 - min(2., cer["cer"])
    if not ref and not text.strip() and pieces:
        # Even punctuation-only DIRECTs are hallucinations in an annotated silent clip.
        quality = -1.
    windows = _switches(changes, ref)
    short_count = short_dels = 0
    local_errors = []
    for lo, hi, short in windows:
        local_errors.append(min(2., sum(costs[lo:hi]) / (hi - lo)))
        if short:
            short_count += 1
            short_dels += all(i in deleted for i in range(lo, hi))
    switch_error = sum(local_errors) / max(1, len(local_errors))
    switch_content = switch_quality_metrics(ref, hyp, changes, operations)
    local_cer = local_switch_cer(text, ref, hyp, changes, operations, hypothesis=hypothesis)
    speakers, attributed_units = speaker_metrics(task, ref, hyp, operations, speaker_offsets, return_details=True)
    timing = timing_metrics(ref, changes,
                            task.get("speaker_changes", []) if task.get("speaker_scorable") else [],
                            [c["end_seconds"] for c in task["chunks"]], onsets, offsets, matches,
                            grace_turns=switch_grace_turns, head_units=SWITCH_HEAD,
                            attributed_units=attributed_units, alignment=task.get("alignment") or [])
    scripts = disjoint_script_metrics(ref, hyp, operations)
    boundaries = sentence_boundary_errors(text, hypothesis, ref, hyp, operations,
                                         trusted=task.get("punctuation_trusted", True),
                                         score_terminal=(task.get("end_policy", "eof") == "eof"
                                                         and task.get("terminal_punctuation_trusted", True)))
    extra = boundaries["extra_internal_stops"]
    punctuation_cost = .60 * extra / (2 + extra)
    punctuation_cost += .12 * boundaries["missing_sentence_stops"] / max(1, boundaries["reference_sentence_stops"])
    short_breaks = boundaries["unsupported_short_breaks"]
    punctuation_cost += .10 * short_breaks / (1 + short_breaks)
    # Punctuation needs exact lexical anchors. A typo can destroy those anchors;
    # never let that evade a larger cost than the typo itself. Combined with the
    # latency cap (.5/N), these auxiliary deductions stay below one CER edit.
    punctuation_penalty = min(.10, .25 / max(1, cer["reference_characters"])) * min(1., punctuation_cost)
    runs = scripts["script_mismatch_runs"]
    # A translated switch can trigger both detectors for the same words.
    language_penalty = .90 * max(switch_content["switch_content_error"], local_cer["switch_cer_cost"], runs / (1 + runs))
    reward = quality - .40 * switch_error - .20 * short_dels / max(1, short_count)
    speaker_attribution_penalty = .40 * speakers["speaker_attribution_error"]
    speaker_penalty = .40 * speakers["speaker_error"] + speaker_attribution_penalty
    reward -= language_penalty + punctuation_penalty + speaker_penalty
    # A bounded latency deduction, gated by *both* overall and local quality.
    # No latency bonus for translating, dropping a switch, or losing a speaker.
    quality_gate = max(0., 1 - cer["cer"]) * (1 - max(
        switch_content["switch_content_error"], local_cer["switch_cer_cost"], speakers["speaker_error"],
        speakers["speaker_attribution_error"]))
    latency_penalty = 0.
    if mode == "quality_latency":
        # Even a maximally stalled exact transcript must not lose to one
        # character error just because the erroneous candidate is fast.
        latency_penalty = min(.10, .5 / max(1, cer["reference_characters"])) * quality_gate * timing["timing_cost"]
        reward -= latency_penalty + .10 * timing["early_units"] / max(1, len(onsets))
    if ref and not matches and cer["cer"] >= .5:
        # No exact lexical coverage plus large character error: unrelated
        # output must not gain from merely containing letters in common.
        # Keep small spelling errors graded by CER even without an exact word.
        reward = -2.5 - .20 * min(1., len(hyp) / len(ref))
    elif reward < -2.0:
        # Preserve ordering among poor partial transcripts while keeping them
        # above an all-WAIT response (-2.5) and a protocol violation (-3).
        excess = -2.0 - reward
        reward = -2.0 - .5 * excess / (.5 + excess)
    return {
        "valid": True, "reason": None, "reward": reward, "quality": quality,
        "reward_version": REWARD_VERSION,
        "unit_mode": "character_cross_boundary" if characters else "mixed",
        "reference_units": len(ref), "matched_units": len(matches), "errors": errors,
        "substitutions": subs, "deletions": len(deleted), "insertions": inserts,
        "error_rate": error_rate, "switch_segments": len(windows), "switch_error_rate": switch_error,
        "short_switch_segments": short_count, "short_switch_deletions": short_dels,
        **cer, **local_cer, **speakers, **switch_content, **scripts, **boundaries, **timing,
        "language_penalty": language_penalty, "punctuation_penalty": punctuation_penalty,
        "punctuation_cost": punctuation_cost, "speaker_attribution_penalty": speaker_attribution_penalty,
        "speaker_penalty": speaker_penalty, "latency_penalty": latency_penalty, "quality_gate": quality_gate,
    }
