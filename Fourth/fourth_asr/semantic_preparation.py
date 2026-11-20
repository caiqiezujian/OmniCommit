"""Group timestamped fragments before assigning them to the streaming grid."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import replace
from itertools import groupby

from .timestamped_source import TAG, Unit, parse_record, reference_text

# Delimiters can be followed by a closing quote or parenthesis.
TRAILING = "”’»›」』】）)]}\"'"
CLAUSE_END = frozenset("。！？!?；;：:、，,،؛؟।॥…．.!؟")
ABBREVIATION = re.compile(r"(?:\b(?:mr|mrs|ms|dr|prof|st|vs|e\.g|i\.e)|\b(?:[A-Za-z]\.)+[A-Za-z])\.$", re.I)
JAPANESE_SUFFIX = frozenset(("か", "か？", "か?", "を", "は", "が", "に", "の", "へ", "も", "で", "と"))
STANDALONE = frozenset(("好", "对", "是", "能", "嗯", "可以", "可以的", "好的", "ok", "yes", "no", "ja", "nein", "oui", "non", "sí",
                        "sim", "não", "да", "нет", "はい", "いいえ", "네", "아니요", "نعم", "لا"))

# Positive evidence for keeping a short tail with its continuation when the
# source repeats a semantic header even at that tail's end.
def unfinished_tail(unit, following):
    lang, _, text = unit.spans[-1]
    tail = text.rstrip().casefold()
    if lang == "zh":
        return (tail.endswith(("但是", "因为", "如果", "然后", "以", "的"))
                or (tail.endswith(("下", "上")) and following.text.lstrip().startswith("半")))
    if lang == "en":
        return bool(re.search(r"\b(?:the|an|because|into|of|but|and)$", tail))
    return False


def closes_clause(text, following=""):
    tail = text.rstrip().rstrip(TRAILING).rstrip()
    if not tail or tail[-1] not in CLAUSE_END:
        return False
    if numeric_continuation(tail, following):
        return False
    return ABBREVIATION.search(tail) is None


def numeric_continuation(left, right):
    # Only directly adjacent digits make a decimal/grouping separator. Source
    # blanks are meaningful ("1, 2" can be an enumeration).
    return (len(left) >= 2 and left[-2].isdigit() and left[-1] in ".,٫٬，"
            and bool(right) and right[0].isdigit())


def _numeric_join(left, right, next_text):
    if numeric_continuation(left, right):
        return True
    if not left or not left[-1].isdigit() or not right:
        return False
    if right[0].isdigit():
        return True
    if right[0] in ".,٫٬，":
        remainder = right[1:] or next_text
        return bool(remainder) and remainder[0].isdigit()
    return False


def _reference_regions(marked):
    """Exact plain-text offsets of annotated utterances in messages, not output headers."""
    state = [None, None]
    regions = []
    pos = cursor = 0
    for match in [*TAG.finditer(marked), None]:
        body = marked[cursor:match.start() if match else len(marked)]
        if body.strip():
            regions.append((pos if regions else 0, tuple(state)))
        pos += len(body)
        if match:
            tag = match[1]
            if tag.startswith("spk") and tag[3:].isdigit():
                state[1] = tag
            else:
                state[0] = tag
            cursor = match.end()
    return [(start, regions[i+1][0] if i+1 < len(regions) else pos,
             (i > 0 and state == regions[i-1][1]) or
             (i+1 < len(regions) and state == regions[i+1][1]))
            for i, (start, state) in enumerate(regions)]


def _split_mixed(units, stats):
    result = []
    for i, unit in enumerate(units):
        if len(unit.spans) > 1:
            children = unit.children
            if (children and all(p.text.strip() for p in children)
                    and (not result or children[0].start_ms >= result[-1].start_ms)
                    and (i + 1 == len(units) or children[-1].start_ms <= units[i+1].start_ms)):
                groups = groupby(children, key=lambda c: c.spans[0][:2])
                for state, members in groups:
                    parts = tuple(members)
                    text = "".join(p.text for p in parts)
                    result.append(Unit(parts[0].start_ms, max(p.end_ms for p in parts),
                                       ((*state, text),), parts))
                stats["split_units"] += len(unit.spans) - 1
                continue
            stats["mixed_units_kept"] += 1
        result.append(unit)
    return result


def _merge(units, regions, short_ms, chunk_ms, stats):
    offsets = [0]
    for unit in units:
        offsets.append(offsets[-1] + len(unit.text))
    boundaries = {end for _, end, _ in regions}
    trusted = {}  # An aligned consecutive pair belongs to one known original semantic region.
    i = 0
    for start, end, reliable in regions:
        while i < len(units) and offsets[i] < end:
            if reliable and start <= offsets[i] and offsets[i+1] <= end:
                trusted[i] = start
            i += 1

    groups = []
    first = 0
    for i, unit in enumerate(units):
        stop = i+1
        at_boundary = offsets[stop] in boundaries
        join = False
        if not at_boundary and stop < len(units) and len(unit.spans) == len(units[stop].spans) == 1:
            right = units[stop]
            lang, speaker, left_text = unit.spans[0]
            right_text = right.text
            gap = right.start_ms - unit.end_ms
            if (right.spans[0][:2] == (lang, speaker) and 0 <= gap <= short_ms
                    and math.ceil(right.end_ms/chunk_ms) <= math.ceil(units[first].end_ms/chunk_ms)+1):
                number_prefix = left_text
                if len(left_text) == 1 and left_text in ".,٫٬，" and first < i:
                    number_prefix = units[i-1].text[-1:] + left_text
                continuation = units[stop+1].text if stop+1 < len(units) else ""
                if stop+1 < len(units) and right.spans[0][:2] != units[stop+1].spans[0][:2]:
                    continuation = ""
                decimal = _numeric_join(number_prefix, right_text, continuation)
                particle = lang == "ja" and right_text.strip() in JAPANESE_SUFFIX
                short_open = (unit.end_ms - unit.start_ms <= short_ms
                              and left_text.strip().casefold() not in STANDALONE
                              and not closes_clause(left_text, right_text)
                              and i in trusted and trusted[i] == trusted.get(stop))
                join = decimal or particle or short_open
        if join:
            continue
        if first == i:
            groups.append(replace(unit, phrase_end=at_boundary))
        else:
            members = units[first:stop]
            state = members[0].spans[0][:2]
            text = "".join(member.text for member in members)
            # A protected group stays indivisible even on a timed-span build.
            groups.append(Unit(members[0].start_ms, members[-1].end_ms, ((*state, text),),
                               phrase_end=at_boundary))
            stats["merged_units"] += len(members) - 1
        first = stop
    return groups


def prepare_units(record, duration_ms, *, chunk_ms, short_ms=300, release_on="semantic-unit"):
    """Build release units from exact source timestamps, independently for each grid."""
    if type(short_ms) is not int or short_ms < 0 or type(chunk_ms) is not int or chunk_ms <= 0:
        raise ValueError("short-ms must be nonnegative and chunk-ms must be positive")
    if release_on not in ("semantic-unit", "timed-span", "merged-unit"):
        raise ValueError("release-on must be semantic-unit, timed-span or merged-unit")
    stats = Counter(merged_units=0, split_units=0, mixed_units_kept=0, overlapping_switches=0)
    units = _split_mixed(parse_record(record, duration_ms), stats)
    stats["overlapping_switches"] = sum(
        old.end_ms > new.start_ms and old.spans[-1][:2] != new.spans[0][:2]
        for old, new in zip(units, units[1:]))
    if release_on == "merged-unit":
        # Step 1 has already chosen the display units. Children are timing
        # evidence for attribution changes only, not new lexical boundaries.
        return units, stats
    regions = _reference_regions(reference_text(record))
    groups = _merge(units, regions, short_ms, chunk_ms, stats)
    if release_on == "timed-span":
        expanded = []
        for i, group in enumerate(groups):
            children = group.children
            if (len(children) > 1 and len(group.spans) == 1 and all(p.text.strip() for p in children)
                    and (not expanded or children[0].start_ms >= expanded[-1].start_ms)
                    and (i+1 == len(groups) or children[-1].start_ms <= groups[i+1].start_ms)):
                expanded.extend(children)
            else:
                expanded.append(group)
        if len(expanded) != len(groups):
            groups = _merge(expanded, regions, short_ms, chunk_ms, stats)
    return groups, stats
