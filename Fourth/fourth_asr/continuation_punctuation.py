"""Edit simple periods at well-evidenced boundaries between original timed segments."""
import re
from collections import Counter
from dataclasses import replace

from .annotation_tail import simple_period_offset


# Inspect the final clause so an earlier conditional in a completed sentence
# cannot turn a later, independent sentence boundary into a comma.
ZH_CLAUSE_BREAK = re.compile(r"[，,。．.！？!?；;：:\n]")
ZH_BA = re.compile(r"(?:我们|你们|他们|她们|大家|我|你|他|她|请|需要|可以|先|再)(?:先|再|都)?把$")
ZH_VIEWPOINT = re.compile(r"(?:从|单从)[\u3400-\u9fff]{1,16}(?:上|中|里|来看)$")
ZH_RESULT = re.compile(r"^(?:那么|那就|(?:我们|你们|他们|她们|大家|我|你|他|她)?就)[\u3400-\u9fff]")
EN_DETERMINER = re.compile(r"\b[a-z]+\s+(?:the|our|their|your)$", re.I)
EN_INFINITIVE = re.compile(r"\b(?:needs?|needed|wants?|wanted|plans?|planned)\s+to$", re.I)
EN_VERB = re.compile(r"^(?:be|check|confirm|discuss|do|find|finish|fix|get|make|review|see|send|set|take|update|use)\b")


def continuation_replacement(language, stem, following):
    """Given left text without its period and nonempty right text, suggest a mark or abstain."""
    if language == "en":
        if not "a" <= following[0] <= "z":
            return None
        if EN_DETERMINER.search(stem) or (EN_INFINITIVE.search(stem) and EN_VERB.match(following)):
            return ""
        return None

    if not "\u3400" <= following[0] <= "\u9fff":
        return None
    clause = ZH_CLAUSE_BREAK.split(stem)[-1].strip()
    if ZH_BA.search(clause):
        return ""
    if ZH_VIEWPOINT.fullmatch(clause) and following.startswith(
        ("也能", "也可以", "还能", "还可以", "就能", "就可以", "可以看", "能够看")
    ):
        return ""
    if clause.startswith(("如果", "假如", "倘若")) and not re.search(r"就|那么|便|则", clause):
        if ZH_RESULT.match(following):
            return "，"
    if clause.startswith(("因为", "由于")) and "所以" not in clause and following.startswith("所以"):
        return "，"
    return None


def revise_continuation_periods(units, max_gap_ms, *, protected=()):
    """Return a training view of validated units. The last segment belongs to the tail policy."""
    stats = Counter(internal_period_candidates=0, internal_periods_removed=0, internal_periods_to_comma=0)
    revised = units
    earlier_end = 0
    for i in range(len(units) - 1):
        left, right = units[i], units[i + 1]
        overlaps_earlier = left.start_ms < earlier_end
        earlier_end = max(earlier_end, left.end_ms)
        if i + 1 in protected or overlaps_earlier or not 0 <= right.start_ms - left.end_ms <= max_gap_ms:
            continue
        if i + 2 < len(units) and units[i + 2].start_ms < right.end_ms:
            continue
        if len(left.spans) != 1 or len(right.spans) != 1 or left.spans[0][:2] != right.spans[0][:2]:
            continue
        tag, speaker, text = left.spans[0]
        language = tag.split("-")[0]
        stripped = text.rstrip()
        if language not in ("en", "zh") or not stripped.endswith((".", "。", "．")):
            continue
        # Protect closing quotes/brackets, ellipses, recognized abbreviations and numbers.
        pos = simple_period_offset(stripped)
        following_text = right.spans[0][2]
        following = following_text.lstrip()
        if pos is None or not stripped[pos - 1].isalpha() or not following or not following[0].isalpha():
            continue
        stats["internal_period_candidates"] += 1
        mark = continuation_replacement(language, stripped[:pos], following)
        if mark is None:
            continue
        stats["internal_periods_to_comma" if mark else "internal_periods_removed"] += 1
        if language == "en" and not mark and pos + 1 == len(text) and not following_text[0].isspace():
            mark = " "  # Without the period, two English words would be joined.
        if revised is units:
            revised = list(units)
        updated_text = text[:pos] + mark + text[pos + 1:]
        revised[i] = replace(left, spans=((tag, speaker, updated_text),), children=())
    return revised, stats
