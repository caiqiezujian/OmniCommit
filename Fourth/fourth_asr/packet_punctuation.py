"""Choose the comma for an ordinary sentence-final period at an HTTP packet boundary."""
import re
import unicodedata

from .annotation_tail import simple_period_offset
from .protocol import SPK_CHANGE
from .semantic_preparation import TRAILING


DEFAULT_PACKET_PERIOD_THRESHOLD_MS = 300
PERIODS = ".。．۔"
CLOSERS = TRAILING + "〉》〕〗〙〛"
CLAUSE_BOUNDARIES = frozenset(".。．۔!?！？؟;；:：,，、،\n")
URL_TAIL = re.compile(
    r"(?:https?://|www\.|[\w.+%-]+@)\S*$|\b(?:[a-z0-9-]+\.)+[a-z]{2,}(?:[/?#]\S*)?\.$",
    re.IGNORECASE | re.ASCII,
)
# Additional forms used in the supported languages beyond simple_period_offset's list.
SHORT_FORM = re.compile(r"\b(?:т\.\s*[едп]|напр|др|г|гг|ул|рис|стр|z\.\s*b|d\.\s*h|"
                        r"hr|frl|usw|p\.\s*ex|avv|dott|dra|drs|pág)\.$", re.IGNORECASE)


def _comma_for(stem, period):
    """Use the final clause's writing system; Han alone uses a full-width comma."""
    if period == "۔":
        return "،"
    han = False
    latin_in_cjk = False
    for char in reversed(stem):
        if char in CLAUSE_BOUNDARIES:
            break
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if name.startswith(("HIRAGANA ", "KATAKANA ", "HALFWIDTH KATAKANA ")):
            return "、"
        if name.startswith(("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")) or char == "々":
            han = True
            continue  # Kana earlier in the same Japanese phrase can disambiguate a kanji ending.
        if name.startswith("HANGUL "):
            return ","
        if han:
            return "，"
        if name.startswith("ARABIC "):
            return "،"
        if period in "。．" and name.startswith("LATIN "):
            latin_in_cjk = True
            continue  # Example: “请打开 API。”
        return ","
    if han or (period in "。．" and not latin_in_cjk):
        return "，"
    return ","


def replace_terminal_period(marked, previous=""):
    """Replace only the final speaker's final period, preserving all other characters."""
    _, change, last = marked.rpartition(SPK_CHANGE)
    end = len(last)
    while end and (last[end - 1].isspace() or last[end - 1] in CLOSERS):
        end -= 1
    if not end or last[end - 1] not in PERIODS:
        return marked

    local = last[:end]
    context = local if change else previous.rpartition(SPK_CHANGE)[2] + local
    period = local[-1]
    stem = context[:-1].rstrip()
    if not stem or stem[-1] in PERIODS + "…!?！？؟,，、،;；:：":
        return marked
    if period == ".":
        # A digit + dot at a streaming boundary may continue as a decimal or an ordinal.
        if stem[-1].isdigit() or SHORT_FORM.search(context) or URL_TAIL.search(context):
            return marked
        # Check both strings: older model turns sometimes omit the inter-word space.
        for text in (context, local):
            if len(text) > 1 and (text[-2].isalpha() or unicodedata.category(text[-2]).startswith("M")):
                if simple_period_offset(text) is None or SHORT_FORM.search(text):
                    return marked
    offset = len(marked) - len(last) + end - 1
    return marked[:offset] + _comma_for(stem, period) + marked[offset + 1:]
