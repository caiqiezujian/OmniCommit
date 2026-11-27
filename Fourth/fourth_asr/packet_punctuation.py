"""Apply caller-reported silence to the last sentence period of one HTTP packet."""
from dataclasses import dataclass
import math
import re
import unicodedata

from .annotation_tail import simple_period_offset
from .protocol import SPK_CHANGE
from .semantic_preparation import TRAILING


DEFAULT_PACKET_PERIOD_THRESHOLD_MS = 150
DEFAULT_PACKET_PERIOD_REMOVE_THRESHOLD_MS = 100
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

# This hint describes the final clause for diagnostics and mixed-script
# typography. Whether a period is removed or replaced depends only on silence.
CLAUSE_BREAK = re.compile("[" + re.escape("".join(sorted(CLAUSE_BOUNDARIES))) + "]")
ZH_CUE = re.compile(r"[你您她它这這们們吗嗎呢吧么麼咱谁沒没说听还给请谢让该预测试问进议项认过顺对嗯]"
                    r"|我[想是会會]|知道|收到|明白了|可以|好[的了]|謝謝|開始了|结束|確認了"
                    r"|今天|明天|早上好|现在|如果|就是|然后|因为|所以|已经|已經|是的|行了|對的|沒錯")
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
EN_CUES = frozenset("""
    the this that these those we you they she your our their my his
    have has been would should could with from there here what when where
    which who why how because already always really
    know think understand agree yes yeah nope right exactly hello thanks thank good morning
    afternoon evening great done ready received please tomorrow
    meeting approved finished see let's i'm we're
    you're they're it's that's i'll we'll don't can't doesn't isn't
""".split())
EN_PHRASE = re.compile(r"\b(?:i (?:am|can|will|see|like|need|want)|it (?:is|was|works)|got it"
                       r"|of course|for sure|all right)\b", re.I)


@dataclass(frozen=True)
class PeriodEdit:
    text: str
    operation: str
    language: str
    threshold_ms: float
    space_separated: bool


def resolve_remove_threshold(upper, lower):
    if type(upper) not in (int, float) or not math.isfinite(upper) or upper < 0:
        raise ValueError("packet_period_threshold_ms must be nonnegative and finite")
    if lower is None:
        return min(DEFAULT_PACKET_PERIOD_REMOVE_THRESHOLD_MS, upper)
    if type(lower) not in (int, float) or not math.isfinite(lower) or not 0 <= lower <= upper:
        raise ValueError("packet_period_remove_threshold_ms must be finite and between 0 and packet_period_threshold_ms")
    return lower


def _english_ending(clause):
    start = len(clause)
    while start and (not clause[start-1].isalpha() or "LATIN" in unicodedata.name(clause[start-1], "")):
        start -= 1
    latin = clause[start:].casefold().replace("’", "'")
    words = WORD.findall(latin)
    return bool(EN_CUES.intersection(words[-8:]) or EN_PHRASE.search(latin) or latin.strip() == "sure")


def _tail_language(stem, local_stem, comma):
    # A separately emitted dot or short continuation can finish an earlier
    # phrase, but evidence must not cross a sentence or a speaker boundary.
    clause = CLAUSE_BREAK.split(stem[-160:])[-1]
    if comma == "，":
        return "zh" if ZH_CUE.search(clause) else "other"
    if comma != ",":
        return "other"
    local = CLAUSE_BREAK.split(local_stem[-160:])[-1]
    if _english_ending(local) or (len(WORD.findall(local)) <= 1 and _english_ending(clause)):
        return "en"
    # E.g. Chinese carrier text followed by an acronym and an ASCII stop.
    if ZH_CUE.search(clause) and re.search(r"[A-Z][A-Z0-9-]+$", clause):
        return "zh"
    return "other"


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


def edit_packet_period(marked, previous, silence_ms, comma_ms, remove_ms):
    """Return one packet-final edit, or None when the caller should keep the text."""
    if silence_ms >= comma_ms:
        return None
    _, change, last = marked.rpartition(SPK_CHANGE)
    end = len(last)
    while end and (last[end - 1].isspace() or last[end - 1] in CLOSERS):
        end -= 1
    if not end or last[end - 1] not in PERIODS:
        return None

    local = last[:end]
    context = local if change else previous.rpartition(SPK_CHANGE)[2].rsplit("\n", 1)[-1] + local
    period = local[-1]
    stem = context[:-1].rstrip()
    if not stem or stem[-1] in PERIODS + "…!?！？؟,，、،;；:：":
        return None
    if period == ".":
        # A digit + dot at a streaming boundary may continue as a decimal or an ordinal.
        if stem[-1].isdigit() or SHORT_FORM.search(context) or URL_TAIL.search(context):
            return None
        # Check both strings: older model turns sometimes omit the inter-word space.
        for text in (context, local):
            if len(text) > 1 and (text[-2].isalpha() or unicodedata.category(text[-2]).startswith("M")):
                if simple_period_offset(text) is None or SHORT_FORM.search(text):
                    return None
    offset = len(marked) - len(last) + end - 1
    comma = _comma_for(stem, period)
    language = _tail_language(stem, local[:-1], comma)
    if language == "zh":
        comma = "，"
    space_separated = comma in (",", "،")
    if silence_ms < remove_ms:
        return PeriodEdit(marked[:offset] + marked[offset + 1:], "remove", language, remove_ms, space_separated)
    return PeriodEdit(marked[:offset] + comma + marked[offset + 1:], "comma", language, comma_ms, space_separated)


def replace_terminal_period(marked, previous=""):
    """Unconditional comma-only compatibility entry point."""
    edit = edit_packet_period(marked, previous, 0, DEFAULT_PACKET_PERIOD_THRESHOLD_MS, 0)
    return edit.text if edit else marked


def join_after_packet_period(previous, marked):
    """Resolve the word separator when the next visible text is available."""
    change = SPK_CHANGE if marked.startswith(SPK_CHANGE) else ""
    body = marked[len(change):]
    right = body.lstrip(" ")
    left = previous.rstrip(" ").rstrip(CLOSERS).rstrip(",،").rstrip(CLOSERS)
    while left and unicodedata.category(left[-1]).startswith("M"):
        left = left[:-1]
    if not left or not right or right.casefold().startswith(("n't", "n’t")):
        return marked
    for char in (left[-1], right[0]):
        # Infer a missing separator only for writing systems that separate
        # words with spaces. Preserve the model's spacing in CJK and Thai.
        if not (char.isdecimal() or char.isalpha() and unicodedata.name(char, "").startswith(
                ("LATIN ", "CYRILLIC ", "GREEK ", "ARABIC ", "HEBREW "))):
            return marked
    separator = "" if previous.endswith(" ") else " "
    return change + separator + right
