"""Score extra sentence breaks at positions supported by the lexical alignment."""

import re


_PERIODS = frozenset(".．。｡۔।॥።։။")
_OTHER_STOPS = frozenset("!?！？؟‼⁇⁈⁉")
_ELLIPSES = frozenset("…⋯")
_DOT_RUN = re.compile(r"[.．](?:\s*[.．])+")
# Common abbreviations that often end before another word. Leave an ambiguous
# dot unscored, including when only the candidate uses the abbreviated form.
_ABBREVIATIONS = frozenset(
    "dr mr mrs ms prof sr jr st vs etc cf fig no vol approx messrs "
    "mme mlle bzw ggf ca usw z b bsp nr str hr fr sra dra pág av "
    "г гг им ул см рис проф т д د أ".split()
)
_DOTTED_SUFFIXES = frozenset(
    "com org net edu gov io ai co cn de fr it es pt ru uk jp kr th ar "
    "pdf html json py txt csv wav mp3 zip".split()
)
_GERMAN_MONTH = re.compile(
    r"\s+(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\b",
    re.IGNORECASE,
)


def _ambiguous_dot(text, index):
    left = text[index - 1] if index else ""
    right = text[index + 1:index + 2]
    if left.isdecimal() and right.isdecimal():
        return True
    if left.isdecimal() and _GERMAN_MONTH.match(text, index + 1):
        return True
    start = index
    while start and text[start - 1].isalpha():
        start -= 1
    word = text[start:index]
    if word.casefold() in _ABBREVIATIONS:
        return True
    if len(word) == 1 and (word.isupper() or start > 0 and text[start - 1] in ".．"):
        return True
    if left.isascii() and left.isalnum() and right.isascii() and right.isalnum():
        # Exempt hostnames, URLs and abbreviated initials. A missing space at
        # a streaming join (e.g. "know.you") does not itself license a period.
        end = index + 1
        while end < len(text) and text[end].isalnum():
            end += 1
        suffix = text[index + 1:end].casefold()
        if (suffix in _DOTTED_SUFFIXES or word.casefold() == "www" or len(word) == len(suffix) == 1
                or text[end:end + 1] == "@"):
            return True
        context = re.split(r"\s", text[max(0, start - 96):start])[-1]
        return "://" in context or "@" in context
    return False


def _gap_mark(text, start, end):
    """Classify punctuation after a lexical unit in the original raw text."""
    gap = text[start:end]
    if not gap:
        return None
    if any(c in _OTHER_STOPS for c in gap):
        return "stop"
    if _DOT_RUN.search(gap) or any(c in _ELLIPSES for c in gap):
        return "ambiguous"
    for offset, char in enumerate(gap, start):
        if char in _PERIODS:
            if char in ".．" and _ambiguous_dot(text, offset):
                return "ambiguous"
            return "period"
    # Some source transcripts are line-broken by speaker or sentence. Either
    # rendering is acceptable across a line break.
    if "\n" in gap or "\r" in gap:
        return "ambiguous"
    return None


def _marks(text, units):
    return [
        _gap_mark(text, unit.end, units[i + 1].start if i + 1 < len(units) else len(text))
        for i, unit in enumerate(units)
    ]


def _local_repetition(units, left, right):
    """Recognize up to four inserted units echoing the adjacent words."""
    count = right - left - 1
    if count < 1 or count > 4:
        return False
    insert = [u.value for u in units[left + 1:right]]
    before = [u.value for u in units[max(0, left + 1 - count):left + 1]]
    after = [u.value for u in units[right:right + count]]
    return insert == before or insert == after


def sentence_boundary_errors(reference_text, hypothesis_text, ref_units, hyp_units, operations):
    """Compare internal periods and auditable reference sentence endings.

    Use the reward's existing edit path. Both adjacent lexical anchors must
    match. Permit an inserted echo between these anchors if the reference is
    clearly continuous, e.g. 我知道。知道你 vs 我知道你. The last lexical unit
    anchors an explicit reference ending at EOF independently of chunking.
    """
    ref_marks = _marks(reference_text, ref_units)
    extra = missing = reference_stops = internal = 0
    strong = ("stop", "period")
    if any(mark in strong for mark in ref_marks):
        hyp_marks = _marks(hypothesis_text, hyp_units)
        anchors = [(r, h) for op, r, h in operations if op == "match"]
        for (r, h), (next_r, next_h) in zip(anchors, anchors[1:]):
            if next_r != r + 1:
                continue
            ref_mark = ref_marks[r]
            if ref_mark == "ambiguous":
                continue
            if next_h != h + 1:
                if ref_mark is not None or not _local_repetition(hyp_units, h, next_h):
                    continue
            seen = hyp_marks[h:next_h]
            if ref_mark in strong:
                if "ambiguous" not in seen:
                    reference_stops += 1
                    missing += not any(mark in strong for mark in seen)
            else:
                internal += sum(mark != "ambiguous" for mark in seen)
                extra += seen.count("period")
        if anchors and anchors[-1] == (len(ref_units) - 1, len(hyp_units) - 1):
            if ref_marks[-1] in strong and hyp_marks[-1] != "ambiguous":
                reference_stops += 1
                missing += hyp_marks[-1] not in strong
    return {
        "punctuation_scorable": bool(reference_stops or internal),
        "reference_sentence_stops": reference_stops,
        "missing_sentence_stops": missing,
        "extra_internal_periods": extra,
        "scored_internal_gaps": internal,
    }
