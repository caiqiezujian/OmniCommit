"""Local reference-word fidelity at annotated language changes.

The Fourth protocol does not predict language IDs. Shared words and
borrowings are scored against the actual transcript, independent of its tags.
"""
from bisect import bisect_left
from functools import lru_cache
from itertools import groupby
import unicodedata


def _near_spelling(a, b):
    """Recognize one edit (including adjacent transposition) in a word."""
    if min(len(a), len(b)) < 3 or abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    if len(a) < len(b):
        return a[i:] == b[i + 1:]
    if a[i + 1:] == b[i + 1:]:
        return True
    return (i + 1 < len(a) and a[i] == b[i + 1] and a[i + 1] == b[i]
            and a[i + 2:] == b[i + 2:])


def switch_quality_metrics(ref, hyp, changes, operations):
    """Score substitutions/deletions on *reference* lexical units after switches.

    Existing utterance and switch-head edit metrics already count insertions.
    The local component measures how much of the new language's reference
    content survives. A unit containing only a number gives no language clue.
    """
    metrics = {"switch_content_error": 0.0, "switch_content_spans": 0,
               "switch_content_fully_missed": 0}
    if not changes:
        return metrics

    losses = [0.0] * len(ref)
    for op, i, j in operations:
        if op == "del":
            losses[i] = 1.0
        elif op == "sub":
            # Keep a spelling slip distinguishable from replacing the word.
            losses[i] = 0.25 if _near_spelling(ref[i].value, hyp[j].value) else 1.0

    starts = [unit.start for unit in ref]
    errors = []
    for start, end in changes:
        lo, hi = bisect_left(starts, start), bisect_left(starts, end)
        lexical = [losses[i] for i in range(lo, hi) if any(c.isalpha() for c in ref[i].value)]
        if not lexical:
            continue
        head = lexical[:3]
        errors.append(0.5 * (sum(head) / len(head) + sum(lexical) / len(lexical)))
        if all(loss == 1.0 for loss in lexical):
            metrics["switch_content_fully_missed"] += 1
    if errors:
        # Keep one entirely missed short switch visible even in a meeting
        # with many correctly recognized switches. Both terms stay in [0, 1].
        metrics["switch_content_error"] = 0.5 * (max(errors) + sum(errors) / len(errors))
        metrics["switch_content_spans"] = len(errors)
    return metrics


@lru_cache(maxsize=8192)
def _script(letter):
    name = unicodedata.name(letter, "")
    if name.startswith(("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH", "IDEOGRAPHIC NUMBER ZERO")):
        return "Han"
    for prefix in ("LATIN", "CYRILLIC", "GREEK", "ARABIC", "THAI", "HANGUL", "HIRAGANA", "KATAKANA"):
        if name.startswith(prefix + " "):
            return prefix
    return None


# In these alphabets many glyphs look alike, especially in names and acronyms.
# Only use the fold to *withhold* an extra penalty; ASR edit scoring stays intact.
_LOOKALIKE_FOLD = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    "к": "k", "т": "t", "м": "m", "і": "i", "в": "b", "н": "h", "ѕ": "s",
    "ј": "j", "ӏ": "l",
    "α": "a", "β": "b", "ε": "e", "ζ": "z", "η": "h", "ι": "i", "κ": "k",
    "μ": "m", "ν": "v", "ο": "o", "ρ": "p", "τ": "t", "υ": "y", "χ": "x",
})
_EUROPEAN_ALPHABETS = {"LATIN", "CYRILLIC", "GREEK"}


def _script_evidence(units):
    letters = "".join(c for unit in units for c in unit.value if c.isalpha())
    scripts = {_script(c) for c in letters}
    return letters, scripts


def _disjoint_writing_systems(left, right):
    a, a_scripts = _script_evidence(left)
    b, b_scripts = _script_evidence(right)
    if min(len(a), len(b)) < 2 or None in a_scripts or None in b_scripts:
        return False
    if not a_scripts.isdisjoint(b_scripts):
        return False
    # Japanese and Korean can both use Han. Kana is one writing system even
    # when a word changes from hiragana to katakana or vice versa.
    for compatible in ({"Han", "HIRAGANA", "KATAKANA"}, {"Han", "HANGUL"}):
        if compatible & a_scripts and compatible & b_scripts:
            return False
    if len(a_scripts) != 1 or len(b_scripts) != 1:
        return False
    if (a_scripts | b_scripts) <= _EUROPEAN_ALPHABETS:
        # Lexical units are already case-folded. The glyph behind e.g. Greek
        # nu could have been uppercase Ν (Latin N) or lowercase ν (Latin v).
        # Give short initials/acronyms with only ambiguous glyphs the benefit
        # of the doubt instead of trying to reconstruct their original case.
        if max(len(a), len(b)) <= 3 and all(c.isascii() or ord(c) in _LOOKALIKE_FOLD for c in a + b):
            return False
        a, b = a.translate(_LOOKALIKE_FOLD), b.translate(_LOOKALIKE_FOLD)
        if a == b or _near_spelling(a, b):
            return False
    return True


def disjoint_script_metrics(ref, hyp, operations):
    """Find unsupported script replacements in monolingual and switching text.

    Inspect entire edit gaps between exact anchors. An insertion or deletion
    in isolation supplies no paired evidence of a changed writing system.
    A run is evidence about the printed transcript only, not acoustic LID.
    """
    runs = ref_units = hyp_units = 0
    for matched, group in groupby(operations, key=lambda edit: edit[0] == "match"):
        if matched:
            continue
        gap = list(group)
        if not any(op == "sub" for op, _, _ in gap):
            continue
        left = [ref[i] for op, i, _ in gap if op != "ins" and any(c.isalpha() for c in ref[i].value)]
        right = [hyp[j] for _, _, j in gap if j is not None and any(c.isalpha() for c in hyp[j].value)]
        if _disjoint_writing_systems(left, right):
            runs += 1
            ref_units += len(left)
            hyp_units += len(right)
    return {"script_mismatch_runs": runs, "script_mismatch_ref_units": ref_units,
            "script_mismatch_hyp_units": hyp_units}
