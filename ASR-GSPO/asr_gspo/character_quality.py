"""True normalized CER, independent of the mixed lexical alignment used for timing."""
import unicodedata

try:
    from rapidfuzz.distance import Levenshtein
except ImportError:
    Levenshtein = None


def characters(text):
    # Keep letters, numbers and combining marks in every language. Never
    # translate, remove accents, or normalize spoken numbers to a different text.
    text = unicodedata.normalize("NFKC", text).casefold()
    def numeric_mark(i, c):
        before = i > 0 and text[i-1].isdecimal()
        after = i + 1 < len(text) and text[i+1].isdecimal()
        return (c in ".,-/:" and before and after) or (c in "+-−" and after) or (c == "%" and before)
    return "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c
                   for i, c in enumerate(text) if unicodedata.category(c)[0] in "LNM" or numeric_mark(i, c))


def _distance(a, b):
    if Levenshtein is not None:
        return Levenshtein.distance(a, b)
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j-1] + 1, previous[j-1] + (x != y)))
        previous = current
    return previous[-1]


def cer_metrics(reference, hypothesis):
    ref, hyp = characters(reference), characters(hypothesis)
    errors = _distance(ref, hyp)
    return dict(cer=errors / max(1, len(ref)), character_errors=errors,
                reference_characters=len(ref), hypothesis_characters=len(hyp))


def _aligned_raw_text(hypothesis, hyp, indices):
    """Recover candidate text, not punctuation-free lexical unit values.

    Adjacent selected units include their intervening raw punctuation. Outer
    signs and percentages belong to an adjacent number even though the mixed
    lexical tokenizer does not retain them. Disjoint selected runs must not
    pull in lexical units assigned to another reference span.
    """
    if not indices:
        return ""
    runs = []
    lo = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            runs.append((lo, previous))
            lo = index
        previous = index
    runs.append((lo, previous))
    pieces = []
    for first, last in runs:
        start, end = hyp[first].start, hyp[last].end
        if (start and unicodedata.normalize("NFKC", hypothesis[start:start+1]).isdecimal()
                and unicodedata.normalize("NFKC", hypothesis[start-1:start]) in ("+", "-", "−")):
            start -= 1
        if (end < len(hypothesis) and unicodedata.normalize("NFKC", hypothesis[end-1:end]).isdecimal()
                and unicodedata.normalize("NFKC", hypothesis[end:end+1]) == "%"):
            end += 1
        pieces.append(hypothesis[start:end])
    # Do not create a new numeric adjacency across unselected lexical text.
    return " ".join(pieces)


def local_switch_cer(reference, ref, hyp, changes, operations, *, hypothesis):
    scores = []
    for start, end in changes:
        positions = {i for i, u in enumerate(ref) if start <= u.start < end}
        # Inserts at a boundary are attached to the following reference unit by
        # the lexical edit path. No text is recovered from the reference here.
        indices = sorted({j for _, i, j in operations if i in positions and j is not None})
        candidate = _aligned_raw_text(hypothesis, hyp, indices)
        scores.append(cer_metrics(reference[start:end], candidate)["cer"])
    bounded = [min(1., x) for x in scores]
    return {"switch_cer": sum(scores) / len(scores) if scores else None,
            "switch_cer_worst": max(scores) if scores else None,
            "switch_cer_cost": .5 * (max(bounded) + sum(bounded)/len(bounded)) if bounded else 0.}
