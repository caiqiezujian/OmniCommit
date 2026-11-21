"""Inspect only the source's final annotation, never rebuild its old segmentation.

The complete assistant transcript remains the text authority. Whitespace and
punctuation differences in an auxiliary tail are tolerated for suffix identity;
no auxiliary text is copied into the training reference. Old WAIT/DIRECT actions
do not decide which words are retained.
"""
from __future__ import annotations

from fractions import Fraction
import json
import math
import re
import unicodedata


_TAGS = re.compile(r"<(?:spk[1-9][0-9]*|[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*)>")
_HEADER_TIME = re.compile(r"\[\d+(?:\.\d+)?ms\](?=<)")
_CLOSERS = frozenset("\"'”’»›」』】）》〉〕〗〙〛)]}")
_TEXT_FIELDS = ("source_text", "text_plain", "text", "word", "char", "alignment_text")


class TailSourceError(ValueError):
    def __init__(self, reason, **details):
        super().__init__(reason)
        self.reason, self.details = reason, details


def _plain(value):
    if not isinstance(value, str):
        raise TailSourceError("invalid_tail_text")
    return _TAGS.sub("", _HEADER_TIME.sub("", value))


def _body(item):
    if not isinstance(item, dict):
        raise TailSourceError("invalid_tail_annotation")
    for field in _TEXT_FIELDS:
        if field in item:
            return _plain(item[field])
    return None


def _lexical(text):
    text = unicodedata.normalize("NFC", text)
    def numeric_mark(i, c):
        before = i > 0 and text[i - 1].isdecimal()
        after = i + 1 < len(text) and text[i + 1].isdecimal()
        return (c in ".,-/:" and before and after) or (c in "+-−" and after) or (c in "%٪‰‱" and before)
    return "".join(c for i, c in enumerate(text)
                   if unicodedata.category(c)[0] in "LNM" or numeric_mark(i, c))


def has_terminal_punctuation(text):
    """Sentence/comma punctuation counts; closing wrappers and numeric % do not."""
    text = text.rstrip()
    while text and (text[-1] in _CLOSERS or unicodedata.category(text[-1]) in ("Pe", "Pf")):
        text = text[:-1].rstrip()
    if not text:
        return False
    last = unicodedata.normalize("NFKC", text[-1])
    if last in ("%", "٪", "‰", "‱") and len(text) > 1 and text[-2].isdecimal():
        return False
    return unicodedata.category(text[-1]).startswith("P")


def _tail_segment(row):
    output = row.get("output")
    if output is None or output == "":
        return None, None
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except ValueError as exc:
            raise TailSourceError("invalid_tail_output_json") from exc
    if not isinstance(output, dict) or not isinstance(output.get("segments"), list):
        raise TailSourceError("invalid_tail_output_segments")
    # Inspect only the trailing content. Intermediate units may contain stale
    # actions, reflowed spaces or bad timestamps and are irrelevant to this gate.
    for index in range(len(output["segments"]) - 1, -1, -1):
        segment = output["segments"][index]
        text = _body(segment)
        if text is None:
            raise TailSourceError("missing_tail_segment_text")
        if text.strip():
            return text, index
    raise TailSourceError("empty_tail_output")


def _tail_entries(row):
    if row.get("timestamps"):
        if not isinstance(row["timestamps"], list):
            raise TailSourceError("invalid_tail_timestamps")
        return row["timestamps"], "timestamps"
    alignment = row.get("alignment")
    if isinstance(alignment, list):
        return alignment, "alignment"
    if isinstance(alignment, dict):
        if alignment.get("status", "ok") != "ok":
            raise TailSourceError("unsuccessful_alignment")
        for key in ("chars", "words", "word_segments"):
            if key in alignment:
                return alignment[key], "alignment." + key
    for key in ("chars", "words", "word_segments"):
        if key in row:
            return row[key], key
    return [], "none"


def _milliseconds(item, edge):
    value = item.get(edge + "_ms") if edge + "_ms" in item else item.get(edge)
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise TailSourceError("invalid_tail_timestamp")
    return Fraction(str(value)) * (1 if edge + "_ms" in item else 1000)


def tail_text_metadata(row, reference_text):
    """Inspect the final body only; do not require or validate any timestamps.

    A final old DIRECT without punctuation is the user's explicit exemption.
    If the authoritative transcript still has a period, preserve its text but
    mark terminal punctuation supervision untrusted for the reward.
    """
    reference = _lexical(reference_text)
    if not reference:
        raise TailSourceError("no_timed_transcript_text")
    # Full-text open endings are safe for this punctuation-only filter even if
    # the old segmentation is missing or damaged. They are not rewritten.
    punctuated = has_terminal_punctuation(reference_text)
    metadata = dict(tail_has_punctuation=punctuated, last_text_end_ms=None,
                    tail_gap_ms=None, tail_decision="keep_unpunctuated",
                    terminal_punctuation_trusted=True, tail_text_source="reference")
    if not punctuated:
        return metadata
    segment, segment_index = _tail_segment(row)
    if segment is not None:
        suffix = _lexical(segment)
        if suffix and not reference.endswith(suffix):
            raise TailSourceError("tail_segment_text_mismatch")
        metadata.update(tail_text_source="final_segment", tail_segment_index=segment_index)
        # The user explicitly exempts an unpunctuated final distilled DIRECT.
        # Only the actual final content-bearing segment may qualify, never an
        # earlier DIRECT before a trailing WAIT that contains more speech.
        if suffix and not has_terminal_punctuation(segment):
            metadata.update(tail_has_punctuation=False, tail_decision="keep_unpunctuated_segment",
                            terminal_punctuation_trusted=False)
            return metadata
    metadata["tail_decision"] = "needs_tail_timestamp"
    return metadata


def inspect_source_tail(row, reference_text, duration_ms, *, min_tail_ms=1000, enforce=True):
    """Return tail metadata; ``enforce=False`` annotates a failed gap without dropping.

    Missing required final timestamps are errors even with enforce=False. A
    punctuation-free ending does not need any timestamp. This function reads no
    audio and never supplies fine-grained alignment to the reward.
    """
    metadata = tail_text_metadata(row, reference_text)
    if not metadata["tail_has_punctuation"]:
        return metadata
    reference = _lexical(reference_text)
    duration = duration_ms if isinstance(duration_ms, Fraction) else Fraction(str(duration_ms))
    segment, segment_index = _tail_segment(row)
    entries, origin = _tail_entries(row)
    if not isinstance(entries, list):
        raise TailSourceError("invalid_tail_timestamps")
    chosen = None
    for index in range(len(entries) - 1, -1, -1):
        entry = entries[index]
        text = _body(entry)
        if text is not None and not _lexical(text):
            continue  # A separately timed full stop is not acoustic speech.
        if text is None:
            if origin == "timestamps" and segment_index == index and segment is not None:
                text = segment
            else:
                raise TailSourceError("missing_trusted_final_text_end")
        suffix = _lexical(text)
        if not suffix or not reference.endswith(suffix):
            raise TailSourceError("tail_timestamp_text_mismatch")
        # Optional exact offsets are verified only for this final anchor. Do not
        # make unrelated intermediate alignment rows a prerequisite for the gate.
        if "char_start" in entry or "char_end" in entry:
            lo, hi = entry.get("char_start"), entry.get("char_end")
            if (type(lo) is not int or type(hi) is not int or not 0 <= lo < hi <= len(reference_text)
                    or _lexical(reference_text[lo:hi]) != suffix
                    or _lexical(reference_text[hi:])):
                raise TailSourceError("tail_timestamp_offsets_mismatch")
        if entry.get("trusted", True) is False:
            raise TailSourceError("missing_trusted_final_text_end")
        end, start = _milliseconds(entry, "end"), _milliseconds(entry, "start")
        if end is None:
            raise TailSourceError("missing_trusted_final_text_end")
        if end < 0 or end > duration or (start is not None and not 0 <= start < end):
            raise TailSourceError("invalid_tail_timestamp_bounds")
        chosen = end
        break
    if chosen is None:
        raise TailSourceError("missing_trusted_final_text_end")
    gap = duration - chosen
    metadata.update(last_text_end_ms=float(chosen), tail_gap_ms=float(gap),
                    tail_timestamp_source=origin, tail_timestamp_index=index,
                    tail_decision="keep_punctuated_with_gap" if gap > min_tail_ms else "reject_punctuated_short_tail")
    if gap <= min_tail_ms and enforce:
        raise TailSourceError("tail_not_greater_than_threshold", **metadata)
    return metadata
