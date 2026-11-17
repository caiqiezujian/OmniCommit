"""A voiced WAIT streak can trigger a single local decode of already heard audio."""
import time
import unicodedata

import numpy as np

from .runtime.streaming import SAMPLE_RATE
from .protocol import DIRECT


def check_recovery_dependencies():
    try:
        import webrtcvad
    except ImportError as exc:
        raise RuntimeError("DIRECT recovery needs webrtcvad-wheels. Install the Third web "
                           "requirements or pass --no-recovery.") from exc
    return webrtcvad


class SpeechActivity:
    """Feed 10 ms PCM16 frames once, including across non-multiple chunk sizes."""
    frame_samples = SAMPLE_RATE // 100

    def __init__(self):
        self.vad = check_recovery_dependencies().Vad(2)
        self.tail = np.array([], dtype=np.int16)

    def observe(self, audio):
        """Return (first voiced sample relative to this chunk, total voiced samples)."""
        pcm = np.clip(np.rint(audio * 32768), -32768, 32767).astype("<i2")
        carry = len(self.tail)
        pcm = np.concatenate((self.tail, pcm))
        size = self.frame_samples
        end = len(pcm) // size * size
        self.tail = pcm[end:].copy()
        first, count = None, 0
        for offset in range(0, end, size):
            frame = pcm[offset:offset + size]
            if self.vad.is_speech(frame.tobytes(), SAMPLE_RATE) and np.any(frame):
                if first is None:
                    first = offset - carry
                count += size
        return first, count


def _lexemes(text):
    """Return comparable words/characters and their end offsets in the original."""
    out = []
    start = None
    for i, char in enumerate(text + " "):
        code = ord(char)
        cjk = (0x3400 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF
               or 0x20000 <= code <= 0x2FA1F)
        if char.isalnum() and not cjk:
            if start is None:
                start = i
            continue
        if start is not None:
            if unicodedata.category(char).startswith("M"):
                continue
            out.append((unicodedata.normalize("NFKC", text[start:i]).casefold(), i))
            start = None
        if cjk and char.isalnum():
            out.append((char, i + 1))
    return out


def novel_continuation(candidate, previous, *, has_overlap):
    """Keep candidate spelling and whitespace; exact lexical overlap only."""
    new = _lexemes(candidate)
    if not new:
        return "", "no_words"
    old = _lexemes(previous)
    words = [word for word, _ in new]
    prior = [word for word, _ in old]
    for count in range(min(len(new), len(old)), 0, -1):
        if prior[-count:] != words[:count]:
            continue
        if count == len(words):
            return "", "already_committed"
        # A common single word or a single CJK character is ambiguous; do
        # not append a duplicate or silently delete a genuine repetition.
        if count == 1 or not has_overlap:
            return "", "ambiguous_overlap"
        end = new[count - 1][1]
        for char in previous[old[-1][1]:]:
            if char.isspace() or end == len(candidate) or candidate[end] != char:
                break
            end += 1
        return candidate[end:], None
    return candidate, None


class VoicedWaitRecovery:
    def __init__(self, options, activity=None):
        self.options = options
        self.activity = activity if activity is not None else SpeechActivity()
        self.reset()

    def reset(self):
        self.first_voice = None
        self.voiced_samples = 0
        self.attempted = False

    def observe(self, session, result):
        turn = session.history[-1]
        length = round((turn.end - turn.start) * SAMPLE_RATE)
        # The model may have seen padding on a short final chunk. It is never
        # evidence of speech or a reason to extend the recovery audio interval.
        audio = turn.audio[:length]
        first, count = self.activity.observe(audio)
        if (self.options.context_policy == "rolling" and result["action"] == "WAIT"
                and not count and length >= SpeechActivity.frame_samples):
            # Require both low RMS (-60 dBFS) and low peak before using VAD
            # silence for eviction. Broadband noise or uncertain speech stays.
            if np.max(np.abs(audio)) <= .005 and np.mean(np.square(audio)) <= 1e-6:
                session.quiet_waits.add(turn.start)
        if result["action"] == "DIRECT":
            self.reset()
            return
        if not count or self.attempted:
            return
        if self.first_voice is None:
            self.first_voice = max(turn.start + first / SAMPLE_RATE, 0.)
        self.voiced_samples += count
        if self.voiced_samples < round(self.options.recovery_voiced_seconds * SAMPLE_RATE):
            return
        self.attempted = True
        start = max(self.first_voice - self.options.recovery_overlap_seconds,
                    session.verified_commit_seconds, 0.)
        stats = result["recovery"] = {"start_seconds": start, "end_seconds": turn.end,
                                     "voiced_seconds": self.voiced_samples / SAMPLE_RATE}
        if turn.end - start > self.options.recovery_window_seconds + 1e-8:
            stats["status"] = "window_exceeded"
            return
        pieces = []
        position = start
        for item in session.history:
            if item.end <= start:
                continue
            if item.start > position + .5 / SAMPLE_RATE:
                # Zero-valued WAITs can have been compacted after a long pause.
                # Only shorten the pre-roll, never skip the pending voiced region.
                if pieces or item.start > self.first_voice + .5 / SAMPLE_RATE:
                    stats["status"] = "audio_unavailable"
                    return
                stats["start_seconds"] = item.start
            low = round(max(0., start - item.start) * SAMPLE_RATE)
            high = round((item.end - item.start) * SAMPLE_RATE)
            pieces.append(item.audio[low:high])
            position = item.end
        if not pieces:
            stats["status"] = "audio_unavailable"
            return
        messages = [{"role": "system", "content": session._context_system_prompt()},
                    {"role": "user", "content": "<audio>"}]
        began = time.perf_counter()
        try:
            reply = session.backend.generate_with_prefix(messages, [np.concatenate(pieces)], DIRECT)
            raw = reply["text"] if isinstance(reply, dict) else reply
            stats["raw_text"] = raw
            try:
                candidate = session.protocol.parse_reply(raw)
            except ValueError:
                stats.update(status="rejected", reason="invalid_reply")
                return
            continuation, reason = novel_continuation(
                candidate, session.committed_tail, has_overlap=stats["start_seconds"] < session.last_text_end)
            if reason:
                stats.update(status="rejected", reason=reason)
                return
            committed = DIRECT + continuation
            result.update(text=committed, display_text=continuation, action="DIRECT",
                          primary_text=turn.reply)
            turn.reply = committed
            stats["status"] = "accepted"
            # The shadow transcript gives no precise word-level consumption
            # time. Bound rolling eviction by the PREVIOUS text turn, its tail
            # guard, and this window's start. Keep the whole newly decoded
            # interval; verified consumption always requires a trusted client.
            if self.options.context_policy == "rolling":
                cutoff = min(session.last_text_end - self.options.rolling_tail_seconds,
                             stats["start_seconds"])
                session.inferred_commit_seconds = max(session.inferred_commit_seconds, cutoff)
                result["inferred_commit_seconds"] = session.inferred_commit_seconds
            self.reset()
        except Exception as exc:
            # Failure of an optional shadow decode preserves the primary WAIT.
            stats.update(status="error", reason=str(exc))
        finally:
            stats["seconds"] = time.perf_counter() - began
