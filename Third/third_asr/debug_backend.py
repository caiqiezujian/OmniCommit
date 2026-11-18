"""Observe a single Third HTTP turn at the interface to the shared inference backend."""
import logging
import time
from uuid import uuid4

from .runtime.streaming import GenerationTruncated


logger = logging.getLogger(__name__)


class DebugBackend:
    """Wrap one session.step() so retries and recovery refer to the same source turn."""

    def __init__(self, backend, callback):
        self.backend = backend
        self.callback = callback
        self.primary_attempts = 0

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def _record(self, event, **fields):
        try:
            self.callback(event, **fields)
        except Exception:
            logger.exception("Failed to record ASR debug event %s", event)

    def generate(self, messages, audios):
        kind = "primary" if self.primary_attempts == 0 else "retry"
        self.primary_attempts += 1
        return self._call(messages, audios, kind, prefix="")

    def generate_with_prefix(self, messages, audios, prefix):
        return self._call(messages, audios, "recovery", prefix)

    def _call(self, messages, audios, kind, prefix):
        ref = {"call_id": uuid4().hex, "kind": kind}
        self._record("model_call", **ref, backend=getattr(self.backend, "backend_name", type(self.backend).__name__),
                     messages=messages, audio_samples=[len(wave) for wave in audios], prefix=prefix)
        # Includes the time spent waiting for BatchBackend as well as model generation.
        started = time.perf_counter()
        try:
            if kind == "recovery":
                reply = self.backend.generate_with_prefix(messages, audios, prefix)
            else:
                reply = self.backend.generate(messages, audios)
        except Exception as exc:
            fields = {}
            if isinstance(exc, GenerationTruncated):
                fields = {"finish_reason": "length", **exc.as_dict()}
            self._record("model_error", **ref, error_type=type(exc).__name__, error=str(exc),
                         elapsed_ms=round((time.perf_counter() - started) * 1000, 3), **fields)
            raise
        # Capture the returned text before protocol parsing, including invalid tags,
        # leading spaces, and the supplied prefix for a forced decode.
        metadata = {}
        if isinstance(reply, dict):
            raw = reply.get("text")
            metadata = {key: reply[key] for key in ("prompt_tokens", "num_cached_tokens", "generated_tokens",
                                                    "finish_reason") if key in reply}
        else:
            raw = reply
        self._record("model_reply", **ref, raw_text=raw, elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                     **metadata)
        return reply
