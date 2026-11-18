"""File-backed meeting traces for the opt-in HTTP debug mode."""
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import tempfile
import threading

from .http_logging import request_event
from .training_audio import SAMPLE_RATE


class DebugArchive:
    """Write each meeting's JSONL and payload files under a path-safe directory."""

    def __init__(self, directory):
        self.directory = Path(directory).expanduser().absolute()
        # Bound the number of locks across arbitrarily many meeting IDs.
        self._locks = [threading.Lock() for _ in range(32)]
        try:
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError as exc:
            self._report("open", exc)

    def _report(self, operation, exc, meeting_id=None, request_id=None):
        request_event("debug_archive_error", level=logging.ERROR, archive_operation=operation,
                      directory=str(self.directory), meeting_id=meeting_id, request_id=request_id,
                      error_type=type(exc).__name__, error=str(exc))

    def _folder(self, meeting_id):
        digest = hashlib.sha256(meeting_id.encode("utf-8", errors="surrogatepass")).hexdigest()
        folder = self.directory / digest
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        return folder

    def event(self, meeting_id, request_id, event, **fields):
        try:
            record = dict(timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                          meeting_id=meeting_id, request_id=request_id, event=event, **fields)
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
            folder = self._folder(meeting_id)
            lock = self._locks[int(folder.name[:2], 16) % len(self._locks)]
            with lock, (folder / "events.jsonl").open("a", encoding="utf-8", errors="backslashreplace",
                                                      newline="\n") as handle:
                handle.write(line)
        except (OSError, TypeError, ValueError, RecursionError) as exc:
            self._report("event", exc, meeting_id, request_id)

    def _artifact(self, meeting_id, request_id, extension, write):
        temp = None
        try:
            # The HTTP service creates UUID4 hex IDs. Never use arbitrary paths from requests.
            if not isinstance(request_id, str) or not re.fullmatch(r"[0-9a-f]{32}", request_id):
                raise ValueError("Expected a server-generated request ID")
            folder = self._folder(meeting_id)
            destination = folder / (request_id + extension)
            with tempfile.NamedTemporaryFile(dir=folder, prefix=".", suffix=".tmp", delete=False) as handle:
                temp = Path(handle.name)
                write(handle)
            temp.replace(destination)
            temp = None
            return destination.name
        except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
            self._report(extension, exc, meeting_id, request_id)
        finally:
            if temp is not None:
                try:
                    temp.unlink(missing_ok=True)
                except OSError as exc:
                    self._report("cleanup", exc, meeting_id, request_id)
        return None

    def body(self, meeting_id, request_id, data):
        """Preserve the exact bytes of the bounded HTTP request body."""
        return self._artifact(meeting_id, request_id, ".body", lambda handle: handle.write(data))

    def audio(self, meeting_id, request_id, pcm):
        """Save the validated 16 kHz mono float32 signal without PCM16 rounding."""
        def write_wav(handle):
            import soundfile as sf
            sf.write(handle, pcm, SAMPLE_RATE, format="WAV", subtype="FLOAT")

        return self._artifact(meeting_id, request_id, ".wav", write_wav)
