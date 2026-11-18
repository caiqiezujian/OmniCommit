"""File-backed meeting traces for the opt-in HTTP debug mode."""
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import threading

from .http_logging import request_event


class DebugArchive:
    """Append diagnostic events to one JSONL file per meeting."""

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
