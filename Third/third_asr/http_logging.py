"""JSONL request diagnostics for the single-process Third HTTP service."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys


DEFAULT_REQUEST_LOG = Path(__file__).resolve().parents[1] / "logs" / "http-requests.jsonl"
logger = logging.getLogger("third_asr.http_requests")


def request_event(event, *, level=logging.INFO, **fields):
    """Callers supply operational metadata; audio, transcripts and headers stay out."""
    if logger.isEnabledFor(level):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": logging.getLevelName(level),
            "event": event,
            **fields,
        }
        logger.log(level, json.dumps(record, ensure_ascii=False, separators=(",", ":")))


@contextmanager
def open_request_log(path):
    """Keep the full request history on disk and print actionable events to stderr."""
    path = Path(path).expanduser().absolute()
    console = logging.StreamHandler(sys.stderr)
    handlers = [console]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(path, maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8")
    except OSError as exc:
        print(f"[http] Cannot open request log {path}: {exc}. Writing request events to stderr.",
              file=sys.stderr, flush=True)
    else:
        console.setLevel(logging.WARNING)
        handlers.append(file_handler)
        print(f"[http] Request log: {path}", file=sys.stderr, flush=True)
    old_level, old_propagate = logger.level, logger.propagate
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    try:
        yield
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(old_level)
        logger.propagate = old_propagate
