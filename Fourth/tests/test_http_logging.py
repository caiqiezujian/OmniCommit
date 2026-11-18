"""Exercise the on-disk request log across rotation and server restarts."""
from datetime import datetime, timedelta
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_asr.http_logging import DEFAULT_REQUEST_LOG, logger, open_request_log, request_event


def test_jsonl_rotation_and_restart(tmp_path):
    assert DEFAULT_REQUEST_LOG == Path(__file__).resolve().parents[1] / "logs" / "http-requests.jsonl"
    path = tmp_path / "logs" / "requests.jsonl"
    before = logger.handlers[:]
    with open_request_log(path):
        handler, = (h for h in logger.handlers if isinstance(h, RotatingFileHandler))
        assert handler.maxBytes == 20 * 1024 * 1024 and handler.backupCount == 5
        handler.maxBytes = 10
        request_event("reserved", meeting_id="中文\n会议")
        request_event("conflict", level=logging.WARNING)
    assert logger.handlers == before
    with open_request_log(path):
        request_event("after_restart")
    assert logger.handlers == before
    rotated = path.with_name(path.name + ".1")
    rows = [json.loads(line) for f in (rotated, path)
            for line in f.read_text(encoding="utf-8").splitlines()]
    assert [r["event"] for r in rows] == ["reserved", "conflict", "after_restart"]
    assert rows[0]["meeting_id"] == "中文\n会议"
    assert rows[1]["level"] == "WARNING"
    assert datetime.fromisoformat(rows[1]["timestamp"]).utcoffset() == timedelta()


def test_unwritable_destination_reports_diagnostics_to_stderr(tmp_path, capsys):
    path = tmp_path / "regular-file"
    path.write_text("block mkdir", encoding="utf-8")
    with open_request_log(path / "http.jsonl"):
        request_event("rejected", status_code=409)
    lines = capsys.readouterr().err.splitlines()
    assert "stderr" in lines[0] and str(path) in lines[0]
    assert json.loads(lines[-1])["status_code"] == 409


def test_console_shows_rejection_and_error_details_and_file_keeps_every_event(tmp_path, capsys):
    path = tmp_path / "requests.jsonl"
    with open_request_log(path):
        request_event("request_finished", request_id="a", status_code=200)
        request_event("request_rejected", level=logging.WARNING, request_id="b", status_code=409,
                      reason="in_flight", blocking_request_id="a", blocking_phase="processing")
        request_event("processing_error", level=logging.ERROR, request_id="c", error_type="RuntimeError")
    file_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    console_rows = [json.loads(line) for line in capsys.readouterr().err.splitlines() if line.startswith("{")]
    assert [row["request_id"] for row in file_rows] == ["a", "b", "c"]
    assert console_rows == file_rows[1:]
