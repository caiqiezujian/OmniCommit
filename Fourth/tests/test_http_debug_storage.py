import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_asr.http_debug import DebugArchive


def test_meeting_uses_one_jsonl_file_for_concurrent_events(tmp_path):
    archive = DebugArchive(tmp_path)
    meeting = "../会议/变更"
    rid = uuid4().hex
    folder = tmp_path / hashlib.sha256(meeting.encode()).hexdigest()
    archive.event(meeting, rid, "packet", text="a\nb")
    # A single meeting can generate writes from the ASGI loop and model workers concurrently.
    def worker(i):
        archive.event(meeting, rid, "model_call", i=i, prompt="问" * 5000)
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        list(pool.map(worker, range(20)))
    records = [json.loads(line) for line in (folder / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert records[0]["text"] == "a\nb"
    assert sorted(r["i"] for r in records[1:]) == list(range(20))
    assert all((r["meeting_id"], r["request_id"]) == (meeting, rid) for r in records)
    assert set(tmp_path.iterdir()) == {folder}
    assert list(folder.iterdir()) == [folder / "events.jsonl"]


def test_disk_and_serialization_errors_leave_session_running(tmp_path, caplog):
    blocked = tmp_path / "existing_file"
    blocked.write_text("keep")
    rid = uuid4().hex
    archive = DebugArchive(blocked)
    archive.event("a", rid, "failure")
    good = DebugArchive(tmp_path / "working")
    good.event("a", rid, "unsupported", data=object())
    good.event("a", rid, "ok")
    assert blocked.read_text() == "keep"
    assert all(json.loads(r.getMessage())["event"] == "debug_archive_error" for r in caplog.records)
    assert len(caplog.records) >= 3
    saved = next(good.directory.glob("*/events.jsonl"))
    assert json.loads(saved.read_text())["event"] == "ok"
