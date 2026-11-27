import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_asr.http_debug import DebugArchive, meeting_folder_name


def test_meeting_uses_one_jsonl_file_for_concurrent_events(tmp_path):
    meeting = "../会议/变更"
    rid = uuid4().hex
    archive = DebugArchive(tmp_path, meeting, rid)
    folder = archive.directory
    assert folder.parent.parent == tmp_path
    assert folder.parent.name.endswith("会议_变更~" + hashlib.sha256(meeting.encode()).hexdigest()[:12])
    archive.note("收包", rid, text="a\nb")
    # A single meeting can generate writes from the ASGI loop and model workers concurrently.
    def worker(i):
        archive.note("逐轮", i=i, 模型正文="问" * 5000)
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        list(pool.map(worker, range(20)))
    records = [json.loads(line) for line in (folder / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert records[0]["text"] == "a\nb"
    assert sorted(r["i"] for r in records[1:]) == list(range(20))
    assert (records[0]["会议号"], records[0]["请求号"]) == (meeting, rid)
    assert set(tmp_path.iterdir()) == {folder.parent}
    assert list(folder.iterdir()) == [folder / "events.jsonl"]


def test_disk_and_serialization_errors_leave_session_running(tmp_path, caplog):
    blocked = tmp_path / "existing_file"
    blocked.write_text("keep")
    rid = uuid4().hex
    archive = DebugArchive(blocked, "a", rid)
    archive.note("收包", rid)
    good = DebugArchive(tmp_path / "working", "a", rid)
    good.note("收包", rid, data=object())
    good.note("返回", rid)
    assert blocked.read_text() == "keep"
    assert all(json.loads(r.getMessage())["event"] == "debug_archive_error" for r in caplog.records)
    assert len(caplog.records) >= 3
    saved = good.directory / "events.jsonl"
    assert json.loads(saved.read_text(encoding="utf-8"))["事件"] == "返回"


def test_readable_meeting_ids_and_separate_runs(tmp_path):
    assert meeting_folder_name("6424126945") == "6424126945"
    assert meeting_folder_name("会议-001") == "会议-001"
    names = [meeting_folder_name(value) for value in ("A/B", "A_B", "A\\B", "..", "CON", "a" * 256)]
    assert len(set(names)) == len(names)
    assert all("/" not in name and "\\" not in name and name not in ("CON", "..") for name in names)
    first, second = (DebugArchive(tmp_path, "6424126945", uuid4().hex) for _ in range(2))
    assert first.directory.parent == second.directory.parent == tmp_path / "6424126945"
    assert first.directory != second.directory
