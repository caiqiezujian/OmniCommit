"""Readable packet journals and optional model traces for one Fourth HTTP session."""
from datetime import datetime, timedelta, timezone
import hashlib
from itertools import count
import json
import logging
from pathlib import Path, PureWindowsPath
import re
import threading

from .http_logging import request_event


BEIJING = timezone(timedelta(hours=8))


def meeting_folder_name(meeting_id):
    """Keep ordinary IDs readable and give escaped or shortened IDs distinct names."""
    slug = re.sub(r"[^\w-]", "_", meeting_id[:48])
    if slug and slug == meeting_id and not PureWindowsPath(slug).is_reserved():
        return slug
    digest = hashlib.sha256(meeting_id.encode("utf-8", errors="surrogatepass")).hexdigest()[:12]
    return f"{slug or 'meeting'}~{digest}"


class DebugArchive:
    """Shared by the upload handler and decoder worker for one meeting run."""

    def __init__(self, directory, meeting_id, session_id, *, detailed=False):
        root = Path(directory).expanduser().absolute()
        meeting_path = root / meeting_folder_name(meeting_id)
        start = datetime.now(BEIJING).strftime("%Y-%m-%d_%H-%M-%S")
        self.directory = meeting_path / f"{start}_{session_id[:12]}"
        self.meeting_id = meeting_id
        self.session_id = session_id
        self.detailed = detailed
        self.packet_numbers = count(1)
        self._lock = threading.Lock()
        try:
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            meeting_path.mkdir(exist_ok=True, mode=0o700)
            self.directory.mkdir(exist_ok=True, mode=0o700)
        except OSError as exc:
            self._report("open", exc)

    def _report(self, operation, exc, request_id=None):
        request_event("debug_archive_error", level=logging.ERROR, archive_operation=operation,
                      directory=str(self.directory), meeting_id=self.meeting_id, request_id=request_id,
                      error_type=type(exc).__name__, error=str(exc))

    def _append(self, filename, record, request_id):
        try:
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
            with self._lock, (self.directory / filename).open("a", encoding="utf-8", errors="backslashreplace",
                                                               newline="\n") as handle:
                handle.write(line)
        except (OSError, TypeError, ValueError, RecursionError) as exc:
            self._report(filename, exc, request_id)

    def note(self, event, request_id=None, **fields):
        row = {"时间": datetime.now(BEIJING).isoformat(timespec="milliseconds"), "事件": event, **fields}
        if request_id:
            row["请求号"] = request_id
            if request_id == self.session_id:
                row["会议号"] = self.meeting_id
        self._append("events.jsonl", row, request_id)

    def event(self, request_id, event, **fields):
        if self.detailed:
            row = dict(timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                       meeting_id=self.meeting_id, request_id=request_id, session_id=self.session_id,
                       event=event, **fields)
            self._append("trace.jsonl", row, request_id)


def turn_fields(result):
    action = result["action"]
    if action is None:
        action = "跳过收尾（缺口已记录）" if result.get("end_reason") == "gap_already_reported" else "解码缺口"
    row = {"轮": result["turn"], "动作": action,
           "本轮耗时_ms": round(result["generation_seconds"] * 1000, 1)}
    if result["event"] == "audio":
        row["新增音频_s"] = [round(result["start_seconds"], 7), round(result["end_seconds"], 7)]
    if result["display_text"]:
        row["模型正文"] = result["display_text"]
    return row


def response_fields(code, body):
    row = {"HTTP": code}
    if code == 200:
        row["src"] = body["text"]
    else:
        row["原因"] = body["error"]
        if body["text"]:
            row["已提交正文"] = body["text"]
    if body.get("buffered_seconds"):
        row["暂存音频_s"] = body["buffered_seconds"]
    if body.get("gaps"):
        row["解码缺口轮"] = [gap["turn"] for gap in body["gaps"]]
    for turn in body["turns"]:
        edit = turn.get("packet_period_edit")
        if edit:
            action = "删除句号" if edit["operation"] == "remove" else "句号改逗号"
            row["标点处理"] = f"第{turn['turn']}轮{action}，{body['silence_ms']} ms < {edit['threshold_ms']} ms"
    return row


REJECTIONS = {
    "in_flight": "同一会议的上一请求仍在处理", "ended": "会议已结束", "failed": "会议此前处理失败",
    "expired": "会议已空闲过期", "unknown_meeting": "找不到该会议", "invalid_meeting_id": "会议号无效",
    "meeting_capacity": "活跃会议数已达上限", "inference_capacity": "并发请求数已达上限",
    "upload_timeout": "接收音频包超时", "invalid_packet": "音频包无效", "request_setup_failed": "服务准备请求失败",
}
