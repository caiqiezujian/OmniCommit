"""HTTP transport for Fourth's streaming protocol. One meeting owns one acoustic history."""
import asyncio
import base64
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from decimal import Decimal
import hashlib
import io
import json
import logging
import math
import sys
import time
import traceback
from typing import Optional
from urllib.parse import quote
from uuid import uuid4
import wave

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.requests import ClientDisconnect

from .debug_backend import DebugBackend
from .http_debug import DebugArchive, REJECTIONS, response_fields, turn_fields
from .http_logging import request_event
from .packet_punctuation import (DEFAULT_PACKET_PERIOD_REMOVE_THRESHOLD_MS, DEFAULT_PACKET_PERIOD_THRESHOLD_MS,
                                 resolve_remove_threshold)
from .runtime.streaming import SAMPLE_RATE
from .service_session import FourthStreamSession


PCM_TYPES = {"audio/pcm", "application/octet-stream"}
WAV_TYPES = {"audio/wav", "audio/wave", "audio/x-wav"}
logger = logging.getLogger(__name__)


class Transcript(BaseModel):
    src: str


def decode_json_audio(body, sample_limit):
    try:
        packet = json.loads(body)
    except (ValueError, RecursionError) as exc:
        raise HTTPException(400, "Invalid JSON audio packet") from exc
    encoded = packet.get("data") if isinstance(packet, dict) else None
    if not isinstance(encoded, str) or not encoded:
        raise HTTPException(400, "Send a JSON object with a nonempty base64 string in data")
    if len(encoded) > 4 * ((sample_limit * 4 + 2) // 3):
        raise HTTPException(413, "Float32 audio packet exceeds configured duration")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise HTTPException(400, "data must contain standard base64") from exc
    if len(raw) > sample_limit * 4:
        raise HTTPException(413, "Float32 audio packet exceeds configured duration")
    if not raw or len(raw) % 4:
        raise HTTPException(400, "Decoded data must contain whole float32 samples")
    samples = np.frombuffer(raw, dtype="<f4")
    if not np.isfinite(samples).all() or np.max(np.abs(samples)) > 1.001:
        raise HTTPException(400, "Decoded float32 samples must be finite and normalized to [-1, 1]")
    silence_fields = [key for key in ("tail_silence", "silence_ms", "trailing_silence_ms") if key in packet]
    if not silence_fields:
        return samples, None
    if len(silence_fields) > 1:
        raise HTTPException(400, "Provide only one silence measurement per audio packet")
    key = silence_fields[0]
    value = packet[key]
    scale_to_ms = 1000 if key == "tail_silence" else 1
    if type(value) not in (int, float) or not 0 <= value <= sys.float_info.max / scale_to_ms:
        raise HTTPException(400, f"{key} must be a nonnegative number representable in finite milliseconds")
    if key == "tail_silence":
        # Decimal scaling keeps a caller's 1.001 seconds exactly at the 1001 ms boundary.
        value = float(Decimal(str(value)) * scale_to_ms)
    return samples, value


async def read_packet_body(request, sample_limit):
    media = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media == "application/json":
        # Base64 expands each three bytes to four characters. Allow a small JSON envelope.
        limit = 4 * ((sample_limit * 4 + 2) // 3) + 4096
    elif media in PCM_TYPES | WAV_TYPES:
        limit = sample_limit * 2 + (65536 if media in WAV_TYPES else 0)
    else:
        raise HTTPException(415, "Send JSON base64 float32 LE, raw PCM16 LE, or PCM16 WAV (mono, 16000 Hz)")
    declared = request.headers.get("content-length")
    if declared is not None:
        if not declared.isdecimal():
            raise HTTPException(400, "Invalid Content-Length")
        if int(declared) > limit:
            raise HTTPException(413, "Audio packet exceeds configured size limit")
    body = bytearray()
    async for piece in request.stream():
        if len(body) + len(piece) > limit:
            raise HTTPException(413, "Audio packet exceeds configured size limit")
        body.extend(piece)
    return media, body


def decode_packet(media, body, sample_limit):
    if media == "application/json":
        return decode_json_audio(body, sample_limit)
    if media in WAV_TYPES:
        if len(body) < 12 or int.from_bytes(body[4:8], "little") + 8 > len(body):
            raise HTTPException(400, "Incomplete WAV container")
        try:
            with wave.open(io.BytesIO(body), "rb") as wav:
                if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getcomptype()) != (
                        1, SAMPLE_RATE, 2, "NONE"):
                    raise HTTPException(415, "WAV must be mono 16000 Hz uncompressed signed PCM16")
                frames = wav.getnframes()
                if frames > sample_limit:
                    raise HTTPException(413, "WAV duration exceeds configured limit")
                body = wav.readframes(frames + 1)  # Also detect an odd trailing byte in a PCM16 data chunk.
                if len(body) != frames * 2:
                    raise HTTPException(400, "Truncated WAV data")
        except (wave.Error, EOFError, RuntimeError) as exc:
            raise HTTPException(400, "Invalid PCM16 WAV container") from exc
    if not body or len(body) % 2:
        raise HTTPException(400, "Send at least one whole PCM16 sample")
    samples = np.frombuffer(body, dtype="<i2").astype(np.float32)
    samples /= 32768.0
    return samples, None


@dataclass
class PacketRequest:
    meeting_id: str
    operation: str
    peer: Optional[str]
    archive: Optional[DebugArchive] = None
    packet_number: Optional[int] = None
    request_id: str = field(default_factory=lambda: uuid4().hex)
    started: float = field(default_factory=time.monotonic)
    phase: str = "upload"

    def elapsed_ms(self):
        return round((time.monotonic() - self.started) * 1000, 1)

    @property
    def detailed(self):
        return self.archive is not None and self.archive.detailed

    def note(self, event, *, include_id=True, **fields):
        if self.archive is not None:
            packet = {"包": self.packet_number} if self.operation == "chunks" else {"操作": "结束"}
            self.archive.note(event, self.request_id if include_id else None, **packet, **fields)

    def debug(self, event, **fields):
        if self.detailed:
            self.archive.event(self.request_id, event, operation=self.operation, **fields)

    def log(self, event, *, level=logging.INFO, **fields):
        request_event(event, level=level, meeting_id=self.meeting_id[:256], request_id=self.request_id,
                      operation=self.operation, peer=self.peer, **fields)
        self.debug(event, peer=self.peer, **fields)

    def reject(self, code, detail, reason, **fields):
        self.log("request_rejected", level=logging.WARNING, status_code=code, reason=reason,
                 duration_ms=self.elapsed_ms(), **fields)
        self.debug("response", status_code=code, body={"detail": detail})
        if self.archive is not None:
            row = {"HTTP": code, "原因": REJECTIONS.get(reason, detail), "耗时_ms": self.elapsed_ms()}
            if reason == "invalid_packet":
                row["详情"] = detail
            if "blocking_request_id" in fields:
                row["占用请求号"] = fields["blocking_request_id"]
                row["占用阶段"] = "接收上传" if fields["blocking_phase"] == "upload" else "转录中"
            if "last_request_id" in fields:
                row["前次请求号"] = fields["last_request_id"]
            self.note("拒绝", **row)
        raise HTTPException(code, detail, headers={"X-Request-ID": self.request_id})

    def log_processing_error(self, exc):
        # Exception messages can contain model output. Keep the stack locations and type.
        stack = [f"{frame.filename}:{frame.lineno} in {frame.name}"
                 for frame in traceback.extract_tb(exc.__traceback__)]
        self.log("processing_error", level=logging.ERROR, error_type=type(exc).__name__, stack=stack)
        self.debug("exception", error_type=type(exc).__name__, error=str(exc), stack=stack)


@dataclass
class ClosedMeeting:
    closed_at: float
    outcome: tuple
    last_request_id: Optional[str]
    archive: Optional[DebugArchive]


@dataclass
class Meeting:
    meeting_id: str
    session: FourthStreamSession
    touched: float
    tail: np.ndarray = None
    last: object = None
    job: object = None
    last_request_id: Optional[str] = None
    archive: Optional[DebugArchive] = None
    received_samples: int = 0
    tail_silence_ms: Optional[float] = None
    packet_period_to_comma: bool = False
    packet_period_threshold_ms: float = DEFAULT_PACKET_PERIOD_THRESHOLD_MS
    packet_period_remove_threshold_ms: float = DEFAULT_PACKET_PERIOD_REMOVE_THRESHOLD_MS

    def step(self, trace, audio=None, *, final_chunk=False, end_of_audio=False):
        session = self.session
        if trace.archive is None:
            return session.step(audio, final_chunk=final_chunk, end_of_audio=end_of_audio)
        turn = session.turn_count + 1
        backend = session.backend
        if trace.detailed:
            start = round(session.heard_seconds * SAMPLE_RATE)
            trace.debug("turn_started", turn=turn, start_sample=start, end_sample=start + (0 if audio is None else len(audio)),
                        sample_rate=SAMPLE_RATE, final_chunk=final_chunk, end_of_audio=end_of_audio)
            # Only this worker owns the session. The underlying batched engine is shared.
            session.backend = DebugBackend(backend, lambda event, **fields: trace.debug(event, turn=turn, **fields))
        try:
            result = session.step(audio, final_chunk=final_chunk, end_of_audio=end_of_audio)
        except Exception as exc:
            trace.note("轮失败", 轮=turn, 原因=str(exc))
            trace.debug("turn_error", turn=turn, error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            if trace.detailed:
                session.backend = backend
        trace.note("收尾轮" if end_of_audio else "逐轮", include_id=False, **turn_fields(result))
        trace.debug("turn_finished", turn=turn, turn_result=result)
        return result

    def process(self, audio, ending, trace, silence_ms=None):
        """Worker thread only. Never submit short intermediate packets as final model turns."""
        if self.tail is not None:
            if ending:
                silence_ms = self.tail_silence_ms
            audio = np.concatenate((self.tail, audio))
            self.tail = None
            self.tail_silence_ms = None
        step_samples = round(self.session.options.chunk_seconds * SAMPLE_RATE)
        offset = 0
        turns = []
        last_direct = None
        correct_period = (self.packet_period_to_comma and silence_ms is not None
                          and silence_ms < self.packet_period_threshold_ms)
        try:
            while len(audio) - offset >= step_samples or (ending and offset < len(audio)):
                stop = min(len(audio), offset + step_samples)
                at_packet_end = stop == len(audio)
                before = self.session.packet_text_state() if correct_period else None
                result = self.step(trace, audio[offset:stop], final_chunk=ending and at_packet_end)
                turns.append(result)
                if correct_period and result["action"] == "DIRECT":
                    last_direct = (result, before)
                elif result.get("truncation", {}).get("status") == "gap":
                    last_direct = None
                offset = stop
            if offset < len(audio):
                self.tail = audio[offset:].copy()
                self.tail_silence_ms = silence_ms
            elif correct_period and last_direct is not None:
                target, before = last_direct
                edit = self.session.apply_packet_period(target, *before, silence_ms=silence_ms,
                                                        comma_ms=self.packet_period_threshold_ms,
                                                        remove_ms=self.packet_period_remove_threshold_ms)
                if edit is not None:
                    event = "packet_period_removed" if edit.operation == "remove" else "packet_period_to_comma"
                    trace.debug(event, turn=target["turn"], silence_ms=silence_ms,
                                threshold_ms=edit.threshold_ms, language=edit.language, text=target["text"])
            if ending:
                turns.append(self.step(trace, end_of_audio=True))
            status = "ended" if ending else ("ok" if turns else "buffered")
            code, error = 200, None
        except Exception as exc:
            # Some earlier turns in this same HTTP packet may have committed.
            # Preserve them for the caller, then retire this meeting.
            code, status, error = 500, "error", str(exc)
            trace.log_processing_error(exc)
        text = "".join(turn["display_text"] for turn in turns)
        gaps = [dict(turn["truncation"], turn=turn["turn"], event=turn["event"]) for turn in turns
                if turn.get("truncation", {}).get("status") == "gap"]
        for gap in gaps:
            logger.warning("Meeting %r ASR gap at %.3f-%.3f seconds (turn %d)",
                           self.meeting_id, gap["start_seconds"], gap["end_seconds"], gap["turn"])
        if gaps and status == "ok":
            status = "partial"
        action = None
        if error is None:
            if text:
                action = "DIRECT"
            elif turns and not gaps and all(turn["action"] == "WAIT" for turn in turns):
                action = "WAIT"
        payload = dict(meeting_id=self.meeting_id, status=status, action=action, text=text, turns=turns, gaps=gaps,
                       heard_seconds=self.session.heard_seconds,
                       buffered_seconds=self.tail.size / SAMPLE_RATE if self.tail is not None else 0.0)
        if silence_ms is not None:
            payload["silence_ms"] = silence_ms
        if error is not None:
            payload["error"] = error
            payload["unprocessed_seconds"] = (len(audio) - offset) / SAMPLE_RATE
        return code, payload


def create_app(backend_factory, options, *, max_meetings=16, max_in_flight=8,
               idle_timeout_seconds=600, max_packet_seconds=8.0,
               upload_timeout_seconds=10, debug_dir=None, debug_model=None, event_log_mode="readable",
               packet_period_to_comma=False, packet_period_threshold_ms=DEFAULT_PACKET_PERIOD_THRESHOLD_MS,
               packet_period_remove_threshold_ms=None):
    """backend_factory creates one thread-safe inference adapter, shared by all meeting workers."""
    for name, value in (("max_meetings", max_meetings), ("max_in_flight", max_in_flight)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (("idle_timeout_seconds", idle_timeout_seconds),
                        ("max_packet_seconds", max_packet_seconds), ("upload_timeout_seconds", upload_timeout_seconds)):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    packet_period_remove_threshold_ms = resolve_remove_threshold(packet_period_threshold_ms,
                                                                  packet_period_remove_threshold_ms)
    sample_limit = int(max_packet_seconds * SAMPLE_RATE)
    if sample_limit < 1:
        raise ValueError("max_packet_seconds must allow at least one PCM sample")
    active = {}
    terminal = OrderedDict()
    reserved = {}  # Maps meeting ID to the one request uploading or processing audio.
    backend = executor = None
    if event_log_mode not in ("readable", "detailed"):
        raise ValueError("event_log_mode must be readable or detailed")

    def retire(meeting, outcome):
        active.pop(meeting.meeting_id)
        meeting.session.close()
        terminal[meeting.meeting_id] = ClosedMeeting(time.monotonic(), outcome, meeting.last_request_id, meeting.archive)
        while len(terminal) > 2 * max_meetings:
            terminal.popitem(last=False)

    def expire():
        cutoff = time.monotonic() - idle_timeout_seconds
        for key, closed in list(terminal.items()):
            if closed.closed_at < cutoff:
                del terminal[key]
        for key, meeting in list(active.items()):
            if key not in reserved and meeting.job is None and meeting.touched < cutoff:
                idle_seconds = round(time.monotonic() - meeting.touched, 3)
                request_event("meeting_expired", meeting_id=key, last_request_id=meeting.last_request_id,
                              idle_seconds=idle_seconds)
                if meeting.archive is not None:
                    meeting.archive.note("会议空闲过期", 空闲_s=idle_seconds, 前次请求号=meeting.last_request_id)
                    meeting.archive.event(meeting.last_request_id, "meeting_expired", idle_seconds=idle_seconds)
                retire(meeting, (410, {"meeting_id": key, "status": "expired"}))

    async def cleanup_loop():
        while True:
            await asyncio.sleep(min(60., idle_timeout_seconds))
            expire()

    @asynccontextmanager
    async def lifespan(app):
        nonlocal backend, executor
        backend = await asyncio.to_thread(backend_factory)
        executor = ThreadPoolExecutor(max_workers=min(max_meetings, max_in_flight),
                                      thread_name_prefix="fourth-meeting")
        cleaner = asyncio.create_task(cleanup_loop())
        try:
            yield
        finally:
            cleaner.cancel()
            await asyncio.gather(cleaner, return_exceptions=True)
            jobs = [meeting.job for meeting in active.values() if meeting.job is not None]
            if jobs:
                await asyncio.gather(*jobs, return_exceptions=True)
            executor.shutdown(wait=True)
            for meeting in active.values():
                meeting.session.close()
            active.clear()
            terminal.clear()
            reserved.clear()
            if callable(getattr(backend, "close", None)):
                await asyncio.to_thread(backend.close)
            backend = None

    app = FastAPI(title="Fourth meeting ASR", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"ready": backend is not None and getattr(backend, "ready", True), "service": "fourth_http",
                "scheduling_mode": getattr(backend, "scheduling_mode", "synchronous"),
                "packet_period_to_comma": packet_period_to_comma,
                "packet_period_threshold_ms": packet_period_threshold_ms,
                "packet_period_remove_threshold_ms": packet_period_remove_threshold_ms,
                "backend": getattr(backend, "backend_name", type(backend).__name__),
                "max_new_tokens": getattr(backend, "max_new_tokens", None),
                "max_context_tokens": getattr(backend, "max_context_tokens", None),
                "action_protocol": options.protocol, "context_policy": options.context_policy,
                "chunk_seconds": options.chunk_seconds, "window_seconds": options.window_seconds,
                "discard_seconds": options.discard_seconds,
                "max_meetings": max_meetings, "active_meetings": len(active),
                "max_in_flight": max_in_flight, "in_flight": len(reserved),
                "max_batch_size": getattr(backend, "max_batch_size", 1),
                "batch_wait_ms": getattr(backend, "batch_wait_ms", 0),
                "batch_stats": backend.stats() if hasattr(backend, "stats") else {},
                "debug_capture_configured": debug_dir is not None,
                "event_log_mode": event_log_mode if debug_dir is not None else "off",
                "idle_timeout_seconds": idle_timeout_seconds,
                "max_packet_seconds": max_packet_seconds, "upload_timeout_seconds": upload_timeout_seconds}

    def respond(outcome, request_id=None):
        code, body = outcome
        headers = {"Cache-Control": "no-store"}
        if request_id is not None:
            headers["X-Request-ID"] = request_id
        return JSONResponse(body, status_code=code, headers=headers)

    def pending(key):
        url = "/v1/meetings/" + quote(key, safe="") + "/result"
        return JSONResponse({"meeting_id": key, "status": "processing", "action": None, "result_url": url},
                            status_code=202, headers={"Location": url, "Cache-Control": "no-store"})

    def reserve(trace):
        key = trace.meeting_id
        if not key.strip() or len(key) > 256 or key in {".", ".."}:
            trace.reject(422, "Invalid meeting_id", "invalid_meeting_id")
        if not getattr(backend, "ready", True):
            trace.reject(503, "Model worker is unavailable", "model_unavailable")
        if key in terminal:
            closed = terminal[key]
            _, body = closed.outcome
            reason = "failed" if body["status"] == "error" else body["status"]
            trace.reject(409, "Meeting has ended, failed or expired. Use a new meeting_id", reason,
                         last_request_id=closed.last_request_id, closed_ms_ago=round((time.monotonic() - closed.closed_at) * 1000, 1))
        meeting = active.get(key)
        if key in reserved:
            blocking = reserved[key]
            trace.reject(409, "A packet is still being processed for this meeting", "in_flight",
                         blocking_request_id=blocking.request_id, blocking_operation=blocking.operation,
                         blocking_phase=blocking.phase, blocking_elapsed_ms=blocking.elapsed_ms())
        if trace.operation == "end" and meeting is None:
            trace.reject(404, "Unknown meeting", "unknown_meeting")
        if not meeting and len(active.keys() | reserved.keys()) >= max_meetings:
            trace.reject(429, "Meeting capacity reached; packet not accepted", "meeting_capacity")
        if len(reserved) >= max_in_flight:
            trace.reject(429, "Inference capacity reached; packet not accepted", "inference_capacity")
        reserved[key] = trace
        trace.log("request_started")

    async def dispatch(meeting, pcm, ending, trace, upload_ms, silence_ms):
        try:
            try:
                outcome = await asyncio.get_running_loop().run_in_executor(
                    executor, meeting.process, pcm, ending, trace, silence_ms)
            except Exception as exc:
                trace.log_processing_error(exc)
                outcome = (500, {"meeting_id": meeting.meeting_id, "status": "error", "action": None,
                                 "text": "", "error": str(exc), "turns": []})
            meeting.last = outcome
            meeting.last_request_id = trace.request_id
            meeting.touched = time.monotonic()
            code, body = outcome
            trace.log("request_finished", level=logging.ERROR if code >= 500 else logging.INFO,
                      status_code=code, status=body["status"], action=body["action"],
                      audio_seconds=round(len(pcm) / SAMPLE_RATE, 6), turns=len(body["turns"]),
                      src_chars=len(body["text"]), upload_ms=upload_ms, duration_ms=trace.elapsed_ms())
            if trace.archive is not None:
                trace.note("返回", **response_fields(code, body),
                           整包耗时_ms=trace.elapsed_ms())
            if trace.detailed:
                trace.debug("response", status_code=code, body={"src": body["text"]} if code == 200 else body,
                            heard_seconds=body.get("heard_seconds"), buffered_seconds=body.get("buffered_seconds"))
            if ending or outcome[0] >= 400:
                retire(meeting, outcome)
            return outcome
        finally:
            meeting.job = None
            reserved.pop(meeting.meeting_id, None)

    async def post(key, request, ending):
        trace = PacketRequest(key, "end" if ending else "chunks", request.client.host if request.client else None)
        expire()
        if debug_dir is not None:
            # A conflict may arrive while the first upload is still creating its meeting.
            owner = reserved.get(key) or active.get(key) or terminal.get(key)
            trace.archive = owner.archive if owner is not None else DebugArchive(
                debug_dir, key, trace.request_id, detailed=event_log_mode == "detailed")
            if not ending:
                trace.packet_number = next(trace.archive.packet_numbers)
            trace.debug("request_received", peer=trace.peer, content_type=request.headers.get("content-type"),
                        content_length=request.headers.get("content-length"))
        reserve(trace)
        job = None
        try:
            if ending:
                trace.note("结束请求")
                pcm = np.empty(0, dtype=np.float32)
                silence_ms = None
                upload_ms = 0.0
            else:
                try:
                    media, body = await asyncio.wait_for(read_packet_body(request, sample_limit), upload_timeout_seconds)
                    upload_ms = trace.elapsed_ms()
                    if trace.detailed:
                        trace.debug("request_body", content_type=media, body_bytes=len(body),
                                    body_sha256=hashlib.sha256(body).hexdigest())
                    pcm, silence_ms = decode_packet(media, body, sample_limit)
                    del body
                except asyncio.TimeoutError:
                    trace.reject(408, "Upload timed out; packet not accepted", "upload_timeout")
                except HTTPException as exc:
                    trace.reject(exc.status_code, exc.detail, "invalid_packet", detail_message=exc.detail)
            meeting = active.get(key)
            if meeting is None:
                meeting = active[key] = Meeting(key, FourthStreamSession(backend, options), time.monotonic(),
                                                archive=trace.archive, packet_period_to_comma=packet_period_to_comma,
                                                packet_period_threshold_ms=packet_period_threshold_ms,
                                                packet_period_remove_threshold_ms=packet_period_remove_threshold_ms)
                if trace.detailed:
                    trace.debug("session_started", model=debug_model, options=asdict(options),
                                backend=getattr(backend, "backend_name", type(backend).__name__),
                                max_context_tokens=getattr(backend, "max_context_tokens", None),
                                max_new_tokens=getattr(backend, "max_new_tokens", None))
            if trace.archive is not None and not ending:
                start = meeting.received_samples
                meeting.received_samples += len(pcm)
                trace.note("收包", 录音区间_s=[start / SAMPLE_RATE, meeting.received_samples / SAMPLE_RATE],
                           上报静音_ms=silence_ms)
                trace.debug("packet_audio", start_sample=start, end_sample=meeting.received_samples,
                            sample_rate=SAMPLE_RATE, silence_ms=silence_ms,
                            buffered_before_samples=0 if meeting.tail is None else len(meeting.tail))
            trace.phase = "processing"
            job = meeting.job = asyncio.create_task(dispatch(meeting, pcm, ending, trace, upload_ms,
                                                             silence_ms))
        except (ClientDisconnect, asyncio.CancelledError):
            trace.note("接收中断", 耗时_ms=trace.elapsed_ms())
            trace.log("request_interrupted", level=logging.WARNING, phase=trace.phase, duration_ms=trace.elapsed_ms())
            raise
        except HTTPException:
            raise
        except Exception as exc:
            trace.log_processing_error(exc)
            trace.reject(500, "Internal Server Error", "request_setup_failed")
        finally:
            if job is None:
                reserved.pop(key, None)
        # A disconnected caller must not cancel work that can already have advanced the session.
        try:
            code, body = await asyncio.shield(job)
        except asyncio.CancelledError:
            trace.note("等待响应已取消", 耗时_ms=trace.elapsed_ms())
            trace.log("response_wait_cancelled", level=logging.WARNING, duration_ms=trace.elapsed_ms())
            raise
        return respond((code, {"src": body["text"]} if code == 200 else body), trace.request_id)

    @app.post("/v1/meetings/{meeting_id}/chunks", summary="Transcribe one audio packet",
              response_model=Transcript,
              openapi_extra={"requestBody": {"required": True,
                  "description": "16 kHz mono audio. JSON uses base64 raw little-endian float32. Binary uses PCM16.",
                  "content": {
                      "application/json": {"schema": {"type": "object", "required": ["data"], "properties": {
                          "data": {"type": "string", "format": "byte",
                                   "description": "Standard base64 of normalized mono float32 LE samples"},
                          "tail_silence": {"type": "number", "minimum": 0,
                                  "description": "Caller-reported silence duration for this packet in seconds. "
                                                 "When enabled, the last committed sentence period is removed below "
                                                 "the lower threshold or changed to a comma below the upper threshold. "
                                                 "The same thresholds apply in every language. "
                                                 "Send only one silence field per packet."},
                          "silence_ms": {"type": "number", "minimum": 0,
                                  "description": "Total silence in this audio packet (milliseconds). Optional. "
                                                 "Uses the same thresholds as tail_silence."},
                          "trailing_silence_ms": {"type": "number", "minimum": 0,
                                  "description": "Legacy measurement field accepted in place of silence_ms."}}}},
                      **{media: {"schema": {"type": "string", "format": "binary"}}
                         for media in ("audio/pcm", "audio/wav", "application/octet-stream")}}}},
              responses={200: {"description": "New transcript text for this packet, empty when nothing was emitted"},
                         409: {"description": "Outstanding operation or closed meeting"},
                         429: {"description": "Capacity reached; packet was not accepted"}})
    async def chunk(meeting_id: str, request: Request):
        return await post(meeting_id, request, False)

    @app.post("/v1/meetings/{meeting_id}/end", response_model=Transcript,
              summary="Finish the meeting and return any final transcript text")
    async def end(meeting_id: str, request: Request):
        return await post(meeting_id, request, True)

    @app.get("/v1/meetings/{meeting_id}/result", include_in_schema=False)
    async def result(meeting_id: str):
        expire()
        if meeting_id in terminal:
            return respond(terminal[meeting_id].outcome)
        meeting = active.get(meeting_id)
        if meeting is None and meeting_id not in reserved:
            raise HTTPException(404, "Unknown meeting or result has expired")
        if meeting_id in reserved or meeting.last is None:
            return pending(meeting_id)
        return respond(meeting.last)

    return app
