"""HTTP transport for Third's streaming protocol. One meeting owns one acoustic history."""
import asyncio
import base64
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
import io
import json
import math
import time
from urllib.parse import quote
import wave

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .runtime.streaming import SAMPLE_RATE
from .service_session import ThirdStreamSession


PCM_TYPES = {"audio/pcm", "application/octet-stream"}
WAV_TYPES = {"audio/wav", "audio/wave", "audio/x-wav"}


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
    return samples


async def read_packet(request, sample_limit):
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
    received_at = time.monotonic()
    if media == "application/json":
        return decode_json_audio(body, sample_limit), received_at
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
    return samples, received_at


@dataclass
class Meeting:
    meeting_id: str
    session: ThirdStreamSession
    touched: float
    tail: np.ndarray = None
    last: object = None
    job: object = None
    needs_collection: bool = False

    def process(self, audio, ending):
        """Worker thread only. Never submit short intermediate packets as final model turns."""
        if self.tail is not None:
            audio = np.concatenate((self.tail, audio))
            self.tail = None
        step_samples = round(self.session.options.chunk_seconds * SAMPLE_RATE)
        offset = 0
        turns = []
        try:
            while len(audio) - offset >= step_samples or (ending and offset < len(audio)):
                stop = min(len(audio), offset + step_samples)
                turns.append(self.session.step(audio[offset:stop], final_chunk=ending and stop == len(audio)))
                offset = stop
            if offset < len(audio):
                self.tail = audio[offset:].copy()
            if ending:
                turns.append(self.session.step(end_of_audio=True))
            status = "ended" if ending else ("ok" if turns else "buffered")
            code, error = 200, None
        except Exception as exc:
            # Some earlier turns in this same HTTP packet may have committed.
            # Preserve them for the caller, then retire this meeting.
            code, status, error = 500, "error", str(exc)
        text = "".join(turn["display_text"] for turn in turns)
        gaps = [dict(turn["truncation"], turn=turn["turn"], event=turn["event"]) for turn in turns
                if turn.get("truncation", {}).get("status") == "gap"]
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
        if error is not None:
            payload["error"] = error
            payload["unprocessed_seconds"] = (len(audio) - offset) / SAMPLE_RATE
        return code, payload


def create_app(backend_factory, options, *, max_meetings=16, max_in_flight=8,
               response_timeout_ms=900, idle_timeout_seconds=600, max_packet_seconds=8.0,
               upload_timeout_seconds=10):
    """backend_factory creates one thread-safe inference adapter, shared by all meeting workers."""
    for name, value in (("max_meetings", max_meetings), ("max_in_flight", max_in_flight)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (("response_timeout_ms", response_timeout_ms), ("idle_timeout_seconds", idle_timeout_seconds),
                        ("max_packet_seconds", max_packet_seconds), ("upload_timeout_seconds", upload_timeout_seconds)):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    sample_limit = int(max_packet_seconds * SAMPLE_RATE)
    if sample_limit < 1:
        raise ValueError("max_packet_seconds must allow at least one PCM sample")
    active = {}
    terminal = OrderedDict()
    reserved = set()  # Includes uploads; bounded before any request body is buffered.
    backend = executor = None

    def retire(meeting, outcome):
        active.pop(meeting.meeting_id)
        meeting.session.close()
        terminal[meeting.meeting_id] = (time.monotonic(), outcome)
        while len(terminal) > 2 * max_meetings:
            terminal.popitem(last=False)

    def expire():
        cutoff = time.monotonic() - idle_timeout_seconds
        for key, (touched, _) in list(terminal.items()):
            if touched < cutoff:
                del terminal[key]
        for key, meeting in list(active.items()):
            if key not in reserved and meeting.job is None and meeting.touched < cutoff:
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
                                      thread_name_prefix="third-meeting")
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

    app = FastAPI(title="Third meeting ASR", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"ready": backend is not None, "service": "third_http",
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
                "response_timeout_ms": response_timeout_ms, "idle_timeout_seconds": idle_timeout_seconds,
                "max_packet_seconds": max_packet_seconds, "upload_timeout_seconds": upload_timeout_seconds}

    def respond(outcome):
        code, body = outcome
        return JSONResponse(body, status_code=code, headers={"Cache-Control": "no-store"})

    def pending(key):
        url = "/v1/meetings/" + quote(key, safe="") + "/result"
        return JSONResponse({"meeting_id": key, "status": "processing", "action": None, "result_url": url},
                            status_code=202, headers={"Location": url, "Cache-Control": "no-store"})

    def reserve(key, ending):
        if not key.strip() or len(key) > 256 or key in {".", ".."}:
            raise HTTPException(422, "Invalid meeting_id")
        expire()
        if key in terminal:
            raise HTTPException(409, "Meeting has ended, failed or expired. GET its result or use a new meeting_id")
        meeting = active.get(key)
        if key in reserved or (meeting is not None and meeting.needs_collection):
            raise HTTPException(409, "Previous packet outstanding. GET the meeting result before sending another")
        if ending and meeting is None:
            raise HTTPException(404, "Unknown meeting")
        if not meeting and len(active.keys() | reserved) >= max_meetings:
            raise HTTPException(429, "Meeting capacity reached; packet not accepted")
        if len(reserved) >= max_in_flight:
            raise HTTPException(429, "Inference capacity reached; packet not accepted")
        reserved.add(key)

    async def dispatch(meeting, pcm, ending):
        try:
            try:
                outcome = await asyncio.get_running_loop().run_in_executor(executor, meeting.process, pcm, ending)
            except Exception as exc:
                outcome = (500, {"meeting_id": meeting.meeting_id, "status": "error", "action": None,
                                 "text": "", "error": str(exc), "turns": []})
            meeting.last = outcome
            meeting.touched = time.monotonic()
            if ending or outcome[0] >= 400:
                retire(meeting, outcome)
            return outcome
        finally:
            meeting.job = None
            reserved.discard(meeting.meeting_id)

    async def post(key, request, ending):
        reserve(key, ending)
        try:
            if ending:
                pcm, received_at = np.empty(0, dtype=np.float32), time.monotonic()
            else:
                try:
                    pcm, received_at = await asyncio.wait_for(read_packet(request, sample_limit), upload_timeout_seconds)
                except asyncio.TimeoutError as exc:
                    raise HTTPException(408, "Upload timed out; packet not accepted") from exc
            meeting = active.get(key)
            if meeting is None:
                meeting = active[key] = Meeting(key, ThirdStreamSession(backend, options), received_at)
            meeting.needs_collection = True
            job = meeting.job = asyncio.create_task(dispatch(meeting, pcm, ending))
        except BaseException:
            reserved.discard(key)
            raise
        remaining = max(0., received_at + response_timeout_ms / 1000 - time.monotonic())
        # Timeout/disconnect releases the HTTP waiter; the model still owns this meeting until it finishes.
        done, _ = await asyncio.wait((job,), timeout=remaining)
        if not done:
            return pending(key)
        outcome = job.result()
        if meeting.last is outcome and meeting.job is None:
            meeting.needs_collection = False
        return respond(outcome)

    @app.post("/v1/meetings/{meeting_id}/chunks", summary="Append one audio packet to a meeting",
              openapi_extra={"requestBody": {"required": True,
                  "description": "16 kHz mono audio. JSON uses base64 raw little-endian float32. Binary uses PCM16.",
                  "content": {
                      "application/json": {"schema": {"type": "object", "required": ["data"], "properties": {
                          "data": {"type": "string", "format": "byte",
                                   "description": "Standard base64 of normalized mono float32 LE samples"}}}},
                      **{media: {"schema": {"type": "string", "format": "binary"}}
                         for media in ("audio/pcm", "audio/wav", "application/octet-stream")}}}},
              responses={200: {"description": "Model result, partial result with explicit gaps, or buffered audio"},
                         202: {"description": "Accepted and still processing; poll the Location URL"},
                         409: {"description": "Outstanding operation or closed meeting"},
                         429: {"description": "Capacity reached; packet was not accepted"}})
    async def chunk(meeting_id: str, request: Request):
        return await post(meeting_id, request, False)

    @app.post("/v1/meetings/{meeting_id}/end")
    async def end(meeting_id: str, request: Request):
        return await post(meeting_id, request, True)

    @app.get("/v1/meetings/{meeting_id}/result")
    async def result(meeting_id: str):
        expire()
        if meeting_id in terminal:
            return respond(terminal[meeting_id][1])
        meeting = active.get(meeting_id)
        if meeting is None and meeting_id not in reserved:
            raise HTTPException(404, "Unknown meeting or result has expired")
        if meeting_id in reserved or meeting.last is None:
            return pending(meeting_id)
        meeting.needs_collection = False
        meeting.touched = time.monotonic()
        return respond(meeting.last)

    return app
