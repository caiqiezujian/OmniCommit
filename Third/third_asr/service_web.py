"""One serialized model worker; bounded admission and per-session state."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import time
from urllib.parse import urlparse

import numpy as np

from .runtime.media import MediaCursor, MediaStore
from .service_session import ThirdStreamSession


def annotate_file_result(row, cursor, source_start, source_end):
    row["source_start_seconds"] = source_start
    row["source_end_seconds"] = source_end
    gap = row.get("truncation", {})
    if gap.get("status") != "gap":
        return
    for key in ("start_seconds", "end_seconds"):
        value = gap[key]
        if value == row["end_seconds"]:
            absolute = source_end
        elif value == row["start_seconds"]:
            absolute = source_start
        else:
            # Earlier completed turns lie on the chunk grid. MediaCursor rounds
            # each nominal millisecond boundary up to a whole source frame.
            rate = cursor.upload.sample_rate
            frame_offset = (round(value * 1000) * rate + 999) // 1000
            absolute = (cursor.start_frame + frame_offset) / rate
        gap["source_" + key] = absolute


def create_app(backend_factory, options, *, max_sessions=1, auto_reference_factory=None,
               max_window_seconds=None):
    if auto_reference_factory is not None:
        raise ValueError("Third transcribes plain text; speaker references are unsupported")
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    if max_sessions < 1:
        raise ValueError("max_sessions must be positive")
    if max_window_seconds is not None and (not math.isfinite(max_window_seconds)
            or max_window_seconds < options.window_seconds):
        raise ValueError("max_window_seconds must cover the default window")
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="omnicommit-model")
    active = set()
    backend = None
    recovery = options.protocol == "explicit" and options.enable_recovery
    protocol_settings = {"action_protocol": options.protocol, "gap_recovery": recovery,
                         "gap_asr": "omni" if recovery else "disabled",
                         "recovery_voiced_seconds": options.recovery_voiced_seconds}
    media_store = MediaStore(max_slots=max(2, max_sessions * 2))

    def window_settings(selected):
        if selected.context_policy != "window":
            return {}
        limit = max_window_seconds
        slots = getattr(backend, "max_audio_turns", None)
        if slots is not None:
            capacity = slots * selected.chunk_seconds
            limit = capacity if limit is None else min(capacity, limit)
        return {"window_seconds": selected.window_seconds, "discard_seconds": selected.discard_seconds,
                "max_window_seconds": limit}

    @asynccontextmanager
    async def lifespan(app):
        nonlocal backend
        async def expire_uploads():
            while True:
                await asyncio.sleep(60)
                media_store.expire()
        expiry_task = None
        try:
            backend = await asyncio.get_running_loop().run_in_executor(executor, backend_factory)
            expiry_task = asyncio.create_task(expire_uploads())
            yield
        finally:
            if expiry_task is not None:
                expiry_task.cancel()
                await asyncio.gather(expiry_task, return_exceptions=True)
            # In-flight generation finishes before shutdown; no model racing.
            executor.shutdown(wait=True, cancel_futures=True)
            media_store.close()

    app = FastAPI(lifespan=lifespan)
    app.state.media_store = media_store
    web_directory = Path(__file__).resolve().parents[1] / "web"
    app.mount("/web", StaticFiles(directory=web_directory), name="web")

    @app.get("/")
    async def index():
        return FileResponse(web_directory / "meeting-v2" / "index.html",
                            headers={"Cache-Control": "no-store"})

    @app.get("/health")
    async def health():
        return JSONResponse({"ready": backend is not None, "protocol": "third", "active_sessions": len(active),
                "max_sessions": max_sessions, "chunk_seconds": options.chunk_seconds,
                "context_policy": options.context_policy,
                "backend": type(backend).__name__ if backend else None,
                "max_new_tokens": getattr(backend, "max_new_tokens", None),
                "max_context_tokens": getattr(backend, "max_context_tokens", None),
                "native_prefix_caching": getattr(backend, "enable_prefix_caching", False),
                "native_mm_processor_cache": getattr(backend, "mm_processor_cache_gb", 0) > 0,
                "manual_kv_reuse": False, "file_input": bool(shutil.which("ffmpeg")),
                "automatic_reference": "disabled",
                **window_settings(options), **protocol_settings}, headers={"Cache-Control": "no-store"})

    async def upload_media(request):
        origin = request.headers.get("origin")
        if origin and urlparse(origin).netloc != request.headers.get("host"):
            return JSONResponse({"error": "Cross-origin upload denied"}, status_code=403)
        upload = None
        decode_task = None
        try:
            upload = media_store.reserve()
            async for chunk in request.stream():
                upload.append(bytes(chunk))
            # Do not delete files while a cancelled FFmpeg worker is still using them.
            decode_task = asyncio.create_task(asyncio.to_thread(upload.decode))
            result = await asyncio.shield(decode_task)
            return JSONResponse(result)
        except asyncio.CancelledError:
            if decode_task is not None:
                try:
                    await asyncio.shield(decode_task)
                except Exception:
                    pass
            if upload is not None:
                media_store.discard(upload.media_id)
            raise
        except Exception as exc:
            if upload is not None:
                media_store.discard(upload.media_id)
            return JSONResponse({"error": str(exc)}, status_code=400)

    upload_media.__annotations__["request"] = Request
    app.post("/api/media")(upload_media)

    # Avoid postponed annotation resolving a function-local WebSocket import.
    async def stream(websocket):
        origin = websocket.headers.get("origin")
        if origin and urlparse(origin).netloc != websocket.headers.get("host"):
            await websocket.close(code=1008, reason="Cross-origin connection denied")
            return
        await websocket.accept()
        session = None
        receive_task = None
        cursor = None
        upload = None
        source_heard = None
        attempted_source_end = None
        task = None
        admitted = False

        try:
            init = await asyncio.wait_for(websocket.receive_json(), timeout=10)
            if init.get("type") != "start" or init.get("sample_rate") != 16000:
                raise ValueError("Start requires sample_rate=16000; send mono PCM16 little-endian")
            if len(active) >= max_sessions:
                raise ValueError("Service session capacity reached; no audio accepted")
            source_kind = init.get("source_kind", "microphone")
            if source_kind not in ("microphone", "system", "file"):
                raise ValueError("Unknown audio source")
            range_keys = ("file_start_seconds", "file_end_seconds")
            if source_kind != "file" and any(key in init for key in range_keys):
                raise ValueError("A file range requires an uploaded file session")
            if "chunk_ms" in init:
                raise ValueError("This service uses a fixed chunk length")
            window_keys = ("window_seconds", "discard_seconds")
            overrides = {key: init[key] for key in window_keys if key in init}
            if overrides and options.context_policy != "window":
                raise ValueError("Window settings require the window context policy")
            selected = replace(options, **overrides) if overrides else options
            settings = window_settings(selected)
            limit = settings.get("max_window_seconds")
            if limit is not None and selected.window_seconds > limit + 1e-8:
                raise ValueError(f"本服务支持的最大窗口为 {limit:g} 秒。")
            if source_kind == "file":
                upload = media_store.claim(init.get("media_id"))
                cursor = MediaCursor(upload, start_seconds=init.get("file_start_seconds", 0),
                                     end_seconds=init.get("file_end_seconds"))
                source_heard = cursor.start_seconds
            session = ThirdStreamSession(backend, selected)
            active.add(session)
            admitted = True
            await websocket.send_json({"type": "ready", "protocol": "third",
                "chunk_seconds": options.chunk_seconds,
                **({"file_start_seconds": cursor.start_seconds, "file_end_seconds": cursor.end_seconds}
                   if cursor is not None else {}),
                "context_policy": options.context_policy, "backend": type(backend).__name__,
                "warning": {"window": "窗口将超长时按整轮丢弃旧音频；连续 WAIT 也使用相同规则。",
                            "rolling": "Rolling mode is an unverified tail heuristic",
                            "verified": "Verified watermarks required before safe context eviction"}[options.context_policy],
                "automatic_reference": "disabled",
                "auto_reference": "disabled",
                **settings, **protocol_settings})
            while True:
                # A user may pause file playback for longer than two minutes.
                # Live capture keeps an inactivity limit; files wait for an
                # explicit resume/end or a transport disconnect.
                receive = websocket.receive()
                receive_task = asyncio.ensure_future(receive if cursor is not None else asyncio.wait_for(receive, 120))
                packet = await receive_task
                receive_task = None
                if packet.get("type") == "websocket.disconnect":
                    break
                binary = packet.get("bytes")
                if binary is not None:
                    if cursor is not None:
                        raise ValueError("File sessions accept only playback-gated server audio")
                    if not binary or len(binary) % 2 or len(binary) > round(options.chunk_seconds * 16000) * 2:
                        raise ValueError("Invalid PCM16 chunk byte length")
                    pcm = np.frombuffer(binary, dtype="<i2").astype(np.float32) / 32768.0
                    call = lambda: session.step(pcm)
                else:
                    event = json.loads(packet.get("text", ""))
                    if "chunk_ms" in event:
                        raise ValueError("chunk_ms can only be selected when starting a session")
                    if any(key in event for key in window_keys):
                        raise ValueError("窗口长度和丢弃长度只能在开始转录时设置。")
                    if any(key in event for key in range_keys):
                        raise ValueError("The file range can only be selected when starting a session")
                    if event.get("type") == "end":
                        if cursor is not None and not cursor.finished:
                            raise ValueError("Drain the played file audio before ending")
                        if cursor is not None and session.turn_count == 0:
                            await websocket.send_json({"type": "file_end_empty", "heard_seconds": 0,
                                                       "source_heard_seconds": source_heard})
                            break
                        attempted_source_end = source_heard
                        call = lambda: session.step(end_of_audio=True)
                    elif event.get("type") == "file_next":
                        if cursor is None:
                            raise ValueError("file_next requires an uploaded file session")
                        # File I/O and scipy resampling should not block other
                        # sockets. Wait for this worker before closing its file.
                        task = asyncio.create_task(asyncio.to_thread(cursor.next_chunk,
                            event.get("release_seconds"), options.chunk_seconds,
                            final=event.get("final", False)))
                        raw = await asyncio.shield(task)
                        task = None
                        if raw is None:
                            await websocket.send_json({"type": "file_done" if cursor.finished else "file_wait"})
                            continue
                        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                        attempted_source_end = cursor.position / upload.sample_rate
                        call = lambda: session.step(pcm, final_chunk=cursor.finished)
                    elif event.get("type") == "control":
                        try:
                            op = event.get("op")
                            if op != "commit" or options.context_policy == "window":
                                raise ValueError(f"Unsupported control for third: {op}")
                            session.verify_commit(event["through_seconds"], verified=event.get("verified", False))
                            snapshot = {"segments": [], "speakers": [], "aliases": {}, "reference_evidence": {}}
                        except Exception as exc:
                            await websocket.send_json({"type": "control_error", "error": str(exc)})
                        else:
                            await websocket.send_json({"type": "control_ack", "op": event["op"], "snapshot": snapshot})
                        continue
                    else:
                        raise ValueError("Unknown message type")
                submitted = time.perf_counter()
                def run():
                    begin = time.perf_counter()
                    value = call()
                    value["queue_wait_seconds"] = begin - submitted
                    return value
                task = asyncio.get_running_loop().run_in_executor(executor, run)
                # Cancellation must not free a session while its worker runs.
                result = await asyncio.shield(task)
                task = None
                if cursor is not None:
                    # Track the decoded source frames separately. The model sees
                    # session-relative 16 kHz packets; a short tail may round differently.
                    annotate_file_result(result, cursor, source_heard, attempted_source_end)
                    source_heard = attempted_source_end
                    attempted_source_end = None
                result["new_segments"] = []
                result["service_seconds"] = time.perf_counter() - submitted
                if cursor is not None:
                    result["file_complete"] = cursor.finished
                await websocket.send_json(dict(result, type="result"))
                if result["event"] == "end_of_audio":
                    break
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            try:
                failure = {"type": "error", "error": str(exc),
                    "heard_seconds": session.heard_seconds if session else 0,
                    "audio_dropped_silently": False}
                if source_heard is not None:
                    failure["source_heard_seconds"] = source_heard
                    if attempted_source_end is not None:
                        failure.update(source_start_seconds=source_heard, source_end_seconds=attempted_source_end)
                await websocket.send_json(failure)
            except Exception:
                pass
        finally:
            if receive_task is not None:
                receive_task.cancel()
                await asyncio.gather(receive_task, return_exceptions=True)
            if task is not None:
                try:
                    await asyncio.shield(task)
                except Exception:
                    pass
            if session:
                session.close()
            if upload is not None:
                media_store.discard(upload.media_id)
            if admitted:
                active.discard(session)
            try:
                await websocket.close()
            except Exception:
                pass

    stream.__annotations__["websocket"] = WebSocket
    app.websocket("/stream")(stream)
    return app


def add_web_arguments(parser):
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--max-sessions", type=int, default=1)
    parser.add_argument("--ssl-certfile")
    parser.add_argument("--ssl-keyfile")
    parser.add_argument("--https-host", action="append", default=[], metavar="IP_OR_NAME",
                        help="Issue/reuse a local HTTPS certificate for this browser-facing IP or name (repeatable)")
    parser.add_argument("--cert-dir", type=Path, help="Persist local HTTPS certificates here (default: recipe certs/)")


def serve_web(parser, args, backend_factory, options, app_factory):
    from .web_tls import TLSConfigError, meeting_tls
    try:
        certfile, keyfile = meeting_tls(args, Path(__file__).resolve().parents[1])
    except (OSError, TLSConfigError) as exc:
        parser.error(str(exc))
    import uvicorn
    app = app_factory(backend_factory, options, max_sessions=args.max_sessions)
    max_ms = round(options.chunk_seconds * 1000)
    uvicorn.run(app, host=args.host, port=args.port, workers=1,
                ws_max_size=max(65536, max_ms * 32),  # mono PCM16 at 16 kHz
                ws_max_queue=4, ssl_certfile=certfile, ssl_keyfile=keyfile)
