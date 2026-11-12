"""One serialized model worker; bounded admission and per-session state."""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import replace
import json
import logging
from pathlib import Path
import shutil
import time
from urllib.parse import urlparse

import numpy as np

from .stream_cli import add_runtime_arguments, build_backend, runtime_options, apply_control
from .streaming import StreamSession
from .media import MediaCursor, MediaStore
from .runtime_protocol import LIVE_CHUNK_MS, LIVE_CHUNK_SECONDS
from .session_identity import SessionLedger

LOG = logging.getLogger("omnicommit.web")


def create_app(backend_factory, options, *, max_sessions=1,
               auto_reference_factory=None, session_factory=StreamSession,
               media_cursor_factory=MediaCursor,
               protocol_name="first", chunk_ms_choices=LIVE_CHUNK_MS):
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    if max_sessions < 1:
        raise ValueError("max_sessions must be positive")
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="omnicommit-model")
    reference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="omnicommit-reference")
    active = set()
    backend = None
    reference_provider = None
    chunk_settings = {"chunk_ms_choices": chunk_ms_choices} if chunk_ms_choices else {}
    media_store = MediaStore(max_slots=max(2, max_sessions * 2))

    @asynccontextmanager
    async def lifespan(app):
        nonlocal backend, reference_provider
        async def expire_uploads():
            while True:
                await asyncio.sleep(60)
                media_store.expire()
        expiry_task = None
        try:
            backend = await asyncio.get_running_loop().run_in_executor(executor, backend_factory)
            if auto_reference_factory is not None:
                reference_provider = auto_reference_factory()
            expiry_task = asyncio.create_task(expire_uploads())
            yield
        finally:
            if expiry_task is not None:
                expiry_task.cancel()
                await asyncio.gather(expiry_task, return_exceptions=True)
            # In-flight generation finishes before shutdown; no model racing.
            executor.shutdown(wait=True, cancel_futures=True)
            reference_executor.shutdown(wait=True, cancel_futures=True)
            reference_model = getattr(reference_provider, "model", None)
            if hasattr(reference_model, "close"):
                reference_model.close()
            media_store.close()

    app = FastAPI(lifespan=lifespan)
    app.state.media_store = media_store
    web_directory = Path(__file__).resolve().parent / "web"
    app.mount("/web", StaticFiles(directory=web_directory), name="web")

    @app.get("/")
    async def index():
        return FileResponse(web_directory / "index.html")

    @app.get("/health")
    async def health():
        return {"ready": backend is not None, "protocol": protocol_name, "active_sessions": len(active),
                "max_sessions": max_sessions, "chunk_seconds": options.chunk_seconds, **chunk_settings,
                "context_policy": options.context_policy,
                "backend": type(backend).__name__ if backend else None,
                "native_prefix_caching": type(backend).__name__ == "VLLMBackend",
                "native_mm_processor_cache": type(backend).__name__ == "VLLMBackend",
                "manual_kv_reuse": False, "file_input": bool(shutil.which("ffmpeg")),
                "automatic_reference": "experimental_qwen" if reference_provider is not None else "disabled"}

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
        ledger = None
        cursor = None
        upload = None
        auto_reference_failed = False
        reference_task = None
        reference_started = None
        pending_evidence = None
        pending_reference_seconds = None
        task = None
        admitted = False

        def collect_reference():
            nonlocal reference_task, auto_reference_failed, pending_evidence, pending_reference_seconds
            if reference_task is None or not reference_task.done():
                return
            try:
                evidence = reference_provider.apply(ledger, reference_task.result())
            except Exception as exc:
                evidence = {"status": "error", "verified": False, "error": str(exc)}
            pending_reference_seconds = time.perf_counter() - reference_started
            reference_task = None
            if evidence is not None:
                pending_evidence = evidence
                if evidence.get("status") == "error":
                    auto_reference_failed = True
                    LOG.warning("Automatic reference disabled for session: %s", evidence.get("error"))
        try:
            init = await asyncio.wait_for(websocket.receive_json(), timeout=10)
            if init.get("type") != "start" or init.get("sample_rate") != 16000:
                raise ValueError("Start requires sample_rate=16000; send mono PCM16 little-endian")
            if len(active) >= max_sessions:
                raise ValueError("Service session capacity reached; no audio accepted")
            source_kind = init.get("source_kind", "microphone")
            if source_kind not in ("microphone", "system", "file"):
                raise ValueError("Unknown audio source")
            session_options = options
            if chunk_ms_choices:
                chunk_ms = init.get("chunk_ms", round(options.chunk_seconds * 1000))
                if type(chunk_ms) is not int or chunk_ms not in chunk_ms_choices:
                    raise ValueError(f"chunk_ms must be one of {list(chunk_ms_choices)} (integer milliseconds)")
                session_options = replace(options, chunk_seconds=chunk_ms / 1000)
            elif "chunk_ms" in init:
                raise ValueError("This service uses a fixed chunk length")
            if source_kind == "file":
                upload = media_store.claim(init.get("media_id"))
                cursor = media_cursor_factory(upload)
            session = session_factory(backend, session_options)
            ledger = SessionLedger(session)
            active.add(session)
            admitted = True
            await websocket.send_json({"type": "ready", "protocol": protocol_name,
                "chunk_seconds": session_options.chunk_seconds, **chunk_settings,
                "context_policy": session_options.context_policy, "backend": type(backend).__name__,
                "warning": "Rolling mode is an unverified tail heuristic" if session_options.context_policy == "rolling" else "Verified watermarks required before safe context eviction",
                "automatic_reference": "experimental_qwen" if reference_provider is not None else "disabled",
                "auto_reference": "experimental_qwen" if reference_provider is not None else "disabled"})
            while True:
                # A user may pause file playback for longer than two minutes.
                # Live capture keeps an inactivity limit; files wait for an
                # explicit resume/end or a transport disconnect.
                packet = await websocket.receive() if cursor is not None else await asyncio.wait_for(websocket.receive(), timeout=120)
                if packet.get("type") == "websocket.disconnect":
                    break
                # Apply a finished reference before building the next model prompt.
                # This is a serial point: no session.step is running for this session.
                collect_reference()
                binary = packet.get("bytes")
                if binary is not None:
                    if cursor is not None:
                        raise ValueError("File sessions accept only playback-gated server audio")
                    if not binary or len(binary) % 2 or len(binary) > round(session_options.chunk_seconds * 16000) * 2:
                        raise ValueError("Invalid PCM16 chunk byte length")
                    pcm = np.frombuffer(binary, dtype="<i2").astype(np.float32) / 32768.0
                    call = lambda: session.step(pcm)
                else:
                    event = json.loads(packet.get("text", ""))
                    if "chunk_ms" in event:
                        raise ValueError("chunk_ms can only be selected when starting a session")
                    if event.get("type") == "end":
                        if cursor is not None and not cursor.finished:
                            raise ValueError("Drain the played file audio before ending")
                        call = lambda: session.step(end_of_audio=True)
                    elif event.get("type") == "file_next":
                        if cursor is None:
                            raise ValueError("file_next requires an uploaded file session")
                        # File I/O and scipy resampling should not block other
                        # sockets. Wait for this worker before closing its file.
                        task = asyncio.create_task(asyncio.to_thread(cursor.next_chunk,
                            event.get("release_seconds"), session_options.chunk_seconds,
                            final=event.get("final", False)))
                        raw = await asyncio.shield(task)
                        task = None
                        if raw is None:
                            await websocket.send_json({"type": "file_done" if cursor.finished else "file_wait"})
                            continue
                        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                        call = lambda: session.step(pcm, final_chunk=cursor.finished)
                    elif event.get("type") == "control":
                        try:
                            op = event.get("op")
                            if op == "correct_segment":
                                snapshot = ledger.correct_segment(event["segment_id"], event["speaker"])
                                pending_evidence = None
                            elif op == "merge_speakers":
                                snapshot = ledger.merge_speakers(event["source"], event["target"])
                                pending_evidence = None
                            elif op == "rename_speaker":
                                snapshot = ledger.rename_speaker(event["speaker"], event["name"])
                            else:
                                if op == "reference" and ledger.resolve(event.get("speaker")) != event.get("speaker"):
                                    raise ValueError("Reference speaker was merged; use the canonical speaker ID")
                                apply_control(session, event)
                                if op == "reference":
                                    ledger.reference_evidence.pop(event.get("speaker"), None)
                                snapshot = ledger.snapshot()
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
                result["new_segments"] = ledger.record(result)
                collect_reference()
                if pending_evidence is not None:
                    result["automatic_reference"] = pending_evidence
                    pending_evidence = None
                if pending_reference_seconds is not None:
                    result["reference_seconds"] = pending_reference_seconds
                    pending_reference_seconds = None
                result["reference_speakers"] = list(session.references)
                if (reference_provider is not None and not auto_reference_failed
                        and reference_task is None and result["new_segments"] and result["event"] == "audio"):
                    try:
                        snapshot = reference_provider.prepare(ledger, result)
                        if snapshot is not None:
                            reference_started = time.perf_counter()
                            reference_task = asyncio.get_running_loop().run_in_executor(reference_executor,
                                                                                       reference_provider.evaluate, snapshot)
                    except Exception as exc:
                        auto_reference_failed = True
                        LOG.warning("Automatic reference preparation failed: %s", exc)
                        result["automatic_reference"] = {"status": "error", "verified": False, "error": str(exc)}
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
                await websocket.send_json({"type": "error", "error": str(exc),
                    "heard_seconds": session.heard_seconds if session else 0,
                    "audio_dropped_silently": False})
            except Exception:
                pass
        finally:
            if task is not None:
                try:
                    await asyncio.shield(task)
                except Exception:
                    pass
            if session:
                session.close()
            if reference_task is not None:
                reference_task.cancel()
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
    parser.add_argument("--auto-reference-model", type=Path,
                        help="Optional local Qwen3-ForcedAligner; creates unverified experimental references")
    parser.add_argument("--auto-reference-python", type=Path,
                        help="Python executable from the separate qwen-asr alignment environment")
    parser.add_argument("--auto-reference-device", default="cuda:0")
    parser.add_argument("--auto-reference-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--auto-reference-timeout", type=float, default=120,
                        help="Maximum seconds per alignment request, including cold model loading")


def serve_web(parser, args, backend_factory, options, app_factory):
    if bool(args.ssl_certfile) != bool(args.ssl_keyfile):
        parser.error("TLS needs both certificate and private key")
    reference_factory = None
    if args.auto_reference_model is not None:
        if not args.auto_reference_model.is_dir():
            parser.error("--auto-reference-model must be an existing local model directory")
        if args.auto_reference_python is None or not args.auto_reference_python.is_file():
            parser.error("--auto-reference-python must be an existing separate alignment Python executable")
        from .auto_reference import QwenAutoReference
        from .aligner_proxy import QwenAlignerProxy
        reference_factory = lambda: QwenAutoReference(str(args.auto_reference_model.resolve()),
            model=QwenAlignerProxy(args.auto_reference_python, args.auto_reference_model,
                                   device=args.auto_reference_device, dtype=args.auto_reference_dtype,
                                   timeout_seconds=args.auto_reference_timeout))
    elif args.auto_reference_python is not None:
        parser.error("--auto-reference-python requires --auto-reference-model")
    import uvicorn
    app = app_factory(backend_factory, options, max_sessions=args.max_sessions,
                      auto_reference_factory=reference_factory)
    uvicorn.run(app, host=args.host, port=args.port, workers=1,
                ws_max_size=max(65536, round(options.chunk_seconds * 16000) * 2),
                ws_max_queue=4, ssl_certfile=args.ssl_certfile,
                ssl_keyfile=args.ssl_keyfile)


def main(argv=None):
    parser = argparse.ArgumentParser(description="OmniCommit PCM streaming Web service")
    add_runtime_arguments(parser)
    add_web_arguments(parser)
    args = parser.parse_args(argv)
    serve_web(parser, args, lambda: build_backend(args, chunk_seconds_choices=LIVE_CHUNK_SECONDS),
              runtime_options(args), create_app)


if __name__ == "__main__":
    main()
