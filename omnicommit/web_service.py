"""One serialized model worker; bounded admission and per-session state."""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import hmac
import json
import os
from pathlib import Path
import time
from urllib.parse import urlparse

import numpy as np

from .stream_cli import add_runtime_arguments, build_backend, runtime_options, apply_control
from .streaming import StreamSession


def create_app(backend_factory, options, *, max_sessions=1, access_token=None):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse
    if max_sessions < 1:
        raise ValueError("max_sessions must be positive")
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="omnicommit-model")
    active = set()
    backend = None

    @asynccontextmanager
    async def lifespan(app):
        nonlocal backend
        try:
            backend = await asyncio.get_running_loop().run_in_executor(executor, backend_factory)
            yield
        finally:
            # In-flight generation finishes before shutdown; no model racing.
            executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    async def index():
        return FileResponse(Path(__file__).resolve().parent / "web" / "index.html")

    @app.get("/health")
    async def health():
        return {"ready": backend is not None, "active_sessions": len(active),
                "max_sessions": max_sessions, "chunk_seconds": options.chunk_seconds,
                "context_policy": options.context_policy,
                "backend": type(backend).__name__ if backend else None,
                "native_prefix_caching": type(backend).__name__ == "VLLMBackend",
                "native_mm_processor_cache": type(backend).__name__ == "VLLMBackend",
                "manual_kv_reuse": False}

    # Avoid postponed annotation resolving a function-local WebSocket import.
    async def stream(websocket):
        origin = websocket.headers.get("origin")
        if origin and urlparse(origin).netloc != websocket.headers.get("host"):
            await websocket.close(code=1008, reason="Cross-origin connection denied")
            return
        await websocket.accept()
        session = None
        task = None
        admitted = False
        try:
            init = await asyncio.wait_for(websocket.receive_json(), timeout=10)
            if access_token and not hmac.compare_digest(str(init.get("token", "")), access_token):
                raise ValueError("Authentication failed")
            if init.get("type") != "start" or init.get("sample_rate") != 16000:
                raise ValueError("Start requires sample_rate=16000; send mono PCM16 little-endian")
            if len(active) >= max_sessions:
                raise ValueError("Service session capacity reached; no audio accepted")
            session = StreamSession(backend, options)
            active.add(session)
            admitted = True
            await websocket.send_json({"type": "ready", "chunk_seconds": options.chunk_seconds,
                "context_policy": options.context_policy, "backend": type(backend).__name__,
                "warning": "Rolling mode is an unverified tail heuristic" if options.context_policy == "rolling" else "Verified watermarks required before safe context eviction"})
            while True:
                packet = await asyncio.wait_for(websocket.receive(), timeout=120)
                if packet.get("type") == "websocket.disconnect":
                    break
                binary = packet.get("bytes")
                if binary is not None:
                    if not binary or len(binary) % 2 or len(binary) > round(options.chunk_seconds * 16000) * 2:
                        raise ValueError("Invalid PCM16 chunk byte length")
                    pcm = np.frombuffer(binary, dtype="<i2").astype(np.float32) / 32768.0
                    call = lambda: session.step(pcm)
                else:
                    event = json.loads(packet.get("text", ""))
                    if event.get("type") == "end":
                        call = lambda: session.step(end_of_audio=True)
                    elif event.get("type") == "control":
                        try:
                            apply_control(session, event)
                        except Exception as exc:
                            await websocket.send_json({"type": "control_error", "error": str(exc)})
                        else:
                            await websocket.send_json({"type": "control_ack", "op": event["op"]})
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
            if admitted:
                active.discard(session)
            try:
                await websocket.close()
            except Exception:
                pass

    stream.__annotations__["websocket"] = WebSocket
    app.websocket("/stream")(stream)
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description="OmniCommit PCM streaming Web service")
    add_runtime_arguments(parser)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--max-sessions", type=int, default=1)
    parser.add_argument("--token-env", default="OMNICOMMIT_WEB_TOKEN")
    parser.add_argument("--ssl-certfile")
    parser.add_argument("--ssl-keyfile")
    args = parser.parse_args(argv)
    token = os.environ.get(args.token_env)
    if args.host not in ("127.0.0.1", "localhost", "::1") and not token:
        parser.error("Remote binding requires an access token in --token-env")
    if bool(args.ssl_certfile) != bool(args.ssl_keyfile):
        parser.error("TLS needs both certificate and private key")
    import uvicorn
    app = create_app(lambda: build_backend(args), runtime_options(args),
                     max_sessions=args.max_sessions, access_token=token)
    uvicorn.run(app, host=args.host, port=args.port, workers=1,
                ws_max_size=65536, ws_max_queue=4, ssl_certfile=args.ssl_certfile,
                ssl_keyfile=args.ssl_keyfile)


if __name__ == "__main__":
    main()
