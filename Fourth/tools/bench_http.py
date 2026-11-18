"""Measure real-time, multi-meeting ASR responses from the caller's side.

All benchmark meetings use distinct, equal-duration excerpts of a 16 kHz mono
PCM16 WAV, including across concurrency tiers. A meeting keeps absolute packet
ready times even if its previous response is late.
"""
import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http.client import HTTPConnection, HTTPException, HTTPSConnection
import json
import math
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
from uuid import uuid4
import wave


RATE = 16000


class HTTPStatusError(ValueError):
    def __init__(self, status, data):
        super().__init__(f"HTTP {status}: {data[:300].decode('utf-8', 'replace')}")
        self.status = status


class Client:
    def __init__(self, url, timeout):
        address = urlsplit(url)
        if address.scheme not in ("http", "https") or not address.hostname or address.query or address.fragment:
            raise ValueError("--url needs an HTTP(S) service base URL")
        cls = HTTPSConnection if address.scheme == "https" else HTTPConnection
        self.connection = cls(address.hostname, address.port, timeout=timeout)
        self.prefix = address.path.rstrip("/")

    def request(self, method, path, body=None, *, content_type="application/octet-stream"):
        headers = {"Content-Type": content_type} if body is not None else {}
        started = time.perf_counter()
        self.connection.request(method, self.prefix + path, body, headers=headers)
        response = self.connection.getresponse()
        data = response.read()
        arrived = time.perf_counter()
        if response.status != 200:
            raise HTTPStatusError(response.status, data)
        result = json.loads(data)
        if not isinstance(result, dict):
            raise ValueError("Service returned a non-object JSON response")
        return result, started, arrived

    def post(self, path, body=None, *, content_type="application/octet-stream"):
        result, started, arrived = self.request("POST", path, body, content_type=content_type)
        if set(result) != {"src"} or not isinstance(result["src"], str):
            raise ValueError("Expected a 200 JSON response with a single string field 'src'")
        return result["src"], started, arrived

    def close(self):
        self.connection.close()


@dataclass
class MeetingStats:
    sent: int = 0
    timely: int = 0
    nonempty_src: int = 0
    empty_src: int = 0
    post_ms: list = field(default_factory=list)
    from_ready_ms: list = field(default_factory=list)
    error: str = ""
    cleanup_error: str = ""


def stream_meeting(url, audio, first, frames, step, meeting, gate, slo_ms, timeout, wire_format):
    stats = MeetingStats()
    client = Client(url, timeout)
    path = "/v1/meetings/" + meeting
    admitted = possibly_admitted = False
    try:
        if wire_format == "json-f32":
            import numpy as np
        with wave.open(str(audio), "rb") as source:
            source.setpos(first)
            gate.wait(timeout=timeout)
            start = time.perf_counter()
            for offset in range(0, frames, step):
                count = min(step, frames - offset)
                body = source.readframes(count)
                if len(body) != count * 2:
                    raise ValueError("WAV contains fewer samples than its header declares")
                ready = start + (offset + count) / RATE
                time.sleep(max(0., ready - time.perf_counter()))
                content_type = "application/octet-stream"
                if wire_format == "json-f32":
                    waveform = np.frombuffer(body, dtype="<i2").astype(np.float32) / 32768.0
                    data = base64.b64encode(waveform.astype("<f4", copy=False).tobytes()).decode("ascii")
                    body = json.dumps({"data": data}).encode("utf-8")
                    content_type = "application/json"
                stats.sent += 1
                possibly_admitted = True
                try:
                    src, started, arrived = client.post(path + "/chunks", body, content_type=content_type)
                except HTTPStatusError as exc:
                    if 400 <= exc.status < 500:
                        possibly_admitted = admitted
                    raise
                admitted = True
                if src:
                    stats.nonempty_src += 1
                else:
                    stats.empty_src += 1
                elapsed = (arrived - started) * 1000
                stats.post_ms.append(elapsed)
                stats.from_ready_ms.append((arrived - ready) * 1000)
                stats.timely += int(elapsed <= slo_ms)
    except (ValueError, HTTPException, OSError, ImportError, threading.BrokenBarrierError, wave.Error) as exc:
        stats.error = str(exc) or type(exc).__name__
        # Each meeting sends in order. Ambiguous failures must not replay audio.
        if not stats.sent:
            gate.abort()
    finally:
        if possibly_admitted:
            try:
                if stats.error:
                    client.close()  # A failed or timed-out read can leave an unusable socket.
                client.post(path + "/end")
            except HTTPStatusError as exc:
                if admitted or exc.status != 404:
                    stats.cleanup_error = str(exc)
            except (ValueError, HTTPException, OSError) as exc:
                stats.cleanup_error = str(exc)
        client.close()
    return stats


def percentiles(values):
    values = sorted(values)
    return {f"p{q}": round(values[math.ceil(q / 100 * len(values)) - 1], 2) if values else None
            for q in (50, 95, 99)}


def report(meetings, chunks, slo_ms):
    planned = chunks * len(meetings)
    sent = sum(m.sent for m in meetings)
    timely = sum(m.timely for m in meetings)
    due = [v for m in meetings for v in m.from_ready_ms]
    errors = Counter(m.error for m in meetings if m.error)
    return dict(concurrency=len(meetings), planned_chunks=planned, sent=sent, unsent=planned - sent,
                nonempty_src=sum(m.nonempty_src for m in meetings), empty_src=sum(m.empty_src for m in meetings),
                responses_within_slo=timely,
                timely_percent_of_planned=round(100 * timely / planned, 2),
                late_from_audio_ready=sum(v > slo_ms for v in due),
                post_latency_ms=percentiles(v for m in meetings for v in m.post_ms),
                audio_ready_to_result_ms=percentiles(due),
                http_4xx=sum(n for e, n in errors.items() if e.startswith("HTTP 4")),
                http_5xx=sum(n for e, n in errors.items() if e.startswith("HTTP 5")),
                errors=dict(errors), cleanup_errors=dict(Counter(m.cleanup_error for m in meetings if m.cleanup_error)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Local uncompressed 16 kHz mono 16-bit WAV")
    parser.add_argument("--url", default="http://127.0.0.1:18892")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--chunk-ms", type=int, help="Defaults to the model step reported by /health")
    parser.add_argument("--wire-format", choices=("pcm16", "json-f32"), default="pcm16",
                        help="Send raw PCM16 or a JSON object with base64 float32 samples (requires NumPy)")
    parser.add_argument("--seconds", type=float, help="Audio seconds per meeting; default up to 30 s")
    parser.add_argument("--slo-ms", type=float, default=1000)
    parser.add_argument("--timeout", type=float, default=15., help="HTTP socket timeout in seconds")
    args = parser.parse_args(argv)
    if min(args.concurrency) < 1 or args.chunk_ms is not None and args.chunk_ms < 1:
        parser.error("--concurrency and --chunk-ms must be positive")
    if not all(math.isfinite(x) and x > 0 for x in (args.timeout, args.slo_ms)) or args.timeout <= args.slo_ms / 1000:
        parser.error("--timeout must exceed the SLO; both must be positive and finite")
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be positive and finite")
    try:
        probe = Client(args.url, args.timeout)
        try:
            health, _, _ = probe.request("GET", "/health")
        finally:
            probe.close()
        if health.get("service") != "third_http" or not health.get("ready"):
            raise ValueError("URL must point to a ready Fourth HTTP ASR service")
        model_chunk_ms = round(health["chunk_seconds"] * 1000)
        chunk_ms = args.chunk_ms or model_chunk_ms
        with wave.open(str(args.audio), "rb") as f:
            if (f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getcomptype()) != (RATE, 1, 2, "NONE"):
                raise ValueError("Use a 16 kHz mono uncompressed PCM16 WAV")
            available = f.getnframes()
        total_meetings = sum(args.concurrency)
        capacity = available // total_meetings
        frames = round(args.seconds * RATE) if args.seconds is not None else min(30 * RATE, capacity)
        if not 0 < frames <= capacity:
            raise ValueError(f"Disjoint clips need {frames * total_meetings / RATE:.3f} s; "
                             f"WAV has {available / RATE:.3f} s. Reduce --seconds or --concurrency.")
    except (ValueError, HTTPException, OSError, KeyError, wave.Error, EOFError) as exc:
        parser.error(str(exc))

    print(json.dumps(dict(url=args.url, wav=str(args.audio), meeting_seconds=frames / RATE, chunk_ms=chunk_ms,
                          model_chunk_ms=model_chunk_ms, wire_format=args.wire_format,
                          slo_ms=args.slo_ms, warmup="none")), flush=True)
    step = RATE * chunk_ms // 1000
    run = uuid4().hex
    failed = False
    first_meeting = 0
    for tier, size in enumerate(args.concurrency):
        gate = threading.Barrier(size)
        with ThreadPoolExecutor(max_workers=size) as pool:
            futures = [pool.submit(stream_meeting, args.url, args.audio, (first_meeting + n) * frames, frames, step,
                                   f"bench-{run}-{tier}-{n}", gate, args.slo_ms, args.timeout, args.wire_format)
                       for n in range(size)]
            row = report([f.result() for f in futures], math.ceil(frames / step), args.slo_ms)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        failed |= bool(row["errors"] or row["cleanup_errors"])
        first_meeting += size
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
