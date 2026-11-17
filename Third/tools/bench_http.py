"""Measure real-time, multi-meeting ASR responses from the caller's side.

All benchmark meetings use distinct, equal-duration excerpts of a 16 kHz mono
PCM16 WAV, including across concurrency tiers. A meeting keeps absolute packet
ready times even if its previous response is late.
"""
import argparse
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
        self.timeout = timeout

    def request(self, method, path, body=None):
        headers = {"Content-Type": "application/octet-stream"} if body is not None else {}
        self.connection.request(method, self.prefix + path, body, headers=headers)
        sent = time.perf_counter()  # Body written to the local socket.
        response = self.connection.getresponse()
        data = response.read()
        arrived = time.perf_counter()
        if response.status >= 400:
            raise HTTPStatusError(response.status, data)
        result = json.loads(data)
        if not isinstance(result, dict):
            raise ValueError("Service returned a non-object JSON response")
        return response.status, result, sent, arrived

    def collect(self, path, status, result, arrived):
        deadline = arrived + self.timeout
        while status == 202:
            if time.perf_counter() >= deadline:
                raise TimeoutError("Delayed model result polling exceeded --timeout")
            time.sleep(.02)
            status, result, _, arrived = self.request("GET", path + "/result")
        if status != 200:
            raise ValueError(f"Unexpected final HTTP {status}")
        return result, arrived

    def close(self):
        self.connection.close()


@dataclass
class MeetingStats:
    sent: int = 0
    delayed: int = 0
    timely: int = 0
    actions: Counter = field(default_factory=Counter)
    first_reply_ms: list = field(default_factory=list)
    result_ms: list = field(default_factory=list)
    from_ready_ms: list = field(default_factory=list)
    error: str = ""
    cleanup_error: str = ""


def stream_meeting(url, audio, first, frames, step, meeting, gate, slo_ms, timeout):
    stats = MeetingStats()
    client = Client(url, timeout)
    path = "/v1/meetings/" + meeting
    admitted = possibly_admitted = False
    try:
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
                stats.sent += 1
                possibly_admitted = True
                try:
                    status, result, sent, arrived = client.request("POST", path + "/chunks", body)
                except HTTPStatusError as exc:
                    if 400 <= exc.status < 500:
                        possibly_admitted = admitted
                    raise
                admitted = True
                stats.first_reply_ms.append((arrived - sent) * 1000)
                delayed = status == 202
                stats.delayed += int(delayed)
                result, arrived = client.collect(path, status, result, arrived)
                if result.get("status") == "buffered":
                    stats.actions["buffered"] += 1
                    continue
                action = result.get("action")
                if action not in ("DIRECT", "WAIT"):
                    raise ValueError(f"Missing model action: {result.get('status')!r}")
                stats.actions[action] += 1
                elapsed = (arrived - sent) * 1000
                stats.result_ms.append(elapsed)
                stats.from_ready_ms.append((arrived - ready) * 1000)
                stats.timely += int(not delayed and elapsed <= slo_ms)
    except (ValueError, HTTPException, OSError, threading.BrokenBarrierError, wave.Error) as exc:
        stats.error = str(exc) or type(exc).__name__
        # Each meeting sends in order. Ambiguous failures must not replay audio.
        if not stats.sent:
            gate.abort()
    finally:
        if possibly_admitted:
            try:
                if stats.error:
                    client.close()  # A failed or timed-out read can leave an unusable socket.
                status, result, _, arrived = client.request("POST", path + "/end")
                client.collect(path, status, result, arrived)
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
    actions = sum((m.actions for m in meetings), Counter())
    due = [v for m in meetings for v in m.from_ready_ms]
    errors = Counter(m.error for m in meetings if m.error)
    return dict(concurrency=len(meetings), planned_chunks=planned, sent=sent, unsent=planned - sent,
                first_post_202=sum(m.delayed for m in meetings),
                direct=actions["DIRECT"], wait=actions["WAIT"], buffered=actions["buffered"],
                first_post_with_model_result_within_slo=timely,
                planned_without_timely_model_result=planned - timely,
                timely_percent_of_planned=round(100 * timely / planned, 2),
                model_results_late_from_audio_ready=sum(v > slo_ms for v in due),
                first_http_reply_ms=percentiles(v for m in meetings for v in m.first_reply_ms),
                final_model_result_ms=percentiles(v for m in meetings for v in m.result_ms),
                audio_ready_to_model_result_ms=percentiles(due),
                http_4xx=sum(n for e, n in errors.items() if e.startswith("HTTP 4")),
                http_5xx=sum(n for e, n in errors.items() if e.startswith("HTTP 5")),
                errors=dict(errors), cleanup_errors=dict(Counter(m.cleanup_error for m in meetings if m.cleanup_error)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Local uncompressed 16 kHz mono 16-bit WAV")
    parser.add_argument("--url", default="http://127.0.0.1:18892")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--chunk-ms", type=int, help="Defaults to the model step reported by /health")
    parser.add_argument("--seconds", type=float, help="Audio seconds per meeting; default up to 30 s")
    parser.add_argument("--slo-ms", type=float, default=1000)
    parser.add_argument("--timeout", type=float, default=15., help="Socket and delayed-result polling timeout in seconds")
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
            _, health, _, _ = probe.request("GET", "/health")
        finally:
            probe.close()
        if health.get("service") != "third_http" or not health.get("ready"):
            raise ValueError("URL must point to a ready Third HTTP ASR service")
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
                          model_chunk_ms=model_chunk_ms, slo_ms=args.slo_ms, warmup="none")), flush=True)
    step = RATE * chunk_ms // 1000
    run = uuid4().hex
    failed = False
    first_meeting = 0
    for tier, size in enumerate(args.concurrency):
        gate = threading.Barrier(size)
        with ThreadPoolExecutor(max_workers=size) as pool:
            futures = [pool.submit(stream_meeting, args.url, args.audio, (first_meeting + n) * frames, frames, step,
                                   f"bench-{run}-{tier}-{n}", gate, args.slo_ms, args.timeout)
                       for n in range(size)]
            row = report([f.result() for f in futures], math.ceil(frames / step), args.slo_ms)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        failed |= bool(row["errors"] or row["cleanup_errors"])
        first_meeting += size
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
