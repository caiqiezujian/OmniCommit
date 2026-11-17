"""Exercise the synchronous public transcription contract on actual HTTP sockets."""
from collections import defaultdict
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import runpy
import struct
import threading
import wave

import pytest


bench = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools" / "bench_http.py"))


@pytest.mark.parametrize(("failure", "wire_format"), [
    ("", "pcm16"), ("", "json-f32"), ("first_429", "pcm16"), ("first_502", "pcm16"),
    ("later_429", "pcm16"), ("unexpected_202", "pcm16"), ("missing_src", "pcm16"),
    ("bad_end", "pcm16")])
def test_real_http_meetings_and_slo_counts(tmp_path, capsys, failure, wire_format):
    blocks = defaultdict(list)
    ended = []
    polls = []

    class Stub(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, code, body):
            encoded = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if self.path == "/health":
                self.send_json(200, dict(ready=True, service="third_http", chunk_seconds=.01))
            else:
                polls.append(self.path)
                self.send_json(404, dict(error="unexpected path"))

        def do_POST(self):
            _, _, _, meeting, op = self.path.split("/")
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if op == "end":
                ended.append(meeting)
                if failure in ("first_502", "later_429"):
                    self.send_json(404, dict(error="unknown meeting"))
                elif failure == "bad_end":
                    self.send_json(200, dict(src=None))
                else:
                    self.send_json(200, dict(src=""))
                return
            assert op == "chunks"
            if wire_format == "json-f32":
                assert self.headers["Content-Type"] == "application/json"
                body = base64.b64decode(json.loads(body)["data"], validate=True)
                samples = struct.unpack("<160f", body)
                assert all(s == samples[0] for s in samples) and samples[0] * 32768 in range(9)
            else:
                assert self.headers["Content-Type"] == "application/octet-stream"
            blocks[meeting].append(body)
            n = len(blocks[meeting])
            if failure == "first_429" or (failure == "later_429" and n == 2):
                self.send_json(429, dict(error="capacity"))
            elif failure == "first_502":
                self.send_json(502, dict(error="gateway"))
            elif failure == "unexpected_202":
                self.send_json(202, dict(status="processing"))
            elif failure == "missing_src":
                self.send_json(200, dict(text="Hi."))
            elif n == 2:
                self.send_json(200, dict(src=" Hi."))
            else:
                self.send_json(200, dict(src=""))

    wav = tmp_path / "distinct.wav"
    with wave.open(str(wav), "wb") as f:
        f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        f.writeframes(b"".join(struct.pack("<h", n) * 160 for n in range(9)))
    server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        code = bench["main"]([str(wav), "--url", f"http://127.0.0.1:{server.server_port}",
                              "--concurrency", "1", "2", "--seconds", ".03", "--wire-format", wire_format])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
    header, *tiers = [json.loads(s) for s in capsys.readouterr().out.splitlines()]
    assert header["chunk_ms"] == 10
    assert header["wire_format"] == wire_format
    assert len(ended) == len(set(ended)) == (0 if failure == "first_429" else 3)
    assert len({data[0] for data in blocks.values()}) == 3
    for n, summary in enumerate(tiers, 1):
        assert summary["planned_chunks"] == n * 3
        if failure in ("first_429", "first_502", "later_429"):
            assert code == 1
            assert summary["http_4xx" if failure.endswith("429") else "http_5xx"] == n
            assert bool(summary["cleanup_errors"]) == (failure == "later_429")
            assert summary["unsent"] == (n if failure == "later_429" else 2 * n)
            assert summary["responses_within_slo"] == (n if failure == "later_429" else 0)
        elif failure in ("unexpected_202", "missing_src"):
            assert code == 1 and summary["unsent"] == n * 2
            assert sum(summary["errors"].values()) == n and summary["responses_within_slo"] == 0
        else:
            assert code == (1 if failure == "bad_end" else 0)
            assert bool(summary["cleanup_errors"]) == (failure == "bad_end")
            assert (summary["empty_src"], summary["nonempty_src"]) == (n * 2, n)
            assert summary["responses_within_slo"] == n * 3 and summary["timely_percent_of_planned"] == 100
            assert summary["unsent"] == 0 and summary["late_from_audio_ready"] == 0
            assert summary["audio_ready_to_result_ms"]["p99"] >= summary["post_latency_ms"]["p99"]
    assert not polls
