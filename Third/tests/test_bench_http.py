"""The load client counts transport acknowledgements separately from model results."""
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import runpy
import struct
import threading
import wave

import pytest


bench = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools" / "bench_http.py"))


@pytest.mark.parametrize("failure", ["", "first_429", "first_502", "later_429"])
def test_real_http_meetings_and_slo_counts(tmp_path, capsys, failure):
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
                assert self.path.endswith("/result")
                polls.append(self.path)
                self.send_json(200, dict(status="ok", action="WAIT", text=""))

        def do_POST(self):
            _, _, _, meeting, op = self.path.split("/")
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if op == "end":
                ended.append(meeting)
                if failure in ("first_502", "later_429"):
                    self.send_json(404, dict(error="unknown meeting"))
                else:
                    self.send_json(200, dict(status="ended"))
                return
            assert op == "chunks" and self.headers.get("Content-Type") == "application/octet-stream"
            blocks[meeting].append(body)
            n = len(blocks[meeting])
            if failure == "first_429" or (failure == "later_429" and n == 2):
                self.send_json(429, dict(error="capacity"))
            elif failure == "first_502":
                self.send_json(502, dict(error="gateway"))
            elif n == 1:
                self.send_json(202, dict(status="processing"))
            elif n == 2:
                self.send_json(200, dict(status="ok", action="DIRECT", text="Hi."))
            else:
                self.send_json(200, dict(status="buffered"))

    wav = tmp_path / "distinct.wav"
    with wave.open(str(wav), "wb") as f:
        f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        f.writeframes(b"".join(struct.pack("<h", n) * 160 for n in range(9)))
    server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        code = bench["main"]([str(wav), "--url", f"http://127.0.0.1:{server.server_port}",
                              "--concurrency", "1", "2", "--seconds", ".03"])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
    header, *tiers = [json.loads(s) for s in capsys.readouterr().out.splitlines()]
    assert header["chunk_ms"] == 10
    assert len(ended) == len(set(ended)) == (0 if failure == "first_429" else 3)
    assert len({data[0] for data in blocks.values()}) == 3
    for n, summary in enumerate(tiers, 1):
        assert summary["planned_chunks"] == n * 3
        if failure:
            assert code == 1
            assert summary["http_4xx" if failure.endswith("429") else "http_5xx"] == n
            assert bool(summary["cleanup_errors"]) == (failure == "later_429")
            assert summary["unsent"] == (n if failure == "later_429" else 2 * n)
            assert summary["timely_percent_of_planned"] == 0
        else:
            assert code == 0
            assert summary["first_post_202"] == summary["wait"] == summary["buffered"] == n
            assert summary["first_post_with_model_result_within_slo"] == summary["direct"] == n
            assert summary["unsent"] == 0
            assert summary["planned_without_timely_model_result"] == 2 * n
            assert summary["audio_ready_to_model_result_ms"]["p99"] >= summary["final_model_result_ms"]["p99"]
    assert len(polls) == (0 if failure.startswith("first") else 3)
