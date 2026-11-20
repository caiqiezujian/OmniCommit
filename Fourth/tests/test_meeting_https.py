"""Real local CA, HTTPS and WSS against the Third and Fourth replay servers."""
import asyncio
from contextlib import contextmanager
from http.client import HTTPSConnection
import importlib
import json
from pathlib import Path
import re
import shutil
import socket
import ssl
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
NEEDS_OPENSSL = pytest.mark.skipif(not shutil.which("openssl"), reason="local CA issuance requires OpenSSL")


@contextmanager
def replay_server(recipe, tmp, tls_args):
    replies = tmp / "replies.json"
    replies.write_text(json.dumps(["<|DIRECT|>hello", "<|WAIT|>"]), encoding="utf-8")
    executable = ROOT / recipe / "tools" / "serve_web.py"
    command = [sys.executable, "-u", str(executable), "--backend", "replay", "--no-recovery",
               "--host", "127.0.0.1", "--port", "0", "--chunk-ms", "1000", "--replies", str(replies),
               *map(str, tls_args)]
    log = tmp / "server.log"
    with log.open("wb") as output:
        server = subprocess.Popen(command, cwd=tmp, stdin=subprocess.DEVNULL, stdout=output,
                                  stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and server.poll() is None:
                match = re.search(r"Uvicorn running on https://127\.0\.0\.1:(\d+)", log.read_bytes().decode(errors="replace"))
                if match:
                    yield int(match[1])
                    return
                time.sleep(.05)
            pytest.fail(f"HTTPS server did not start:\n{log.read_bytes().decode(errors='replace')}")
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)


def tls_client(ca=None):
    # Isolated trust store: do not rely on or modify the workstation's root certificates.
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    if ca is not None:
        ctx.load_verify_locations(cafile=str(ca))
    return ctx


def peer_certificate(port, context, hostname):
    with socket.create_connection(("127.0.0.1", port), timeout=5) as tcp:
        with context.wrap_socket(tcp, server_hostname=hostname) as tls:
            return tls.getpeercert(binary_form=True)


async def send_computer_audio(ws_module, port, context, protocol):
    async with ws_module.connect(f"wss://127.0.0.1:{port}/stream", ssl=context,
                                 origin=f"https://127.0.0.1:{port}") as ws:
        await ws.send(json.dumps({"type": "start", "sample_rate": 16000, "source_kind": "system"}))
        ready = json.loads(await ws.recv())
        assert (ready["type"], ready["protocol"]) == ("ready", protocol)
        await ws.send(bytes(16000 * 2))
        result = json.loads(await ws.recv())
        assert (result["action"], result["display_text"]) == ("DIRECT", "hello")
        await ws.send(json.dumps({"type": "end"}))
        assert json.loads(await ws.recv())["action"] == "WAIT"


@pytest.mark.parametrize("recipe", ["Third", "Fourth"])
@NEEDS_OPENSSL
def test_real_https_and_system_source_wss(recipe, tmp_path, monkeypatch):
    for package in ("fastapi", "uvicorn"):
        pytest.importorskip(package)
    websockets = pytest.importorskip("websockets")
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
    monkeypatch.setenv("no_proxy", "localhost,127.0.0.1")
    certs = tmp_path / "certs"
    with replay_server(recipe, tmp_path, ["--https-host", "meeting.example", "--cert-dir", certs]) as port:
        ctx = tls_client(certs / "root-ca.crt")
        assert peer_certificate(port, ctx, "meeting.example")
        with pytest.raises(ssl.SSLCertVerificationError):
            peer_certificate(port, ctx, "different.example")
        with pytest.raises(ssl.SSLCertVerificationError):
            peer_certificate(port, tls_client(), "meeting.example")
        http = HTTPSConnection("127.0.0.1", port, context=ctx, timeout=5)
        try:
            http.request("GET", "/")
            page = http.getresponse()
            assert page.status == 200 and b'value="system"' in page.read()
            http.request("GET", "/health")
            health = json.load(http.getresponse())
            assert health["ready"] and health["protocol"] == recipe.lower()
        finally:
            http.close()
        asyncio.run(asyncio.wait_for(send_computer_audio(websockets, port, ctx, recipe.lower()), 15))


@pytest.mark.parametrize("recipe", ["Third", "Fourth"])
@NEEDS_OPENSSL
def test_restart_reuses_root_and_leaf_and_renews_only_leaf_on_host_change(recipe, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / recipe))
    tls = importlib.import_module(f"{recipe.lower()}_asr.web_tls")
    first = tls.prepare_local_tls(["first.example"], tmp_path)
    root_bytes, leaf_bytes = first.ca_certfile.read_bytes(), first.certfile.read_bytes()
    again = tls.prepare_local_tls(["first.example"], tmp_path)
    assert again.ca_certfile.read_bytes() == root_bytes and again.certfile.read_bytes() == leaf_bytes
    expanded = tls.prepare_local_tls(["first.example", "second.example"], tmp_path)
    assert expanded.ca_certfile.read_bytes() == root_bytes
    assert expanded.certfile.read_bytes() != leaf_bytes
    args = SimpleNamespace(ssl_certfile=expanded.certfile, ssl_keyfile=expanded.keyfile,
                           cert_dir=None, https_host=[], port=0)
    assert tls.meeting_tls(args, tmp_path) == (str(expanded.certfile), str(expanded.keyfile))


@pytest.mark.parametrize("recipe", ["Third", "Fourth"])
@pytest.mark.parametrize("extra, message", [
    (["--cert-dir", "certs"], "--cert-dir requires"),
    (["--ssl-certfile", "leaf.pem"], "both"),
    (["--ssl-certfile", "leaf.pem", "--ssl-keyfile", "leaf.key", "--https-host", "localhost"], "choose"),
    (["--https-host", "0.0.0.0"], "listening address"),
])
def test_invalid_tls_cli_exits_before_serving(recipe, tmp_path, extra, message):
    cmd = [sys.executable, str(ROOT / recipe / "tools" / "serve_web.py"),
           "--backend", "replay", "--no-recovery", *extra]
    result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, timeout=15)
    assert result.returncode == 2
    assert message in result.stderr.decode(errors="replace").lower()
