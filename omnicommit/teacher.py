"""Explicit OpenAI-compatible input_audio transport; no SDK or secret in traces."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from urllib.parse import urlsplit

from .protocol import parse_teacher


def prompt_text(path, number_format):
    prompt = Path(path).read_text(encoding="utf-8").strip()
    modes = {
        "verbatim": "NUMBER_FORMAT=VERBATIM。保留音频中数词的原始口语词形，不转换成阿拉伯数字。",
        "normalized": "NUMBER_FORMAT=NORMALIZED。完整数量表达确认后转成阿拉伯数字；未闭合的数值暂缓，不猜数量级。",
    }
    return prompt + "\n\n" + modes[number_format]


class AudioTeacher:
    def __init__(self, api_base, model, generation, prompt, timeout=300, api_key=None, session=None):
        import requests
        if not api_base.startswith(("http://", "https://")):
            raise ValueError("api_base must be an HTTP(S) endpoint")
        url = urlsplit(api_base)
        if not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("api_base must not contain credentials, query parameters or a fragment")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty service model ID")
        if not isinstance(generation, dict):
            raise ValueError("generation config must be an object")
        if set(generation) & {"messages", "model", "stream", "audio", "input_audio", "api_key", "headers", "authorization"}:
            raise ValueError("generation config cannot replace model/messages/audio or enable streaming")
        self.api_base, self.model = api_base.rstrip("/"), model
        self.generation, self.prompt, self.timeout = generation, prompt, timeout
        self.session = session or requests.Session()
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = "Bearer " + api_key

    def transcribe(self, wav_bytes):
        # Reference transcript, existing teacher output and future audio never enter this request.
        body = {"model": self.model, "messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {
                "data": base64.b64encode(wav_bytes).decode("ascii"), "format": "wav"}},
            {"type": "text", "text": self.prompt}]}], **self.generation}
        response = self.session.post(self.api_base + "/chat/completions",
                                     headers=self.headers, json=body, timeout=self.timeout)
        try:
            response.raise_for_status()
            choice = response.json()["choices"][0]
            if choice.get("finish_reason") not in (None, "stop"):
                raise ValueError("Teacher response incomplete: " + str(choice.get("finish_reason")))
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Teacher content is empty or not text")
            return parse_teacher(json.loads(content))
        finally:
            response.close()

    def close(self):
        self.session.close()


class ReplayTeacher:
    """Ordered JSONL replay, one complete source record resident at a time."""
    def __init__(self, path):
        self.stream = open(path, encoding="utf-8-sig")
        self.record = None

    def select(self, identifier):
        # Replay entries may omit source records rejected before any inference.
        # A second source row with the same ID still needs its own replay entry.
        self.record = None
        while self.record is None or str(self.record.get("id")) != identifier:
            line = self.stream.readline()
            if not line:
                raise ValueError("Replay has no remaining record for id=" + identifier)
            if line.strip():
                self.record = json.loads(line)
        return self

    def full(self):
        return parse_teacher(self.record["full"])

    def prefix(self, end_ms):
        matches = [row for row in self.record.get("prefixes", [])
                   if abs(row["end_ms"] - end_ms) < 1e-6]
        if len(matches) != 1:
            raise ValueError("Replay requires exactly one matching prefix endpoint")
        return parse_teacher({"segments": matches[0]["segments"]})

    def close(self):
        self.stream.close()
