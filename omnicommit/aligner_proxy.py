"""Persistent, timeout-bounded Qwen alignment subprocess; no ASR dependency mixing."""
from __future__ import annotations
import base64
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import threading
from types import SimpleNamespace
import uuid

import numpy as np


class QwenAlignerProxy:
    def __init__(self, python, model, *, device='cuda:0', dtype='bfloat16', max_seconds=5,
                 timeout_seconds=60, worker_path=None):
        self.python = Path(python).resolve(strict=True)
        self.model = Path(model).resolve(strict=True)
        self.worker = Path(worker_path or Path(__file__).with_name('aligner_worker.py')).resolve(strict=True)
        if not self.python.is_file() or not self.model.is_dir() or not self.worker.is_file():
            raise ValueError('Expected Python executable, local model directory and worker script')
        if dtype not in ('float32', 'float16', 'bfloat16') or not isinstance(device, str) or not device:
            raise ValueError('Invalid device or dtype')
        if isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= 30:
            raise ValueError('max_seconds must be 1..30')
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError('timeout_seconds must be positive')
        self.device, self.dtype, self.max_seconds, self.timeout = device, dtype, max_seconds, timeout_seconds
        self.lock = threading.Lock()
        self.process = None
        self.responses = None

    def _start(self):
        if self.process is not None and self.process.poll() is None:
            return
        command = [str(self.python), '-u', str(self.worker), '--model', str(self.model), '--device', self.device,
                   '--dtype', self.dtype, '--max-seconds', str(self.max_seconds)]
        kwargs = {'stdin': subprocess.PIPE, 'stdout': subprocess.PIPE, 'stderr': None,
                  'text': True, 'encoding': 'utf-8', 'bufsize': 1}
        if os.name == 'nt':
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        process = subprocess.Popen(command, **kwargs)
        responses = queue.Queue()
        self.process, self.responses = process, responses
        def read_output():
            try:
                while True:
                    line = process.stdout.readline(1024 * 1024)
                    if not line:
                        break
                    if not line.endswith('\n'):
                        responses.put(RuntimeError('Alignment worker response exceeds limit'))
                        return
                    responses.put(line)
            except Exception as exc:
                responses.put(exc)
            finally:
                responses.put(EOFError('Alignment worker exited'))
        threading.Thread(target=read_output, name='qwen-aligner-output', daemon=True).start()

    def _stop(self):
        process = self.process
        self.process, self.responses = None, None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
        if process.stdin:
            process.stdin.close()
        if process.stdout:
            process.stdout.close()

    def close(self):
        with self.lock:
            self._stop()

    def align(self, *, audio, text, language):
        if not isinstance(audio, tuple) or len(audio) != 2 or audio[1] != 16000:
            raise ValueError('Expected (mono float PCM, 16000)')
        pcm = np.asarray(audio[0], dtype=np.float32)
        if pcm.ndim != 1 or not 0 < len(pcm) <= int(self.max_seconds * 16000) or not np.isfinite(pcm).all() or np.max(np.abs(pcm)) > 1.001:
            raise ValueError('Invalid bounded PCM audio')
        if not isinstance(text, str) or not text.strip() or len(text) > 10000 or not isinstance(language, str) or not language:
            raise ValueError('Invalid text or language')
        request_id = uuid.uuid4().hex
        request = {'op': 'align', 'id': request_id, 'text': text, 'language': language,
                   'pcm_f32le': base64.b64encode(pcm.astype('<f4', copy=False).tobytes()).decode('ascii')}
        with self.lock:
            try:
                self._start()
                process, responses = self.process, self.responses
                def send_request():
                    try:
                        process.stdin.write(json.dumps(request, ensure_ascii=True) + '\n')
                        process.stdin.flush()
                    except Exception as exc:
                        responses.put(exc)
                # The worker may hang before reading even its first request.
                # Include pipe writes in the deadline, not just response reads.
                threading.Thread(target=send_request, name='qwen-aligner-input', daemon=True).start()
                raw = self.responses.get(timeout=self.timeout)
                if isinstance(raw, Exception):
                    raise raw
                response = json.loads(raw)
                if response.get('id') != request_id:
                    raise ValueError('Alignment worker response ID mismatch')
                if response.get('ok') is not True:
                    raise RuntimeError(str(response.get('error', 'Alignment failed')))
                items = response.get('items')
                if not isinstance(items, list) or not items:
                    raise ValueError('Empty alignment result')
                normalized = []
                for item in items:
                    if not isinstance(item, dict) or not isinstance(item.get('text'), str):
                        raise ValueError('Invalid alignment item')
                    start, end = item.get('start_time'), item.get('end_time')
                    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (start, end)) or not 0 <= start < end <= len(pcm) / 16000 + 1e-6:
                        raise ValueError('Invalid alignment timestamps')
                    normalized.append(SimpleNamespace(text=item['text'], start_time=start, end_time=end))
                return [SimpleNamespace(items=normalized)]
            except queue.Empty as exc:
                self._stop()
                raise TimeoutError('Alignment worker timed out') from exc
            except Exception:
                self._stop()
                raise
