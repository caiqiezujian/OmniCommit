import sys
import time
import numpy as np
import pytest

from omnicommit.aligner_proxy import QwenAlignerProxy


@pytest.fixture
def paths(tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    worker = tmp_path / 'fake_worker.py'
    worker.write_text('''import json, sys, time
for line in sys.stdin:
    obj=json.loads(line)
    if obj['text']=='sleep': time.sleep(2)
    if obj['text']=='bad':
        print(json.dumps({'id':obj['id'],'ok':False,'error':'failed'}), flush=True)
    else:
        print(json.dumps({'id':obj['id'],'ok':True,'items':[{'text':obj['text'],'start_time':0.0,'end_time':0.5}]}), flush=True)
''', encoding='utf-8')
    return model, worker


def test_proxy_persistent_and_compatible_with_reference_aligner(paths):
    model, worker = paths
    proxy = QwenAlignerProxy(sys.executable, model, worker_path=worker, timeout_seconds=2)
    try:
        audio = (np.full(16000, .1, np.float32), 16000)
        first = proxy.align(audio=audio, text='hello', language='English')
        pid = proxy.process.pid
        second = proxy.align(audio=audio, text='world', language='English')
        assert proxy.process.pid == pid
        assert first[0].items[0].text == 'hello' and second[0].items[0].end_time == .5
        with pytest.raises(RuntimeError, match='failed'):
            proxy.align(audio=audio, text='bad', language='English')
        assert proxy.process is None
        assert proxy.align(audio=audio, text='again', language='English')[0].items[0].text == 'again'
    finally:
        proxy.close()


def test_worker_timeout_including_blocked_request_write(paths):
    model, worker = paths
    worker.write_text('import time\ntime.sleep(10)\n', encoding='utf-8')
    proxy = QwenAlignerProxy(sys.executable, model, worker_path=worker, timeout_seconds=.3)
    try:
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            proxy.align(audio=(np.ones(80000, np.float32), 16000), text='hello', language='English')
        assert time.monotonic() - start < 3 and proxy.process is None
    finally:
        proxy.close()


def test_pcm_bounds_checked_without_starting_worker(paths):
    model, worker = paths
    proxy = QwenAlignerProxy(sys.executable, model, worker_path=worker)
    with pytest.raises(ValueError, match='Invalid bounded PCM'):
        proxy.align(audio=(np.zeros(5*16000+1), 16000), text='hello', language='English')
    assert proxy.process is None


def test_real_worker_protocol_with_fake_qwen_logs_isolated(tmp_path, monkeypatch):
    model = tmp_path / 'model'
    model.mkdir()
    (tmp_path / 'torch.py').write_text("float32='f32'\nfloat16='f16'\nbfloat16='bf16'\n", encoding='utf-8')
    (tmp_path / 'qwen_asr.py').write_text('''import os
from types import SimpleNamespace
print('third party stdout')
os.write(1, b'low-level output\\n')
class Qwen3ForcedAligner:
    @classmethod
    def from_pretrained(cls, *a, **kw):
        obj=cls(); obj.aligner_processor=SimpleNamespace(); return obj
    def align(self, audio, text, language):
        assert audio[1]==16000
        return [SimpleNamespace(items=[SimpleNamespace(text=text,start_time=.1,end_time=.5)])]
''', encoding='utf-8')
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    proxy = QwenAlignerProxy(sys.executable, model, device='cpu', dtype='float32', timeout_seconds=5)
    try:
        response = proxy.align(audio=(np.zeros(16000, np.float32), 16000), text='hello', language='English')
        assert response[0].items[0].text == 'hello'
        assert response[0].items[0].start_time == .1
    finally:
        proxy.close()
