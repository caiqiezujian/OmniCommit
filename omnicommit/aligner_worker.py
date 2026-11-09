"""Private line-JSON RPC worker: actual Qwen alignment in a separate environment.

STDOUT is reserved exclusively for protocol responses; all third-party output
is redirected at the file-descriptor level to STDERR before importing them.
"""
from __future__ import annotations
import argparse
import base64
import json
import os
from pathlib import Path
import sys


def main():
    protocol = os.fdopen(os.dup(1), 'w', encoding='utf-8', buffering=1)
    os.dup2(2, 1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dtype', choices=('float32', 'float16', 'bfloat16'), default='bfloat16')
    parser.add_argument('--max-seconds', type=float, default=5)
    args = parser.parse_args()
    model_path = Path(args.model).resolve(strict=True)
    if not model_path.is_dir() or not 1 <= args.max_seconds <= 30:
        raise ValueError('Expected existing model directory and 1..30 seconds')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    model = None
    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            request_id = request['id']
            if request.get('op') != 'align':
                raise ValueError('Unknown operation')
            text, language = request.get('text'), request.get('language')
            if not isinstance(text, str) or not text.strip() or len(text) > 10000 or not isinstance(language, str) or not language:
                raise ValueError('Invalid text or language')
            raw = base64.b64decode(request['pcm_f32le'], validate=True)
            if not raw or len(raw) % 4 or len(raw) > int(args.max_seconds * 16000) * 4:
                raise ValueError('Invalid audio length')
            import numpy as np
            pcm = np.frombuffer(raw, dtype='<f4').astype(np.float32)
            if not np.isfinite(pcm).all() or np.max(np.abs(pcm)) > 1.001:
                raise ValueError('Invalid PCM audio')
            if model is None:
                import torch
                from qwen_asr import Qwen3ForcedAligner
                from omnicommit.forced_alignment import strict_qwen_timestamps
                model = Qwen3ForcedAligner.from_pretrained(str(model_path), dtype=getattr(torch, args.dtype), device_map=args.device)
                model.aligner_processor.fix_timestamp = strict_qwen_timestamps
            result = model.align(audio=(pcm, 16000), text=text, language=language)
            if len(result) != 1:
                raise ValueError('Expected one alignment result')
            items = [{'text': i.text, 'start_time': float(i.start_time), 'end_time': float(i.end_time)} for i in result[0].items]
            response = {'id': request_id, 'ok': True, 'items': items}
            protocol.write(json.dumps(response, ensure_ascii=True, allow_nan=False) + '\n')
        except Exception as exc:
            protocol.write(json.dumps({'id': request_id, 'ok': False, 'error': f'{type(exc).__name__}: {exc}'}, ensure_ascii=True) + '\n')
    protocol.close()


if __name__ == '__main__':
    main()
