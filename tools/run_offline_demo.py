#!/usr/bin/env python3
"""Exercise the entire CPU pipeline using a tone and scripted hypotheses, NOT real ASR."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/offline-demo'))
    args = parser.parse_args()
    folder = args.output_dir.absolute().resolve()
    if folder.exists():
        parser.error('Demo output directory already exists; use a new path')
    folder.mkdir(parents=True)
    import numpy as np
    import soundfile as sf

    sr = 16000
    time = np.arange(sr * 7 // 2, dtype=np.float32) / sr
    audio = folder / 'synthetic-tone.wav'
    sf.write(str(audio), .05 * np.sin(2 * np.pi * 220 * time), sr, subtype='PCM_16')

    def segment(text, language='zh', speaker='spk1'):
        return {'language': language, 'speaker': speaker, 'text': text}

    chinese = segment('我有两只猫。')
    english = segment('Any questions?', 'en')
    reply = segment('没有。', speaker='spk2')
    full = [chinese, english, reply]
    raw = {'id': 'synthetic-demo', 'audios': [str(audio)], 'messages': [
        {'role': 'system', 'content': 'Transcribe speech with language and speaker tags.'},
        {'role': 'user', 'content': '<audio>**Entity Reference:** []'},
        {'role': 'assistant', 'content': '<zh><spk1>我有两只猫。<en><spk1>Any questions?<zh><spk2>没有。'}]}
    replay = {'id': raw['id'], 'full': {'segments': full}, 'prefixes': [
        {'end_ms': 500, 'segments': [segment('我有粮')]},
        {'end_ms': 1000, 'segments': [segment('我有')]},
        {'end_ms': 1500, 'segments': [segment('我有凉')]},
        {'end_ms': 2000, 'segments': [chinese]},
        {'end_ms': 2500, 'segments': [chinese, english]},
        {'end_ms': 3000, 'segments': full}]}
    for name, value in [('raw.jsonl', raw), ('responses.jsonl', replay)]:
        (folder / name).write_text(json.dumps(value, ensure_ascii=False) + '\n', encoding='utf-8')
    root = Path(__file__).resolve().parents[1]
    commands = [
        ['Data-Scripts/collect_prefixes.py', str(folder / 'raw.jsonl'), '--output', str(folder / 'traces.jsonl'),
         '--replay-responses', str(folder / 'responses.jsonl'), '--chunk-seconds', '0.5'],
        ['Data-Scripts/build_trajectories.py', str(folder / 'traces.jsonl'), '--output', str(folder / 'trajectories.jsonl')],
        ['SFT-Scripts/prepare_sft.py', str(folder / 'trajectories.jsonl'), '--output', str(folder / 'train.jsonl')],
        ['SFT-Scripts/check_sft.py', str(folder / 'train.jsonl'), '--rows', '1',
         '--tokens-file', str(folder / 'train.jsonl.tokens.txt')],
    ]
    for command in commands:
        print('[demo]', command[0], flush=True)
        subprocess.run([sys.executable, '-X', 'utf8', str(root / command[0]), *command[1:]], check=True)
    train = json.loads((folder / 'train.jsonl').read_text(encoding='utf-8').splitlines()[0])
    print('\nScripted assistant outputs (synthetic protocol test, not real speech):')
    for message in train['messages']:
        if message['role'] == 'assistant':
            print(message['content'])
    print(f'Artifacts: {folder}')


if __name__ == '__main__':
    main()
