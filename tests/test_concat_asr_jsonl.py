import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf

from omnicommit.alignment import annotation_units


SCRIPT = Path(__file__).resolve().parents[1] / 'Data-Scripts' / 'concat_asr_jsonl.py'


def record(audio, transcript, segments):
    annotation = {
        'segments': [{'source_text': text, 'action': action} for text, action in segments],
        'flush': segments[-1][1] == 'WAIT',
        'error': None,
    }
    return {
        'audios': [audio],
        'messages': [
            {'role': 'system', 'content': 'Transcribe the audio.'},
            {'role': 'user', 'content': '<audio>**Entity Reference:** []'},
            {'role': 'assistant', 'content': transcript},
        ],
        'output': json.dumps(annotation, ensure_ascii=False),
    }


def write_jsonl(path, rows):
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def run_cli(a, b, tmp_path):
    out = tmp_path / 'joined.jsonl'
    audio_dir = tmp_path / 'joined_wavs'
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(a), str(b), str(out), '--audio-dir', str(audio_dir)],
        cwd=tmp_path, capture_output=True, encoding='utf-8', errors='replace',
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return [json.loads(line) for line in out.read_text(encoding='utf-8').splitlines()], audio_dir


def test_seam_languages_audio_and_annotations_match_real_alignment_entry(tmp_path):
    # A ends in Italian. The eligible B starts in Chinese (also A's FIRST
    # language). The ineligible B starts in Italian and ends in English.
    a_file, b_file = tmp_path / 'A' / 'input.jsonl', tmp_path / 'B' / 'input.jsonl'
    a = record('speech.wav', '<zh><spk1>比分<zh><spk1>70比71。<it><spk2>Va bene.', [
        ('<zh><spk1>比分', 'DIRECT'), ('70比71', 'WAIT'), ('<zh><spk1>。<it><spk2>Va', 'WAIT'),
        ('<it><spk2> bene.', 'DIRECT'),
    ])
    b = record('speech.wav', '<zh><spk1>好的。<spk10>继续吧。<spk1>行。', [
        ('<zh><spk1>好', 'DIRECT'), ('<zh><spk1>的。', 'DIRECT'), ('<zh><spk10>继', 'WAIT'),
        ('<zh><spk10>续吧。', 'DIRECT'), ('<zh><spk1>行。', 'WAIT'),
    ])
    same_language = record('other.wav', '<it><spk1>Sì.<en>All right.', [
        ('<it><spk1>Sì.', 'DIRECT'), ('<en><spk1>All', 'WAIT'), ('<en><spk1> right.', 'DIRECT'),
    ])
    write_jsonl(a_file, [a])
    write_jsonl(b_file, [same_language, b])
    # Bad line isolation with a valid row after it.
    b_file.write_bytes(b'{malformed\n' + b_file.read_bytes())
    a_samples = np.arange(1600, dtype=np.int16) + 500
    b_samples = -np.arange(1800, dtype=np.int16) - 300
    sf.write(a_file.parent / 'speech.wav', a_samples, 16000)
    sf.write(b_file.parent / 'speech.wav', b_samples, 16000)
    sf.write(b_file.parent / 'other.wav', b_samples, 16000)

    rows, audio_dir = run_cli(a_file, b_file, tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == {'audios', 'messages', 'output'}
    assert len(row['audios']) == 1
    wav = Path(row['audios'][0])
    assert wav.is_absolute() and wav.parent.resolve() == audio_dir.resolve()
    assert sf.info(wav).subtype == 'PCM_16'
    samples, rate = sf.read(wav, dtype='int16')
    assert rate == 16000 and samples.ndim == 1
    np.testing.assert_array_equal(samples, np.concatenate([a_samples, b_samples]))
    assert [message['role'] for message in row['messages']] == ['system', 'user', 'assistant']
    assert row['messages'][:2] == a['messages'][:2]
    assert row['messages'][2]['content'].startswith(a['messages'][2]['content'])
    assert isinstance(row['output'], str)
    combined = json.loads(row['output'])
    a_segments = json.loads(a['output'])['segments']
    assert combined['segments'][:len(a_segments)] == a_segments
    b_segments = combined['segments'][len(a_segments):]
    assert [segment['action'] for segment in b_segments] == ['WAIT', 'DIRECT', 'WAIT', 'DIRECT', 'WAIT']

    # The same downstream parser used before timestamp recasting must see the
    # original words and speaker turns, including inherited and partial headers.
    annotated, units, flush = annotation_units(row)
    assert flush
    b_speaker1 = annotated[len(a_segments)]['attributed'][0]['speaker']
    b_speaker2 = annotated[len(a_segments) + 2]['attributed'][0]['speaker']
    assert len({'spk1', 'spk2', b_speaker1, b_speaker2}) == 4
    assert b_segments[0]['source_text'] == f'<zh><{b_speaker1}>好'
    assert annotated[-1]['attributed'][0]['speaker'] == b_speaker1
    assert ''.join(char for _, _, char in units) == '比分70比71。Va bene.好的。继续吧。行。'


def test_different_sample_rates_and_channels_preserve_duration_and_order(tmp_path):
    a_file, b_file = tmp_path / 'a.jsonl', tmp_path / 'b.jsonl'
    write_jsonl(a_file, [record('a.wav', '<de><spk1>Gut.', [('<de><spk1>Gut.', 'DIRECT')])])
    write_jsonl(b_file, [record('b.wav', '<en><spk1>Very good.', [
        ('<en><spk1>Very', 'DIRECT'), ('<en><spk1> good.', 'DIRECT'),
    ])])
    sf.write(tmp_path / 'a.wav', np.full((800, 2), 0.25), 8000)
    sf.write(tmp_path / 'b.wav', np.full(4800, -0.125), 48000)
    rows, _ = run_cli(a_file, b_file, tmp_path)
    assert len(rows) == 1
    audio = Path(rows[0]['audios'][0])
    samples, rate = sf.read(audio)
    assert rate == 16000 and samples.shape == (3200,)
    np.testing.assert_allclose(samples[80:1520], 0.25, atol=0.002)
    np.testing.assert_allclose(samples[1680:-80], -0.125, atol=0.002)


def test_matching_does_not_use_the_only_eligible_b_for_the_wrong_a(tmp_path):
    def make(name, language):
        file = name + '.wav'
        sf.write(tmp_path / file, np.full(1600, 0.25), 16000)
        header = f'<{language}><spk1>'
        return record(file, header + name + ' end.', [
            (header + name, 'DIRECT'), (header + ' end.', 'DIRECT'),
        ])

    a_file, b_file = tmp_path / 'a.jsonl', tmp_path / 'b.jsonl'
    write_jsonl(a_file, [make('a1', 'zh'), make('a2', 'en')])
    write_jsonl(b_file, [make('b1', 'fr'), make('b2', 'en')])
    rows, _ = run_cli(a_file, b_file, tmp_path)
    assert len(rows) == 2
    paired_texts = {''.join(char for _, _, char in annotation_units(row)[1]) for row in rows}
    assert paired_texts == {'a1 end.b2 end.', 'a2 end.b1 end.'}


def test_bad_candidates_do_not_break_a_single_terminal_wait_pair(tmp_path):
    a_file, b_file = tmp_path / 'a.jsonl', tmp_path / 'b.jsonl'
    a = record('a.wav', '<zh><spk1>请。', [('<zh><spk1>请。', 'DIRECT')])
    # B's only segment already waits. Its words, leading spaces and final flush
    # survive the join so that the actual end-of-audio turn can release them.
    b = record('b.wav', '  <en><spk1>Paris.', [('<en><spk1>  Paris.', 'WAIT')])
    prefix = '<audio>**Entity Reference:** '
    a['messages'][1]['content'] = prefix + '["Li"]'
    b['messages'][1]['content'] = prefix + '["Paris", "Li"]'
    missing = record('missing.wav', '<fr><spk1>Oui.', [('<fr><spk1>Oui.', 'DIRECT')])
    bad = record('b.wav', '<it><spk1>Bene.', [('<it><spk1>Ciao.', 'DIRECT')])
    write_jsonl(a_file, [a])
    write_jsonl(b_file, [missing, missing, bad, bad, b])
    for name in ['a.wav', 'b.wav']:
        sf.write(tmp_path / name, np.zeros(1600, dtype='int16'), 16000)

    rows, audio_dir = run_cli(a_file, b_file, tmp_path)
    assert len(rows) == len(list(audio_dir.glob('*.wav'))) == 1
    row = rows[0]
    assert row['messages'][1]['content'].startswith(prefix)
    assert json.loads(row['messages'][1]['content'][len(prefix):]) == ['Li', 'Paris']
    segments, units, flush = annotation_units(row)
    assert flush and [s['action'] for s in segments] == ['DIRECT', 'WAIT']
    assert segments[1]['attributed'] == [{'text': '  Paris.', 'language': 'en', 'speaker': 'spk2'}]
    assert ''.join(char for _, _, char in units) == '请。  Paris.'
