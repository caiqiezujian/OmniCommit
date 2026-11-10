"""Distribution diagnostics must not conflate EOS, mixed languages or errors."""
import base64
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('diagnose_sft_targets', Path(__file__).resolve().parents[1] / 'SFT-Scripts' / 'diagnose_sft_targets.py')
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def reference(start, end):
    value = {'path': '/missing/audio.wav', 'start': start, 'end': end, 'input_seconds': max(.5, end - start)}
    return diagnostic.VIRTUAL_PREFIX + base64.urlsafe_b64encode(json.dumps(value).encode()).decode()


def row(targets, eos='<|WAIT|>', durations=None):
    durations = durations or [.5] * len(targets)
    result = {'audios': [], 'messages': [{'role': 'system', 'content': 'Streaming ASR.'}]}
    start = 0
    for target, duration in zip(targets, durations):
        result['audios'].append(reference(start, start + duration))
        start += duration
        result['messages'].extend([{'role': 'user', 'content': '<audio>'}, {'role': 'assistant', 'content': target}])
    result['messages'].extend([{'role': 'user', 'content': '<end_of_audio>'}, {'role': 'assistant', 'content': eos}])
    return result


def write(path, rows):
    path.write_text(''.join(json.dumps(value, ensure_ascii=False) + '\n' for value in rows), encoding='utf-8')
    return path


def test_english_text_emitted_only_at_eos_is_visible(tmp_path):
    source = write(tmp_path / 'train.jsonl', [row(['<|WAIT|>', '<|WAIT|>'], '<|DIRECT|><en><spk1>Hello world.')])
    report = diagnostic.scan_dataset(source)
    metrics = report['groups']['monolingual']['en']
    assert metrics['audio_wait_ratio'] == 1
    assert metrics['all_text_at_eos_rows'] == 1
    assert metrics['eos_text_character_ratio'] == 1
    assert metrics['first_audio_direct_seconds']['count'] == 0
    assert metrics['longest_wait_seconds_per_row']['max'] == 1
    assert report['english']['text_characters'] == len('Hello world.')


def test_mixed_waits_not_assigned_to_monolingual_english(tmp_path):
    source = write(tmp_path / 'mixed.jsonl', [row(['<|WAIT|>', '<|DIRECT|><zh><spk1>你好<en><spk2>Hello.'])])
    report = diagnostic.scan_dataset(source)
    assert report['groups']['monolingual'] == {}
    metrics = report['groups']['mixed']['en+zh']
    assert metrics['audio_wait_turns'] == 1
    assert report['english']['mixed_rows'] == 1
    assert report['english']['monolingual_rows'] == 0
    assert report['english']['text_characters'] == 6


def test_bad_rows_continue_and_error_details_are_bounded(tmp_path, capsys):
    source = tmp_path / 'mixed-errors.jsonl'
    good = row(['<|DIRECT|><en><spk1>Go.'])
    bad = row(['<|DIRECT|>missing headers'])
    source.write_text('{bad json\n' + json.dumps(good) + '\n' + json.dumps(bad) + '\n', encoding='utf-8')
    assert diagnostic.main([str(source), '--max-details', '1']) == 1
    report = json.loads(capsys.readouterr().out)
    assert (report['rows_read'], report['rows_valid'], report['rows_invalid']) == (3, 1, 2)
    assert len(report['errors']) == 1 and report['errors'][0]['line'] == 1
    assert report['english']['rows_with_en'] == 1


def test_no_audio_or_sidecar_access_and_report_never_overwrites(tmp_path, capsys):
    source = write(tmp_path / 'train.jsonl', [row(['<|DIRECT|><en><spk1>Keep spaces. '])])
    original = source.read_bytes()
    target = tmp_path / 'report.json'
    assert not Path(str(source) + '.tokens.txt').exists()
    assert diagnostic.main([str(source), '--report', str(target)]) == 0
    capsys.readouterr()
    assert json.loads(target.read_text(encoding='utf-8'))['english']['text_characters'] == len('Keep spaces. ')
    saved_report = target.read_bytes()
    with pytest.raises(SystemExit) as exc:
        diagnostic.main([str(source), '--report', str(target)])
    assert exc.value.code == 2
    assert target.read_bytes() == saved_report and source.read_bytes() == original


def test_wait_duration_uses_real_tail_and_first_direct_endpoint(tmp_path):
    source = write(tmp_path / 'timed.jsonl', [row(
        ['<|WAIT|>', '<|WAIT|>', '<|DIRECT|><en><spk1>Ready.', '<|WAIT|>'],
        durations=[.5, .5, .5, .2])])
    metrics = diagnostic.scan_dataset(source)['total']
    assert metrics['first_audio_direct_seconds']['mean'] == 1.5
    assert metrics['longest_wait_seconds_per_row']['max'] == 1
    assert metrics['audio_wait_ratio'] == .75
    assert metrics['eos_text_character_ratio'] == 0
