import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('one_step_asr', ROOT / 'Data-Scripts/build_asr_sft.py')
one = importlib.util.module_from_spec(spec)
spec.loader.exec_module(one)


@pytest.fixture
def record(tmp_path):
    audio = tmp_path / 'speech.wav'
    sf.write(str(audio), np.zeros(48000, dtype=np.float32), 16000)
    parts = [('<zh><spk1>请联系', 'DIRECT', 50, 300, 'zh', 'spk1'),
             ('张伟', 'WAIT', 400, 850, 'zh', 'spk1'),
             ('<zh><spk1>确认方案。', 'DIRECT', 1100, 1800, 'zh', 'spk1'),
             ('<en><spk2>Hello, ', 'WAIT', 2100, 2400, 'en', 'spk2'),
             ('<en><spk2>yes.', 'DIRECT', 2600, 2900, 'en', 'spk2')]
    segments = [{'source_text': t, 'action': a} for t, a, *_ in parts]
    timestamps = []
    for i, (text, action, start, end, lang, speaker) in enumerate(parts, 1):
        plain = one.TAG.sub('', text)
        timestamps.append({'index': i, 'source_text': text, 'text_plain': plain,
                           'start_ms': start, 'end_ms': end, 'language': lang, 'speaker': speaker,
                           'spans': [{'start_ms': start, 'end_ms': end, 'text_plain': plain,
                                      'language': lang, 'speaker': speaker}]})
    return {'id': 'test', 'audios': [str(audio)],
            'messages': [{'role': 'assistant', 'content': '<zh><spk1>请联系张伟确认方案。<en><spk2>Hello, yes.'}],
            'output': json.dumps({'segments': segments, 'flush': False, 'error': None}, ensure_ascii=False),
            'timestamps': timestamps, 'tags': ['<|'+s['action']+'|>' for s in segments]}


def test_compact_training_data_is_checkable_and_preserves_text(record, tmp_path):
    prepared, tokens = one.build(record, tmp_path, .5, 'normalized')
    assert set(prepared) == {'audios', 'messages'}
    assert len(prepared['audios']) == 6
    assert prepared['messages'][-2] == {'role': 'user', 'content': '<end_of_audio>'}
    contents = [m['content'] for m in prepared['messages'] if m['role'] == 'assistant']
    decoded = ''.join(one.TAG.sub('', c.replace('<|DIRECT|>', '').replace('<|WAIT|>', '')) for c in contents)
    assert decoded == '请联系张伟确认方案。Hello, yes.'
    assert {'<|WAIT|>', '<|DIRECT|>', '<zh>', '<en>', '<spk1>', '<spk2>'} == tokens
    one.check_row(prepared, decode=True, token_inventory=tokens)


def test_cli_800ms_recasts_switch_at_grid_boundary_and_keeps_short_tail(record, tmp_path):
    from omnicommit_audio import decode_chunk_ref

    sf.write(record['audios'][0], np.zeros(53760, dtype=np.float32), 16000)  # 3.36 s
    switch = record['timestamps'][3]
    switch.update(start_ms=2400, end_ms=2550)
    switch['spans'][0].update(start_ms=2400, end_ms=2550)
    source, target = tmp_path / 'timed.jsonl', tmp_path / 'train.800ms.jsonl'
    source.write_text(json.dumps(record) + '\n', encoding='utf-8')
    assert one.main([str(source), '--output', str(target), '--chunk-seconds', '0.8']) == 0

    row = json.loads(target.read_text(encoding='utf-8'))
    refs = [decode_chunk_ref(ref) for ref in row['audios']]
    assert [(r['start'], r['end']) for r in refs] == [
        (0, .8), (.8, 1.6), (1.6, 2.4), (2.4, 3.2), (3.2, 3.36)]
    assert refs[-1]['input_seconds'] == .5
    assert [m['content'] for m in row['messages'] if m['role'] == 'assistant'] == [
        '<|DIRECT|><zh><spk1>请联系', '<|WAIT|>', '<|DIRECT|><zh><spk1>张伟确认方案。',
        '<|WAIT|>', '<|DIRECT|><en><spk2>Hello, yes.', '<|WAIT|>']
    tokens = set(Path(str(target) + '.tokens.txt').read_text(encoding='utf-8').splitlines())
    one.check_row(row, decode=True, token_inventory=tokens)


def test_unannotated_audio_gap_reported_but_not_assumed_silence_or_dropped(record, tmp_path):
    sf.write(record['audios'][0], np.zeros(6 * 16000, dtype=np.float32), 16000)
    diagnostics = {}
    prepared, _ = one.build(record, tmp_path, .5, 'normalized', diagnostics)
    assert len(prepared['audios']) == 12
    assert diagnostics['unannotated_gaps'] == [{'start_ms': 2900.0, 'end_ms': 6000.0}]


def test_review_file_and_optional_gap_rejection(record, tmp_path):
    sf.write(record['audios'][0], np.zeros(6 * 16000, dtype=np.float32), 16000)
    source = tmp_path / 'input.jsonl'
    source.write_text(json.dumps(record, ensure_ascii=False) + '\n', encoding='utf-8')
    target = tmp_path / 'retained.jsonl'
    assert one.main([str(source), '--output', str(target)]) == 0
    review = [json.loads(line) for line in Path(str(target) + '.review.jsonl').read_text().splitlines()]
    assert len(review) == 1 and review[0]['start_ms'] == 2900 and review[0]['end_ms'] == 6000
    assert review[0]['reason'] == 'unannotated_interval_not_confirmed_silence'
    assert len(target.read_text(encoding='utf-8').splitlines()) == 1
    rejected = tmp_path / 'reject.jsonl'
    assert one.main([str(source), '--output', str(rejected), '--max-unannotated-gap-seconds', '2']) == 1
    assert rejected.read_text() == ''
    assert 'unannotated interval' in Path(str(rejected) + '.rejected.jsonl').read_text()
    assert len(Path(str(rejected) + '.review.jsonl').read_text().splitlines()) == 1


def test_unannotated_gap_inside_multiple_timed_spans_is_not_hidden():
    aligned = {'duration_ms': 4000, 'segments': [{'start_ms': 0, 'end_ms': 4000,
               'attribution_spans': [{'start_ms': 0, 'end_ms': 300},
                                     {'start_ms': 3200, 'end_ms': 4000}]}]}
    assert one.unannotated_gaps(aligned) == [{'start_ms': 300, 'end_ms': 3200}]


def test_bad_rows_and_bad_utf8_do_not_stop_later_record(record, tmp_path):
    invalid = copy.deepcopy(record)
    invalid['tags'][1] = '<|DIRECT|>'
    source, target = tmp_path / 'input.jsonl', tmp_path / 'train.jsonl'
    good = json.dumps(record, ensure_ascii=False).encode('utf-8')
    source.write_bytes(b'{bad\n\xff\n' + json.dumps(invalid).encode() + b'\n' + good + b'\n')
    assert one.main([str(source), '--output', str(target)]) == 0
    output = [json.loads(s) for s in target.read_text(encoding='utf-8').splitlines()]
    assert len(output) == 1 and set(output[0]) == {'audios', 'messages'}
    assert len(Path(str(target)+'.rejected.jsonl').read_text().splitlines()) == 3
    assert '<zh>' in Path(str(target)+'.tokens.txt').read_text(encoding='utf-8')


def test_timestamps_must_match_segment_and_attribution(record):
    bad = copy.deepcopy(record)
    bad['timestamps'][1]['text_plain'] = '张三'
    with pytest.raises(ValueError, match='text_plain'):
        one.make_acoustic(bad)
    bad = copy.deepcopy(record)
    bad['timestamps'][1]['spans'][0]['speaker'] = 'spk2'
    with pytest.raises(ValueError, match='attribution'):
        one.make_acoustic(bad)
    bad = copy.deepcopy(record)
    bad['timestamps'][1]['spans'][0]['text'] = '<zh><spk2>张伟'
    with pytest.raises(ValueError, match='contradicting'):
        one.make_acoustic(bad)
    bad = copy.deepcopy(record)
    bad['timestamps'][1]['index'] = 1
    with pytest.raises(ValueError, match='index'):
        one.make_acoustic(bad)


def test_internal_switch_requires_precisely_timed_attribution_spans(record, tmp_path):
    segments = [{'source_text': '<zh><spk1>好', 'action': 'DIRECT'},
                {'source_text': '<zh><spk1>你好<en><spk2>Hello', 'action': 'WAIT'},
                {'source_text': '<en><spk2>world', 'action': 'DIRECT'}]
    record['messages'][0]['content'] = '<zh><spk1>好你好<en><spk2>Helloworld'
    record['output'] = json.dumps({'segments': segments, 'flush': False, 'error': None})
    record.pop('tags')
    record['timestamps'] = [
        {'index': 1, 'start_ms': 100, 'end_ms': 200, 'source_text': segments[0]['source_text'],
         'spans': [{'start_ms': 100, 'end_ms': 200, 'text_plain': '好', 'language': 'zh', 'speaker': 'spk1'}]},
        {'index': 2, 'start_ms': 300, 'end_ms': 800, 'source_text': segments[1]['source_text'],
         'spans': [{'start_ms': 300, 'end_ms': 400, 'text_plain': '你好', 'language': 'zh', 'speaker': 'spk1'},
                   {'start_ms': 600, 'end_ms': 800, 'text_plain': 'Hello', 'language': 'en', 'speaker': 'spk2'}]},
        {'index': 3, 'start_ms': 900, 'end_ms': 1100, 'source_text': segments[2]['source_text'],
         'spans': [{'start_ms': 900, 'end_ms': 1100, 'text_plain': 'world', 'language': 'en', 'speaker': 'spk2'}]},
    ]
    result, _ = one.build(record, tmp_path, .5, 'verbatim')
    answers = [m['content'] for m in result['messages'] if m['role'] == 'assistant']
    assert answers[0] == '<|DIRECT|><zh><spk1>好'
    assert answers[1] == '<|WAIT|>'  # switch at .6s, not unit start .3s
    assert answers[2] == '<|DIRECT|><zh><spk1>你好<en><spk2>Helloworld'
    record['timestamps'][1].pop('spans')
    with pytest.raises(ValueError, match='needs timed spans'):
        one.make_acoustic(record)
