from types import SimpleNamespace
import numpy as np
import pytest

from omnicommit.session_identity import SessionLedger
from omnicommit.streaming import AudioTurn, SpeakerReference


def session():
    audio = np.full(8000, .1, dtype=np.float32)
    return SimpleNamespace(speakers=['spk1', 'spk2', 'spk3'], references={},
        heard_seconds=1.5, history=[AudioTurn(i*.5, (i+1)*.5, audio.copy(), '<|WAIT|>') for i in range(3)],
        options=SimpleNamespace(max_reference_seconds=5, max_total_reference_seconds=30),
        backend=SimpleNamespace(allowed_tokens=None))


def result(turn, speaker, text):
    return {'turn': turn, 'action': 'DIRECT',
            'segments': [{'speaker': speaker, 'language': 'zh', 'text': text}]}


def ref(speaker):
    return SpeakerReference(speaker, 'zh', '好', 0, 1, np.ones(16000, dtype=np.float32))


def test_correction_preserves_raw_ids_and_invalidates_references():
    s = session()
    ledger = SessionLedger(s)
    ledger.record(result(1, 'spk1', '你好'))
    ledger.record(result(2, 'spk2', '再见'))
    s.references.update(spk1=ref('spk1'), spk2=ref('spk2'))
    snapshot = ledger.correct_segment(2, 'spk1')
    assert snapshot['segments'][1]['model_speaker'] == 'spk2'
    assert snapshot['segments'][1]['display_speaker'] == 'spk1'
    assert not s.references
    assert s.speakers == ['spk1', 'spk2', 'spk3']
    assert snapshot['segments'][1]['turn'] == 2 and snapshot['segments'][1]['segment_index'] == 0
    with pytest.raises(ValueError):
        ledger.correct_segment(2, 'spk4')


def test_merge_alias_applies_to_future_results_and_display_name():
    s = session()
    ledger = SessionLedger(s)
    ledger.record(result(1, 'spk3', '你好'))
    s.references.update(spk1=ref('spk1'), spk3=ref('spk3'))
    ledger.merge_speakers('spk3', 'spk1')
    ledger.rename_speaker('spk1', '张伟')
    later = ledger.record(result(2, 'spk3', '又见面了'))
    snapshot = ledger.snapshot()
    assert later[0]['model_speaker'] == 'spk3' and later[0]['display_speaker'] == 'spk1'
    assert snapshot['segments'][0]['display_speaker'] == 'spk1'
    assert snapshot['speakers'][0] == {'speaker': 'spk1', 'name': '张伟'}
    assert not s.references and s.speakers == ['spk1', 'spk2', 'spk3']
    with pytest.raises(ValueError):
        ledger.merge_speakers('spk3', 'spk1')


def test_estimated_reference_requires_independent_aligned_identity_evidence():
    s = session()
    ledger = SessionLedger(s)
    item = ledger.record(result(3, 'spk1', '请联系张伟。'))[0]
    evidence = {'source': 'independent_online', 'single_speaker': True, 'overlap': False,
                'start': 0, 'end': 1, 'alignment_confidence': .96, 'identity_confidence': .95,
                'speaker': 'spk1', 'language': 'zh', 'text': '请联系张伟。'}
    for changes in ({'source': 'model_current_chunk'}, {'overlap': True}, {'end': 2},
                    {'text': '张伟。'}, {'identity_confidence': .5}, {'start': .6, 'end': 1}):
        assert ledger.consider_reference(item['segment_id'], dict(evidence, **changes)) is None
    status = ledger.consider_reference(item['segment_id'], evidence)
    assert status['status'] == 'estimated' and len(s.references['spk1'].audio) == 16000
    assert ledger.consider_reference(item['segment_id'], evidence) is None
    ledger.correct_segment(item['segment_id'], 'spk2')
    assert not s.references and not ledger.reference_evidence


def test_silence_evicted_or_existing_manual_reference_is_not_admitted():
    s = session()
    ledger = SessionLedger(s)
    item = ledger.record(result(1, 'spk1', '你好'))[0]
    ev = {'source': 'independent_online', 'single_speaker': True, 'overlap': False,
          'start': 0, 'end': 1, 'alignment_confidence': .96, 'identity_confidence': .95,
          'speaker': 'spk1', 'language': 'zh', 'text': '你好'}
    s.references['spk1'] = ref('spk1')
    assert ledger.consider_reference(item['segment_id'], ev) is None
    s.references.clear()
    s.history.pop(0)
    assert ledger.consider_reference(item['segment_id'], ev) is None
    s.history.insert(0, AudioTurn(0, .5, np.zeros(8000, dtype=np.float32), '<|WAIT|>'))
    s.history[1].audio[:] = 0
    assert ledger.consider_reference(item['segment_id'], ev) is None
