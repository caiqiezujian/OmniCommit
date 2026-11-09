import types

import numpy as np

from omnicommit.auto_reference import QwenAutoReference
from omnicommit.session_identity import SessionLedger
from omnicommit.streaming import StreamOptions, StreamSession, ReplayBackend


class FakeAligner:
    def __init__(self, first=.05, last=.95):
        self.calls = []
        self.first, self.last = first, last

    def align(self, *, audio, text, language):
        self.calls.append((audio, text, language))
        item = types.SimpleNamespace(text='Hello', start_time=self.first, end_time=self.last)
        return [types.SimpleNamespace(items=[item])]


def prepared():
    session = StreamSession(ReplayBackend(['<|WAIT|>', '<|DIRECT|><en><spk1>Hello.']),
                            StreamOptions(chunk_seconds=.5))
    ledger = SessionLedger(session)
    audio = np.full(8000, .1, dtype=np.float32)
    ledger.record(session.step(audio))
    result = session.step(audio)
    ledger.record(result)
    return session, ledger, result


def test_qwen_candidate_uses_received_history_and_reports_unverified():
    session, ledger, result = prepared()
    fake = FakeAligner()
    provider = QwenAutoReference('fake', model=fake)
    item = provider.consider(ledger, result)
    assert len(fake.calls) == 1
    (audio, rate), text, language = fake.calls[0]
    assert len(audio) == 16000 and rate == 16000 and text == 'Hello.' and language == 'English'
    assert item['status'] == 'experimental' and item['verified'] is False
    assert item['identity_verified'] is item['overlap_verified'] is False
    assert 'score' not in item and session.references['spk1'].text == 'Hello.'
    assert len(session.references['spk1'].audio) == 14400
    assert provider.consider(ledger, result) is None  # Existing refs never replaced.


def test_bad_alignment_does_not_create_reference_or_abort():
    session, ledger, result = prepared()
    provider = QwenAutoReference('fake', model=FakeAligner(last=1.1))
    assert provider.consider(ledger, result) is None
    assert session.references == {}
    fake = FakeAligner()
    fake.align = lambda **kw: (_ for _ in ()).throw(RuntimeError('align failed'))
    error = QwenAutoReference('fake', model=fake).consider(ledger, result)
    assert error['status'] == 'error' and not error['verified']
    assert not session.failed and not session.references


def test_unsuitable_multiline_reference_is_skipped_without_disabling_provider():
    session = StreamSession(ReplayBackend(['<|DIRECT|><en><spk1>Hello.\nNext.']), StreamOptions())
    ledger = SessionLedger(session)
    result = session.step(np.full(8000, .1, dtype=np.float32))
    ledger.record(result)
    provider = QwenAutoReference('fake', model=FakeAligner())
    assert provider.prepare(ledger, result) is None
    assert not session.failed


def test_background_snapshot_is_causal_and_correction_invalidates_even_when_reverted():
    session, ledger, result = prepared()
    session.speakers.append('spk2')
    provider = QwenAutoReference('fake', model=FakeAligner())
    snapshot = provider.prepare(ledger, result)
    assert snapshot['audio'].flags.writeable is False
    candidate = provider.evaluate(snapshot)
    assert candidate['status'] == 'candidate'
    assert not session.references
    ledger.correct_segment(1, 'spk2')
    ledger.correct_segment(1, 'spk1')
    assert provider.apply(ledger, candidate) is None
    assert session.references == {}


def test_background_candidate_does_not_override_manual_reference_or_closed_session():
    session, ledger, result = prepared()
    provider = QwenAutoReference('fake', model=FakeAligner())
    candidate = provider.evaluate(provider.prepare(ledger, result))
    session.register_reference('spk1', 'en', 'Hello.', .05, .95,
                               verified=True, clean=True, overlap=False)
    assert provider.apply(ledger, candidate) is None
    assert 'spk1' not in ledger.reference_evidence
    session.close()
    assert provider.apply(ledger, candidate) is None
