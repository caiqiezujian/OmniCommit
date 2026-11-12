"""Experimental automatic references from *past* audio and Qwen alignment.

Qwen forced alignment does not verify transcription, overlap, or identity.
Candidates produced here are explicitly unverified and must never be treated
as trusted commit watermarks. Enable only as an evaluated experiment.
"""
from __future__ import annotations

import math
import threading
import numpy as np

from .forced_alignment import QWEN_LANGUAGES, restore_qwen_spans, strict_qwen_timestamps
from .runtime_protocol import speaker_reference_text
from .streaming import SAMPLE_RATE, SpeakerReference


class QwenAutoReference:
    def __init__(self, model_name, device='cuda:0', dtype='bfloat16', *, model=None, max_window_seconds=5):
        if not model_name or not 1 <= max_window_seconds <= 30:
            raise ValueError('Model path and a bounded alignment window are required')
        self.model_name, self.device, self.dtype = model_name, device, dtype
        self.model, self.max_window_seconds = model, float(max_window_seconds)
        self._lock = threading.Lock()

    def _model(self):
        if self.model is None:
            import torch
            from qwen_asr import Qwen3ForcedAligner
            self.model = Qwen3ForcedAligner.from_pretrained(
                self.model_name, dtype=getattr(torch, self.dtype), device_map=self.device)
            self.model.aligner_processor.fix_timestamp = strict_qwen_timestamps
        return self.model

    def consider(self, ledger, result):
        """Synchronous compatibility helper. Web uses prepare/evaluate/apply."""
        try:
            snapshot = self.prepare(ledger, result)
            return self.apply(ledger, self.evaluate(snapshot)) if snapshot else None
        except Exception as exc:
            return {'status': 'error', 'error': f'{type(exc).__name__}: {exc}',
                    'verified': False, 'provider': 'qwen_forced_alignment_experimental'}

    def prepare(self, ledger, result):
        """Capture an immutable audio snapshot on the session/event-loop thread."""
        session = ledger.session
        segments = result.get('segments', [])
        if result.get('event') != 'audio' or result.get('action') != 'DIRECT' or len(segments) != 1:
            return None
        part = segments[0]
        speaker, language, text = part['speaker'], part['language'], part['text']
        if language not in QWEN_LANGUAGES or speaker in session.references or ledger.resolve(speaker) != speaker:
            return None
        try:
            speaker_reference_text(speaker, language, text)
        except ValueError:
            # A single unsuitable segment should not disable future candidates.
            return None
        if (not ledger.segments or result['turn'] != session.turn_count
                or ledger.segments[-1]['turn'] != result['turn']
                or ledger.segments[-1]['model_speaker'] != speaker
                or ledger.segments[-1]['language'] != language or ledger.segments[-1]['text'] != text):
            return None
        if ledger.segments[-1]['display_speaker'] != speaker or not session.history:
            return None
        # Use only retained, already received samples, with a conservative
        # finite look-back. A preceding DIRECT is not a verified boundary.
        end = session.heard_seconds
        begin = max(session.history[0].start, end - self.max_window_seconds)
        pieces, position = [], begin
        for turn in session.history:
            low, high = max(begin, turn.start), min(end, turn.end)
            if high <= low:
                continue
            if abs(low - position) > 1 / SAMPLE_RATE:
                return None
            if any(pair != (language, speaker) for pair in turn.attributions):
                return None
            pieces.append(turn.audio[round((low-turn.start)*SAMPLE_RATE):round((high-turn.start)*SAMPLE_RATE)])
            position = high
        if not pieces or abs(position-end) > 1 / SAMPLE_RATE:
            return None
        audio = np.concatenate(pieces).astype(np.float32, copy=True)
        if not len(audio) or not np.isfinite(audio).all():
            return None
        audio.setflags(write=False)
        return {'ledger_id': id(ledger), 'session_id': id(session),
                'identity_revision': ledger.identity_revision,
                'segment': dict(ledger.segments[-1]), 'begin': begin, 'heard_end': end,
                'max_reference_seconds': session.options.max_reference_seconds, 'audio': audio}

    def evaluate(self, snapshot):
        """Run alignment on an immutable snapshot; never access live session state."""
        if snapshot is None:
            return None
        try:
            return self._evaluate(snapshot)
        except Exception as exc:
            return {'status': 'error', 'error': f'{type(exc).__name__}: {exc}',
                    'verified': False, 'provider': 'qwen_forced_alignment_experimental'}

    def _evaluate(self, snapshot):
        audio = snapshot['audio']
        part = snapshot['segment']
        text, language = part['text'], part['language']
        # One worker at a time owns this model; loading cannot race either.
        with self._lock:
            aligned = self._model().align(audio=(audio, SAMPLE_RATE), text=text, language=QWEN_LANGUAGES[language])
        if len(aligned) != 1 or not aligned[0].items:
            return None
        words = restore_qwen_spans(text, aligned[0].items)
        previous_end = 0.0
        for word in words:
            start, stop = word['start'], word['end']
            if (not math.isfinite(start) or not math.isfinite(stop) or start < previous_end - 1e-6
                    or stop <= start or stop > len(audio) / SAMPLE_RATE + 1e-6):
                return None
            previous_end = stop
        first, last = words[0]['start'], words[-1]['end']
        if not .8 <= last-first <= snapshot['max_reference_seconds']:
            return None
        crop = audio[round(first*SAMPLE_RATE):round(last*SAMPLE_RATE)].copy()
        rms = float(np.sqrt(np.mean(crop.astype(np.float64) ** 2)))
        clipping = float(np.mean(np.abs(crop) >= .999))
        if rms < .005 or clipping > .02:
            return None
        start, stop = snapshot['begin']+first, snapshot['begin']+last
        if stop > snapshot['heard_end'] + 1e-6:
            return None
        crop.setflags(write=False)
        return {'status': 'candidate', 'snapshot': snapshot, 'audio': crop, 'start': start, 'end': stop}

    def apply(self, ledger, candidate):
        """Commit only if identity, session and reference budgets remain valid."""
        if not candidate or candidate.get('status') != 'candidate':
            return candidate
        session = ledger.session
        snapshot = candidate['snapshot']
        part = snapshot['segment']
        speaker, language, text = part['model_speaker'], part['language'], part['text']
        index = part['segment_id'] - 1
        if (id(ledger) != snapshot['ledger_id'] or id(session) != snapshot['session_id']
                or session.closed or session.failed or ledger.identity_revision != snapshot['identity_revision']
                or not 0 <= index < len(ledger.segments) or ledger.segments[index] != part
                or speaker not in session.speakers or ledger.resolve(speaker) != speaker
                or part['display_speaker'] != speaker or speaker in session.references):
            return None
        start, stop, audio = candidate['start'], candidate['end'], candidate['audio']
        if (not 0 <= start < stop <= min(snapshot['heard_end'], session.heard_seconds) + 1e-6
                or not .8 <= stop-start <= session.options.max_reference_seconds
                or abs(len(audio) / SAMPLE_RATE - (stop-start)) > 2/SAMPLE_RATE
                or sum(ref.end-ref.start for ref in session.references.values()) + stop-start > session.options.max_total_reference_seconds):
            return None
        session.references[speaker] = SpeakerReference(speaker, language, text, start, stop, audio.copy())
        evidence = {'status': 'experimental', 'verified': False,
                    'provider': 'qwen_forced_alignment_experimental',
                    'identity_verified': False, 'overlap_verified': False,
                    'start': start, 'end': stop, 'segment_id': part['segment_id']}
        ledger.reference_evidence[speaker] = evidence
        return dict(evidence)
