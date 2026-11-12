"""Session-local display identities and evidence-gated speaker references.

Model IDs remain unchanged: corrections apply to the UI mapping, while cached
references affected by a correction are invalidated. This does not rewrite
prior model context or establish the ground truth of an estimated reference.
"""
from __future__ import annotations

import math
import numpy as np

from .runtime_protocol import speaker_reference_text
from .streaming import SAMPLE_RATE, SpeakerReference


class SessionLedger:
    def __init__(self, session):
        self.session = session
        self.segments = []
        self.aliases = {}
        self.names = {}
        self.reference_evidence = {}
        self.identity_revision = 0

    def resolve(self, speaker):
        seen = set()
        while speaker in self.aliases:
            if speaker in seen:
                raise ValueError('Speaker alias cycle')
            seen.add(speaker)
            speaker = self.aliases[speaker]
        return speaker

    def _known(self, speaker):
        if speaker not in self.session.speakers:
            raise ValueError('Unknown model speaker ID')
        return self.resolve(speaker)

    def _invalidate(self, *speakers):
        # A wrong identity must never remain in the acoustic prompt.
        for speaker in set(speakers):
            self.session.references.pop(speaker, None)
            self.reference_evidence.pop(speaker, None)

    def record(self, result):
        turn = result.get('turn')
        if not isinstance(turn, int) or isinstance(turn, bool) or turn < 1:
            raise ValueError('Invalid result turn')
        if result.get('action') == 'WAIT':
            return []
        if result.get('action') != 'DIRECT':
            raise ValueError('Unsupported result action')
        added = []
        for i, segment in enumerate(result.get('segments', [])):
            speaker = segment['speaker']
            display = self._known(speaker)
            item = {'segment_id': len(self.segments) + 1, 'turn': turn, 'segment_index': i,
                    'model_speaker': speaker, 'display_speaker': display,
                    'language': segment['language'], 'text': segment['text']}
            self.segments.append(item)
            added.append(dict(item))
        return added

    def snapshot(self):
        speakers = []
        for raw in self.session.speakers:
            speaker = self.resolve(raw)
            if not any(x['speaker'] == speaker for x in speakers):
                speakers.append({'speaker': speaker, 'name': self.names.get(speaker, speaker)})
        return {'segments': [dict(s) for s in self.segments], 'speakers': speakers,
                'aliases': dict(self.aliases), 'reference_evidence': dict(self.reference_evidence)}

    def correct_segment(self, segment_id, speaker):
        if isinstance(segment_id, bool) or not isinstance(segment_id, int) or not 1 <= segment_id <= len(self.segments):
            raise ValueError('Unknown segment ID')
        target = self._known(speaker)
        item = self.segments[segment_id - 1]
        old = item['display_speaker']
        if old != target:
            self._invalidate(old, target, item['model_speaker'])
            item['display_speaker'] = target
            self.identity_revision += 1
        return self.snapshot()

    def merge_speakers(self, source, target):
        source, target = self._known(source), self._known(target)
        if source == target:
            raise ValueError('Source and target already refer to the same speaker')
        self.aliases[source] = target
        self.identity_revision += 1
        for item in self.segments:
            item['display_speaker'] = self.resolve(item['display_speaker'])
        self._invalidate(source, target)
        self.names.pop(source, None)
        return self.snapshot()

    def rename_speaker(self, speaker, name):
        speaker = self._known(speaker)
        if not isinstance(name, str) or not name.strip() or len(name) > 80 or any(c in name for c in '\r\n'):
            raise ValueError('Name must be a nonempty single line of at most 80 characters')
        self.names[speaker] = name.strip()
        return self.snapshot()

    def consider_reference(self, segment_id, evidence, *, min_alignment=.9, min_identity=.9):
        """Optionally accept a candidate from independent online evidence.

        `evidence` MUST originate in external, causal text/audio alignment and
        single-speaker identity/overlap detection. Both confidence scores are
        estimates, not ground truth; thresholds need domain calibration. This
        method supplies no detector. Returns None when evidence is absent/weak.
        Manual/trusted references are never silently overwritten.
        """
        if isinstance(segment_id, bool) or not isinstance(segment_id, int) or not 1 <= segment_id <= len(self.segments):
            raise ValueError('Unknown segment ID')
        if not isinstance(evidence, dict) or evidence.get('source') != 'independent_online':
            return None
        if evidence.get('single_speaker') is not True or evidence.get('overlap') is not False:
            return None
        values = [evidence.get(k) for k in ('start', 'end', 'alignment_confidence', 'identity_confidence')]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            return None
        start, end, alignment, identity = values
        item = self.segments[segment_id - 1]
        speaker = item['model_speaker']
        if (self.resolve(speaker) != item['display_speaker'] or self.resolve(speaker) != speaker
                or evidence.get('speaker') != speaker or evidence.get('language') != item['language']
                or evidence.get('text') != item['text'] or not 0 <= start < end <= self.session.heard_seconds
                or not 0 <= alignment <= 1 or not 0 <= identity <= 1
                or alignment < min_alignment or identity < min_identity
                or not .8 <= end - start <= self.session.options.max_reference_seconds):
            return None
        if speaker in self.session.references and speaker not in self.reference_evidence:
            return None
        score = min(alignment, identity)
        if speaker in self.reference_evidence and score <= self.reference_evidence[speaker].get('score', -1):
            return None
        total = sum(r.end-r.start for s, r in self.session.references.items() if s != speaker) + end-start
        if total > self.session.options.max_total_reference_seconds:
            return None
        pieces, cursor = [], start
        for turn in self.session.history:
            low, high = max(start, turn.start), min(end, turn.end)
            if high <= low:
                continue
            if abs(low-cursor) > 1/SAMPLE_RATE:
                return None
            pieces.append(turn.audio[round((low-turn.start)*SAMPLE_RATE):round((high-turn.start)*SAMPLE_RATE)])
            cursor = high
        if not pieces or abs(cursor-end) > 1/SAMPLE_RATE:
            return None
        audio = np.concatenate(pieces).astype(np.float32, copy=True)
        if not len(audio) or not np.isfinite(audio).all():
            return None
        rms = float(np.sqrt(np.mean(audio.astype(np.float64)**2)))
        clipping = float(np.mean(np.abs(audio) >= .999))
        # Detect obvious silence or saturation only; this is NOT overlap or
        # identity verification. Evidence provider must perform those checks.
        if rms < .005 or clipping > .02:
            return None
        speaker_reference_text(speaker, item['language'], item['text'],
                               getattr(self.session.backend, 'allowed_tokens', None))
        self.session.references[speaker] = SpeakerReference(speaker, item['language'], item['text'],
                                                            start, end, audio)
        self.reference_evidence[speaker] = {'status': 'estimated', 'source': 'independent_online',
                                            'score': score, 'start': start, 'end': end,
                                            'segment_id': segment_id}
        return dict(self.reference_evidence[speaker])
