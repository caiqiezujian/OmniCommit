#!/usr/bin/env python3
"""Recast in-row ASR timestamps directly into compact audio-interleaved SFT JSONL.

Input: original audios/messages/output plus timestamps and optional tags. Output
contains only audios/messages per row; audio references point to the original
file. No teacher, guessed timestamp, new WAV, or audio after EOS is created.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'SFT-Scripts'))
from omnicommit.alignment import TAG, annotation_units, align_annotation, finite
from omnicommit.recast import CHUNK_SECONDS, recast_row
from prepare_sft import prepare_row, sort_tokens
from check_sft import check_row


def attributed_chars(segments):
    return [(v['language'], v['speaker'], ch) for v in segments for ch in v['text']]


def check_embedded_tags(text, language, speaker):
    if any(match.group(1) not in (language, speaker) for match in TAG.finditer(text)):
        raise ValueError('timestamp text has tags contradicting its language or speaker')


def make_acoustic(row):
    """Validate timing metadata against the existing annotation without rewriting it."""
    segments, _, _ = annotation_units(row)
    times = row.get('timestamps')
    if not isinstance(times, list) or len(times) != len(segments):
        raise ValueError('timestamps count must equal output.segments count')
    if any(not isinstance(t, dict) for t in times):
        raise ValueError('each timestamp must be an object')
    if any('index' in t for t in times):
        indices = [t.get('index') for t in times]
        if any(type(i) is not int for i in indices) or sorted(indices) != list(range(1, len(times)+1)):
            raise ValueError('timestamp index must uniquely cover 1..N')
        times = sorted(times, key=lambda t: t['index'])
    tags = row.get('tags')
    if tags is not None and (not isinstance(tags, list) or tags != ['<|'+s['action']+'|>' for s in segments]):
        raise ValueError('top-level tags disagree with output actions')
    words, timed_spans = [], []
    for index, (timestamp, segment) in enumerate(zip(times, segments), 1):
        if 'source_text' in timestamp and timestamp['source_text'] != segment['source_text']:
            raise ValueError(f'timestamp {index} source_text differs from annotation')
        expected = ''.join(s['text'] for s in segment['attributed'])
        if 'text_plain' in timestamp and timestamp['text_plain'] != expected:
            raise ValueError(f'timestamp {index} text_plain differs from annotation')
        if 'text' in timestamp and TAG.sub('', timestamp['text']) != expected:
            raise ValueError(f'timestamp {index} text differs from annotation')
        begin = finite(timestamp.get('start_ms'), 'timestamp.start_ms')
        end = finite(timestamp.get('end_ms'), 'timestamp.end_ms')
        if not 0 <= begin < end:
            raise ValueError(f'timestamp {index} interval invalid')
        attributed = segment['attributed']
        spans = timestamp.get('spans')
        if spans is None:
            if len(attributed) != 1:
                raise ValueError(f'timestamp {index} needs timed spans for internal attribution change')
            spans = [{'start_ms': begin, 'end_ms': end, **attributed[0]}]
        if not isinstance(spans, list) or not spans:
            raise ValueError(f'timestamp {index} spans must be nonempty')
        mapped = []
        for span in spans:
            if not isinstance(span, dict):
                raise ValueError('timed span must be an object')
            start = finite(span.get('start_ms'), 'span.start_ms')
            stop = finite(span.get('end_ms'), 'span.end_ms')
            if not begin <= start < stop <= end:
                raise ValueError(f'timestamp {index} span outside segment interval')
            plain = span.get('text_plain')
            if plain is None:
                raw = span.get('text')
                if not isinstance(raw, str):
                    raise ValueError('timed span needs text or text_plain')
                plain = TAG.sub('', raw)
            if not isinstance(plain, str) or not plain:
                raise ValueError('timed span text must be nonempty')
            if 'text' in span and (not isinstance(span['text'], str) or TAG.sub('', span['text']) != plain):
                raise ValueError('timed span text disagrees with text_plain')
            lang, speaker = span.get('language'), span.get('speaker')
            if not isinstance(lang, str) or not isinstance(speaker, str):
                raise ValueError('timed span requires language and speaker')
            if 'text' in span:
                check_embedded_tags(span['text'], lang, speaker)
            mapped.append({'start_ms': start, 'end_ms': stop, 'language': lang, 'speaker': speaker, 'text': plain})
            words.append({'text': plain, 'start_ms': start, 'end_ms': stop})
        if attributed_chars(mapped) != attributed_chars(attributed):
            raise ValueError(f'timestamp {index} span text or attribution differs from original')
        if len(attributed) == 1:
            if timestamp.get('language', attributed[0]['language']) != attributed[0]['language'] or timestamp.get('speaker', attributed[0]['speaker']) != attributed[0]['speaker']:
                raise ValueError(f'timestamp {index} attribution differs from original')
            if 'text' in timestamp:
                check_embedded_tags(timestamp['text'], attributed[0]['language'], attributed[0]['speaker'])
        timed_spans.append(mapped)
    return {'words': words, 'backend': 'input-segment-and-span-timestamps'}, timed_spans


def unannotated_gaps(aligned, minimum_ms=2000):
    """Report uncovered timing intervals; they are not necessarily silence."""
    gaps, previous_end = [], 0.0
    intervals = [span for segment in aligned['segments']
                 for span in segment.get('attribution_spans', [segment])]
    for span in sorted(intervals, key=lambda s: s['start_ms']):
        begin, end = span['start_ms'], span['end_ms']
        if begin > previous_end and begin - previous_end >= minimum_ms:
            gaps.append({'start_ms': previous_end, 'end_ms': begin})
        previous_end = max(previous_end, end)
    if aligned['duration_ms'] > previous_end and aligned['duration_ms'] - previous_end >= minimum_ms:
        gaps.append({'start_ms': previous_end, 'end_ms': aligned['duration_ms']})
    return gaps


def build(row, audio_root, chunk_seconds, number_format, diagnostics=None):
    acoustic, spans = make_acoustic(row)
    aligned = align_annotation(row, acoustic, audio_root)
    for segment, timed in zip(aligned['segments'], spans):
        segment['attribution_spans'] = timed
    if diagnostics is not None:
        diagnostics['unannotated_gaps'] = unannotated_gaps(aligned)
        diagnostics['all_unannotated_gaps'] = unannotated_gaps(aligned, minimum_ms=0)
    trajectory = recast_row(aligned, chunk_seconds=chunk_seconds, number_format=number_format)
    prepared, tokens = prepare_row(trajectory, audio_root)
    compact = {'audios': prepared['audios'], 'messages': prepared['messages']}
    check_row(compact, decode=False, token_inventory=tokens)
    return compact, tokens


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Timed annotation JSONL')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--chunk-seconds', type=float, choices=CHUNK_SECONDS, default=.5)
    parser.add_argument('--number-format', choices=('normalized', 'verbatim'), default='normalized')
    parser.add_argument('--audio-root', type=Path)
    parser.add_argument('--max-unannotated-gap-seconds', type=float,
                        help='Optional rejection threshold; disabled by default. Gaps are not assumed silence.')
    args = parser.parse_args(argv)
    tokens_path = Path(str(args.output) + '.tokens.txt')
    reject_path = Path(str(args.output) + '.rejected.jsonl')
    review_path = Path(str(args.output) + '.review.jsonl')
    paths = [args.output, tokens_path, reject_path, review_path]
    if len({p.resolve() for p in [args.input]+paths}) != len(paths)+1:
        parser.error('input, output and sidecars must have distinct paths')
    if not args.input.is_file():
        parser.error('input JSONL does not exist')
    if args.max_unannotated_gap_seconds is not None and (not math.isfinite(args.max_unannotated_gap_seconds) or args.max_unannotated_gap_seconds <= 0):
        parser.error('--max-unannotated-gap-seconds must be positive and finite')
    if any(p.exists() for p in paths):
        parser.error('output or sidecar exists; choose new paths')
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
    counts = {'read': 0, 'kept': 0, 'rejected': 0,
              'rows_with_unannotated_gaps_ge_2s': 0, 'unannotated_gaps_ge_2s': 0}
    gap_examples = []
    inventory = {'<|WAIT|>', '<|DIRECT|>'}
    with args.input.open('rb') as src, args.output.open('x', encoding='utf-8') as out, reject_path.open('x', encoding='utf-8') as rejects, review_path.open('x', encoding='utf-8') as review:
        for line_number, raw in enumerate(src, 1):
            if not raw.strip():
                continue
            counts['read'] += 1
            row = None
            try:
                row = json.loads(raw.decode('utf-8-sig' if line_number == 1 else 'utf-8'),
                                 parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
                if not isinstance(row, dict):
                    raise ValueError('input row must be an object')
                diagnostics = {}
                compact, used = build(row, args.audio_root or args.input.resolve().parent,
                                      args.chunk_seconds, args.number_format, diagnostics)
                serialized = json.dumps(compact, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
                serialized.encode('utf-8')
            except Exception as exc:
                counts['rejected'] += 1
                record = {'line': line_number, 'id': row.get('id', row.get('key')) if isinstance(row, dict) else None,
                          'error': str(exc)}
                rejects.write(json.dumps(record, ensure_ascii=True) + '\n')
                rejects.flush()
                detail = f': {exc}' if isinstance(exc, ModuleNotFoundError) else ''
                print(f'[reject] line={line_number}: {type(exc).__name__}{detail}', file=sys.stderr, flush=True)
                continue
            gaps = diagnostics['unannotated_gaps']
            counts['rows_with_unannotated_gaps_ge_2s'] += bool(gaps)
            counts['unannotated_gaps_ge_2s'] += len(gaps)
            for gap in gaps:
                item = {'line': line_number, 'id': row.get('id', row.get('key')),
                        'audio': row['audios'][0], **gap,
                        'reason': 'unannotated_interval_not_confirmed_silence'}
                review.write(json.dumps(item, ensure_ascii=True) + '\n')
                if len(gap_examples) < 5:
                    gap_examples.append({'line': line_number, **gap})
            if gaps:
                review.flush()
            maximum = args.max_unannotated_gap_seconds
            over_limit = next((g for g in diagnostics['all_unannotated_gaps']
                               if maximum is not None and g['end_ms'] - g['start_ms'] > maximum * 1000), None)
            if over_limit is not None:
                counts['rejected'] += 1
                rejects.write(json.dumps({'line': line_number, 'id': row.get('id', row.get('key')),
                                           'error': 'unannotated interval exceeds configured maximum',
                                           'start_ms': over_limit['start_ms'], 'end_ms': over_limit['end_ms'],
                                           'max_seconds': maximum}, ensure_ascii=True) + '\n')
                rejects.flush()
                continue
            out.write(serialized + '\n')
            out.flush()
            inventory.update(used)
            counts['kept'] += 1
    tokens_path.write_text('\n'.join(sort_tokens(inventory)) + '\n', encoding='utf-8')
    print(json.dumps(dict(counts, unannotated_gap_examples=gap_examples,
                          unannotated_gap_note='Unannotated interval, not verified silence; review audio.',
                          tokens=str(tokens_path), rejected_file=str(reject_path), review_file=str(review_path)), ensure_ascii=False), flush=True)
    return 0 if counts['kept'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
