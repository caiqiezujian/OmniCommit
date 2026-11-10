#!/usr/bin/env python3
"""Read-only target-distribution diagnostics for legacy WAIT/DIRECT SFT JSONL.

Only the Python standard library is used. Audio references are decoded as JSON
metadata; source audio, model weights and .tokens.txt files are never opened.
This complements structural/encoded-length audits, not acoustic validation.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path
import re
import sys


WAIT = "<|WAIT|>"
DIRECT = "<|DIRECT|>"
END_OF_AUDIO = "<end_of_audio>"
VIRTUAL_PREFIX = "omnicommitaudio:v1:"
HEADER = re.compile(r"<([a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*)><(spk[1-9][0-9]*)>")


def finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + ' must be finite')
    return float(value)


def audio_range(reference):
    """Return actual source interval, or None for nonvirtual audio paths."""
    if not isinstance(reference, str) or not reference:
        raise ValueError('audio reference must be a nonempty string')
    if not reference.startswith(VIRTUAL_PREFIX):
        return None
    try:
        raw = base64.b64decode(reference[len(VIRTUAL_PREFIX):], altchars=b'-_', validate=True)
        value = json.loads(raw.decode('utf-8'))
        if not isinstance(value, dict):
            raise ValueError('virtual payload is not an object')
        start = finite_number(value.get('start'), 'virtual.start')
        end = finite_number(value.get('end'), 'virtual.end')
        if start < 0 or end <= start:
            raise ValueError('virtual interval must be positive')
        return start, end
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ValueError('invalid virtual audio reference: ' + str(exc)) from exc


def parse_target(content):
    if content == WAIT:
        return 'WAIT', []
    if not isinstance(content, str) or not content.startswith(DIRECT):
        raise ValueError('assistant must be exactly WAIT or DIRECT with attributed text')
    body = content[len(DIRECT):]
    matches = list(HEADER.finditer(body))
    if not matches or matches[0].start() != 0:
        raise ValueError('DIRECT needs a complete initial language/speaker header')
    parts = []
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        text = body[match.end():stop]
        if not text.strip() or '<' in text or '>' in text:
            raise ValueError('DIRECT contains empty or malformed attributed text')
        parts.append((match.group(1), text))
    return 'DIRECT', parts


def diagnose_row(row):
    if not isinstance(row, dict):
        raise ValueError('row must be a JSON object')
    messages, refs = row.get('messages'), row.get('audios')
    if not isinstance(messages, list) or len(messages) < 3 or len(messages) % 2 != 1:
        raise ValueError('expected system and user/assistant pairs')
    if not isinstance(refs, list) or not refs:
        raise ValueError('audios must be a nonempty list')
    if not isinstance(messages[0], dict) or messages[0].get('role') != 'system':
        raise ValueError('first message must be system')
    if row.get('speaker_references'):
        raise ValueError('speaker-reference conditioning is outside this compact-row diagnostic')
    ranges = [audio_range(ref) for ref in refs]
    timing_available = all(value is not None for value in ranges)
    if timing_available:
        for earlier, later in zip(ranges, ranges[1:]):
            if abs(earlier[1] - later[0]) > 1e-6:
                raise ValueError('live virtual audio ranges are not contiguous')
    counts = {'audio_turns': 0, 'audio_wait_turns': 0, 'audio_direct_turns': 0,
              'eos_turns': 0, 'eos_direct_turns': 0, 'text_characters': 0,
              'eos_text_characters': 0, 'language_text_characters': {},
              'language_lexical_characters': {}, 'timing_available': timing_available,
              'first_audio_direct_seconds': None, 'longest_wait_seconds': None}
    wait_run = longest_wait = 0.0
    seen_eos = False
    for offset in range(1, len(messages), 2):
        user, assistant = messages[offset:offset + 2]
        if not isinstance(user, dict) or not isinstance(assistant, dict):
            raise ValueError('messages must be objects')
        if user.get('role') != 'user' or assistant.get('role') != 'assistant':
            raise ValueError('expected alternating user/assistant roles')
        action, parts = parse_target(assistant.get('content'))
        content = user.get('content')
        if content == END_OF_AUDIO:
            if seen_eos or offset != len(messages) - 2 or counts['audio_turns'] != len(refs):
                raise ValueError('end_of_audio must occur once, after all audio, in the final pair')
            seen_eos = True
            counts['eos_turns'] += 1
            counts['eos_direct_turns'] += action == 'DIRECT'
            counts['eos_text_characters'] += sum(len(text) for _, text in parts)
        elif content == '<audio>':
            if seen_eos or counts['audio_turns'] >= len(refs):
                raise ValueError('audio message/reference count mismatch')
            index = counts['audio_turns']
            counts['audio_turns'] += 1
            counts['audio_wait_turns'] += action == 'WAIT'
            counts['audio_direct_turns'] += action == 'DIRECT'
            if timing_available:
                start, end = ranges[index]
                if action == 'WAIT':
                    wait_run += end - start
                    longest_wait = max(longest_wait, wait_run)
                else:
                    if counts['first_audio_direct_seconds'] is None:
                        counts['first_audio_direct_seconds'] = end - ranges[0][0]
                    wait_run = 0.0
        else:
            raise ValueError('user content must be <audio> or final <end_of_audio>')
        for language, text in parts:
            counts['text_characters'] += len(text)
            counts['language_text_characters'][language] = counts['language_text_characters'].get(language, 0) + len(text)
            lexical = sum(char.isalnum() for char in text)
            counts['language_lexical_characters'][language] = counts['language_lexical_characters'].get(language, 0) + lexical
    if counts['audio_turns'] != len(refs):
        raise ValueError('audio message/reference count mismatch')
    if timing_available:
        counts['longest_wait_seconds'] = longest_wait
    counts['languages'] = sorted(counts['language_text_characters'])
    counts['all_text_at_eos'] = bool(counts['text_characters'] and counts['text_characters'] == counts['eos_text_characters'])
    return counts


def empty_metrics():
    return {'rows': 0, 'audio_turns': 0, 'audio_wait_turns': 0, 'audio_direct_turns': 0,
            'eos_turns': 0, 'eos_direct_turns': 0, 'text_characters': 0,
            'eos_text_characters': 0, 'all_text_at_eos_rows': 0,
            'rows_without_audio_direct': 0, 'rows_with_timing': 0,
            '_first': {'count': 0, 'sum': 0., 'min': None, 'max': None},
            '_wait': {'count': 0, 'sum': 0., 'min': None, 'max': None}}


def update_distribution(values, value):
    values['count'] += 1
    values['sum'] += value
    values['min'] = value if values['min'] is None else min(values['min'], value)
    values['max'] = value if values['max'] is None else max(values['max'], value)


def update_metrics(metrics, row):
    metrics['rows'] += 1
    for key in ('audio_turns', 'audio_wait_turns', 'audio_direct_turns', 'eos_turns',
                'eos_direct_turns', 'text_characters', 'eos_text_characters'):
        metrics[key] += row[key]
    metrics['all_text_at_eos_rows'] += row['all_text_at_eos']
    metrics['rows_without_audio_direct'] += row['audio_direct_turns'] == 0
    metrics['rows_with_timing'] += row['timing_available']
    if row['first_audio_direct_seconds'] is not None:
        update_distribution(metrics['_first'], row['first_audio_direct_seconds'])
    if row['longest_wait_seconds'] is not None:
        update_distribution(metrics['_wait'], row['longest_wait_seconds'])


def distribution(values):
    return {'count': values['count'], 'min': values['min'],
            'mean': values['sum'] / values['count'] if values['count'] else None,
            'max': values['max']}


def finish_metrics(metrics):
    result = {key: value for key, value in metrics.items() if not key.startswith('_')}
    result['audio_wait_ratio'] = metrics['audio_wait_turns'] / metrics['audio_turns'] if metrics['audio_turns'] else None
    result['eos_text_character_ratio'] = metrics['eos_text_characters'] / metrics['text_characters'] if metrics['text_characters'] else None
    result['first_audio_direct_seconds'] = distribution(metrics['_first'])
    result['longest_wait_seconds_per_row'] = distribution(metrics['_wait'])
    return result


def scan_dataset(path, max_details=20):
    total = empty_metrics()
    groups = {'monolingual': {}, 'mixed': {}, 'unattributed': {}}
    languages = {}
    english = {'rows_with_en': 0, 'monolingual_rows': 0, 'mixed_rows': 0,
               'text_characters': 0, 'lexical_characters': 0}
    read = invalid = blanks = 0
    errors = []
    with Path(path).open('rb') as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                blanks += 1
                continue
            read += 1
            try:
                def bad_constant(value):
                    raise ValueError('nonfinite JSON constant: ' + value)
                value = json.loads(raw.decode('utf-8-sig' if line_number == 1 else 'utf-8'), parse_constant=bad_constant)
                row = diagnose_row(value)
            except (ValueError, TypeError, KeyError, UnicodeError) as exc:
                invalid += 1
                if len(errors) < max_details:
                    errors.append({'line': line_number, 'error': str(exc)})
                continue
            update_metrics(total, row)
            names = row['languages']
            kind = 'monolingual' if len(names) == 1 else 'mixed' if names else 'unattributed'
            key = '+'.join(names) if names else 'no_attributed_target_text'
            update_metrics(groups[kind].setdefault(key, empty_metrics()), row)
            for language in names:
                counts = languages.setdefault(language, {'rows_with_text': 0, 'text_characters': 0, 'lexical_characters': 0})
                counts['rows_with_text'] += 1
                counts['text_characters'] += row['language_text_characters'][language]
                counts['lexical_characters'] += row['language_lexical_characters'][language]
            if 'en' in names:
                english['rows_with_en'] += 1
                english['monolingual_rows'] += kind == 'monolingual'
                english['mixed_rows'] += kind == 'mixed'
                english['text_characters'] += row['language_text_characters']['en']
                english['lexical_characters'] += row['language_lexical_characters']['en']
    return {'schema_version': 1, 'dataset': str(Path(path).resolve()),
            'rows_read': read, 'rows_valid': total['rows'], 'rows_invalid': invalid,
            'blank_lines': blanks, 'total': finish_metrics(total),
            'groups': {kind: {key: finish_metrics(value) for key, value in sorted(values.items())}
                       for kind, values in groups.items()},
            'languages': dict(sorted(languages.items())), 'english': english,
            'errors': errors, 'error_details_limit': max_details,
            'notes': ['Target labels, not audio-path language, determine language groups.',
                      'Labels are annotations, not acoustic language verification. Latin script is not assumed to be English.',
                      'Language/text statistics cover protocol-valid rows; inspect invalid-row counts before drawing distribution conclusions.',
                      'Mixed-row WAITs are not attributed to one language.',
                      'Timing uses virtual source intervals; padding is excluded. Nonvirtual paths have no timing metrics.',
                      'Text characters include original spaces and punctuation; lexical characters use str.isalnum().',
                      'No audio decoding, model loading, token sidecar access or dataset rewriting.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--max-details', type=int, default=20)
    args = parser.parse_args(argv)
    if not args.dataset.is_file():
        parser.error('dataset does not exist')
    if args.max_details < 0:
        parser.error('--max-details must be nonnegative')
    if args.report and (args.report.exists() or args.report.resolve() == args.dataset.resolve()):
        parser.error('report must be a new file distinct from dataset')
    report = scan_dataset(args.dataset, args.max_details)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('x', encoding='utf-8') as stream:
            stream.write(serialized + '\n')
    print(serialized)
    return 0 if report['rows_valid'] and not report['rows_invalid'] else 1


if __name__ == '__main__':
    sys.exit(main())
