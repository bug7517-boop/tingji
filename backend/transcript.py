"""Reading paragraphs are views; timed ASR segments remain the source of truth."""
import re
import hashlib
import json


def join_text(parts):
    result = ''
    for text in parts:
        if result and re.search(r'[A-Za-z0-9]$', result) and re.match(r'[A-Za-z0-9]', text):
            result += ' '
        result += text
    return result


def paragraphs(segments, seconds=30):
    result = []
    for index, segment in enumerate(segments):
        ids = segment.get('speakers') or ([segment['speaker']] if segment.get('speaker') else [])
        signature = (tuple(sorted(ids)), 'speaker' in segment, bool(segment.get('quality_review')))
        if (not result or not seconds or result[-1]['signature'] != signature or
                segment['end'] - result[-1]['start'] > seconds or segment['start'] - result[-1]['end'] > 3 or
                len(result[-1]['parts']) >= 50):
            result.append({**segment, 'parts': [], 'signature': signature})
        group = result[-1]
        group['parts'].append({**segment, 'segmentIndex': index})
        group['end'] = segment['end']
        group['text'] = join_text(p['text'] for p in group['parts'])
        if group.get('language') != segment.get('language'):
            group['language'] = 'mixed'
    return result


def revision(segments):
    values = [(s['start'], s['end'], s['text']) for s in segments]
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode('utf-8')).hexdigest()


def apply_text(job, changes):
    changed = False
    for segment, text in changes:
        if segment['text'] != text:
            from .transcription import text_language
            segment.setdefault('text_original', segment['text'])
            segment.update(text=text, text_manual=text != segment['text_original'], language=text_language(text))
            changed = True
    if changed:
        job.update(summary_parts=[], summary_stale=bool(job.get('summary')))
    return changed
