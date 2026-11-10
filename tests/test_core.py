import json
from pathlib import Path
import threading
import wave

from fastapi.testclient import TestClient
import numpy as np
import pytest

from backend import app as service
from backend import audio, summary


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'JOBS', tmp_path / 'jobs')
    monkeypatch.setattr(service, 'MODELS', tmp_path / 'models')
    (service.MODELS / 'small').mkdir(parents=True)
    (service.MODELS / 'small' / 'model.bin').touch()
    # Exercise API queue semantics without triggering model inference in unit tests.
    monkeypatch.setattr(service, 'process', lambda *args: None)
    with TestClient(service.app) as connection:
        yield connection


def create(client, content=b'not a real wav', name='會議.wav'):
    response = client.post('/api/jobs', files={'file': (name, content)},
                           data={'language': 'zh', 'model': 'small', 'auto_summary': 'false'})
    assert response.status_code == 201
    return response.json()['id']


def test_formats_and_origin_rejected(client):
    assert client.post('/api/jobs', files={'file': ('bad.exe', b'x')}).status_code == 400
    assert client.post('/api/jobs', files={'file': ('empty.wav', b'')}).status_code == 400
    assert client.post('/api/jobs', files={'file': ('test.wav', b'x')}, headers={'Origin': 'https://evil.example'}).status_code == 403
    assert client.get('/api/health', headers={'Host': 'evil.example'}).status_code == 400
    assert client.get('/api/jobs/not-an-id').status_code == 404


def test_export_resume_delete_and_persistence(client):
    job_id = create(client)
    assert client.post(f'/api/jobs/{job_id}/resume').status_code == 409
    assert client.delete(f'/api/jobs/{job_id}').status_code == 409
    service.update(job_id, status='paused', segments=[{'start': 3600.125, 'end': 3602.5, 'text': '下週確認預算。'}], completed_chunks=12)
    response = client.get(f'/api/jobs/{job_id}/export/srt')
    assert response.status_code == 200
    assert '01:00:00,125 --> 01:00:02,500' in response.text
    assert "filename*=UTF-8''" in response.headers['content-disposition']
    assert '下週確認預算。' in client.get(f'/api/jobs/{job_id}/export/txt').text
    assert client.post(f'/api/jobs/{job_id}/resume').status_code == 200
    assert service.read(job_id)['completed_chunks'] == 12
    assert service.read(job_id)['segments'][0]['start'] == 3600.125
    service.update(job_id, status='done', transcript_done=True)
    assert client.get(f'/api/jobs/{job_id}/export/summary').status_code == 409
    assert client.delete(f'/api/jobs/{job_id}').status_code == 200
    assert client.get(f'/api/jobs/{job_id}').status_code == 404


def make_wav(path, seconds=1, rate=8000):
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(np.zeros(round(seconds * rate), dtype=np.int16).tobytes())


def test_real_audio_resampling_and_chunks(tmp_path):
    source, target = tmp_path / 'source.wav', tmp_path / 'audio.wav'
    make_wav(source, 2)
    assert audio.normalize(source, target) == 2
    chunk, offset = audio.read_chunk(target, 0)
    assert offset == 0 and len(chunk) == 32000
    assert chunk.dtype == np.float32


def test_duration_guard_checks_decoded_samples(tmp_path, monkeypatch):
    source, target = tmp_path / 'source.wav', tmp_path / 'audio.wav'
    make_wav(source, 2)
    monkeypatch.setattr(audio, 'MAX_SECONDS', 1)
    with pytest.raises(ValueError, match='3 小時'):
        audio.normalize(source, target)
    assert not target.exists()


def test_three_hour_boundary_chunk_has_correct_offset(tmp_path):
    # Sparse 3-hour PCM file: exercise the actual final chunk without RAM allocation.
    path = tmp_path / 'three-hours.wav'
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(audio.SAMPLE_RATE)
        out.setnframes(audio.MAX_SECONDS * audio.SAMPLE_RATE)
        out.writeframesraw(b'')
    with path.open('r+b') as stream:
        length = audio.MAX_SECONDS * audio.SAMPLE_RATE * 2
        stream.seek(4)
        stream.write((36 + length).to_bytes(4, 'little'))
        stream.seek(40)
        stream.write(length.to_bytes(4, 'little'))
        stream.truncate(44 + length)
    chunk, offset = audio.read_chunk(path, 35)
    assert offset == 10498
    assert len(chunk) == 302 * audio.SAMPLE_RATE
    assert audio.timecode(offset) == '02:54:58'


def test_transcript_language_classification():
    from opencc import OpenCC
    from backend.transcription import text_language
    text = OpenCC('s2twp').convert('我们下周 review API budget，Alice will send the report by Friday.')
    assert text == '我們下週 review API budget，Alice will send the report by Friday.'
    assert text in audio.export_transcript([{'start': 0, 'end': 3, 'text': text}], 'srt')
    assert text_language(text) == 'mixed'
    assert text_language('我們下週確認預算。') == 'zh'
    assert text_language('Alice will send the report.') == 'en'


@pytest.mark.parametrize('language,expected,detect', [('zh', None, True), ('zh-only', 'zh', False), ('en', 'en', False)])
def test_language_modes_preserve_region_offsets(monkeypatch, language, expected, detect):
    from types import SimpleNamespace
    import faster_whisper.vad
    from backend.transcription import transcribe_regions
    monkeypatch.setattr(faster_whisper.vad, 'get_speech_timestamps',
                        lambda *_: [{'start': 16000, 'end': 32000}, {'start': 48000, 'end': 64000}])
    calls = []
    class Model:
        def transcribe(self, pcm, **kwargs):
            calls.append(kwargs)
            return iter([SimpleNamespace(start=0, end=1, text='test')]), None
    result = list(transcribe_regions(Model(), np.zeros(64000, dtype=np.float32), language))
    assert [s[:2] for s in result] == [(1, 2), (3, 4)]
    assert all(c['language'] == expected and c['multilingual'] == detect and c['task'] == 'transcribe' for c in calls)


def test_plain_txt_export_and_srt_keep_original_text_and_timing(client):
    job_id = create(client)
    segments = [{'start': 60.5, 'end': 62, 'text': '我們下週確認預算。'},
                {'start': 63, 'end': 65, 'text': 'Alice will review the budget.'}]
    service.update(job_id, status='done', segments=segments, transcript_done=True)
    plain = client.get(f'/api/jobs/{job_id}/export/txt?timestamps=false').text.lstrip('\ufeff')
    assert plain == '我們下週確認預算。\nAlice will review the budget.\n'
    assert '[00:01:00' in client.get(f'/api/jobs/{job_id}/export/txt?timestamps=true').text
    assert '00:01:00,500 --> 00:01:02,000' in client.get(f'/api/jobs/{job_id}/export/srt?timestamps=false').text
    assert service.read(job_id)['segments'] == segments


def test_full_chinese_language_is_accepted_and_saved(client):
    response = client.post('/api/jobs', files={'file': ('中文會議.wav', b'audio')},
                           data={'language': 'zh-only', 'model': 'small', 'auto_summary': 'false'})
    assert response.status_code == 201
    assert service.read(response.json()['id'])['language'] == 'zh-only'


def test_traditional_chinese_conversion_preserves_english():
    from opencc import OpenCC
    text = OpenCC('s2twp').convert('我们下周 review API budget，Alice will send the report by Friday.')
    assert text == '我們下週 review API budget，Alice will send the report by Friday.'
    assert text in audio.export_transcript([{'start': 0, 'end': 3, 'text': text}], 'srt')


def test_summary_covers_all_input_and_resumes_checkpoint():
    segments = [{'start': i * 100, 'text': f'片段{i}：' + '討論預算與時程。' * 250} for i in range(24)]
    calls = []

    class FakeEngine:
        def select_records(self, records, key_only=False):
            calls.append((records, key_only))
            return {'key_points': [r['id'] for r in records], 'decisions': [], 'actions': [], 'questions': []}

    saved = []
    result = summary.summarize(segments, [], lambda p: saved.append(list(p)), lambda _: None, lambda: False, FakeEngine())
    for i in range(24):
        assert any(any(r['id'] == i for r in records) for records, _ in calls)
    assert '## 會議重點' in result
    for line in result.splitlines():
        if line.startswith('- ['):
            assert line.split('] ', 1)[1] in [s['text'] for s in segments]
    checkpoint = saved[-1]
    calls.clear()
    summary.summarize(segments, checkpoint, lambda _: None, lambda _: None, lambda: False, FakeEngine())
    assert all(key_only for _, key_only in calls)


def test_summary_schema_rejects_fabricated_sources():
    assert not summary.valid_selection({'key_points': [2023], 'decisions': [], 'actions': [], 'questions': []}, {0, 1})
    assert not summary.valid_selection({'key_points': ['0'], 'decisions': [], 'actions': [], 'questions': []}, {0})


def test_summary_failure_preserves_transcript(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'JOBS', tmp_path / 'jobs')
    job_id = 'a' * 32
    service.directory(job_id).mkdir(parents=True)
    service.write({'id': job_id, 'status': 'queued', 'transcript_done': True, 'auto_summary': True,
                   'segments': [{'start': 0, 'end': 3, 'text': '測試會議'}], 'summary_parts': []})
    monkeypatch.setattr(summary, 'summarize', lambda *args: (_ for _ in ()).throw(RuntimeError('模型測試失敗')))
    service.process(job_id, 'summary')
    job = service.read(job_id)
    assert job['status'] == 'done'
    assert job['summary_error'] == '模型測試失敗'
    assert job['segments'][0]['text'] == '測試會議'
