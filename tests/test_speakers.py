import copy
import queue
import threading

from fastapi.testclient import TestClient
import pytest

from backend import app as service, audio, speakers


def test_global_ids_and_ambiguous_turns_preserve_transcript():
    segments = [{'start': 1, 'end': 3, 'text': '中文 English'},
                {'start': 5, 'end': 9, 'text': '交接內容'},
                {'start': 3600, 'end': 3602, 'text': '同一位再發言'},
                {'start': 3700, 'end': 3701, 'text': '未對齊'}]
    original = copy.deepcopy(segments)
    turns = [{'start': 0, 'end': 6, 'speaker': 'speaker_1'},
             {'start': 6, 'end': 10, 'speaker': 'speaker_2'},
             {'start': 3599, 'end': 3603, 'speaker': 'speaker_1'}]
    result = speakers.label_segments(segments, turns)
    assert [s['speaker'] for s in result] == ['speaker_1', None, 'speaker_1', None]
    assert set(result[1]['speakers']) == {'speaker_1', 'speaker_2'}
    assert result[3]['speakers'] == []
    assert segments == original
    assert [{k: s[k] for k in ('start', 'end', 'text')} for s in result] == original
    assert '請回聽' in speakers.label(result[1])


def test_speaker_exports_with_and_without_timestamps():
    segments = [{'start': 1, 'end': 3, 'text': '繁體 English', 'speaker': 'speaker_2'}]
    assert audio.export_transcript(segments, 'txt', False) == '【講者 2】繁體 English\n'
    assert '[00:00:01' in audio.export_transcript(segments, 'txt')
    srt = audio.export_transcript(segments, 'srt', False)
    assert '00:00:01,000 --> 00:00:03,000' in srt and '【講者 2】繁體 English' in srt


@pytest.fixture
def stored(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'JOBS', tmp_path / 'jobs')
    monkeypatch.setattr(service, 'TASKS', queue.Queue())
    job_id = 'd' * 32
    service.directory(job_id).mkdir(parents=True)
    service.write({'id': job_id, 'status': 'done', 'filename': '舊錄音.wav', 'source': 'source.wav',
                   'created_at': '2026-11-16', 'model': 'small', 'language': 'zh', 'progress': 100,
                   'duration': 10, 'transcript_done': True, 'segments': [{'start': 1, 'end': 3, 'text': '繁體 English'}],
                   'auto_summary': True, 'summary': '原本摘要', 'summary_parts': [], 'summary_error': None})
    return job_id


def test_missing_models_and_busy_requests_do_not_mutate(stored, monkeypatch):
    client = TestClient(service.app)
    original = service.read(stored)
    monkeypatch.setattr(speakers, 'ready', lambda: False)
    assert client.post(f'/api/jobs/{stored}/speakers', json={'count': 2}).status_code == 409
    assert service.read(stored) == original
    monkeypatch.setattr(speakers, 'ready', lambda: True)
    assert client.post(f'/api/jobs/{stored}/speakers', json={'count': 21}).status_code == 422
    service.update(stored, status='diarizing')
    assert client.post(f'/api/jobs/{stored}/speakers', json={'count': 2}).status_code == 409


def test_new_noise_upload_persists_speaker_settings(stored, monkeypatch, tmp_path):
    monkeypatch.setattr(service, 'MODELS', tmp_path / 'models')
    (service.MODELS / 'small').mkdir(parents=True)
    (service.MODELS / 'small' / 'model.bin').touch()
    monkeypatch.setattr(speakers, 'ready', lambda: False)
    client = TestClient(service.app)
    data = {'diarization': 'true', 'speaker_count': '4', 'noise_mode': 'true'}
    assert client.post('/api/jobs', files={'file': ('test.wav', b'x')}, data=data).status_code == 409
    monkeypatch.setattr(speakers, 'ready', lambda: True)
    result = client.post('/api/jobs', files={'file': ('test.wav', b'x')}, data=data)
    assert result.status_code == 201
    job = service.read(result.json()['id'])
    assert job['diarization'] and job['speaker_count'] == 4 and job['workflow'] == 'preview'


def test_old_job_speakers_only_preserves_summary_and_text(stored, monkeypatch):
    client = TestClient(service.app)
    monkeypatch.setattr(speakers, 'ready', lambda: True)
    monkeypatch.setattr(service, 'ensure_normalized', lambda *args: 'audio.wav')
    monkeypatch.setattr(speakers, 'detect', lambda *args: [{'start': 0, 'end': 4, 'speaker': 'speaker_1'}])
    assert client.post(f'/api/jobs/{stored}/speakers', json={'count': 2}).status_code == 200
    assert service.TASKS.get_nowait() == (stored, 'speakers')
    service.process(stored, 'speakers')
    result = service.read(stored)
    assert result['status'] == 'done' and result['speakers_done']
    assert result['summary'] == '原本摘要'
    assert result['segments'][0]['text'] == '繁體 English'
    assert result['segments'][0]['speaker'] == 'speaker_1'


def test_speaker_failure_preserves_text(stored, monkeypatch):
    service.update(stored, diarization=True, auto_summary=False)
    monkeypatch.setattr(service, 'ensure_normalized', lambda *args: 'audio.wav')
    monkeypatch.setattr(speakers, 'detect', lambda *args: (_ for _ in ()).throw(RuntimeError('模型無法載入')))
    service.process(stored, 'speakers')
    result = service.read(stored)
    assert result['status'] == 'done' and '模型無法載入' in result['speaker_error']
    assert result['segments'][0]['text'] == '繁體 English' and result['summary'] == '原本摘要'


def test_pause_resumes_saved_turns_without_repeating_inference(stored, monkeypatch):
    turns = [{'start': 0, 'end': 4, 'speaker': 'speaker_1'}]
    service.update(stored, diarization=True, speaker_turns=turns)
    service.STOP_EVENTS[stored] = threading.Event()
    service.STOP_EVENTS[stored].set()
    service.process(stored, 'speakers')
    assert service.read(stored)['status'] == 'paused'
    monkeypatch.setattr(service, 'ensure_normalized', lambda *args: 'audio.wav')
    monkeypatch.setattr(speakers, 'detect', lambda *args: pytest.fail('must reuse saved turns'))
    service.process(stored, 'speakers')
    assert service.read(stored)['speakers_done']


def test_rename_persists_and_all_exports_use_names(stored):
    segments = [{'start': 1, 'end': 3, 'text': '繁體 English', 'speaker': 'speaker_1', 'speakers': ['speaker_1']},
                {'start': 3, 'end': 5, 'text': '交接原文', 'speaker': None, 'speakers': ['speaker_1', 'speaker_2']}]
    service.update(stored, segments=segments, speakers_done=True)
    client = TestClient(service.app)
    path = f'/api/jobs/{stored}/speaker-names'
    assert client.patch(path, json={'names': {'speaker_1': '  王小明 ', 'speaker_2': 'Alice'}}).status_code == 200
    assert service.read(stored)['speaker_names'] == {'speaker_1': '王小明', 'speaker_2': 'Alice'}
    assert client.get(f'/api/jobs/{stored}').json()['speaker_names']['speaker_1'] == '王小明'
    for kind in ('txt', 'srt'):
        for timestamps in ('true', 'false'):
            body = client.get(f'/api/jobs/{stored}/export/{kind}?timestamps={timestamps}').text
            assert '【王小明】繁體 English' in body
            assert '王小明／Alice（交接或重疊，請回聽）' in body
            assert '講者 1' not in body and '講者 2' not in body
    assert service.read(stored)['segments'] == segments
    assert service.read(stored)['summary'] == '原本摘要'
    assert client.patch(path, json={'names': {'speaker_1': ''}}).status_code == 200
    body = client.get(f'/api/jobs/{stored}/export/txt').text
    assert '講者 1' in body and 'Alice' in body


def test_rename_validation_is_atomic_and_names_are_per_meeting(stored):
    service.update(stored, segments=[{'start': 1, 'end': 2, 'text': '原文', 'speaker': 'speaker_1'}], speaker_names={'speaker_1': '原姓名'})
    client = TestClient(service.app)
    path = f'/api/jobs/{stored}/speaker-names'
    for names in ({'speaker_9': '不存在'}, {'speaker_1': 'x'*51}, {'speaker_1': 'Alice\n'}, {'speaker_1': 'A\x00B'}):
        assert client.patch(path, json={'names': names}).status_code == 400
        assert service.read(stored)['speaker_names'] == {'speaker_1': '原姓名'}
    other = {**service.read(stored), 'id': 'e'*32, 'speaker_names': {}}
    service.directory(other['id']).mkdir()
    service.write(other)
    assert client.patch(path, json={'names': {'speaker_1': '新姓名'}}).status_code == 200
    assert service.read(other['id'])['speaker_names'] == {}
    service.update(stored, status='diarizing')
    assert client.patch(path, json={'names': {'speaker_1': '不可改'}}).status_code == 409


def test_rerun_clears_names_when_new_groups_are_published(stored, monkeypatch):
    service.update(stored, diarization=True, speaker_names={'speaker_1': '舊姓名'})
    monkeypatch.setattr(service, 'ensure_normalized', lambda *args: 'audio.wav')
    monkeypatch.setattr(speakers, 'detect', lambda *args: [{'start': 0, 'end': 4, 'speaker': 'speaker_1'}])
    service.process(stored, 'speakers')
    assert service.read(stored)['speaker_names'] == {}
