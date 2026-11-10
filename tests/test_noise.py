import io
import queue
import threading
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import app as service, audio, noise, transcription


def write_wav(path, samples):
    with wave.open(str(path), 'wb') as out:
        out.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        out.writeframes(samples.astype(np.int16).tobytes())


def samples(path):
    with wave.open(str(path), 'rb') as inp:
        return np.frombuffer(inp.readframes(inp.getnframes()), dtype=np.int16)


@pytest.mark.parametrize('profile', ['light', 'standard'])
def test_denoising_keeps_timeline_and_reduces_hum(tmp_path, profile):
    t = np.arange(16000 * 4) / 16000
    source = (2500 * np.sin(2 * np.pi * 40 * t)).astype(np.int16)
    source[32000] = 28000
    inp, out = tmp_path / 'in.wav', tmp_path / 'out.wav'
    write_wav(inp, source)
    before = inp.read_bytes()
    noise.enhance(inp, out, profile)
    processed = samples(out)
    assert len(processed) == len(source)
    assert abs(np.argmax(np.abs(processed)) - 32000) <= 1
    assert np.std(processed[16000:30000]) < np.std(source[16000:30000]) * .8
    assert inp.read_bytes() == before


def test_extract_keeps_exact_original_offset_and_clamps_at_end(tmp_path):
    inp, out = tmp_path / 'in.wav', tmp_path / 'clip.wav'
    source = np.arange(16000 * 8, dtype=np.int16)
    write_wav(inp, source)
    start, duration = noise.extract(inp, out, 5.125, 180)
    assert start == 5.125 and duration == 2.875
    assert np.array_equal(samples(out), source[82000:])
    with pytest.raises(ValueError, match='起點'):
        noise.extract(inp, out, 8, 180)


def test_cancel_denoising_does_not_publish_partial_audio(tmp_path):
    inp, out = tmp_path / 'in.wav', tmp_path / 'out.wav'
    write_wav(inp, np.zeros(16000, dtype=np.int16))
    with pytest.raises(InterruptedError):
        noise.enhance(inp, out, 'standard', cancelled=lambda: True)
    assert not out.exists() and not out.with_suffix('.partial.wav').exists()


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'JOBS', tmp_path / 'jobs')
    monkeypatch.setattr(service, 'MODELS', tmp_path / 'models')
    monkeypatch.setattr(service, 'TASKS', queue.Queue())
    monkeypatch.setattr(service, 'process', lambda *args: None)
    (service.MODELS / 'small').mkdir(parents=True)
    (service.MODELS / 'small' / 'model.bin').touch()
    with TestClient(service.app) as client:
        yield client


def upload(client, **data):
    return client.post('/api/jobs', files={'file': ('noise.wav', b'audio')},
                       data={'language': 'zh', 'model': 'small', 'noise_mode': 'true', **data})


def test_preview_must_finish_before_full_and_only_tested_profile_can_be_used(api_client):
    job_id = upload(api_client, preview_minutes='5', preview_start='120').json()['id']
    assert service.read(job_id)['pending_action'] == 'preview'
    assert api_client.post(f'/api/jobs/{job_id}/start-full', json={'profile': 'standard'}).status_code == 409
    preview = {'audio_ready': True, 'profile': 'standard',
               'original': {'done': True, 'segments': []}, 'denoised': {'done': True, 'segments': []}}
    service.update(job_id, status='awaiting_choice', duration=600, preview=preview)
    assert api_client.post(f'/api/jobs/{job_id}/start-full', json={'profile': 'light'}).status_code == 400
    result = api_client.post(f'/api/jobs/{job_id}/start-full', json={'profile': 'standard'})
    assert result.status_code == 200
    job = service.read(job_id)
    assert job['workflow'] == 'full' and job['audio_profile'] == 'standard'
    assert job['segments'] == [] and not job['transcript_done'] and job['preview'] == preview
    assert not result.json()['audio_ready']
    assert api_client.post(f'/api/jobs/{job_id}/start-full', json={'profile': 'original'}).status_code == 409
    assert api_client.post(f'/api/jobs/{job_id}/preview', json={'minutes': 3}).status_code == 409


def test_preview_retry_validation_and_persisted_history(api_client):
    assert upload(api_client, preview_minutes='2').status_code == 400
    assert upload(api_client, preview_start='-1').status_code == 422
    job_id = upload(api_client).json()['id']
    service.update(job_id, status='awaiting_choice', duration=400,
                   preview={'original': {'done': True}, 'denoised': {'done': True}})
    assert api_client.post(f'/api/jobs/{job_id}/preview', json={'minutes': 7}).status_code == 422
    assert api_client.post(f'/api/jobs/{job_id}/preview', json={'start': 500}).status_code == 400
    assert api_client.post(f'/api/jobs/{job_id}/preview', json={'start': 60, 'minutes': 4, 'profile': 'light'}).status_code == 200
    job = service.read(job_id)
    assert job['preview_start'] == 60 and job['noise_profile'] == 'light' and job['preview'] == {}
    assert api_client.get(f'/api/jobs/{job_id}/preview/audio/original').status_code == 409
    assert 'preview' not in api_client.get('/api/jobs').json()[0]


def test_preview_checkpoint_resume_and_bilingual_absolute_timestamps(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'JOBS', tmp_path / 'jobs')
    monkeypatch.setattr(service, 'TASKS', queue.Queue())
    monkeypatch.setattr(service, 'local_model', lambda _: object())
    monkeypatch.setattr(transcription, 'transcribe_regions', lambda *args: iter([(1, 2, '我们下周 review API budget。')]))
    job_id = 'e' * 32
    folder = service.directory(job_id)
    folder.mkdir(parents=True)
    write_wav(folder / 'audio.wav', np.zeros(16000 * 140, dtype=np.int16))
    service.write({'id': job_id, 'filename': 'noise.wav', 'duration': 140, 'status': 'queued',
                   'model': 'small', 'language': 'zh', 'progress': 0, 'noise_mode': True,
                   'workflow': 'preview', 'noise_profile': 'standard', 'preview_start': 120,
                   'preview_minutes': 3, 'preview': {}, 'segments': [], 'transcript_done': False,
                   'summary': '', 'auto_summary': False})
    original_transcribe = noise.transcribe_clip
    called = []
    should_pause = [True]
    def transcribe(model, path, *args):
        called.append(path.name)
        if path.name == 'preview-denoised.wav' and should_pause[0]:
            raise InterruptedError()
        return original_transcribe(model, path, *args)
    monkeypatch.setattr(noise, 'transcribe_clip', transcribe)
    service.process(job_id, 'preview')
    paused = service.read(job_id)
    assert paused['status'] == 'paused' and paused['preview']['original']['done']
    should_pause[0] = False
    service.process(job_id, 'preview')
    job = service.read(job_id)
    assert job['status'] == 'awaiting_choice' and not job['transcript_done'] and not job['summary']
    assert called.count('preview-original.wav') == 1
    assert job['preview']['duration'] == 20
    for variant in ['original', 'denoised']:
        segment = job['preview'][variant]['segments'][0]
        assert segment['start'] == 121 and segment['end'] == 122
        assert segment['text'] == '我們下週 review API budget。' and segment['language'] == 'mixed'
        assert '00:02:01,000' in audio.export_transcript([segment], 'srt')
