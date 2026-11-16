import copy

from fastapi.testclient import TestClient
import pytest

from backend import app as service, transcript


@pytest.fixture
def saved(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'JOBS', tmp_path / 'jobs')
    job_id = 'c'*32
    service.directory(job_id).mkdir(parents=True)
    segments = [{'start': 0, 'end': 2, 'text': '我們先確認預蒜。', 'speaker': 'speaker_1', 'speakers': ['speaker_1'], 'language': 'zh'},
                {'start': 2, 'end': 4, 'text': '預蒜和預蒜都需要 review API。', 'speaker': 'speaker_1', 'speakers': ['speaker_1'], 'language': 'mixed'},
                {'start': 4, 'end': 6, 'text': 'Budget is ready.', 'speaker': 'speaker_2', 'speakers': ['speaker_2'], 'language': 'en'}]
    job = {'id': job_id, 'filename': '測試.wav', 'status': 'done', 'transcript_done': True,
           'segments': segments, 'duration': 6, 'summary': '舊摘要', 'summary_parts': [{'old': True}],
           'speaker_names': {'speaker_1': '王小明', 'speaker_2': 'Alice'}}
    service.write(job)
    return TestClient(service.app), job_id, copy.deepcopy(job)


def test_text_edit_is_atomic_persistent_and_in_all_exports(saved):
    client, job_id, original = saved
    url = f'/api/jobs/{job_id}'
    part = original['segments'][1]
    edit = {'index': 1, 'start': part['start'], 'end': part['end'], 'expected_text': part['text'], 'text': '預算都需要 review API。'}
    assert client.patch(url+'/segments/text', json={'edits': [edit]}).status_code == 200
    job = client.get(url).json()
    assert job['segments'][0] == original['segments'][0] and job['segments'][2] == original['segments'][2]
    assert job['segments'][1]['speaker'] == 'speaker_1'
    assert job['segments'][1]['text_original'] == part['text'] and job['segments'][1]['text_manual']
    assert job['summary_stale'] and job['summary'] == original['summary'] and service.read(job_id)['summary_parts'] == []
    for kind in ('txt', 'srt'):
        for timestamps in ('true', 'false'):
            body = client.get(url+f'/export/{kind}?timestamps={timestamps}').text
            assert '預算都需要 review API。' in body and '預蒜和預蒜' not in body
            assert '王小明' in body
    assert client.patch(url+'/segments/text', json={'edits': [edit]}).status_code == 409
    current = job['segments'][1]['text']
    assert client.patch(url+'/segments/text', json={'edits': [{**edit, 'expected_text': current, 'text': part['text']}]}).status_code == 200
    assert service.read(job_id)['segments'][1]['text_manual'] is False


def test_invalid_multi_edit_changes_nothing(saved):
    client, job_id, original = saved
    path = f'/api/jobs/{job_id}/segments/text'
    first = {'index': 0, 'start': 0, 'end': 2, 'expected_text': original['segments'][0]['text'], 'text': '新文字'}
    bad = {**first, 'index': 1, 'start': 2, 'end': 4, 'expected_text': '過期文字'}
    assert client.patch(path, json={'edits': [first, bad]}).status_code == 409
    assert service.read(job_id) == original
    assert client.patch(path, json={'edits': [first, first]}).status_code == 400
    for text in (' ', 'bad\x00text'):
        assert client.patch(path, json={'edits': [{**first, 'text': text}]}).status_code == 400
    service.update(job_id, status='diarizing')
    assert client.patch(path, json={'edits': [first]}).status_code == 409


def test_replacement_previews_all_occurrences_and_updates_exports(saved):
    client, job_id, original = saved
    url = f'/api/jobs/{job_id}'
    data = {'find': '預蒜', 'replace': '預算'}
    preview = client.post(url+'/text-replacements/preview', json=data).json()
    assert preview['occurrences'] == 3 and preview['segments'] == 2
    assert service.read(job_id) == original
    assert client.patch(url+'/text-replacements', json=data).status_code == 409
    assert client.patch(url+'/text-replacements', json={**data, 'revision': preview['revision']}).status_code == 200
    job = service.read(job_id)
    assert job['segments'][2] == original['segments'][2] and job['speaker_names'] == original['speaker_names']
    assert '預蒜' not in ''.join(s['text'] for s in job['segments'])
    assert job['summary_stale']
    for kind in ('txt', 'srt'):
        body = client.get(url+f'/export/{kind}').text
        assert '預蒜' not in body and '預算' in body and 'review API' in body
    assert client.patch(url+'/text-replacements', json={**data, 'revision': preview['revision']}).status_code == 409


def test_replacement_is_literal_case_sensitive_and_blocks_empty_result(saved):
    client, job_id, _ = saved
    url = f'/api/jobs/{job_id}'
    for find, count in [('budget', 0), ('Budget', 1), ('.*', 0), ('不存在', 0)]:
        response = client.post(url+'/text-replacements/preview', json={'find': find, 'replace': '替換'})
        assert response.status_code == 200 and response.json()['occurrences'] == count
    assert client.post(url+'/text-replacements/preview', json={'find': 'Budget is ready.', 'replace': ''}).status_code == 400
    assert client.post(url+'/text-replacements/preview', json={'find': 'a\n', 'replace': 'b'}).status_code == 400
    assert client.post(url+'/text-replacements/preview', json={'find': '', 'replace': 'b'}).status_code == 422


def test_paragraphs_do_not_merge_speakers_or_long_pauses_and_keep_source(saved):
    client, job_id, original = saved
    groups = transcript.paragraphs(original['segments'], 30)
    assert len(groups) == 2 and len(groups[0]['parts']) == 2
    assert (groups[0]['start'], groups[0]['end']) == (0, 4)
    assert transcript.join_text(['We', 'review', 'API。', '中文']) == 'We review API。中文'
    assert len(transcript.paragraphs(original['segments'], 0)) == 3
    source = [{**s, 'start': s['start']*10, 'end': s['end']*10} for s in original['segments']]
    assert len(transcript.paragraphs(source, 30)) == 3
    txt = client.get(f'/api/jobs/{job_id}/export/txt?paragraphs=30').text
    srt = client.get(f'/api/jobs/{job_id}/export/srt?paragraphs=30').text
    assert txt.count('【王小明】') == 1 and srt.count('【王小明】') == 2
    assert service.read(job_id) == original


def test_multiline_edit_does_not_create_empty_srt_cues(saved):
    client, job_id, original = saved
    edit = {'index': 0, 'start': 0, 'end': 2, 'expected_text': original['segments'][0]['text'], 'text': '第一行\n\n第二行'}
    assert client.patch(f'/api/jobs/{job_id}/segments/text', json={'edits': [edit]}).status_code == 200
    srt = client.get(f'/api/jobs/{job_id}/export/srt').text
    assert '第一行\n第二行' in srt and '第一行\n\n第二行' not in srt
