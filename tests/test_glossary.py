import queue
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
import pytest

from backend import app as service, glossary, transcription


def test_domain_catalog_and_custom_terms_are_bounded():
    terms=glossary.terms_for('real_estate',['自訂地名','農地'])
    assert terms[0]=='自訂地名' and terms.count('農地')==1
    assert terms.index('市地重劃')<10
    assert {'農地','都市計畫','市地重劃','抵費地','容積率'}.issubset(terms)
    assert glossary.terms_for('general',[])==[]
    assert len(glossary.hints(terms))<=400
    assert glossary.split('甲、乙，丙\n甲')==['甲','乙','丙']
    with pytest.raises(ValueError):glossary.normalize(['x'*51])
    with pytest.raises(ValueError):glossary.normalize(['bad\x00'])
    assert len(glossary.terms_for('real_estate',[str(i) for i in range(200)]))>200


def test_saved_dictionary_and_job_snapshot_are_independent(tmp_path,monkeypatch):
    monkeypatch.setattr(service,'DATA',tmp_path/'data')
    monkeypatch.setattr(service,'JOBS',tmp_path/'jobs')
    monkeypatch.setattr(service,'MODELS',tmp_path/'models')
    monkeypatch.setattr(service,'TASKS',queue.Queue())
    (service.MODELS/'small').mkdir(parents=True);(service.MODELS/'small'/'model.bin').touch()
    client=TestClient(service.app)
    assert client.patch('/api/glossary',json={'terms':[' 自訂地名 ','自訂地名']}).status_code==200
    assert client.get('/api/glossary').json()['custom_terms']==['自訂地名']
    response=client.post('/api/jobs',files={'file':('test.wav',b'x')},data={'domain':'real_estate','auto_summary':'false'})
    assert response.status_code==201;job_id=response.json()['id']
    assert '自訂地名' in service.read(job_id)['glossary_terms']
    client.patch('/api/glossary',json={'terms':['其他地名']})
    assert '其他地名' not in service.read(job_id)['glossary_terms']
    response=client.post('/api/jobs',files={'file':('test.wav',b'x')},data={'domain':'general','glossary_extra':'','glossary_source':'provided'})
    assert response.status_code==201 and service.read(response.json()['id'])['glossary_terms']==[]
    assert client.post('/api/jobs',files={'file':('test.wav',b'x')},data={'domain':'bad'}).status_code==400


def test_transcription_passes_only_vocabulary_and_preserves_language(monkeypatch):
    import faster_whisper.vad as vad
    monkeypatch.setattr(vad,'get_speech_timestamps',lambda *args:[{'start':0,'end':16000}])
    calls=[]
    class Model:
        def transcribe(self,pcm,**kwargs):
            calls.append(kwargs)
            return iter([SimpleNamespace(start=0,end=1,text='Review API，農地。')]),None
    parts=list(transcription.transcribe_regions(Model(),np.zeros(16000,dtype=np.float32),'zh',terms=['農地','都市計畫']))
    assert calls[0]['hotwords']=='農地、都市計畫' and calls[0]['task']=='transcribe' and calls[0]['language'] is None
    assert parts[0][2]=='Review API，農地。'
