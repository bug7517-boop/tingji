import copy
import queue

from fastapi.testclient import TestClient
import pytest

from backend import app as service, speaker_refinement as refinement, speakers


@pytest.fixture
def ready_job(tmp_path,monkeypatch):
    monkeypatch.setattr(service,'JOBS',tmp_path/'jobs')
    monkeypatch.setattr(service,'TASKS',queue.Queue())
    model=tmp_path/'advanced.onnx';model.touch();monkeypatch.setattr(speakers,'ADVANCED',model)
    job_id='9'*32;service.directory(job_id).mkdir(parents=True)
    segments=[{'start':0,'end':4,'text':'手動一','speaker':'speaker_1','speakers':['speaker_1'],'speaker_manual':True},
              {'start':4,'end':8,'text':'手動二','speaker':'speaker_1','speakers':['speaker_1'],'speaker_manual':True},
              {'start':16,'end':18,'text':'需要修正 English','speaker':'speaker_2','speakers':['speaker_2']},
              {'start':18,'end':20,'text':'不確定','speaker':'speaker_2','speakers':['speaker_2']}]
    job={'id':job_id,'filename':'校準.wav','status':'done','duration':20,'transcript_done':True,'segments':segments,'auto_summary':False,
         'summary':'原摘要','speaker_count':2,'speaker_names':{'speaker_1':'甲','speaker_2':'乙'}}
    service.write(job)
    return TestClient(service.app),job_id,copy.deepcopy(job)


def test_reference_calibration_protects_manual_and_defers_weak_matches():
    segments=[{'start':0,'end':2,'speaker_manual':True,'speaker':'speaker_2'},
              {'start':2,'end':4},{'start':4,'end':6},{'start':6,'end':10}]
    turns=[{'start':0,'end':4,'speaker':'speaker_1','score':.9,'margin':.3,'qualified':True},
           {'start':4,'end':6,'speaker':'speaker_1','score':.5,'margin':.01,'qualified':False},
           {'start':6,'end':8,'speaker':'speaker_1','score':.9,'margin':.3,'qualified':True}]
    assignments,review=refinement.decide_segments(segments,turns,[])
    assert [a['index'] for a in assignments]==[1] and review==[2,3]
    assignments,review=refinement.decide_segments(segments,turns,[{'speaker':'speaker_2','start':4,'end':6}])
    assert assignments[1]['index']==2 and assignments[1]['speaker']=='speaker_2' and assignments[1]['source']=='reference'


def test_optimize_preserves_names_text_manual_and_undo_preserves_later_edits(ready_job,monkeypatch):
    client,job_id,original=ready_job;url=f'/api/jobs/{job_id}'
    assert client.post(url+'/speakers/optimize',json={'references':[{'speaker':'speaker_2','start':8,'end':16}]}).status_code==200
    monkeypatch.setattr(service,'ensure_normalized',lambda *args:'audio.wav')
    monkeypatch.setattr(speakers,'optimize',lambda *args:{'assignments':[{'index':0,'speaker':'speaker_2','score':.9,'margin':.3},{'index':2,'speaker':'speaker_1','score':.9,'margin':.3}],
        'review':[3],'model':'test-model','reference_clips':4,'turns':[]})
    service.process(job_id,'speaker-optimize');job=service.read(job_id)
    assert job['status']=='done' and job['speaker_names']==original['speaker_names']
    assert job['segments'][:2]==original['segments'][:2]
    assert job['segments'][2]['speaker']=='speaker_1' and job['segments'][3]['quality_review']
    assert job['speaker_optimization']['changed']==1 and job['speaker_optimization']['manual_protected']==2
    info=client.get(url).json()['speaker_optimization'];assert 'after' not in info and 'backup' not in info
    assert '乙（待確認）' in client.get(url+'/export/txt').text
    service.update(job_id,segments=[*job['segments'][:2],{**job['segments'][2],'text':'後來校正的文字'},job['segments'][3]])
    assert client.patch(url+'/segments/3/speaker',json={'speaker':'speaker_1','start':18,'end':20}).status_code==200
    assert client.post(url+'/speakers/undo-optimization').status_code==200
    result=service.read(job_id)
    assert result['segments'][2]['text']=='後來校正的文字' and result['segments'][2]['speaker']=='speaker_2'
    assert result['segments'][3]['speaker']=='speaker_1' and result['segments'][3]['speaker_manual']
    assert result['speaker_names']==original['speaker_names'] and result['summary']=='原摘要'


def test_reference_validation_and_insufficient_samples_do_not_mutate(ready_job):
    client,job_id,original=ready_job;url=f'/api/jobs/{job_id}/speakers/optimize'
    assert client.post(url,json={'references':[]}).status_code==409
    assert client.post(url,json={'references':[{'speaker':'speaker_2','start':18,'end':25}]}).status_code==400
    assert client.post(url,json={'references':[{'speaker':'speaker_9','start':8,'end':16}]}).status_code==400
    assert client.post(url,json={'references':[{'speaker':'speaker_2','start':0,'end':8}]}).status_code==400
    assert client.post(url,json={'references':[{'speaker':'speaker_1','start':8,'end':16},{'speaker':'speaker_2','start':10,'end':18}]}).status_code==400
    assert service.read(job_id)==original


def test_reference_vectors_handle_degenerate_audio():
    import numpy as np
    with pytest.raises(ValueError):refinement.unit([0,0])
    with pytest.raises(ValueError):refinement.unit([float('nan'),1])
    profiles=refinement.prototypes([np.array([1.,0.]),np.array([0.,1.])],['a','b'])
    label,score,margin=refinement.classify(np.array([1.,0.]),profiles)
    assert label=='a' and score==1 and margin==1
