"""Reference-based speaker verification; vectors and audio stay on this PC."""
import wave


def references(job):
    spans = []
    for segment in job['segments']:
        if not segment.get('speaker_manual') or not segment.get('speaker'):
            continue
        if spans and spans[-1]['speaker'] == segment['speaker'] and 0 <= segment['start']-spans[-1]['end'] <= .25:
            spans[-1]['end'] = segment['end']
        else:
            spans.append({k: segment[k] for k in ('start', 'end', 'speaker')})
    spans.extend(job.get('speaker_references') or [])
    clips = []
    for span in spans:
        start = span['start']
        while span['end'] - start >= 1.8:
            end = min(start + 4, span['end'])
            clips.append({'start': start, 'end': end, 'speaker': span['speaker']})
            start = end
    return clips


def unit(vector):
    import numpy as np
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or norm < 1e-8:
        raise ValueError('聲音樣本無法產生有效特徵，請選擇較清楚的單人說話片段。')
    return vector/norm


def extract(extractor, stream, start, end):
    import numpy as np
    stream.setpos(max(0, min(round(start*16000), stream.getnframes())))
    pcm = np.frombuffer(stream.readframes(max(0, round((end-start)*16000))), dtype='<i2').astype(np.float32)/32768
    from faster_whisper.vad import get_speech_timestamps,VadOptions
    speech=get_speech_timestamps(pcm,VadOptions(min_silence_duration_ms=150,speech_pad_ms=100))
    if not speech or sum(r['end']-r['start'] for r in speech)<.8*16000:
        raise ValueError('參考聲音中有效語音太少，請選擇較清楚的單人說話片段。')
    pcm=np.concatenate([pcm[r['start']:r['end']] for r in speech])
    state = extractor.create_stream()
    state.accept_waveform(sample_rate=16000, waveform=pcm)
    state.input_finished()
    if not extractor.is_ready(state):
        raise ValueError('聲音樣本太短，請提供至少 2 秒的清楚語音。')
    return unit(extractor.compute(state))


def prototypes(vectors, labels):
    import numpy as np
    result={}
    for label in sorted(set(labels)):
        values=np.asarray([v for v,target in zip(vectors,labels) if target==label])
        if len(values)>=3:
            similarity=values@values.T
            medoid=int(np.argmax(np.mean(similarity,axis=1)))
            keep=similarity[medoid]>=.4
            if keep.sum()>=2:values=values[keep]
        result[label]=unit(np.mean(values,axis=0))
    return result


def classify(vector, profiles):
    scores = sorted(((label, float(vector@profile)) for label, profile in profiles.items()), key=lambda item: item[1], reverse=True)
    first = scores[0]
    return first[0], first[1], first[1] - scores[1][1] if len(scores)>1 else 0


def evaluate(path, model, clips):
    import sherpa_onnx as so
    extractor = so.SpeakerEmbeddingExtractor(so.SpeakerEmbeddingExtractorConfig(model=str(model), num_threads=2))
    with wave.open(str(path), 'rb') as stream:
        vectors = [extract(extractor, stream, clip['start'], clip['end']) for clip in clips]
    labels = [clip['speaker'] for clip in clips]
    correct = 0
    per_speaker = {}
    margins = []
    for index, vector in enumerate(vectors):
        targets = labels[:index] + labels[index+1:]
        if labels[index] not in targets:
            continue
        profiles = prototypes(vectors[:index]+vectors[index+1:], targets)
        predicted, score, margin = classify(vector, profiles)
        correct += predicted == labels[index]
        values = per_speaker.setdefault(labels[index], [0, 0])
        values[0] += predicted == labels[index]
        values[1] += 1
        margins.append(margin if predicted == labels[index] else -margin)
    import numpy as np
    return {'clips': len(clips), 'correct': int(correct), 'per_speaker': per_speaker,
            'balanced_accuracy': float(np.mean([good/total for good,total in per_speaker.values()])),
            'mean_margin': float(np.mean(margins))}


def decide_segments(segments, turns, explicit):
    """Require both temporal coverage and a clear voice match; preserve humans."""
    changes, review = [], []
    for index, segment in enumerate(segments):
        if segment.get('speaker_manual'):
            continue
        duration = segment['end']-segment['start']
        if duration <= 0:
            review.append(index)
            continue
        confirmed = next((r for r in explicit if min(segment['end'],r['end'])-max(segment['start'],r['start']) >= .9*duration),None)
        if confirmed:
            changes.append({'index':index,'speaker':confirmed['speaker'],'score':1,'margin':1,'source':'reference'})
            continue
        votes, scores, margins = {}, {}, {}
        for turn in turns:
            overlap = min(segment['end'],turn['end'])-max(segment['start'],turn['start'])
            if overlap <= 0 or not turn.get('qualified'):
                continue
            target = turn['speaker']
            votes[target] = votes.get(target,0)+overlap
            scores[target] = min(scores.get(target,1),turn['score'])
            margins[target] = min(margins.get(target,1),turn['margin'])
        ranked = sorted(votes,key=votes.get,reverse=True)
        if ranked and votes[ranked[0]]/duration >= .6 and votes[ranked[0]]/sum(votes.values()) >= .8:
            target=ranked[0]
            changes.append({'index':index,'speaker':target,'score':round(scores[target],3),
                            'margin':round(margins[target],3),'source':'voice-reference'})
        else:
            review.append(index)
    return changes,review


def optimize_local(path, job, progress=lambda _: None):
    import numpy as np
    import sherpa_onnx as so
    from . import speakers
    model=speakers.ADVANCED
    clips=references(job)
    extractor=so.SpeakerEmbeddingExtractor(so.SpeakerEmbeddingExtractorConfig(model=str(model),num_threads=2))
    vectors=[];labels=[]
    with wave.open(str(path),'rb') as stream:
        for clip in clips:
            try:vector=extract(extractor,stream,clip['start'],clip['end'])
            except ValueError:continue
            vectors.append(vector);labels.append(clip['speaker'])
    from collections import Counter
    counts=Counter(labels)
    if any(counts[s]<2 for s in speakers.catalog(job)):
        raise ValueError('部分講者的有效參考聲音不足，請補充清楚的單人片段後重試。')
    profiles=prototypes(vectors,labels)
    turns=job.get('speaker_quality_turns')
    if turns is None:
        turns=speakers.detect_local(path,job.get('speaker_count') or len(profiles),
            progress=lambda ratio:progress(ratio*.6),embedding_model=model)
    verified=[]
    with wave.open(str(path),'rb') as stream:
        for index,turn in enumerate(turns):
            progress(.6+(index+1)/max(1,len(turns))*.4)
            duration=turn['end']-turn['start']
            if duration<1.8:
                continue
            # Multiple independent windows expose turns that contain a real speaker switch.
            starts=[turn['start']] if duration<=5 else [turn['start'],max(turn['start'],(turn['start']+turn['end'])/2-2),turn['end']-4]
            predictions=[]
            for start in starts:
                try:vector=extract(extractor,stream,start,min(turn['end'],start+4))
                except ValueError:continue
                predictions.append(classify(vector,profiles))
            if len(predictions)!=len(starts):continue
            label=predictions[0][0]
            score=min(p[1] for p in predictions);margin=min(p[2] for p in predictions)
            qualified=all(p[0]==label for p in predictions) and score>=.65 and margin>=.15
            verified.append({'start':turn['start'],'end':turn['end'],'speaker':label,
                             'score':score,'margin':margin,'qualified':qualified})
    assignments,review=decide_segments(job['segments'],verified,job.get('speaker_references') or [])
    return {'assignments':assignments,'review':review,'model':'ERes2NetV2 + verified references',
            'reference_clips':len(vectors),'reference_speakers':len(profiles),'turns':turns}
