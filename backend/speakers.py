"""Local diarization across the entire recording; IDs never reset at ASR chunks."""
from pathlib import Path
import importlib.util
import wave

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'models' / 'speakers'
SEGMENTATION = MODELS / 'segmentation.onnx'
EMBEDDING = MODELS / 'embedding.onnx'


def ready():
    return (importlib.util.find_spec('sherpa_onnx') is not None
            and SEGMENTATION.is_file() and EMBEDDING.is_file())


def detect(path, count=0, cancelled=lambda: False, progress=lambda _: None):
    import numpy as np
    import sherpa_onnx as sherpa
    if not ready():
        raise RuntimeError('講者模型尚未就緒，請執行「下載講者模型.cmd」。')
    if cancelled():
        raise InterruptedError()
    config = sherpa.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(SEGMENTATION)),
            num_threads=2, provider='cpu'),
        embedding=sherpa.SpeakerEmbeddingExtractorConfig(model=str(EMBEDDING), num_threads=2, provider='cpu'),
        clustering=sherpa.FastClusteringConfig(num_clusters=count if count else -1, threshold=0.5),
        min_duration_on=0.3, min_duration_off=0.5)
    if not config.validate():
        raise RuntimeError('講者模型無效，請重新下載講者模型。')
    engine = sherpa.OfflineSpeakerDiarization(config)
    # The global clustering needs all audio. Fill float32 directly in blocks to
    # avoid holding both the complete PCM bytes and another float64 copy.
    with wave.open(str(path), 'rb') as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (16000, 1, 2):
            raise ValueError('講者辨識需要 16 kHz 單聲道 PCM。')
        samples = np.empty(stream.getnframes(), dtype=np.float32)
        offset = 0
        while chunk := stream.readframes(16000 * 30):
            if cancelled():
                raise InterruptedError()
            values = np.frombuffer(chunk, dtype='<i2')
            samples[offset:offset + len(values)] = values
            offset += len(values)
        samples /= 32768.0
    def callback(done, total):
        progress(done / max(1, total))
        return 1 if cancelled() else 0
    result = engine.process(samples, callback=callback).sort_by_start_time()
    if cancelled():
        raise InterruptedError()
    ids = {}
    turns = []
    for turn in result:
        if turn.speaker not in ids:
            ids[turn.speaker] = f'speaker_{len(ids) + 1}'
        turns.append({'start': round(turn.start, 3), 'end': round(turn.end, 3), 'speaker': ids[turn.speaker]})
    return turns


def label_segments(segments, turns):
    """Preserve transcript verbatim. Flag cross-speaker segments, never guess words."""
    result = []
    turns = sorted(turns, key=lambda t: t['start'])
    first = 0
    for segment in segments:
        scores = {}
        while first < len(turns) and turns[first]['end'] <= segment['start']:
            first += 1
        for index in range(first, len(turns)):
            turn = turns[index]
            if turn['start'] >= segment['end']:
                break
            overlap = min(segment['end'], turn['end']) - max(segment['start'], turn['start'])
            if overlap > 0:
                scores[turn['speaker']] = scores.get(turn['speaker'], 0) + overlap
        ranked = sorted(scores, key=scores.get, reverse=True)
        selected = [s for s in ranked if scores[s] >= max(0.2, (segment['end'] - segment['start']) * 0.15)]
        # Very short or unaligned speech remains explicitly unknown.
        clean = {k: v for k, v in segment.items() if k not in {'speaker_original', 'speaker_manual'}}
        result.append({**clean, 'speaker': selected[0] if len(selected) == 1 else None,
                       'speakers': selected, 'speaker_uncertain': len(selected) != 1})
    return result


def speaker_ids(segments):
    originals = [s['speaker_original'] for s in segments if s.get('speaker_original')]
    return {value for segment in [*segments, *originals] for value in
            (segment.get('speakers') or ([segment['speaker']] if segment.get('speaker') else []))}


def catalog(job):
    return (speaker_ids(job['segments']) | set(job.get('speaker_names') or {}) |
            {t['speaker'] for t in job.get('speaker_turns') or []})


def label(segment, names=None):
    ids = segment.get('speakers') or ([segment['speaker']] if segment.get('speaker') else [])
    if not ids:
        return '講者未辨識' if 'speaker' in segment else ''
    display = '／'.join((names or {}).get(value) or '講者 ' + value.removeprefix('speaker_') for value in ids)
    return display + ('（交接或重疊，請回聽）' if len(ids) > 1 else '')
