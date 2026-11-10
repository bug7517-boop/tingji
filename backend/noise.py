"""Local, streaming FFmpeg denoising through PyAV. Sample count stays unchanged."""
from fractions import Fraction
from pathlib import Path
import wave

from .audio import SAMPLE_RATE

PROFILES = {'light': '輕度降噪', 'standard': '標準降噪', 'original': '原音'}
# afftdn at 16 kHz delays the signal by 25 ms. Compensate rather than shift timestamps.
DENOISE_DELAY = SAMPLE_RATE // 40


def extract(source: Path, target: Path, start: float, seconds: float):
    with wave.open(str(source), 'rb') as inp:
        total = inp.getnframes()
        first = round(start * SAMPLE_RATE)
        if first >= total:
            raise ValueError('試轉起點已超過錄音長度，請選擇較早的位置。')
        frames = min(round(seconds * SAMPLE_RATE), total - first)
        inp.setpos(first)
        with wave.open(str(target), 'wb') as out:
            out.setparams((1, 2, SAMPLE_RATE, 0, 'NONE', 'not compressed'))
            remaining = frames
            while remaining:
                data = inp.readframes(min(remaining, SAMPLE_RATE * 5))
                if not data:
                    raise ValueError('音訊資料不完整，請重新匯入。')
                out.writeframesraw(data)
                remaining -= len(data) // 2
    return first / SAMPLE_RATE, frames / SAMPLE_RATE


def enhance(source: Path, target: Path, profile: str, cancelled=lambda: False, progress=lambda _: None):
    import av
    import numpy as np
    if profile not in {'light', 'standard'}:
        raise ValueError('不支援的降噪設定。')
    graph = av.filter.Graph()
    src = graph.add_abuffer(sample_rate=SAMPLE_RATE, format='s16', layout='mono',
                           time_base=Fraction(1, SAMPLE_RATE))
    highpass = graph.add('highpass', 'f=70')
    denoise = graph.add('afftdn', 'nr=8:nf=-40:tn=1' if profile == 'light' else 'nr=16:nf=-35:tn=1')
    output_format = graph.add('aformat', 'sample_fmts=s16:sample_rates=16000:channel_layouts=mono')
    sink = graph.add('abuffersink')
    src.link_to(highpass)
    highpass.link_to(denoise)
    denoise.link_to(output_format)
    output_format.link_to(sink)
    graph.configure()
    temp = target.with_suffix('.partial.wav')
    try:
        with wave.open(str(source), 'rb') as inp, wave.open(str(temp), 'wb') as out:
            total = inp.getnframes()
            out.setparams((1, 2, SAMPLE_RATE, 0, 'NONE', 'not compressed'))
            received, written, last_report = 0, 0, -1
            discard = DENOISE_DELAY

            def drain():
                nonlocal written, discard
                while True:
                    try:
                        frame = sink.pull()
                    except (av.error.BlockingIOError, av.error.EOFError):
                        break
                    data = frame.to_ndarray().astype(np.int16).tobytes()
                    skipped = min(discard, len(data) // 2)
                    data = data[skipped * 2:]
                    discard -= skipped
                    # Filters may pad a final FFT frame; never lengthen the recording.
                    data = data[:max(0, (total - written) * 2)]
                    out.writeframesraw(data)
                    written += len(data) // 2

            while data := inp.readframes(SAMPLE_RATE):
                if cancelled():
                    raise InterruptedError()
                frame = av.AudioFrame.from_ndarray(np.frombuffer(data, dtype=np.int16).reshape(1, -1),
                                                   format='s16', layout='mono')
                frame.sample_rate = SAMPLE_RATE
                frame.time_base = Fraction(1, SAMPLE_RATE)
                frame.pts = received
                received += frame.samples
                src.push(frame)
                drain()
                percent = int(received / total * 100)
                if percent != last_report:
                    last_report = percent
                    progress(received / total)
            padding = av.AudioFrame.from_ndarray(np.zeros((1, DENOISE_DELAY), dtype=np.int16), format='s16', layout='mono')
            padding.sample_rate = SAMPLE_RATE
            padding.time_base = Fraction(1, SAMPLE_RATE)
            padding.pts = received
            src.push(padding)
            drain()
            src.push(None)
            drain()
            if written < total:
                out.writeframesraw(bytes((total - written) * 2))
        if cancelled():
            raise InterruptedError()
        temp.replace(target)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def transcribe_clip(model, path: Path, language: str, offset: float, cancelled=lambda: False):
    from opencc import OpenCC
    from . import audio, transcription
    pcm, _ = audio.read_chunk(path, 0)
    with wave.open(str(path), 'rb') as inp:
        duration = inp.getnframes() / SAMPLE_RATE
    convert = OpenCC('s2twp').convert
    result = []
    for start, end, raw in transcription.transcribe_regions(model, pcm, language, cancelled):
        text = convert(raw.strip())
        start = max(0, start)
        end = min(duration, end)
        if text and end > start:
            result.append({'start': round(offset + start, 3), 'end': round(offset + end, 3),
                           'text': text, 'language': transcription.text_language(text)})
    return result
