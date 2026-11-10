"""Stream compressed input to disk; keep memory independent of meeting length."""
import wave
from pathlib import Path

MAX_SECONDS = 3 * 60 * 60
SAMPLE_RATE = 16000
CHUNK_SECONDS = 300
OVERLAP = 2


def normalize(source: Path, target: Path, progress=lambda _: None):
    import av
    import numpy as np
    total = 0
    try:
        with av.open(str(source)) as container:
            streams = list(container.streams.audio)
            if not streams:
                raise ValueError('檔案沒有可辨識的音軌。')
            stream = streams[0]
            estimate = float(stream.duration * stream.time_base) if stream.duration else 0
            if estimate > MAX_SECONDS + 0.1:
                raise ValueError('錄音超過 3 小時，請先分割檔案。')
            resampler = av.AudioResampler(format='s16', layout='mono', rate=SAMPLE_RATE)

            def write(frames):
                nonlocal total
                for frame in frames:
                    pcm = frame.to_ndarray().astype(np.int16).tobytes()
                    total += len(pcm) // 2
                    if total > MAX_SECONDS * SAMPLE_RATE:
                        raise ValueError('錄音超過 3 小時，請先分割檔案。')
                    out.writeframesraw(pcm)

            with wave.open(str(target), 'wb') as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(SAMPLE_RATE)
                last = -1
                for frame in container.decode(stream):
                    write(resampler.resample(frame))
                    seconds = total / SAMPLE_RATE
                    if int(seconds) // 10 != last:
                        last = int(seconds) // 10
                        progress(seconds / estimate if estimate else None)
                write(resampler.resample(None))
        if total == 0:
            raise ValueError('錄音是空的，請選擇其他檔案。')
        return total / SAMPLE_RATE
    except Exception:
        target.unlink(missing_ok=True)
        raise


def read_chunk(path: Path, index: int):
    import numpy as np
    core_start = index * CHUNK_SECONDS
    start = max(0, core_start - OVERLAP)
    with wave.open(str(path), 'rb') as stream:
        stream.setpos(min(start * SAMPLE_RATE, stream.getnframes()))
        data = stream.readframes((CHUNK_SECONDS + OVERLAP * 2) * SAMPLE_RATE)
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0, start


def timecode(seconds, srt=False):
    millis = max(0, round(seconds * 1000))
    sec, ms = divmod(millis, 1000)
    minutes, sec = divmod(sec, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:02}:{minutes:02}:{sec:02}' + (f',{ms:03}' if srt else '')


def export_transcript(segments, kind, include_timestamps=True):
    if kind == 'srt':
        return '\n\n'.join(f"{i}\n{timecode(s['start'], True)} --> {timecode(s['end'], True)}\n{s['text']}" for i, s in enumerate(segments, 1)) + '\n'
    if not include_timestamps:
        return '\n'.join(s['text'] for s in segments) + '\n'
    return '\n'.join(f"[{timecode(s['start'])} → {timecode(s['end'])}] {s['text']}" for s in segments) + '\n'
