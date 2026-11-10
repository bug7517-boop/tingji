from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import re
import threading
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Literal
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import audio, summary, transcription, noise

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('TINGJI_DATA', str(ROOT / 'data')))
JOBS = DATA / 'jobs'
MODELS = ROOT / 'models' / 'whisper'
MAX_BYTES = 4 * 1024**3
ALLOWED_EXT = {'.mp3', '.m4a', '.wav', '.flac', '.ogg', '.aac', '.mp4', '.webm', '.wma', '.opus'}
ACTIVE = {'queued', 'normalizing', 'loading', 'transcribing', 'summarizing', 'denoising', 'previewing'}
LOCK = threading.RLock()
TASKS = queue.Queue()
MODEL_LOCK = threading.Lock()
STOP_EVENTS = {}
WHISPER = {}


def directory(job_id):
    if not re.fullmatch(r'[a-f0-9]{32}', job_id):
        raise HTTPException(404, '找不到這份會議紀錄。')
    return JOBS / job_id


def read(job_id):
    with LOCK:
        path = directory(job_id) / 'job.json'
        if not path.exists():
            raise HTTPException(404, '找不到這份會議紀錄。')
        return json.loads(path.read_text(encoding='utf-8'))


def write(job):
    with LOCK:
        path = directory(job['id']) / 'job.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(job, ensure_ascii=False), encoding='utf-8')
        temp.replace(path)


def update(job_id, **fields):
    with LOCK:
        job = read(job_id)
        job.update(fields)
        write(job)
        return job


def public(job, details=True):
    hidden = {'summary_parts', 'completed_chunks', 'source'}
    if not details:
        hidden.update({'segments', 'summary', 'preview'})
    result = {k: v for k, v in job.items() if k not in hidden}
    result['audio_ready'] = bool(job.get('duration') and (job.get('audio_profile', 'original') == 'original' or
                                                       job.get('enhanced_profile') == job.get('audio_profile')))
    return result


def local_model(name):
    # Never initiate a model download from a transcription request.
    from faster_whisper import WhisperModel
    with MODEL_LOCK:
        if name not in WHISPER:
            model_dir = MODELS / name
            if not (model_dir / 'model.bin').exists():
                raise RuntimeError(f'辨識模型 {name} 尚未下載，請執行「下載模型.cmd」。')
            WHISPER.clear()
            WHISPER[name] = WhisperModel(str(model_dir), device='cpu', compute_type='int8',
                                        cpu_threads=max(1, min(6, (os.cpu_count() or 4) - 1)),
                                        local_files_only=True)
        return WHISPER[name]


def ensure_normalized(job_id, event):
    job = read(job_id)
    normalized = directory(job_id) / 'audio.wav'
    if not job.get('duration') or not normalized.exists():
        update(job_id, status='normalizing', message='準備音訊，檢查長度與格式', progress=0, error=None)

        def on_decode(ratio):
            if event.is_set():
                raise InterruptedError()
            update(job_id, progress=round(min(1, ratio or 0) * 5, 1))

        duration = audio.normalize(directory(job_id) / job['source'], normalized, on_decode)
        update(job_id, duration=duration)
    return normalized


def process_preview(job_id, event):
    normalized = ensure_normalized(job_id, event)
    job = read(job_id)
    folder = directory(job_id)
    preview = job.get('preview', {})
    original, denoised = folder / 'preview-original.wav', folder / 'preview-denoised.wav'
    if not preview.get('audio_ready'):
        start, length = noise.extract(normalized, original, job['preview_start'], job['preview_minutes'] * 60)
        update(job_id, status='denoising', message='準備降噪試聽音訊', progress=5)
        noise.enhance(original, denoised, job['noise_profile'], event.is_set,
                      lambda ratio: update(job_id, progress=round(5 + ratio * 10, 1)))
        preview = {'audio_ready': True, 'start': start, 'duration': length, 'profile': job['noise_profile'],
                   'original': {'done': False, 'segments': []}, 'denoised': {'done': False, 'segments': []}}
        update(job_id, preview=preview)
    update(job_id, status='loading', message='載入試轉辨識模型', error=None)
    model = local_model(job['model'])
    for variant, path, label in [('original', original, '原音'), ('denoised', denoised, '降噪版')]:
        if event.is_set():
            raise InterruptedError()
        if preview[variant]['done']:
            continue
        update(job_id, status='previewing', message=f'試轉{label}，完成後可並排比較', progress=20 if variant == 'original' else 60)
        segments = noise.transcribe_clip(model, path, job['language'], preview['start'], event.is_set)
        preview[variant] = {'done': True, 'segments': segments}
        update(job_id, preview=preview)
    if event.is_set():
        raise InterruptedError()
    update(job_id, status='awaiting_choice', progress=100, error=None,
           message='試轉完成，請比較原音與降噪版，再選擇整份處理方式')


def process(job_id, action='transcribe'):
    event = STOP_EVENTS.setdefault(job_id, threading.Event())
    try:
        job = update(job_id, pending_action=action)
        if event.is_set():
            raise InterruptedError()
        if action == 'preview':
            process_preview(job_id, event)
            return
        if action != 'summary' and not job.get('transcript_done'):
            normalized = ensure_normalized(job_id, event)
            job = read(job_id)
            profile = job.get('audio_profile', 'original')
            if profile != 'original':
                enhanced = directory(job_id) / 'enhanced.wav'
                if not enhanced.exists() or job.get('enhanced_profile') != profile:
                    update(job_id, status='denoising', message='套用選定降噪設定至整份錄音', progress=0)
                    noise.enhance(normalized, enhanced, profile, event.is_set,
                                  lambda ratio: update(job_id, progress=round(ratio * 5, 1)))
                    update(job_id, enhanced_profile=profile)
                normalized = enhanced
            duration = job['duration']
            update(job_id, status='loading', message='載入本機辨識模型', progress=max(job['progress'], 5), error=None)
            model = local_model(job['model'])
            from opencc import OpenCC
            convert = OpenCC('s2twp').convert
            count = math.ceil(duration / audio.CHUNK_SECONDS)
            for index in range(job.get('completed_chunks', 0), count):
                if event.is_set():
                    raise InterruptedError()
                update(job_id, status='transcribing', message=f'辨識錄音 {index + 1}/{count} 段',
                       progress=round(5 + index / count * 90, 1))
                pcm, offset = audio.read_chunk(normalized, index)
                iterator = transcription.transcribe_regions(model, pcm, job['language'], event.is_set)
                new_segments = []
                core_start = index * audio.CHUNK_SECONDS
                core_end = min(duration, core_start + audio.CHUNK_SECONDS)
                for local_start, local_end, raw_text in iterator:
                    if event.is_set():
                        raise InterruptedError()
                    start, end = local_start + offset, local_end + offset
                    # Keep one copy of overlapping text based on the segment midpoint.
                    if not (core_start <= (start + end) / 2 < core_end):
                        continue
                    text = convert(raw_text.strip())
                    if text:
                        new_segments.append({'start': round(max(core_start, start), 3),
                                             'end': round(min(duration, end), 3), 'text': text,
                                             'language': transcription.text_language(text)})
                with LOCK:
                    latest = read(job_id)
                    existing = latest['segments']
                    for segment in new_segments:
                        if existing:
                            segment['start'] = max(segment['start'], existing[-1]['end'])
                        if segment['end'] > segment['start']:
                            existing.append(segment)
                    latest.update(completed_chunks=index + 1, segments=existing)
                    write(latest)
                job = read(job_id)
            job = update(job_id, transcript_done=True, progress=100,
                         message='逐字稿已完成' if job['segments'] else '未偵測到可辨識的語音')
        job = read(job_id)
        if (job['auto_summary'] or action == 'summary') and job['segments']:
            # Release the ASR model before loading the summary model on modest PCs.
            with MODEL_LOCK:
                WHISPER.clear()
            import gc
            if 'model' in locals():
                del model
            gc.collect()
            update(job_id, status='summarizing', summary_error=None, message='載入本機摘要模型', progress=100)
            engine = summary.LocalSummarizer()
            try:
                result = summary.summarize(job['segments'], job.get('summary_parts', []),
                    lambda parts: update(job_id, summary_parts=parts),
                    lambda msg: update(job_id, message=msg), event.is_set, engine)
                from opencc import OpenCC
                update(job_id, summary=OpenCC('s2twp').convert(result))
            except InterruptedError:
                raise
            except Exception as exc:
                update(job_id, summary_error=str(exc))
            finally:
                engine.stop()
        if event.is_set():
            raise InterruptedError()
        latest = read(job_id)
        update(job_id, status='done', progress=100,
               message='逐字稿已完成；摘要需要重試' if latest.get('summary_error') else
               ('轉錄完成' if latest['segments'] else '未偵測到可辨識的語音'), error=None)
    except InterruptedError:
        update(job_id, status='paused', message='已暫停，完成的分段已保存', error=None)
    except Exception as exc:
        update(job_id, status='error', message='處理未完成，可重試續跑', error=str(exc))
    finally:
        with LOCK:
            STOP_EVENTS.pop(job_id, None)


def worker():
    while True:
        task = TASKS.get()
        try:
            if task is None:
                return
            process(*task)
        finally:
            TASKS.task_done()


@asynccontextmanager
async def lifespan(app):
    JOBS.mkdir(parents=True, exist_ok=True)
    for path in JOBS.glob('*/job.json'):
        try:
            job = json.loads(path.read_text(encoding='utf-8'))
            if job['status'] in ACTIVE:
                job.update(status='paused', message='上次處理已中斷，可從完成的分段續跑')
                write(job)
        except (ValueError, KeyError, OSError):
            continue
    thread = threading.Thread(target=worker, daemon=True, name='meeting-worker')
    thread.start()
    yield
    for event in list(STOP_EVENTS.values()):
        event.set()
    TASKS.put(None)
    thread.join(timeout=5)


app = FastAPI(title='聽記', lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', 'testserver'])


@app.middleware('http')
async def protect_local(request: Request, call_next):
    if request.method in {'POST', 'PUT', 'PATCH', 'DELETE'}:
        origin = request.headers.get('origin')
        if origin and origin != str(request.base_url).rstrip('/') and origin not in {
                'http://127.0.0.1:8765', 'http://localhost:8765',
                'http://127.0.0.1:5173', 'http://localhost:5173'}:
            return JSONResponse({'detail': '只允許從本機工具操作。'}, status_code=403)
        if request.headers.get('sec-fetch-site') == 'cross-site':
            return JSONResponse({'detail': '不允許跨站操作。'}, status_code=403)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Cache-Control'] = 'no-store' if request.url.path.startswith('/api') else 'no-cache'
    return response


@app.get('/api/health')
def health():
    return {'ok': True, 'local': True, 'noise_mode_available': True, 'summary_ready': summary.ready(),
            'models': {name: (MODELS / name / 'model.bin').exists() for name in ['tiny', 'base', 'small', 'medium']}}


@app.get('/api/jobs')
def list_jobs():
    jobs = []
    for path in JOBS.glob('*/job.json'):
        try:
            jobs.append(public(read(path.parent.name), details=False))
        except (ValueError, OSError):
            continue
    return sorted(jobs, key=lambda j: j['created_at'], reverse=True)


@app.post('/api/jobs', status_code=201)
async def create_job(file: UploadFile = File(...), language: str = Form('zh'),
                     model: str = Form('small'), auto_summary: bool = Form(True),
                     noise_mode: bool = Form(False), preview_start: float = Form(0, ge=0, lt=10800),
                     preview_minutes: int = Form(3), noise_profile: str = Form('standard')):
    ext = Path(file.filename or '').suffix.lower()
    if ext not in ALLOWED_EXT:
        await file.close()
        raise HTTPException(400, '請選擇支援的音訊檔案，例如 MP3、M4A、WAV。')
    if (model not in {'tiny', 'base', 'small', 'medium'} or language not in {'zh', 'zh-only', 'en', 'auto', 'ja'} or
            preview_minutes not in {3, 4, 5} or noise_profile not in {'light', 'standard'}):
        await file.close()
        raise HTTPException(400, '無效的語言或模型設定。')
    if not (MODELS / model / 'model.bin').exists():
        await file.close()
        raise HTTPException(409, f'{model} 模型尚未下載，請先執行「下載模型.cmd」。')
    job_id = uuid.uuid4().hex
    folder = directory(job_id)
    folder.mkdir(parents=True)
    source = folder / ('source' + ext)
    size = 0
    try:
        with source.open('wb') as target:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, '檔案超過 4 GB，請壓縮或分割後再匯入。')
                target.write(chunk)
        if size == 0:
            raise HTTPException(400, '檔案是空的。')
        job = {'id': job_id, 'filename': (file.filename or '錄音')[:240], 'source': source.name,
               'created_at': datetime.now(timezone.utc).isoformat(), 'size': size, 'language': language,
               'model': model, 'auto_summary': auto_summary, 'status': 'queued', 'message': '排隊等待處理',
               'progress': 0, 'duration': None, 'completed_chunks': 0, 'segments': [],
               'transcript_done': False, 'summary': '', 'summary_parts': [], 'summary_error': None, 'error': None}
        job.update(noise_mode=noise_mode, workflow='preview' if noise_mode else 'full',
                   preview_start=preview_start, preview_minutes=preview_minutes, noise_profile=noise_profile,
                   preview={}, audio_profile='original', pending_action='preview' if noise_mode else 'transcribe')
        write(job)
        with LOCK:
            STOP_EVENTS[job_id] = threading.Event()
            TASKS.put((job_id, job['pending_action']))
        return public(job)
    except Exception:
        source.unlink(missing_ok=True)
        if not (folder / 'job.json').exists():
            folder.rmdir()
        raise
    finally:
        await file.close()


@app.get('/api/jobs/{job_id}')
def get_job(job_id: str):
    return public(read(job_id))


@app.post('/api/jobs/{job_id}/pause')
def pause(job_id: str):
    with LOCK:
        job = read(job_id)
        if job['status'] not in ACTIVE:
            raise HTTPException(409, '這份錄音目前沒有在處理。')
        STOP_EVENTS.setdefault(job_id, threading.Event()).set()
        update(job_id, message='正在暫停，請稍候…')
    return {'ok': True}


@app.post('/api/jobs/{job_id}/resume')
def resume(job_id: str):
    with LOCK:
        job = read(job_id)
        if job['status'] in ACTIVE:
            raise HTTPException(409, '這份錄音已在處理。')
        if job['status'] == 'done':
            raise HTTPException(409, '逐字稿已完成，請使用重新產生摘要。')
        update(job_id, status='queued', message='排隊等待續跑', error=None)
        STOP_EVENTS[job_id] = threading.Event()
        TASKS.put((job_id, job.get('pending_action', 'transcribe')))
    return {'ok': True}


@app.post('/api/jobs/{job_id}/summary')
def regenerate_summary(job_id: str):
    with LOCK:
        job = read(job_id)
        if job['status'] in ACTIVE or not job['transcript_done'] or not job['segments']:
            raise HTTPException(409, '請等待逐字稿完成後再產生摘要。')
        update(job_id, status='queued', message='排隊等待摘要', summary_error=None)
        STOP_EVENTS[job_id] = threading.Event()
        TASKS.put((job_id, 'summary'))
    return {'ok': True}


class PreviewSettings(BaseModel):
    start: float = Field(default=0, ge=0, lt=10800)
    minutes: Literal[3, 4, 5] = 3
    profile: Literal['light', 'standard'] = 'standard'


class FullSettings(BaseModel):
    profile: Literal['original', 'light', 'standard']


@app.post('/api/jobs/{job_id}/preview')
def retry_preview(job_id: str, settings: PreviewSettings):
    with LOCK:
        job = read(job_id)
        if not job.get('noise_mode') or job.get('workflow') != 'preview':
            raise HTTPException(409, '請從雜音錄音模式匯入檔案，整份處理開始後無法更改試轉設定。')
        if job['status'] in ACTIVE:
            raise HTTPException(409, '請等待目前試轉完成，或先暫停處理。')
        if job.get('duration') is not None and settings.start >= job['duration']:
            raise HTTPException(400, '試轉起點已超過錄音長度。')
        update(job_id, preview={}, preview_start=settings.start, preview_minutes=settings.minutes,
               noise_profile=settings.profile, status='queued', message='排隊等待重新試轉', progress=0,
               pending_action='preview', error=None)
        STOP_EVENTS[job_id] = threading.Event()
        TASKS.put((job_id, 'preview'))
    return public(read(job_id))


@app.post('/api/jobs/{job_id}/start-full')
def start_full(job_id: str, settings: FullSettings):
    with LOCK:
        job = read(job_id)
        preview = job.get('preview', {})
        if (not job.get('noise_mode') or job.get('workflow') != 'preview' or
                job['status'] != 'awaiting_choice' or
                not all(preview.get(v, {}).get('done') for v in ['original', 'denoised'])):
            raise HTTPException(409, '請先完成原音與降噪版的試轉比較，再處理整份錄音。')
        if settings.profile not in {'original', preview.get('profile')}:
            raise HTTPException(400, '請先試轉這個降噪設定，再套用到整份錄音。')
        update(job_id, workflow='full', audio_profile=settings.profile, status='queued',
               message='排隊等待處理整份錄音', pending_action='transcribe', progress=0, error=None,
               segments=[], completed_chunks=0, transcript_done=False, summary='', summary_parts=[], summary_error=None)
        STOP_EVENTS[job_id] = threading.Event()
        TASKS.put((job_id, 'transcribe'))
    return public(read(job_id))


@app.get('/api/jobs/{job_id}/preview/audio/{variant}')
def get_preview_audio(job_id: str, variant: Literal['original', 'denoised']):
    job = read(job_id)
    path = directory(job_id) / f'preview-{variant}.wav'
    if not job.get('preview', {}).get('audio_ready') or not path.exists():
        raise HTTPException(409, '試聽音訊尚未準備完成。')
    return FileResponse(path, media_type='audio/wav')


@app.delete('/api/jobs/{job_id}')
def delete_job(job_id: str):
    import shutil
    with LOCK:
        job = read(job_id)
        if job['status'] in ACTIVE:
            raise HTTPException(409, '請先暫停處理，再刪除錄音。')
        folder = directory(job_id).resolve()
        if folder.parent != JOBS.resolve():
            raise HTTPException(400, '無效的紀錄路徑。')
        shutil.rmtree(folder)
    return {'ok': True}


@app.get('/api/jobs/{job_id}/audio')
def get_audio(job_id: str):
    job = read(job_id)
    if job.get('audio_profile', 'original') != 'original':
        enhanced = directory(job_id) / 'enhanced.wav'
        if job.get('enhanced_profile') != job['audio_profile'] or not enhanced.exists():
            raise HTTPException(409, '整份降噪音訊尚未準備完成。')
        return FileResponse(enhanced, media_type='audio/wav')
    normalized = directory(job_id) / 'audio.wav'
    # Playback of normalized audio is compatible with the browser even for WMA inputs.
    if normalized.exists() and job.get('duration'):
        return FileResponse(normalized, media_type='audio/wav')
    return FileResponse(directory(job_id) / job['source'])


@app.get('/api/jobs/{job_id}/export/{kind}')
def export(job_id: str, kind: str, timestamps: bool = True):
    from urllib.parse import quote
    job = read(job_id)
    if kind == 'summary':
        if not job['summary']:
            raise HTTPException(409, '摘要尚未完成。')
        body, ext = job['summary'], 'md'
    elif kind in {'txt', 'srt'}:
        if not job['segments']:
            raise HTTPException(409, '目前還沒有可匯出的逐字稿。')
        body, ext = audio.export_transcript(job['segments'], kind, include_timestamps=timestamps), kind
    elif kind == 'json':
        body, ext = json.dumps(public(job), ensure_ascii=False, indent=2), 'json'
    else:
        raise HTTPException(400, '不支援的匯出格式。')
    filename = Path(job['filename']).stem + ('-摘要' if kind == 'summary' else '-逐字稿') + '.' + ext
    return Response(('\ufeff' + body).encode('utf-8') if ext in {'txt', 'srt'} else body.encode('utf-8'),
                    media_type='application/json' if ext == 'json' else 'text/plain; charset=utf-8',
                    headers={'Content-Disposition': "attachment; filename*=UTF-8''" + quote(filename)})


if (ROOT / 'web' / 'dist').exists():
    app.mount('/', StaticFiles(directory=ROOT / 'web' / 'dist', html=True), name='web')
