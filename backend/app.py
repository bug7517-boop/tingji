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

from . import audio, summary, transcription, noise, speakers, transcript, speaker_refinement, glossary

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('TINGJI_DATA', str(ROOT / 'data')))
JOBS = DATA / 'jobs'
MODELS = ROOT / 'models' / 'whisper'
MAX_BYTES = 4 * 1024**3
ALLOWED_EXT = {'.mp3', '.m4a', '.wav', '.flac', '.ogg', '.aac', '.mp4', '.webm', '.wma', '.opus'}
ACTIVE = {'queued', 'normalizing', 'loading', 'transcribing', 'summarizing', 'denoising', 'previewing', 'diarizing'}
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
        for attempt in range(6):
            try:
                temp.replace(path)
                break
            except PermissionError:
                if attempt==5:raise
                import time
                time.sleep(.02*(attempt+1))


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
    if result.get('speaker_optimization'):
        result['speaker_optimization']={k:v for k,v in result['speaker_optimization'].items() if k not in {'after','backup'}}
    result.pop('speaker_quality_turns',None)
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
        extra={'terms':job['glossary_terms']} if job.get('glossary_terms') else {}
        segments = noise.transcribe_clip(model, path, job['language'], preview['start'], event.is_set, **extra)
        preview[variant] = {'done': True, 'segments': segments}
        update(job_id, preview=preview)
    if event.is_set():
        raise InterruptedError()
    update(job_id, status='awaiting_choice', progress=100, error=None,
           message='試轉完成，請比較原音與降噪版，再選擇整份處理方式')


SPEAKER_FIELDS = ('speaker', 'speakers', 'speaker_uncertain', 'quality_review', 'speaker_optimized',
                  'speaker_match_score', 'speaker_match_margin')


def process_speaker_optimization(job_id, event):
    normalized=ensure_normalized(job_id,event)
    job=read(job_id)
    with MODEL_LOCK:WHISPER.clear()
    import gc
    gc.collect()
    update(job_id,status='diarizing',progress=0,message='以已確認的參考聲音校準講者，手動修正會保留')
    result=speakers.optimize(normalized,job,event.is_set,lambda ratio:update(job_id,progress=round(ratio*95,1)))
    if event.is_set():
        raise InterruptedError()
    with LOCK:
        latest=read(job_id)
        backup='speaker-optimization-'+uuid.uuid4().hex+'.json'
        (directory(job_id)/backup).write_text(json.dumps(latest,ensure_ascii=False),encoding='utf-8')
        changed=0
        for item in result['assignments']:
            segment=latest['segments'][item['index']]
            if segment.get('speaker_manual'):
                continue
            different=segment.get('speakers',[])!=[item['speaker']] or segment.get('speaker')!=item['speaker']
            changed+=different
            segment.update(speaker=item['speaker'],speakers=[item['speaker']],speaker_uncertain=False,
                           quality_review=False,speaker_optimized=bool(different),
                           speaker_match_score=item['score'],speaker_match_margin=item['margin'])
        for index in result['review']:
            if not latest['segments'][index].get('speaker_manual'):
                latest['segments'][index]['quality_review']=True
        after=[{key:s[key] for key in SPEAKER_FIELDS if key in s} for s in latest['segments']]
        latest.update(speaker_optimization={'model':result['model'],'changed':changed,'review':len(result['review']),
            'manual_protected':sum(bool(s.get('speaker_manual')) for s in latest['segments']),
            'reference_clips':result['reference_clips'],'backup':backup,'after':after},
            speaker_quality_turns=result['turns'],status='done',progress=100,error=None,
            message=f'講者優化完成：修正 {changed} 段，{len(result["review"])} 段請回聽確認')
        write(latest)


def process(job_id, action='transcribe'):
    event = STOP_EVENTS.setdefault(job_id, threading.Event())
    try:
        job = update(job_id, pending_action=action)
        if event.is_set():
            raise InterruptedError()
        if action == 'preview':
            process_preview(job_id, event)
            return
        if action == 'speaker-optimize':
            process_speaker_optimization(job_id,event)
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
                extra={'terms':job['glossary_terms']} if job.get('glossary_terms') else {}
                iterator = transcription.transcribe_regions(model, pcm, job['language'], event.is_set, **extra)
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
        if job.get('diarization') and not job.get('speakers_done') and job['segments'] and action != 'summary':
            try:
                with MODEL_LOCK:
                    WHISPER.clear()
                if 'model' in locals():
                    del model
                import gc
                gc.collect()
                normalized = ensure_normalized(job_id, event)
                update(job_id, status='diarizing', message='正在本機區分講者，長錄音需要一些時間', speaker_error=None)
                turns = job.get('speaker_turns')
                if turns is None:
                    turns = speakers.detect(normalized, job.get('speaker_count', 0), event.is_set,
                        lambda ratio: update(job_id, progress=round(ratio * 95, 1), message=
                            '正在比對整份錄音的講者聲音，完成後顯示講者標籤' if ratio >= 1 else '正在分析語音區間與講者聲音'))
                    update(job_id, speaker_turns=turns)
                if event.is_set():
                    raise InterruptedError()
                update(job_id, segments=speakers.label_segments(job['segments'], turns), speakers_done=True,
                       speaker_names={},speaker_quality_turns=None,speaker_optimization=None,speaker_references=[])
            except InterruptedError:
                raise
            except Exception as exc:
                update(job_id, speaker_error=f'講者辨識未完成：{exc}')
            job = read(job_id)
        if (job['auto_summary'] or action == 'summary') and job['segments'] and action != 'speakers':
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
                update(job_id, summary=OpenCC('s2twp').convert(result), summary_stale=False)
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
               message='逐字稿已完成；講者辨識需要重試' if latest.get('speaker_error') else
               '逐字稿已完成；摘要需要重試' if latest.get('summary_error') else
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
    return {'ok': True, 'local': True, 'noise_mode_available': True, 'summary_ready': summary.ready(), 'speakers_ready': speakers.ready(),
            'speaker_optimization_ready':speakers.ready() and speakers.ADVANCED.is_file(),
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


class CustomGlossary(BaseModel):
    terms: list[str] = Field(max_length=200)


@app.get('/api/glossary')
def get_glossary():
    data=glossary.catalog()
    with LOCK:
        custom=glossary.load_custom(DATA)
    return {'catalog':data,'custom_terms':custom}


@app.patch('/api/glossary')
def update_glossary(settings: CustomGlossary):
    try:
        with LOCK:
            terms=glossary.save_custom(DATA,settings.terms)
    except ValueError as exc:
        raise HTTPException(400,str(exc)) from exc
    return {'ok':True,'custom_terms':terms}


@app.post('/api/jobs', status_code=201)
async def create_job(file: UploadFile = File(...), language: str = Form('zh'),
                     model: str = Form('small'), auto_summary: bool = Form(True),
                     diarization: bool = Form(False), speaker_count: int = Form(0, ge=0, le=20),
                     domain: str = Form('general'), glossary_extra: str | None = Form(None,max_length=12000),
                     glossary_source: Literal['saved','provided'] = Form('saved'),
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
    if diarization and not speakers.ready():
        await file.close()
        raise HTTPException(409, '請先執行 setup.cmd 及「下載講者模型.cmd」準備講者辨識。')
    try:
        with LOCK:
            custom=glossary.load_custom(DATA) if glossary_source=='saved' and not glossary_extra else glossary.split(glossary_extra or '')
        vocabulary=glossary.terms_for(domain,custom)
    except ValueError as exc:
        await file.close()
        raise HTTPException(400,str(exc)) from exc
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
               'transcript_done': False, 'summary': '', 'summary_parts': [], 'summary_error': None, 'error': None,
               'diarization': diarization, 'speaker_count': speaker_count, 'speakers_done': False, 'speaker_error': None,
               'domain':domain,'glossary_terms':vocabulary}
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


class TextCorrection(BaseModel):
    index: int = Field(ge=0)
    start: float
    end: float
    expected_text: str = Field(max_length=10000)
    text: str = Field(min_length=1, max_length=10000)


class TextCorrections(BaseModel):
    edits: list[TextCorrection] = Field(min_length=1, max_length=100)


@app.patch('/api/jobs/{job_id}/segments/text')
def correct_text(job_id: str, settings: TextCorrections):
    with LOCK:
        job = read(job_id)
        if job['status'] in ACTIVE or not job.get('transcript_done'):
            raise HTTPException(409, '請等待逐字稿完成後再校正文字。')
        if len({edit.index for edit in settings.edits}) != len(settings.edits):
            raise HTTPException(400, '同一段落不能重複修改。')
        validated = []
        for edit in settings.edits:
            if edit.index >= len(job['segments']):
                raise HTTPException(404, '找不到這段逐字稿，請重新整理。')
            segment = job['segments'][edit.index]
            if (segment['start'], segment['end'], segment['text']) != (edit.start, edit.end, edit.expected_text):
                raise HTTPException(409, '文字已更新，請重新開啟編輯以避免覆蓋其他修改。')
            text = edit.text.replace('\r\n', '\n').strip()
            if not text or any((ord(c) < 32 and c not in '\n\t') or ord(c) == 127 for c in text):
                raise HTTPException(400, '文字不能空白或包含無效控制字元。')
            validated.append((segment, text))
        if transcript.apply_text(job, validated):
            write(job)
    return {'ok': True}


class Replacement(BaseModel):
    find: str = Field(min_length=1, max_length=2000)
    replace: str = Field(max_length=2000)
    revision: str | None = None


def replacement_plan(job, settings):
    if job['status'] in ACTIVE or not job.get('transcript_done'):
        raise HTTPException(409, '請等待逐字稿完成後再批次修正。')
    if any(ord(c) < 32 or ord(c) == 127 for c in settings.find + settings.replace):
        raise HTTPException(400, '批次替換不能包含換行或控制字元。')
    changes, preview, count = [], [], 0
    for index, segment in enumerate(job['segments']):
        occurrences = segment['text'].count(settings.find)
        if not occurrences:
            continue
        text = segment['text'].replace(settings.find, settings.replace).strip()
        if not text:
            raise HTTPException(400, '替換會使某段內容空白，請改用逐段編輯。')
        if len(text) > 10000:
            raise HTTPException(400, '替換後單段文字超過 10000 字。')
        changes.append((segment, text))
        count += occurrences
        if len(preview) < 5:
            preview.append({'index': index, 'before': segment['text'], 'after': text})
    return changes, {'occurrences': count, 'segments': len(changes), 'preview': preview,
                     'revision': transcript.revision(job['segments'])}


@app.post('/api/jobs/{job_id}/text-replacements/preview')
def preview_replacement(job_id: str, settings: Replacement):
    with LOCK:
        _, result = replacement_plan(read(job_id), settings)
        return result


@app.patch('/api/jobs/{job_id}/text-replacements')
def replace_text(job_id: str, settings: Replacement):
    with LOCK:
        job = read(job_id)
        changes, result = replacement_plan(job, settings)
        if not settings.revision or settings.revision != result['revision']:
            raise HTTPException(409, '逐字稿已變更，請重新預覽後再套用。')
        if transcript.apply_text(job, changes):
            write(job)
        return {'ok': True, 'occurrences': result['occurrences'], 'segments': result['segments']}


class SpeakerNames(BaseModel):
    names: dict[str, str] = Field(max_length=100)


@app.patch('/api/jobs/{job_id}/speaker-names')
def rename_speakers(job_id: str, settings: SpeakerNames):
    with LOCK:
        job = read(job_id)
        if job['status'] in ACTIVE:
            raise HTTPException(409, '請等待目前處理完成後再修改講者姓名。')
        known = speakers.catalog(job)
        if not set(settings.names).issubset(known):
            raise HTTPException(400, '包含這份逐字稿不存在的講者。')
        names = dict(job.get('speaker_names') or {})
        for speaker, value in settings.names.items():
            name = value.strip()
            if len(name) > 50 or any(ord(c) < 32 or ord(c) == 127 for c in value):
                raise HTTPException(400, '講者姓名最多 50 字，不能包含換行或控制字元。')
            if name:
                names[speaker] = name
            else:
                names.pop(speaker, None)
        update(job_id, speaker_names=names)
    return {'ok': True}


class SegmentSpeaker(BaseModel):
    speaker: str | None = None
    restore: bool = False
    start: float
    end: float


@app.patch('/api/jobs/{job_id}/segments/{index}/speaker')
def correct_segment_speaker(job_id: str, index: int, settings: SegmentSpeaker):
    with LOCK:
        job = read(job_id)
        if job['status'] in ACTIVE or not job.get('transcript_done'):
            raise HTTPException(409, '請等待逐字稿完成後再修改段落講者。')
        if not 0 <= index < len(job['segments']):
            raise HTTPException(404, '找不到這段逐字稿，請重新整理。')
        segment = job['segments'][index]
        if (segment['start'], segment['end']) != (settings.start, settings.end):
            raise HTTPException(409, '段落已更新，請重新整理後再修改。')
        if settings.speaker is not None and settings.speaker not in speakers.catalog(job):
            raise HTTPException(400, '這份紀錄不存在此講者。')
        keys = ('speaker', 'speakers', 'speaker_uncertain', 'quality_review')
        if settings.restore:
            original = segment.pop('speaker_original', None)
            if original is not None:
                for key in keys:
                    segment.pop(key, None)
                segment.update(original)
            segment.pop('speaker_manual', None)
        else:
            if 'speaker_original' not in segment:
                segment['speaker_original'] = {key: segment[key] for key in keys if key in segment}
            segment.update(speaker=settings.speaker, speakers=[settings.speaker] if settings.speaker else [],
                           speaker_uncertain=settings.speaker is None, speaker_manual=True)
            segment.pop('quality_review',None)
        write(job)
    return {'ok': True}


class SpeakerSettings(BaseModel):
    count: int = Field(default=0, ge=0, le=20)


class VoiceReference(BaseModel):
    speaker: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)


class SpeakerOptimization(BaseModel):
    references: list[VoiceReference] | None = Field(default=None,max_length=20)


@app.post('/api/jobs/{job_id}/speakers/optimize')
def optimize_speakers(job_id: str, settings: SpeakerOptimization):
    with LOCK:
        job=read(job_id)
        if job['status'] in ACTIVE or not job.get('transcript_done') or not job['segments']:
            raise HTTPException(409,'請等逐字稿完成後再優化講者。')
        if not speakers.ADVANCED.is_file():
            raise HTTPException(409,'請先執行「下載進階講者模型.cmd」。')
        known=speakers.catalog(job)
        refs=[r.model_dump() for r in settings.references] if settings.references is not None else job.get('speaker_references',[])
        for ref in refs:
            if ref['speaker'] not in known or ref['end']-ref['start']<2 or ref['end']>job['duration']:
                raise HTTPException(400,'參考聲音需屬於既有講者，至少 2 秒，且位於錄音範圍內。')
            if any(s.get('speaker_manual') and s.get('speaker') and s['speaker']!=ref['speaker'] and
                   min(s['end'],ref['end'])-max(s['start'],ref['start'])>=.8*(s['end']-s['start']) for s in job['segments']):
                raise HTTPException(400,'參考時間與已手動校正的講者不一致，請先確認時間或校正結果。')
        for i,ref in enumerate(refs):
            if any(other['speaker']!=ref['speaker'] and min(other['end'],ref['end'])>max(other['start'],ref['start']) for other in refs[i+1:]):
                raise HTTPException(400,'不同講者的參考聲音時間不可重疊。')
        candidate={**job,'speaker_references':refs}
        from collections import Counter
        counts=Counter(c['speaker'] for c in speaker_refinement.references(candidate))
        if len(known)<2 or any(counts[s]<2 for s in known):
            raise HTTPException(409,'每位講者至少需要兩個清楚聲音小段；可先手動校正，或提供約 8 秒以上的單人參考聲音。')
        update(job_id,speaker_references=refs,status='queued',pending_action='speaker-optimize',
               message='排隊等待依已確認的聲音優化講者',error=None)
        STOP_EVENTS[job_id]=threading.Event()
        TASKS.put((job_id,'speaker-optimize'))
    return {'ok':True}


@app.post('/api/jobs/{job_id}/speakers/undo-optimization')
def undo_speaker_optimization(job_id: str):
    with LOCK:
        job=read(job_id)
        if job['status'] in ACTIVE:
            raise HTTPException(409,'請先等待目前處理完成。')
        info=job.get('speaker_optimization') or {}
        name=info.get('backup','')
        if not re.fullmatch(r'speaker-optimization-[a-f0-9]{32}\.json',name):
            raise HTTPException(409,'沒有可還原的講者優化紀錄。')
        original=json.loads((directory(job_id)/name).read_text(encoding='utf-8'))
        if len(job['segments'])!=len(original['segments']):
            raise HTTPException(409,'逐字稿分段已改變，無法套用此備份。')
        for index,(current,before) in enumerate(zip(job['segments'],original['segments'])):
            if current.get('speaker_manual') or (current['start'],current['end'])!=(before['start'],before['end']):
                continue
            actual={key:current[key] for key in SPEAKER_FIELDS if key in current}
            if actual!=info['after'][index]:
                continue
            for key in SPEAKER_FIELDS:
                current.pop(key,None)
                if key in before:current[key]=before[key]
        job.pop('speaker_optimization',None)
        write(job)
    return {'ok':True}


@app.post('/api/jobs/{job_id}/speakers')
def identify_speakers(job_id: str, settings: SpeakerSettings):
    with LOCK:
        job = read(job_id)
        if job['status'] in ACTIVE or not job['transcript_done'] or not job['segments']:
            raise HTTPException(409, '請等待逐字稿完成後再辨識講者。')
        if not speakers.ready():
            raise HTTPException(409, '請先執行 setup.cmd 及「下載講者模型.cmd」。')
        update(job_id, status='queued', pending_action='speakers', message='排隊等待辨識講者',
               diarization=True, speaker_count=settings.count, speakers_done=False, speaker_turns=None, speaker_error=None)
        STOP_EVENTS[job_id] = threading.Event()
        TASKS.put((job_id, 'speakers'))
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
def export(job_id: str, kind: str, timestamps: bool = True, paragraphs: int = 0):
    from urllib.parse import quote
    job = read(job_id)
    if paragraphs not in {0, 30, 60}:
        raise HTTPException(400, '閱讀分段只支援原始、30 秒或 60 秒。')
    if kind == 'summary':
        if not job['summary']:
            raise HTTPException(409, '摘要尚未完成。')
        body, ext = job['summary'], 'md'
    elif kind in {'txt', 'srt'}:
        if not job['segments']:
            raise HTTPException(409, '目前還沒有可匯出的逐字稿。')
        segments = transcript.paragraphs(job['segments'], paragraphs) if kind == 'txt' and paragraphs else job['segments']
        body, ext = audio.export_transcript(segments, kind, include_timestamps=timestamps,
                                          speaker_names=job.get('speaker_names')), kind
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
