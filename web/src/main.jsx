import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { AudioLines, Mic, FileText, BookOpen, ShieldCheck, Laptop, Upload, FileUp, Check, X, Pause, Play, RefreshCw, Download, Search, Trash2, LoaderCircle, ArrowRight, AlertCircle } from 'lucide-react';
import { api, active, clock, statusText } from './api';
import './style.css';
import NoisePreview, { profileName } from './NoisePreview';
import TranscriptDisplay from './TranscriptDisplay';

function Sidebar({page, setPage}) {
  return <aside className="sidebar">
    <div className="brand"><AudioLines size={36}/><span>聽記</span></div>
    <nav aria-label="主要導覽">{[[Mic,'new','新增錄音'],[FileText,'history','會議紀錄'],[BookOpen,'help','使用說明']].map(([Icon,id,label]) =>
      <button key={id} onClick={() => setPage(id)} className={page === id ? 'nav-item selected' : 'nav-item'}><Icon size={23}/>{label}</button>)}</nav>
    <div className="privacy"><ShieldCheck size={19}/><span>本機處理 · 檔案不上傳</span></div>
  </aside>;
}

function UploadPanel({health, onCreated, notify}) {
  const [file,setFile] = useState(null), [language,setLanguage] = useState('zh');
  const [model,setModel] = useState('small'), [summary,setSummary] = useState(true);
  const [drag,setDrag] = useState(false), [busy,setBusy] = useState(false), [percent,setPercent] = useState(0);
  const [noiseMode,setNoiseMode] = useState(false), [trialStart,setTrialStart] = useState(0);
  const [trialMinutes,setTrialMinutes] = useState(3), [noiseProfile,setNoiseProfile] = useState('standard');
  const input = useRef(null), request = useRef(null);
  useEffect(() => () => request.current?.abort(), []);
  const choose = chosen => {
    if (!chosen) return;
    if (!/\.(mp3|m4a|wav|flac|ogg|aac|mp4|webm|wma|opus)$/i.test(chosen.name)) return notify('請選擇 MP3、M4A、WAV 等支援的音訊格式。');
    if (chosen.size > 4 * 1024 ** 3) return notify('檔案超過 4 GB，請先壓縮或分割。');
    if (!chosen.size) return notify('檔案是空的，請選擇其他錄音。');
    setFile(chosen);
  };
  async function submit() {
    if (!file) return;
    if (noiseMode && (!Number.isFinite(Number(trialStart)) || Number(trialStart) < 0 || Number(trialStart) >= 180)) return notify('試轉起點請填入 0 至 180 分鐘以內的位置。');
    const form = new FormData();
    form.append('file',file); form.append('language',language); form.append('model',model); form.append('auto_summary',String(summary));
    form.append('noise_mode',String(noiseMode)); form.append('preview_start',String(Number(trialStart)*60));
    form.append('preview_minutes',String(trialMinutes)); form.append('noise_profile',noiseProfile);
    setBusy(true); setPercent(0);
    const xhr = new XMLHttpRequest(); request.current = xhr;
    xhr.open('POST','/api/jobs');
    xhr.upload.onprogress = e => {if(e.lengthComputable) setPercent(Math.round(e.loaded/e.total*100));};
    xhr.onload = () => {
      setBusy(false); request.current = null;
      let body;
      try { body=JSON.parse(xhr.responseText); } catch { return notify('伺服器回應異常，請確認工具仍在執行。'); }
      if(xhr.status >= 200 && xhr.status < 300) {onCreated(body);setFile(null);}
      else notify(typeof body.detail === 'string' ? body.detail : '匯入失敗，請重試。');
    };
    xhr.onerror = () => {setBusy(false);notify('無法連線到本機服務，請確認啟動視窗仍開著。');};
    xhr.onabort = () => setBusy(false);
    xhr.send(form);
  }
  const missing = health && !health.models[model];
  return <section className="upload-panel panel" aria-label="匯入錄音">
    <div className="recording-modes" role="group" aria-label="錄音處理模式"><button aria-pressed={!noiseMode} disabled={busy} className={!noiseMode?'mode selected':'mode'} onClick={()=>setNoiseMode(false)}>一般錄音</button><button aria-pressed={noiseMode} disabled={busy || !health?.noise_mode_available} className={noiseMode?'mode selected':'mode'} onClick={()=>setNoiseMode(true)}>雜音錄音模式</button></div>
    {health && !health.noise_mode_available ? <p className="setup-note">雜音模式會在目前任務完成、服務更新後啟用。</p> : null}
    <input ref={input} type="file" hidden accept=".mp3,.m4a,.wav,.flac,.ogg,.aac,.mp4,.webm,.wma,.opus" onChange={e => {choose(e.target.files[0]); e.target.value='';}}/>
    <div className={`dropzone ${drag?'dragging':''} ${file?'has-file':''}`} onDragOver={e=>{e.preventDefault();if(!busy)setDrag(true);}} onDragLeave={()=>setDrag(false)} onDrop={e=>{e.preventDefault();setDrag(false);if(!busy)choose(e.dataTransfer.files[0]);}}>
      {file ? <><div className="file-icon"><AudioLines size={42}/></div><strong className="filename">{file.name}</strong><span className="muted">{(file.size/1024/1024).toFixed(1)} MB · 匯入後將檢查錄音長度</span><button disabled={busy} className="text-button" onClick={()=>setFile(null)}><X size={14}/>重新選擇</button></> :
      <><FileUp className="upload-icon" size={70} strokeWidth={1.35}/><h2>將錄音拖曳至此，或<button className="choose-file" disabled={busy} onClick={()=>input.current.click()}>選擇檔案</button></h2><p>MP3、M4A、WAV · 最長 3 小時</p></>}
    </div>
    <div className="settings">
      <label>語言<select aria-label="語言" disabled={busy} value={language} onChange={e=>setLanguage(e.target.value)}><option value="zh">中英混合（繁體）</option><option value="zh-only">全中文（繁體）</option><option value="auto">自動偵測</option><option value="en">英文</option><option value="ja">日文</option></select></label>
      <label>辨識模型<select aria-label="辨識模型" disabled={busy} value={model} onChange={e=>setModel(e.target.value)}>{['tiny','base','small','medium'].map(m=><option key={m} value={m}>{m}{health&&!health.models[m]?'（未下載）':''}</option>)}</select></label>
      <label>摘要<select aria-label="摘要" disabled={busy} value={String(summary)} onChange={e=>setSummary(e.target.value==='true')}><option value="true">自動產生</option><option value="false">稍後產生</option></select></label>
      <button className="primary start" disabled={!file||busy||missing||!health} onClick={submit}>{busy?<><LoaderCircle className="spin" size={18}/>匯入 {percent}%</>:<>{noiseMode?'先試轉比較':'開始轉錄'}<ArrowRight size={18}/></>}</button>
    </div>
    {noiseMode ? <div className="noise-options"><p>先試轉原音與降噪版，再選擇整份錄音的處理方式。</p><div className="preview-settings">
      <label>試轉起點（分鐘）<input aria-label="試轉起點（分鐘）" type="number" min="0" max="179.99" step="0.5" value={trialStart} disabled={busy} onChange={e=>setTrialStart(e.target.value)}/></label>
      <label>試轉長度<select aria-label="試轉長度" value={trialMinutes} disabled={busy} onChange={e=>setTrialMinutes(Number(e.target.value))}>{[3,4,5].map(m=><option key={m} value={m}>{m} 分鐘</option>)}</select></label>
      <label>降噪強度<select aria-label="降噪強度" value={noiseProfile} disabled={busy} onChange={e=>setNoiseProfile(e.target.value)}><option value="light">輕度降噪</option><option value="standard">標準降噪</option></select></label>
    </div></div> : null}
    {missing?<p className="setup-note"><AlertCircle size={15}/>請先執行「下載模型.cmd」準備 {model} 模型。</p>:null}
    {summary&&health&&!health.summary_ready?<p className="setup-note"><AlertCircle size={15}/>摘要模型尚未就緒；逐字稿仍可產生，之後可補做摘要。</p>:null}
  </section>;
}

function Workflow() {
  return <aside className="workflow panel"><h2>從錄音到行動</h2><ol>{[
    ['匯入會議錄音','支援 MP3、M4A、WAV 等常見音訊格式。'],
    ['產生時間戳逐字稿','自動辨識語音，生成帶時間戳的逐字稿，方便快速定位重點。'],
    ['整理摘要與待辦','自動產生會議摘要，提煉重點並整理待辦事項，協助落實行動。']
  ].map(([title,text],i)=><li key={title}><span className="step">0{i+1}</span><div><h3>{title}</h3><p>{text}</p></div></li>)}</ol></aside>;
}

function SummaryText({text}) {
  // Render plain text safely; never inject model output as HTML.
  return <div className="summary-text">{text.split('\n').map((line,i)=>line.startsWith('## ')?<h3 key={i}>{line.slice(3)}</h3>:line.startsWith('# ')?<h2 key={i}>{line.slice(2)}</h2>:<p key={i}>{line.replace(/\*\*/g,'')||'\u00a0'}</p>)}</div>;
}

function Results({job, onAction, notify, showTimestamps, onTimestampsChange}) {
  const [tab,setTab] = useState('transcript'), [query,setQuery] = useState('');
  const [current,setCurrent] = useState(0);
  const player = useRef(null);
  useEffect(()=>{setTab('transcript');setQuery('');setCurrent(0);},[job?.id]);
  const download = kind => {window.location.href=`/api/jobs/${job.id}/export/${kind}${kind==='txt'?`?timestamps=${showTimestamps}`:''}`;};
  const rows = job?.segments?.filter(s=>s.text.toLocaleLowerCase().includes(query.toLocaleLowerCase())) || [];
  const isActive=job&&active(job.status);
  return <section className="results panel" aria-label="轉錄結果">
    <div className="result-toolbar"><div role="tablist" aria-label="結果類型" className="tabs"><button role="tab" aria-selected={tab==='transcript'} className={tab==='transcript'?'tab current':'tab'} onClick={()=>setTab('transcript')}>逐字稿</button><button role="tab" aria-selected={tab==='summary'} className={tab==='summary'?'tab current':'tab'} onClick={()=>setTab('summary')}>會議摘要</button></div>
    <div className="exports"><span><Upload size={17}/>匯出</span><button disabled={!job?.segments?.length} onClick={()=>download('txt')}>TXT</button><button disabled={!job?.segments?.length} onClick={()=>download('srt')}>SRT</button>{tab==='summary'?<button disabled={!job?.summary} onClick={()=>download('summary')}>摘要</button>:null}</div></div>
    {!job?<div className="empty-result"><FileText size={68} strokeWidth={1.35}/><p>完成轉錄後，文字會顯示在這裡。</p></div>:<>
      <div className="job-info"><div><h3>{job.filename}</h3><span className="muted">{job.duration?clock(job.duration)+' · ':''}{statusText[job.status]} · {job.model}</span></div><div className="job-controls">
      {isActive?<button className="secondary compact" onClick={()=>onAction('pause')}><Pause size={15}/>暫停</button>:['paused','error'].includes(job.status)?<button className="secondary compact" onClick={()=>onAction('resume')}><Play size={15}/>繼續處理</button>:null}
      <button disabled={!job.transcript_done||!job.segments.length||isActive} className="secondary compact" onClick={()=>onAction('summary')}><RefreshCw size={15}/>{job.summary?'重新產生摘要':'產生摘要'}</button></div></div>
      <div className="job-state" role="status"><span>{isActive?<LoaderCircle className="spin" size={15}/>:job.status==='done'?<Check size={15}/>:null}{job.message}</span>{isActive?<span>{job.progress}%</span>:null}</div>
      {isActive?<div className={`progress ${job.status==='summarizing'?'indeterminate':''}`}><div style={{width:job.progress+'%'}}/></div>:null}
      {job.error?<div className="inline-error"><AlertCircle size={17}/><span>{job.error}</span></div>:null}
      {job.noise_mode?<p className="applied-profile">雜音錄音模式 · 整份處理使用{profileName[job.audio_profile] || '原音'} · 時間戳對應原始錄音</p>:null}
      {job.audio_ready?<div className="player"><audio key={`${job.id}-${job.audio_profile || 'original'}`} ref={player} src={`/api/jobs/${job.id}/audio`} controls preload="metadata" onTimeUpdate={e=>setCurrent(e.target.currentTime)}/><span>{showTimestamps?'點選時間戳可跳至錄音位置':'可使用播放器回聽錄音'}</span></div>:null}
      <div role="tabpanel" className="result-body">
      {tab==='transcript'?<>
        <div className="transcript-tools"><span>TXT 依此設定匯出；SRT 保留字幕時間</span><TranscriptDisplay showTimestamps={showTimestamps} onChange={onTimestampsChange}/></div>
        {job.segments.length?<label className="search"><Search size={17}/><input aria-label="搜尋逐字稿" value={query} placeholder="搜尋逐字稿…" onChange={e=>setQuery(e.target.value)}/><span>{rows.length} 段</span></label>:null}
        {!rows.length?<div className="small-empty">{job.segments.length?'找不到符合的文字。':job.transcript_done?'未偵測到可辨識的語音。':'正在等待辨識結果，完成一段就會保存。'}</div>:<div className="segments">{rows.map((s,i)=><div key={s.start+'-'+i} className={`segment ${current>=s.start&&current<s.end?'playing':''}`}>
          {showTimestamps?<div className="segment-stamp"><button className="timecode" title="播放此段" onClick={()=>{if(player.current){player.current.currentTime=s.start;player.current.play().catch(()=>notify('請使用播放器開始播放。'));}}}>{clock(s.start)}</button>{s.language ? <span className="language-label">{{zh:"繁體中文",en:"English",mixed:"中英混合",other:"其他"}[s.language]}</span> : null}</div>:null}<p>{s.text}</p></div>)}</div>}
      </>:<>
        {job.summary_error?<div className="inline-error"><AlertCircle size={17}/><span>{job.summary_error}</span></div>:null}
        {job.summary?<><p className="summary-note">AI 摘要可能有遺漏，請對照時間戳與原始錄音確認。</p><SummaryText text={job.summary}/></>:<div className="small-empty">{job.status==='summarizing'?'正在本機整理摘要，長錄音需要一些時間。':'摘要完成後，會議重點、決議與待辦會顯示在這裡。'}</div>}
      </>}
      </div>
    </>}
  </section>;
}

function History({jobs, select, remove}) {
  return <section className="panel history"><div className="section-heading"><h2>會議紀錄</h2><span className="muted">{jobs.length} 份錄音</span></div>{!jobs.length?<div className="empty-result"><FileText size={56}/><p>還沒有會議紀錄，從新增錄音開始。</p></div>:jobs.map(job=><div className="history-row" key={job.id}><div className="history-icon"><AudioLines size={22}/></div><button className="history-select" onClick={()=>select(job.id)}><strong>{job.filename}</strong><span>{new Date(job.created_at).toLocaleString('zh-TW')} · {job.duration?clock(job.duration)+' · ':''}{statusText[job.status]}</span></button><button className="icon-button danger" aria-label={`刪除 ${job.filename}`} disabled={active(job.status)} onClick={()=>remove(job)}><Trash2 size={18}/></button></div>)}</section>;
}

function Help({health}) {
  return <section className="panel help"><h2>讓每場會議都有跡可循</h2><p>選擇錄音後按「開始轉錄」。工具會檢查檔案、分段辨識，並在這台電腦上保存逐字稿及摘要。</p>
    <h3>第一次使用</h3><p>執行專案內的「下載模型.cmd」下載 small 辨識模型與本機摘要模型。首次準備需要網路，完成後錄音轉錄與摘要都可以離線使用。</p>
    <div className="model-status"><span className={health?.models?.small?'ready':''}><Check size={16}/>small 辨識模型：{health?.models?.small?'已就緒':'尚未下載'}</span><span className={health?.summary_ready?'ready':''}><Check size={16}/>本機摘要模型：{health?.summary_ready?'已就緒':'尚未下載'}</span></div>
    <h3>三小時長錄音</h3><p>每份錄音最長 3 小時、檔案最大 4 GB。以約 5 分鐘為一段辨識並保存，邊界保留重疊音訊。可以暫停；重開工具後，從「會議紀錄」選擇錄音並續跑。暫停會等目前模型操作結束。</p>
    <h3>雜音錄音模式</h3><p>選擇「雜音錄音模式」，設定試轉起點及 3、4 或 5 分鐘長度，先並排比較原音與降噪版的逐字稿，也可以分別回聽。覺得效果不好時，換位置或降噪強度重新試轉，無須重新上傳。選擇原音或已試轉的降噪設定後，才會處理整份錄音及摘要。不足設定長度時會處理至錄音結尾。</p><p>降噪以本機處理，不覆蓋原始檔、不刪除停頓、不改變速度。逐字稿與試轉時間戳都對應原始錄音。降噪無法保證改善爆音、回音或多人重疊說話，請比較文字是否完整，而不是只比較音量。</p>
    <h3>模型與速度</h3><p>small 為預設；tiny、base 較快但辨識精度可能較低，medium 需要更多記憶體。可在終端執行 .venv\Scripts\python.exe scripts\download_models.py --whisper medium --skip-summary 下載其他模型。CPU 處理長錄音可能花數小時，請保持啟動視窗開啟。</p>
    <h3>繁體中文與英文逐字稿</h3><p>預設使用「中英混合（繁體）」，適合中文、英文交錯的會議；中文轉為繁體中文，英文保留原文。全中文會議可選「全中文（繁體）」，固定以中文辨識，不在每段重新猜測語言。全英文會議可選「英文」。專有名詞、人名與多人同時發言仍需校對。</p>
    <h3>時間戳與匯出</h3><p>逐字稿與試轉比較都可選擇「顯示時間戳記」或「不顯示時間戳記」，切換不必重新轉錄。TXT 會依目前設定匯出；SRT 作為字幕，仍保留必要的時間資訊。顯示時間戳時可點擊回聽，摘要可下載為 Markdown。時間戳為辨識模型估計，並非逐字精準對齊。</p>
    <h3>摘要與隱私</h3><p>摘要會讀取完整逐字稿，分類挑出重點、決議、待辦與待確認事項，附上原文及來源時間戳；英文原文也會保留。分類可能有誤或遺漏，重要決議請回聽確認。此版本不自動區分發言者。錄音、暫存音訊與結果保存在專案的 data/jobs；從會議紀錄刪除會移除該份資料。</p>
  </section>;
}

function App() {
  const [page,setPage]=useState('new'), [health,setHealth]=useState(null), [jobs,setJobs]=useState([]), [job,setJob]=useState(null);
  const [error,setError]=useState(''), [deleteTarget,setDeleteTarget]=useState(null);
  const [showTimestamps,setShowTimestamps]=useState(()=>{try{return localStorage.getItem('tingji.transcript-display.v1')!=='hide';}catch{return true;}});
  function changeTimestamps(show){setShowTimestamps(show);try{localStorage.setItem('tingji.transcript-display.v1',show?'show':'hide');}catch{/* The display setting still works when browser storage is unavailable. */}}
  const selectedId=useRef(null), notified=useRef(false);
  const notify=message=>setError(message);
  useEffect(()=>{
    let live=true, timer;
    async function poll(){
      try{
        const [h,list]=await Promise.all([api('/health'),api('/jobs')]);
        if(!live)return;setHealth(h);setJobs(list);notified.current=false;
        if(selectedId.current){const detail=await api('/jobs/'+selectedId.current);if(live&&detail.id===selectedId.current)setJob(detail);}
      }catch(e){if(live&&!notified.current){setError(e.message==='Failed to fetch'?'無法連線到本機服務，請確認啟動視窗仍開著。':e.message);notified.current=true;}}
      finally{if(live)timer=setTimeout(poll,2500);}
    }
    poll();return()=>{live=false;clearTimeout(timer);};
  },[]);
  async function select(id){try{const detail=await api('/jobs/'+id);selectedId.current=id;setJob(detail);setPage('new');}catch(e){notify(e.message);}}
  async function action(name){try{await api(`/jobs/${job.id}/${name}`,{method:'POST'});setJob(await api('/jobs/'+job.id));}catch(e){notify(e.message);}}
  async function remove(){try{await api('/jobs/'+deleteTarget.id,{method:'DELETE'});setJobs(items=>items.filter(j=>j.id!==deleteTarget.id));if(selectedId.current===deleteTarget.id){selectedId.current=null;setJob(null);}setDeleteTarget(null);}catch(e){notify(e.message);}}
  return <div className="app-shell"><Sidebar page={page} setPage={setPage}/><main>
    <header className="page-header"><div><h1>{page==='new'?'把會議留下，讓重點清楚。':page==='history'?'每一場討論，都有紀錄。':'安心記錄，專注討論。'}</h1><p>{page==='new'?'將錄音轉成逐字稿、時間戳與會議摘要。':page==='history'?'回到會議現場，找到需要的重點。':'本機處理的使用方式與常見問題。'}</p></div><div className="local-mode"><Laptop size={18}/>本機模式</div></header>
    {error?<div role="alert" className="notice"><AlertCircle size={19}/><span>{error}</span><button aria-label="關閉訊息" onClick={()=>setError('')}><X size={17}/></button></div>:null}
    {page==='new'?<><div className="workspace"><UploadPanel health={health} notify={notify} onCreated={detail=>{selectedId.current=detail.id;setJob(detail);setJobs(prev=>[detail,...prev]);}}/><Workflow/></div>{job?.workflow==='preview'?<NoisePreview job={job} onUpdated={setJob} onAction={action} notify={notify} showTimestamps={showTimestamps} onTimestampsChange={changeTimestamps}/>:<><Results job={job} onAction={action} notify={notify} showTimestamps={showTimestamps} onTimestampsChange={changeTimestamps}/>{job?.noise_mode&&job.preview?.audio_ready?<details className="saved-preview"><summary>查看這份錄音的試轉比較</summary><NoisePreview job={job} onUpdated={setJob} onAction={action} notify={notify} showTimestamps={showTimestamps} onTimestampsChange={changeTimestamps} readonly/></details>:null}</>}</>:page==='history'?<History jobs={jobs} select={select} remove={setDeleteTarget}/>:<Help health={health}/>}
    <footer className="mobile-privacy"><ShieldCheck size={15}/>本機處理 · 檔案不上傳</footer>
  </main>{deleteTarget?<div className="modal-backdrop"><div role="dialog" aria-modal="true" aria-labelledby="delete-title" className="dialog"><h2 id="delete-title">刪除這份會議紀錄？</h2><p>「{deleteTarget.filename}」的錄音副本、逐字稿與摘要將從工具中刪除。</p><div><button className="secondary" onClick={()=>setDeleteTarget(null)}>取消</button><button className="primary delete" onClick={remove}>刪除紀錄</button></div></div></div>:null}</div>;
}

createRoot(document.getElementById('root')).render(<App/>);



