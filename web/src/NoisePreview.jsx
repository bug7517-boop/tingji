import React, { useEffect, useRef, useState } from 'react';
import { AlertCircle, Check, LoaderCircle, Pause, Play, RefreshCw } from 'lucide-react';
import { api, active, clock } from './api';
import './noise.css';
import TranscriptDisplay from './TranscriptDisplay';

export const profileName = { original: '原音', light: '輕度降噪', standard: '標準降噪' };

function PreviewCard({ job, variant, onChoose, busy, readonly, notify, showTimestamps }) {
  const player = useRef(null);
  const preview = job.preview || {};
  const result = preview[variant];
  const original = variant === 'original';
  const title = original ? '原音' : profileName[preview.profile || job.noise_profile];
  const profile = original ? 'original' : preview.profile;
  const segments = result?.segments || [];
  return <section className="preview-card" aria-label={`${title}試轉結果`}>
    <div className="preview-card-heading"><h3>{title}</h3><span>{result?.done ? `${segments.length} 段逐字稿` : '等待試轉'}</span></div>
    {preview.audio_ready ? <audio key={`${job.id}-${preview.start}-${preview.profile}`} ref={player} controls preload="metadata"
      src={`/api/jobs/${job.id}/preview/audio/${variant}?profile=${preview.profile}&start=${preview.start}`}
      onPlay={e => { const current = e.currentTarget; current.closest('.noise-comparison').querySelectorAll('audio').forEach(a => { if(a !== current) a.pause(); }); }}/>:null}
    <div className="preview-transcript">{segments.length ? segments.map((s, i) => <div className="segment" key={`${s.start}-${i}`}>
      {showTimestamps?<div className="segment-stamp"><button className="timecode" title="播放此段" onClick={() => {
        if(player.current) { player.current.currentTime = Math.max(0, s.start - preview.start); player.current.play().catch(() => notify('請使用播放器開始播放。')); }
      }}>{clock(s.start)}</button><span className="language-label">{{ zh: '繁體中文', en: 'English', mixed: '中英混合', other: '其他' }[s.language]}</span></div>:null}<p>{s.text}</p>
    </div>) : <p className="small-empty">{result?.done ? '這個版本未偵測到可辨識的語音。' : '完成此版本的試轉後，逐字稿會顯示在這裡。'}</p>}</div>
    {!readonly ? <button className={original ? 'secondary' : 'primary'} disabled={busy || job.status !== 'awaiting_choice'} onClick={() => onChoose(profile)}><Check size={16}/>{original ? '用原音處理整份' : '用這個降噪設定處理整份'}</button> : null}
  </section>;
}

export default function NoisePreview({ job, onUpdated, onAction, notify, readonly = false, showTimestamps, onTimestampsChange }) {
  const [start, setStart] = useState(job.preview_start / 60 || 0);
  const [minutes, setMinutes] = useState(job.preview_minutes || 3);
  const [profile, setProfile] = useState(job.noise_profile || 'standard');
  const [busy, setBusy] = useState(false);
  useEffect(() => { setStart(job.preview_start / 60 || 0); setMinutes(job.preview_minutes || 3); setProfile(job.noise_profile || 'standard'); }, [job.id]);
  const isActive = active(job.status);
  const preview = job.preview || {};
  async function request(path, body) {
    setBusy(true);
    try { onUpdated(await api(`/jobs/${job.id}/${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })); }
    catch(e) { notify(e.message); }
    finally { setBusy(false); }
  }
  return <section className="noise-comparison panel" aria-label="雜音錄音試轉比較">
    <div className="comparison-heading"><div><h2>先比較，再處理整份</h2><p>{job.filename}</p></div>{!readonly && isActive ? <button className="secondary compact" onClick={() => onAction('pause')}><Pause size={15}/>暫停</button> : !readonly && ['paused', 'error'].includes(job.status) ? <button className="secondary compact" onClick={() => onAction('resume')}><Play size={15}/>繼續試轉</button> : null}</div>
    {!readonly ? <div className="preview-settings">
      <label>試轉起點（分鐘）<input aria-label="重新試轉起點（分鐘）" type="number" min="0" max="179.99" step="0.5" value={start} disabled={busy || isActive} onChange={e => setStart(e.target.value)}/></label>
      <label>試轉長度<select aria-label="重新試轉長度" disabled={busy || isActive} value={minutes} onChange={e => setMinutes(Number(e.target.value))}>{[3,4,5].map(m => <option key={m} value={m}>{m} 分鐘</option>)}</select></label>
      <label>降噪強度<select aria-label="重新試轉降噪強度" disabled={busy || isActive} value={profile} onChange={e => setProfile(e.target.value)}><option value="light">輕度降噪</option><option value="standard">標準降噪</option></select></label>
      <button className="secondary" disabled={busy || isActive || !Number.isFinite(Number(start)) || Number(start) < 0 || Number(start) >= 180} onClick={() => request('preview', { start: Number(start) * 60, minutes, profile })}><RefreshCw size={15}/>重新試轉</button>
    </div> : null}
    <p className="comparison-note">{preview.audio_ready ? `比較範圍 ${clock(preview.start)} → ${clock(preview.start + preview.duration)}。` : ''}時間戳對應原始錄音；剩餘錄音不足設定長度時，試轉至檔案結尾。</p>
    {!readonly ? <div className="comparison-state" role="status">{isActive ? <LoaderCircle size={16} className="spin"/> : job.status === 'awaiting_choice' ? <Check size={16}/> : null}{job.message}</div> : null}
    {job.error && !readonly ? <div className="inline-error"><AlertCircle size={17}/><span>{job.error}</span></div> : null}
    <div className="transcript-tools"><span>兩個版本使用相同的顯示設定</span><TranscriptDisplay showTimestamps={showTimestamps} onChange={onTimestampsChange} label="試轉逐字稿時間戳記"/></div>
    <div className="comparison-columns">{['original','denoised'].map(variant => <PreviewCard key={variant} job={job} variant={variant} busy={busy} readonly={readonly} notify={notify} showTimestamps={showTimestamps} onChoose={selected => request('start-full', { profile: selected })}/>)}</div>
    <p className="comparison-note">請回聽並比較漏字、錯字及人聲是否完整；降噪不一定比較好。兩個版本都不理想時，可換試轉位置或降噪強度。</p>
  </section>;
}
