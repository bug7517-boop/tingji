import React, { useEffect, useState } from 'react';
import { clock } from './api';
import { knownSpeakerIds } from './Speakers';
import './quality.css';

export function parseReferenceTime(value){
  if(!/^\d+(?::\d{1,2}){0,2}(?:\.\d+)?$/.test(value.trim()))return NaN;
  const parts=value.trim().split(':').map(Number);
  if(parts.length>1&&parts.slice(1).some(n=>n>=60))return NaN;
  return parts.reduce((total,n)=>total*60+n,0);
}

export default function SpeakerQuality({job,health,disabled,onAction,notify}){
  const ids=knownSpeakerIds(job), refsKey=JSON.stringify(job.speaker_references || []);
  const [ranges,setRanges]=useState({}),[busy,setBusy]=useState(false);
  useEffect(()=>{setRanges(Object.fromEntries(JSON.parse(refsKey).map(r=>[r.speaker,{start:clock(r.start),end:clock(r.end)}])));},[job.id,refsKey]);
  if(!ids.length||!job.transcript_done)return null;
  async function optimize(){
    const references=[];
    for(const id of ids){
      const range=ranges[id];if(!range?.start&&!range?.end)continue;
      const start=parseReferenceTime(range.start || ''),end=parseReferenceTime(range.end || '');
      if(!Number.isFinite(start)||!Number.isFinite(end)||end-start<2||end>job.duration){notify('請填寫正確起訖時間，例如 00:07:50～00:08:15，且至少 2 秒。');return;}
      references.push({speaker:id,start,end});
    }
    setBusy(true);try{await onAction('speakers/optimize',{references});}finally{setBusy(false);}
  }
  return <details className="speaker-quality"><summary>講者容易混淆？使用參考聲音優化</summary>
    <p>會保留已手動校正的段落與姓名，用這些聲音搭配進階模型重新比對。每位講者可提供一段 8～25 秒、沒有其他人插話的清楚語音；若已手動校正足夠片段，可留白。</p>
    {!health?.speaker_optimization_ready?<p>請先執行「下載進階講者模型.cmd」（約 68 MB）。</p>:null}
    <div className="reference-ranges">{ids.map(id=><fieldset key={id}><legend>{job.speaker_names?.[id] || '講者 '+id.replace('speaker_','')}</legend>
      {['start','end'].map(field=><label key={field}>{field==='start'?'起點':'終點'}<input aria-label={`${id} 參考${field==='start'?'起點':'終點'}`} placeholder="00:00:00" disabled={disabled||busy} value={ranges[id]?.[field] || ''} onChange={e=>setRanges(prev=>({...prev,[id]:{...prev[id],[field]:e.target.value}}))}/></label>)}
    </fieldset>)}</div><button className="secondary compact" disabled={disabled||busy||!health?.speaker_optimization_ready} onClick={optimize}>依參考聲音優化講者</button>
    {job.speaker_optimization?<div className="optimization-result"><p>已調整 {job.speaker_optimization.changed} 段；保留 {job.speaker_optimization.manual_protected} 段手動校正；{job.speaker_optimization.review} 段仍待確認，請回聽。</p><button className="secondary compact" disabled={disabled||busy} onClick={()=>onAction('speakers/undo-optimization')}>還原上次講者優化</button></div>:null}
  </details>;
}
