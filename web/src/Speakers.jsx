import React, { useEffect, useState } from 'react';
import './speakers.css';

export function knownSpeakerIds(job) {
  const ids=new Set(Object.keys(job.speaker_names || {}));
  for(const s of [...job.segments,...(job.speaker_turns || []),...job.segments.flatMap(s=>s.speaker_original?[s.speaker_original]:[])]) {
    for(const id of s.speakers?.length?s.speakers:s.speaker?[s.speaker]:[])ids.add(id);
  }
  return [...ids].sort((a,b)=>Number(a.replace('speaker_',''))-Number(b.replace('speaker_','')));
}

export function SpeakerCount({value, onChange, disabled, label='講者人數'}) {
  return <label>{label}<select aria-label={label} value={value} disabled={disabled} onChange={e=>onChange(Number(e.target.value))}>
    <option value="0">自動估計</option>{Array.from({length:20},(_,i)=>i+1).map(n=><option key={n} value={n}>{n} 位</option>)}
  </select></label>;
}

export function SpeakerLabel({segment, names={}}) {
  if (!Object.hasOwn(segment,'speaker')) return null;
  const ids=segment.speakers?.length?segment.speakers:segment.speaker?[segment.speaker]:[];
  return <span className={`speaker-label ${ids.length!==1?'uncertain':''}`}>
    {ids.length?ids.map(id=>names[id] || '講者 '+id.replace('speaker_','')).join('／'):'講者未辨識'}
    {ids.length>1?'（交接或重疊，請回聽）':''}
  </span>;
}

export function SpeakerNames({job, disabled, onSave}) {
  const ids=knownSpeakerIds(job);
  const namesKey=JSON.stringify(job.speaker_names || {});
  const [draft,setDraft]=useState(job.speaker_names || {});
  const [busy,setBusy]=useState(false), [saved,setSaved]=useState(false);
  useEffect(()=>{setDraft(JSON.parse(namesKey));},[job.id,namesKey]);
  if(!ids.length)return null;
  async function save(e){
    e.preventDefault();setBusy(true);setSaved(false);
    try {setSaved(await onSave(Object.fromEntries(ids.map(id=>[id,(draft[id] || '').trim()]))));}
    finally {setBusy(false);}
  }
  return <form className="speaker-names" onSubmit={save}>
    <h4>講者姓名</h4><p>儲存後，逐字稿與 TXT、SRT 匯出都使用這些姓名。留白可恢復講者編號；重新辨識講者完成後需重新設定姓名。</p>
    <div className="speaker-name-fields">{ids.map(id=><label key={id}>{'講者 '+id.replace('speaker_','')}<input aria-label={'講者 '+id.replace('speaker_','')+'姓名'} maxLength={50} disabled={disabled||busy} value={draft[id] || ''} placeholder="輸入實際姓名" onChange={e=>{setDraft(prev=>({...prev,[id]:e.target.value}));setSaved(false);}}/></label>)}</div>
    <button className="secondary compact" type="submit" disabled={disabled||busy}>{busy?'儲存中…':'儲存講者姓名'}</button>
    {saved?<span className="speaker-save-status" role="status">姓名已儲存，匯出也會套用。</span>:null}
  </form>;
}

export function SegmentSpeaker({segment, ids, names={}, disabled, onSave}) {
  const [busy,setBusy]=useState(false);
  const current=segment.speakers?.length>1?'mixed':segment.speaker || 'unknown';
  async function save(value){
    setBusy(true);
    try {await onSave({speaker:value==='unknown'||value==='restore'?null:value,restore:value==='restore',start:segment.start,end:segment.end});}
    finally {setBusy(false);}
  }
  return <div className="segment-speaker-edit"><label>此段講者<select aria-label={`第 ${segment.segmentIndex+1} 段講者`} value={current} disabled={disabled||busy} onChange={e=>save(e.target.value)}>
    {current==='mixed'?<option value="mixed" disabled>多位講者／待確認</option>:null}
    <option value="unknown">講者未辨識</option>{ids.map(id=><option value={id} key={id}>{names[id] || '講者 '+id.replace('speaker_','')}</option>)}
    {segment.speaker_manual?<option value="restore">恢復自動辨識結果</option>:null}
  </select></label><span role="status">{busy?'儲存中…':segment.speaker_manual?'已手動修正':''}</span></div>;
}
