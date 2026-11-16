import React, { useState } from 'react';
import { api, clock } from './api';
import { SpeakerLabel, SegmentSpeaker } from './Speakers';
import './editor.css';

function joinText(parts){
  let text='';
  for(const part of parts){if(/[A-Za-z0-9]$/.test(text)&&/^[A-Za-z0-9]/.test(part.text))text+=' ';text+=part.text;}
  return text;
}

export function buildParagraphs(segments, seconds){
  const groups=[];
  segments.forEach((segment,segmentIndex)=>{
    const ids=segment.speakers?.length?segment.speakers:segment.speaker?[segment.speaker]:[];
    const signature=JSON.stringify([[...ids].sort(),Object.hasOwn(segment,'speaker')]);
    let group=groups.at(-1);
    if(!group||!seconds||group.signature!==signature||segment.end-group.start>seconds||segment.start-group.end>3||group.parts.length>=50){
      group={...segment,parts:[],signature,segmentIndex};groups.push(group);
    }
    group.parts.push({...segment,segmentIndex});group.end=segment.end;group.text=joinText(group.parts);
    if(group.language!==segment.language)group.language='mixed';
  });
  return groups;
}

export function ParagraphRow({group, job, ids, disabled, current, showTimestamps, seek, onAction}){
  const [snapshot,setSnapshot]=useState(null),[draft,setDraft]=useState({}),[busy,setBusy]=useState(false);
  const [showSpeakers,setShowSpeakers]=useState(false);
  function edit(){setSnapshot(group.parts);setDraft(Object.fromEntries(group.parts.map(p=>[p.segmentIndex,p.text])));setShowSpeakers(false);}
  async function save(e){
    e.preventDefault();
    const edits=snapshot.filter(p=>draft[p.segmentIndex]!==p.text).map(p=>({index:p.segmentIndex,start:p.start,end:p.end,expected_text:p.text,text:draft[p.segmentIndex]}));
    if(!edits.length){setSnapshot(null);return;}
    setBusy(true);try{if(await onAction('segments/text',{edits},'PATCH'))setSnapshot(null);}finally{setBusy(false);}
  }
  return <div className={`segment ${current>=group.start&&current<group.end?'playing':''}`}>
    {showTimestamps?<div className="segment-stamp"><button className="timecode" title="播放此段" onClick={()=>seek(group.start)}>{clock(group.start)}</button>
      {group.language?<span className="language-label">{{zh:'繁體中文',en:'English',mixed:'中英混合',other:'其他'}[group.language]}</span>:null}</div>:null}
    <div className="paragraph-content"><SpeakerLabel segment={group} names={job.speaker_names || {}}/>
      {snapshot?<form className="text-editor" onSubmit={save}><p className="editor-hint">依原始時間小段校正，保留字幕同步。儲存後閱讀段落會自動合併。</p>
        {snapshot.map(p=><label className="text-edit-field" key={p.segmentIndex}><span>第 {p.segmentIndex+1} 段 · {clock(p.start)}～{clock(p.end)}</span>
          <textarea aria-label={`第 ${p.segmentIndex+1} 段文字`} value={draft[p.segmentIndex]} maxLength={10000} required disabled={busy||disabled} onChange={e=>setDraft(prev=>({...prev,[p.segmentIndex]:e.target.value}))}/>
          {p.text_original!==undefined?<button type="button" className="text-button" disabled={busy||disabled} onClick={()=>setDraft(prev=>({...prev,[p.segmentIndex]:p.text_original}))}>使用原始辨識文字</button>:null}
        </label>)}<div className="editor-actions"><button type="submit" className="secondary compact" disabled={busy||disabled}>{busy?'儲存中…':'儲存文字'}</button><button type="button" className="secondary compact" disabled={busy} onClick={()=>setSnapshot(null)}>取消</button></div></form>:
        <><p className="paragraph-text">{group.text}</p><div className="editor-actions"><button type="button" className="text-button" disabled={disabled} aria-label={`編輯第 ${group.segmentIndex+1} 段文字`} onClick={edit}>編輯文字</button>{group.parts.some(p=>p.text_manual)?<span className="edited-label">文字已校正</span>:null}
          {ids.length&&group.parts.length>1?<button type="button" className="text-button" disabled={disabled} onClick={()=>setShowSpeakers(!showSpeakers)}>{showSpeakers?'收起講者修正':'修正此段講者'}</button>:null}</div></>}
      {!snapshot&&ids.length&&(group.parts.length===1||showSpeakers)?<div className="paragraph-speakers">{group.parts.map(p=><div key={p.segmentIndex}>
        {group.parts.length>1?<p className="source-line">{clock(p.start)} · {p.text}</p>:null}<SegmentSpeaker segment={p} ids={ids} names={job.speaker_names || {}} disabled={disabled} onSave={payload=>onAction(`segments/${p.segmentIndex}/speaker`,payload,'PATCH')}/>
      </div>)}</div>:null}
    </div>
  </div>;
}

export function BatchCorrection({job, disabled, onAction, notify}){
  const [find,setFind]=useState(''),[replace,setReplace]=useState(''),[preview,setPreview]=useState(null),[busy,setBusy]=useState(false),[saved,setSaved]=useState(false);
  async function inspect(e){e.preventDefault();setBusy(true);setSaved(false);setPreview(null);try{setPreview(await api(`/jobs/${job.id}/text-replacements/preview`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({find,replace})}));}catch(e){notify(e.message);}finally{setBusy(false);}}
  async function apply(){setBusy(true);try{if(await onAction('text-replacements',{find,replace,revision:preview.revision},'PATCH')){setPreview(null);setSaved(true);}}finally{setBusy(false);}}
  return <details className="batch-correction"><summary>批次修正相同錯字</summary><p className="editor-hint">只套用這份錄音，依完全相同文字替換（英文區分大小寫），不會更動講者姓名。先預覽再儲存。</p>
    <form onSubmit={inspect}><div className="batch-fields"><label>尋找錯字<input aria-label="尋找錯字" maxLength={2000} required value={find} disabled={disabled||busy} onChange={e=>{setFind(e.target.value);setPreview(null);setSaved(false);}}/></label><label>替換成<input aria-label="替換成" maxLength={2000} value={replace} disabled={disabled||busy} onChange={e=>{setReplace(e.target.value);setPreview(null);setSaved(false);}}/></label></div>
      <button className="secondary compact" disabled={disabled||busy||!find||find===replace} type="submit">{busy?'處理中…':'預覽修正'}</button></form>
    {preview?<div className="replacement-preview"><p role="status">找到 {preview.occurrences} 處，影響 {preview.segments} 個原始小段。{preview.segments>5?'以下顯示前 5 段。':''}</p>
      {preview.preview.map(p=><div className="replacement-example" key={p.index}><strong>第 {p.index+1} 段</strong><p>原文：{p.before}</p><p>修改：{p.after}</p></div>)}
      {preview.occurrences?<button type="button" className="secondary compact" disabled={disabled||busy} onClick={apply}>儲存全部修正（{preview.occurrences} 處）</button>:null}</div>:null}
    {saved?<p role="status" className="edited-label">批次修正已儲存，TXT、SRT 會套用。</p>:null}
  </details>;
}
