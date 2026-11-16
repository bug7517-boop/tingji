import React, { useEffect, useRef, useState } from 'react';
import { api } from './api';
import './glossary.css';

export default function DomainGlossary({domain,onDomainChange,text,onTextChange,disabled,notify}){
  const [catalog,setCatalog]=useState(null),[busy,setBusy]=useState(false),[saved,setSaved]=useState(false);
  const edited=useRef(false);
  useEffect(()=>{let live=true;api('/glossary').then(data=>{if(live){setCatalog(data.catalog);if(!edited.current)onTextChange(data.custom_terms.join('、'));}}).catch(()=>{});return()=>{live=false;};},[onTextChange]);
  async function save(){
    setBusy(true);try{const terms=text.split(/[\n\r,，、;；]+/).map(t=>t.trim()).filter(Boolean);const result=await api('/glossary',{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({terms})});onTextChange(result.custom_terms.join('、'));setSaved(true);}catch(e){notify(e.message);}finally{setBusy(false);}
  }
  return <div className="domain-glossary"><label>專業領域<select aria-label="專業領域" value={domain} disabled={disabled} onChange={e=>onDomainChange(e.target.value)}><option value="real_estate">農地與不動產</option><option value="general">一般會議</option></select></label>
    <p>{domain==='real_estate'?'已加入農地、都市計畫、市地重劃、農牧用地、農舍、建蔽率、容積率等專業用詞提示。':'一般辨識；仍可加入自訂專有名詞。'}詞庫幫助辨識用詞，無法保證完全正確，不會自動補寫內容。</p>
    <details><summary>查看詞庫／新增常用詞</summary>{domain==='real_estate'&&catalog?<div className="domain-categories">{catalog.categories.map(c=><div key={c.name}><strong>{c.name}</strong><p>{c.terms.join('、')}</p></div>)}<p>詞彙參考：{catalog.sources.map((s,i)=><React.Fragment key={s.url}>{i?'、':''}<a href={s.url} target="_blank" rel="noreferrer">{s.title}</a></React.Fragment>)}。此處不提供法律判斷。</p></div>:null}
      <label>自訂專有名詞<textarea aria-label="自訂專有名詞" maxLength={12000} placeholder="例如：地名、人名、公司名稱，以逗號或換行分隔" disabled={disabled||busy} value={text} onChange={e=>{edited.current=true;onTextChange(e.target.value);setSaved(false);}}/></label>
      <button className="secondary compact" type="button" disabled={disabled||busy} onClick={save}>儲存為常用詞</button>{saved?<span role="status">已存入本機詞庫，後續匯入可沿用。</span>:null}
      <p>最多 200 個自訂詞，每詞最多 50 字。儲存詞庫不會改寫既有逐字稿；每次轉錄只依提示長度限制取用詞彙。</p>
    </details></div>;
}
