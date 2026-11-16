import React from 'react';
import './speakers.css';

export function SpeakerCount({value, onChange, disabled, label='講者人數'}) {
  return <label>{label}<select aria-label={label} value={value} disabled={disabled} onChange={e=>onChange(Number(e.target.value))}>
    <option value="0">自動估計</option>{Array.from({length:20},(_,i)=>i+1).map(n=><option key={n} value={n}>{n} 位</option>)}
  </select></label>;
}

export function SpeakerLabel({segment}) {
  if (!Object.hasOwn(segment,'speaker')) return null;
  const ids=segment.speakers?.length?segment.speakers:segment.speaker?[segment.speaker]:[];
  return <span className={`speaker-label ${ids.length!==1?'uncertain':''}`}>
    {ids.length?ids.map(id=>'講者 '+id.replace('speaker_','')).join('／'):'講者未辨識'}
    {ids.length>1?'（交接或重疊，請回聽）':''}
  </span>;
}
