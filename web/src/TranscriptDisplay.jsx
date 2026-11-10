import React from 'react';
import './transcript.css';

export default function TranscriptDisplay({ showTimestamps, onChange, label = '逐字稿時間戳記' }) {
  return <label className="transcript-display">時間戳記<select aria-label={label} value={showTimestamps ? 'show' : 'hide'} onChange={e => onChange(e.target.value === 'show')}>
    <option value="show">顯示時間戳記</option><option value="hide">不顯示時間戳記</option>
  </select></label>;
}
