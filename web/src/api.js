export async function api(path, options = {}) {
  const res = await fetch('/api' + path, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(typeof body.detail === 'string' ? body.detail : '操作失敗，請稍後重試。');
  }
  return res.json();
}
export const active = status => ['queued', 'normalizing', 'loading', 'transcribing', 'summarizing', 'denoising', 'previewing'].includes(status);
export const clock = value => {
  const secs = Math.floor(value || 0);
  return [Math.floor(secs / 3600), Math.floor(secs / 60) % 60, secs % 60].map(x => String(x).padStart(2, '0')).join(':');
};
export const statusText = {queued: '排隊中', normalizing: '準備音訊', loading: '載入模型', transcribing: '轉錄中', summarizing: '整理摘要', denoising: '降噪中', previewing: '試轉中', awaiting_choice: '待選擇整份處理方式', done: '已完成', paused: '已暫停', error: '需要重試'};
