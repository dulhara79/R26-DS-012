/** Canvas drawing helpers for timeline, radar, forecast, and sparklines */

export function drawSpark(canvas, data, color) {
  if (!canvas || data.length < 2) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const mn = Math.min(...data), mx = Math.max(...data), rng = mx - mn || 1;
  const step = W / (data.length - 1);
  ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = 'round';
  data.forEach((v, i) => {
    const x = i * step, y = H - ((v - mn) / rng) * (H - 8) - 4;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
}

export function drawTimeline(canvas, scores, threshold) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (scores.length < 2) {
    ctx.fillStyle = '#64748b'; ctx.font = '13px Inter'; ctx.textAlign = 'center';
    ctx.fillText('Awaiting data...', W / 2, H / 2); return;
  }
  const pad = { l: 50, r: 20, t: 15, b: 25 };
  const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;
  const mx = Math.max(...scores, threshold * 1.5) * 1.1;
  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.04)'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (ch / 4) * i;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.fillStyle = '#475569'; ctx.font = '10px JetBrains Mono'; ctx.textAlign = 'right';
    ctx.fillText((mx * (1 - i / 4)).toFixed(2), pad.l - 6, y + 3);
  }
  // Threshold
  const thY = pad.t + ch * (1 - threshold / mx);
  ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]);
  ctx.beginPath(); ctx.moveTo(pad.l, thY); ctx.lineTo(W - pad.r, thY); ctx.stroke();
  ctx.setLineDash([]);
  // Bars
  const bw = Math.max(2, cw / scores.length - 1);
  scores.forEach((s, i) => {
    const x = pad.l + (i / scores.length) * cw;
    const bh = (s / mx) * ch;
    ctx.fillStyle = s > threshold ? 'rgba(239,68,68,0.7)' : 'rgba(16,185,129,0.5)';
    ctx.fillRect(x, pad.t + ch - bh, bw, bh);
  });
}

export function drawRadar(canvas, zScores, featureNames) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const n = featureNames.length, cx = W / 2, cy = H / 2 + 10;
  const maxR = Math.min(cx, cy) - 40, step = (2 * Math.PI) / n, start = -Math.PI / 2;
  // Rings
  [0.25, 0.5, 0.75, 1].forEach(f => {
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const a = start + i * step, x = cx + Math.cos(a) * maxR * f, y = cy + Math.sin(a) * maxR * f;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.lineWidth = 1; ctx.stroke();
  });
  // Spokes + labels
  featureNames.forEach((fname, i) => {
    const a = start + i * step;
    const ex = cx + Math.cos(a) * maxR, ey = cy + Math.sin(a) * maxR;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ex, ey);
    ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.stroke();
    const lx = cx + Math.cos(a) * (maxR + 18), ly = cy + Math.sin(a) * (maxR + 18);
    ctx.fillStyle = '#64748b'; ctx.font = '9px Inter'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(fname.replace('mean_', '').replace('std_', 's_'), lx, ly);
  });
  if (!zScores) return;
  // Polygon
  const maxZ = 5;
  ctx.beginPath();
  featureNames.forEach((f, i) => {
    const z = Math.min(Math.abs(zScores[f] || 0), maxZ), r = (z / maxZ) * maxR;
    const a = start + i * step, x = cx + Math.cos(a) * r, y = cy + Math.sin(a) * r;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.closePath(); ctx.fillStyle = 'rgba(99,102,241,0.15)'; ctx.fill();
  ctx.strokeStyle = '#6366f1'; ctx.lineWidth = 2; ctx.stroke();
  // Dots
  featureNames.forEach((f, i) => {
    const z = Math.min(Math.abs(zScores[f] || 0), maxZ), r = (z / maxZ) * maxR;
    const a = start + i * step, x = cx + Math.cos(a) * r, y = cy + Math.sin(a) * r;
    ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fillStyle = z > 3 ? '#ef4444' : z > 1.5 ? '#f59e0b' : '#10b981'; ctx.fill();
  });
}

export function drawForecast(canvas, forecasts) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (forecasts.length < 2) {
    ctx.fillStyle = '#64748b'; ctx.font = '13px Inter'; ctx.textAlign = 'center';
    ctx.fillText('Awaiting forecast data...', W / 2, H / 2); return;
  }
  const pad = { l: 40, r: 15, t: 15, b: 25 };
  const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;
  const dy = pad.t + ch * 0.5;
  ctx.fillStyle = 'rgba(239,68,68,0.04)'; ctx.fillRect(pad.l, pad.t, cw, dy - pad.t);
  ctx.strokeStyle = 'rgba(239,68,68,0.3)'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(pad.l, dy); ctx.lineTo(W - pad.r, dy); ctx.stroke();
  ctx.setLineDash([]); ctx.fillStyle = '#ef4444'; ctx.font = '9px JetBrains Mono'; ctx.textAlign = 'left';
  ctx.fillText('P=0.5', pad.l + 4, dy - 4);
  ctx.fillStyle = '#475569'; ctx.font = '10px JetBrains Mono'; ctx.textAlign = 'right';
  ctx.fillText('1.0', pad.l - 5, pad.t + 4); ctx.fillText('0.0', pad.l - 5, pad.t + ch + 4);
  const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + ch);
  grad.addColorStop(0, 'rgba(239,68,68,0.8)'); grad.addColorStop(0.5, 'rgba(245,158,11,0.8)'); grad.addColorStop(1, 'rgba(16,185,129,0.8)');
  ctx.beginPath(); ctx.strokeStyle = grad; ctx.lineWidth = 2; ctx.lineJoin = 'round';
  forecasts.forEach((p, i) => {
    const x = pad.l + (i / (forecasts.length - 1)) * cw, y = pad.t + (1 - p) * ch;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke(); ctx.lineTo(pad.l + cw, pad.t + ch); ctx.lineTo(pad.l, pad.t + ch);
  ctx.closePath(); ctx.fillStyle = 'rgba(99,102,241,0.06)'; ctx.fill();
}
