// Shared Canvas 2D drawing utilities for complex number visualization

export function cmul(ar, ai, br, bi) { return [ar*br - ai*bi, ar*bi + ai*br]; }
export function cabs(r, i) { return Math.sqrt(r*r + i*i + 1e-12); }
export function cconj(r, i) { return [r, -i]; }
export function cphase(r, i) { return Math.atan2(i, r); }

export function phaseToHue(phase) { return ((phase / Math.PI + 1) * 180) % 360; }

export function complexToColor(r, i, alpha = 1) {
  const mag = cabs(r, i);
  const hue = phaseToHue(cphase(r, i));
  const light = Math.min(70, 30 + mag * 40);
  return `hsla(${hue},80%,${light}%,${alpha})`;
}

export function magPhaseColor(mag, phase, maxMag = 1) {
  const hue = phaseToHue(phase);
  const brightness = Math.min(90, 20 + (mag / maxMag) * 70);
  return `hsl(${hue},75%,${brightness}%)`;
}

export function generateRandomComplex(n, scale = 1) {
  const out = [];
  for (let i = 0; i < n; i++) {
    out.push([(Math.random() - 0.5) * 2 * scale, (Math.random() - 0.5) * 2 * scale]);
  }
  return out;
}

export function drawArrow(ctx, x1, y1, x2, y2, color, lineWidth = 2) {
  const headLen = 8;
  const angle = Math.atan2(y2 - y1, x2 - x1);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - headLen * Math.cos(angle - 0.35), y2 - headLen * Math.sin(angle - 0.35));
  ctx.lineTo(x2 - headLen * Math.cos(angle + 0.35), y2 - headLen * Math.sin(angle + 0.35));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

export function drawComplexArrow(ctx, cx, cy, real, imag, scale, color) {
  const x2 = cx + real * scale;
  const y2 = cy - imag * scale;
  drawArrow(ctx, cx, cy, x2, y2, color || complexToColor(real, imag));
}

export function drawAxes(ctx, cx, cy, size, label) {
  ctx.save();
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cx - size, cy); ctx.lineTo(cx + size, cy);
  ctx.moveTo(cx, cy - size); ctx.lineTo(cx, cy + size);
  ctx.stroke();
  ctx.fillStyle = '#484f58';
  ctx.font = '10px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Re', cx + size - 5, cy + 14);
  ctx.fillText('Im', cx + 12, cy - size + 8);
  if (label) {
    ctx.fillStyle = '#8b949e';
    ctx.font = '12px Inter, sans-serif';
    ctx.fillText(label, cx, cy - size - 8);
  }
  ctx.restore();
}

export function drawUnitCircle(ctx, cx, cy, radius) {
  ctx.save();
  ctx.strokeStyle = '#21262d';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
}

export function drawComplexHeatmap(ctx, x, y, cellW, cellH, matrix, maxMag) {
  const d = matrix.length;
  if (!maxMag) {
    maxMag = 0;
    for (let r = 0; r < d; r++)
      for (let c = 0; c < (matrix[r]?.length || 0); c++)
        maxMag = Math.max(maxMag, cabs(matrix[r][c][0], matrix[r][c][1]));
    maxMag = maxMag || 1;
  }
  for (let r = 0; r < d; r++) {
    for (let c = 0; c < (matrix[r]?.length || 0); c++) {
      const [re, im] = matrix[r][c];
      const mag = cabs(re, im);
      const phase = cphase(re, im);
      ctx.fillStyle = magPhaseColor(mag, phase, maxMag);
      ctx.fillRect(x + c * cellW, y + r * cellH, cellW - 1, cellH - 1);
    }
  }
}

export function drawVectorBars(ctx, x, y, w, h, vec, label, maxVal) {
  const n = vec.length;
  if (!n) return;
  const barW = Math.min(w / n - 1, 20);
  const mid = y + h / 2;
  if (!maxVal) {
    maxVal = 0;
    for (const [r, i] of vec) maxVal = Math.max(maxVal, Math.abs(r), Math.abs(i));
    maxVal = maxVal || 1;
  }
  const scale = (h / 2 - 4) / maxVal;
  ctx.save();
  for (let i = 0; i < n; i++) {
    const bx = x + i * (barW + 1);
    const [re, im] = vec[i];
    ctx.fillStyle = '#4a90d9';
    const rh = -re * scale;
    ctx.fillRect(bx, mid, barW / 2, rh);
    ctx.fillStyle = '#e74c3c';
    const ih = -im * scale;
    ctx.fillRect(bx + barW / 2, mid, barW / 2, ih);
  }
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 0.5;
  ctx.beginPath(); ctx.moveTo(x, mid); ctx.lineTo(x + w, mid); ctx.stroke();
  if (label) {
    ctx.fillStyle = '#8b949e';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(label, x, y + 12);
  }
  ctx.restore();
}

export function drawLabel(ctx, x, y, text, opts = {}) {
  ctx.save();
  ctx.fillStyle = opts.color || '#c9d1d9';
  ctx.font = opts.font || '12px Inter, sans-serif';
  ctx.textAlign = opts.align || 'left';
  ctx.textBaseline = opts.baseline || 'top';
  ctx.fillText(text, x, y);
  ctx.restore();
}

export function drawBox(ctx, x, y, w, h, fill, stroke) {
  ctx.save();
  if (fill) { ctx.fillStyle = fill; ctx.fillRect(x, y, w, h); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.strokeRect(x, y, w, h); }
  ctx.restore();
}

export function clearCanvas(ctx, canvas) {
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

export function fitCanvas(canvas) {
  const par = canvas.parentElement;
  const dpr = Math.min(window.devicePixelRatio, 2);
  const w = par.clientWidth;
  const h = par.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  return { w, h, ctx };
}
