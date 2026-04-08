// Complex Arithmetic / Phase Explorer - Interactive Canvas 2D
// Modes: cmul, ComplexNorm, ModReLU, Complex RoPE, Phase Alignment Score
import {
  cmul, cabs, cphase, complexToColor, phaseToHue, magPhaseColor,
  drawArrow, drawComplexArrow, drawAxes, drawUnitCircle, drawLabel,
  drawBox, clearCanvas, fitCanvas,
} from './canvas-utils.js';

let canvas, ctx, W, H;
let animId = null;
let currentMode = 'cmul';
let dragging = null;
let frameCount = 0;

// Interactive state
let vecA = [0.7, 0.5];
let vecB = [0.3, 0.8];
let vectors = [];
let ropeTheta = 0.5;
let ropePos = 6;

const MODES = ['cmul', 'norm', 'modrelu', 'rope', 'score'];

export function init(canvasEl) {
  canvas = canvasEl;
  vectors = Array.from({ length: 6 }, () => [(Math.random() - 0.5) * 1.6, (Math.random() - 0.5) * 1.6]);

  canvas.addEventListener('mousedown', onMouseDown);
  canvas.addEventListener('mousemove', onMouseMove);
  canvas.addEventListener('mouseup', () => { dragging = null; });
  canvas.addEventListener('mouseleave', () => { dragging = null; });
}

export function setMode(m) { currentMode = m; }
export function getMode() { return currentMode; }
export function getModes() { return MODES; }

function canvasToComplex(mx, my, cx, cy, scale) {
  return [(mx - cx) / scale, -(my - cy) / scale];
}

function onMouseDown(e) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio, 2);
  const mx = (e.clientX - rect.left);
  const my = (e.clientY - rect.top);
  const cx = W / 2, cy = H / 2;
  const scale = Math.min(W, H) / 3;

  if (currentMode === 'cmul' || currentMode === 'score') {
    const dA = Math.hypot(mx - (cx + vecA[0] * scale), my - (cy - vecA[1] * scale));
    const dB = Math.hypot(mx - (cx + vecB[0] * scale), my - (cy - vecB[1] * scale));
    if (dA < 20) dragging = 'A';
    else if (dB < 20) dragging = 'B';
  } else if (currentMode === 'norm' || currentMode === 'modrelu') {
    for (let i = 0; i < vectors.length; i++) {
      const dx = mx - (cx + vectors[i][0] * scale);
      const dy = my - (cy - vectors[i][1] * scale);
      if (Math.hypot(dx, dy) < 15) { dragging = i; break; }
    }
  }
}

function onMouseMove(e) {
  if (dragging === null) return;
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const cx = W / 2, cy = H / 2;
  const scale = Math.min(W, H) / 3;
  const [re, im] = canvasToComplex(mx, my, cx, cy, scale);

  if (dragging === 'A') { vecA = [re, im]; }
  else if (dragging === 'B') { vecB = [re, im]; }
  else if (typeof dragging === 'number') { vectors[dragging] = [re, im]; }
}

export function update(cfg) { /* no config dependency for now */ }
export function start() {
  if (animId) return;
  function loop() {
    animId = requestAnimationFrame(loop);
    frameCount++;
    draw();
  }
  loop();
}
export function stop() {
  if (animId) { cancelAnimationFrame(animId); animId = null; }
}

function draw() {
  const fit = fitCanvas(canvas);
  W = fit.w; H = fit.h; ctx = fit.ctx;
  clearCanvas(ctx, { width: W, height: H });

  drawLabel(ctx, W / 2, 12, 'Complex Arithmetic Explorer', {
    font: 'bold 14px Inter', color: '#58a6ff', align: 'center',
  });

  const cx = W / 2, cy = H / 2 + 10;
  const scale = Math.min(W, H) / 3;

  switch (currentMode) {
    case 'cmul': drawCmul(cx, cy, scale); break;
    case 'norm': drawNorm(cx, cy, scale); break;
    case 'modrelu': drawModReLU(cx, cy, scale); break;
    case 'rope': drawRope(cx, cy, scale); break;
    case 'score': drawScore(cx, cy, scale); break;
  }
}

function drawCmul(cx, cy, scale) {
  drawAxes(ctx, cx, cy, scale + 20);
  drawUnitCircle(ctx, cx, cy, scale);

  const [cr, ci] = cmul(vecA[0], vecA[1], vecB[0], vecB[1]);

  drawComplexArrow(ctx, cx, cy, vecA[0], vecA[1], scale, '#e06c75');
  drawComplexArrow(ctx, cx, cy, vecB[0], vecB[1], scale, '#61afef');
  drawComplexArrow(ctx, cx, cy, cr, ci, scale, '#98c379');

  // Dot labels near tips
  drawDotLabel(cx + vecA[0] * scale, cy - vecA[1] * scale, 'A', '#e06c75');
  drawDotLabel(cx + vecB[0] * scale, cy - vecB[1] * scale, 'B', '#61afef');
  drawDotLabel(cx + cr * scale, cy - ci * scale, 'A·B', '#98c379');

  const magA = cabs(vecA[0], vecA[1]), magB = cabs(vecB[0], vecB[1]), magC = cabs(cr, ci);
  const phA = cphase(vecA[0], vecA[1]) * 180 / Math.PI;
  const phB = cphase(vecB[0], vecB[1]) * 180 / Math.PI;
  const phC = cphase(cr, ci) * 180 / Math.PI;

  const infoY = H - 80;
  drawLabel(ctx, 20, infoY, `A: |${magA.toFixed(2)}| ∠${phA.toFixed(0)}°`, { font: '12px "Fira Code"', color: '#e06c75' });
  drawLabel(ctx, 20, infoY + 16, `B: |${magB.toFixed(2)}| ∠${phB.toFixed(0)}°`, { font: '12px "Fira Code"', color: '#61afef' });
  drawLabel(ctx, 20, infoY + 32, `A·B: |${magC.toFixed(2)}| ∠${phC.toFixed(0)}°`, { font: '12px "Fira Code"', color: '#98c379' });
  drawLabel(ctx, 20, infoY + 52, '|A·B| = |A|·|B|  and  arg(A·B) = arg(A)+arg(B)', { font: '11px Inter', color: '#8b949e' });
  drawLabel(ctx, W / 2, H - 12, 'Drag A (red) or B (blue) to explore multiplication', { font: '10px Inter', color: '#484f58', align: 'center' });
}

function drawNorm(cx, cy, scale) {
  drawAxes(ctx, cx, cy, scale + 20);
  drawUnitCircle(ctx, cx, cy, scale);

  let rmsSum = 0;
  for (const [r, i] of vectors) rmsSum += r * r + i * i;
  const rms = Math.sqrt(rmsSum / vectors.length + 1e-8);

  for (let i = 0; i < vectors.length; i++) {
    const [r, im] = vectors[i];
    drawComplexArrow(ctx, cx, cy, r, im, scale, 'rgba(150,150,150,0.4)');
    const nr = r / rms, ni = im / rms;
    drawComplexArrow(ctx, cx, cy, nr, ni, scale, complexToColor(nr, ni));
    drawDotLabel(cx + nr * scale, cy - ni * scale, `${i}`, complexToColor(nr, ni));
  }

  drawLabel(ctx, 20, H - 60, 'ComplexNorm: RMS-normalize magnitudes, preserve phase angles', { font: '12px Inter', color: '#c9d1d9' });
  drawLabel(ctx, 20, H - 40, `RMS = ${rms.toFixed(3)}  |  Gray = original, Color = normalized`, { font: '11px "Fira Code"', color: '#8b949e' });
  drawLabel(ctx, W / 2, H - 12, 'Drag vectors to see normalization in action', { font: '10px Inter', color: '#484f58', align: 'center' });
}

function drawModReLU(cx, cy, scale) {
  drawAxes(ctx, cx, cy, scale + 20);
  drawUnitCircle(ctx, cx, cy, scale);

  const threshold = 0.3;
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, threshold * scale, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(231,76,60,0.08)';
  ctx.fill();
  ctx.strokeStyle = 'rgba(231,76,60,0.4)';
  ctx.setLineDash([3, 3]);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();

  for (let i = 0; i < vectors.length; i++) {
    const [r, im] = vectors[i];
    const mag = cabs(r, im);
    drawComplexArrow(ctx, cx, cy, r, im, scale, 'rgba(150,150,150,0.3)');
    if (mag > threshold) {
      const newMag = mag - threshold;
      const phase = cphase(r, im);
      const nr = Math.cos(phase) * newMag;
      const ni = Math.sin(phase) * newMag;
      drawComplexArrow(ctx, cx, cy, nr, ni, scale, '#98c379');
    } else {
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx + r * scale, cy - im * scale, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#e74c3c';
      ctx.fill();
      ctx.restore();
    }
  }

  drawLabel(ctx, 20, H - 60, `ModReLU: max(0, |z| - b) · z/|z|  (b = ${threshold})`, { font: '12px Inter', color: '#c9d1d9' });
  drawLabel(ctx, 20, H - 40, 'Red zone = killed (|z| < threshold).  Green = survived, phase preserved.', { font: '11px Inter', color: '#8b949e' });
  drawLabel(ctx, W / 2, H - 12, 'Drag vectors across the threshold boundary', { font: '10px Inter', color: '#484f58', align: 'center' });
}

function drawRope(cx, cy, scale) {
  drawAxes(ctx, cx, cy, scale + 20, 'Position-dependent rotation');
  drawUnitCircle(ctx, cx, cy, scale);

  const T = 8;
  const theta = ropeTheta;
  ropePos = (frameCount * 0.01) % T;

  for (let t = 0; t < T; t++) {
    const angle = t * theta;
    const baseR = 0.6 * Math.cos(t * 0.5);
    const baseI = 0.6 * Math.sin(t * 0.5);
    const [rr, ri] = cmul(baseR, baseI, Math.cos(angle), Math.sin(angle));

    ctx.save();
    ctx.globalAlpha = 0.2;
    drawComplexArrow(ctx, cx, cy, baseR, baseI, scale, '#484f58');
    ctx.restore();

    const color = `hsl(${(t / T) * 360}, 70%, 55%)`;
    drawComplexArrow(ctx, cx, cy, rr, ri, scale, color);

    ctx.save();
    ctx.beginPath();
    ctx.arc(cx + rr * scale, cy - ri * scale, 3, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.font = '9px Inter';
    ctx.textAlign = 'center';
    ctx.fillText(`t${t}`, cx + rr * scale, cy - ri * scale - 8);
    ctx.restore();
  }

  // Show relative angle between consecutive tokens
  if (T >= 2) {
    const a1 = 0 * theta, a2 = 1 * theta;
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, scale * 0.3, -a1, -a2, a2 < a1);
    ctx.strokeStyle = '#f0883e';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.restore();
    drawLabel(ctx, cx + scale * 0.35, cy - 10, `Δθ=${(theta * 180 / Math.PI).toFixed(0)}°`, { font: '10px Inter', color: '#f0883e' });
  }

  drawLabel(ctx, 20, H - 60, `RoPE: x_t → x_t · e^{i·t·θ}   (θ = ${theta.toFixed(2)} rad)`, { font: '12px Inter', color: '#c9d1d9' });
  drawLabel(ctx, 20, H - 40, 'Each position t gets multiplied by a rotation. Relative position = relative angle.', { font: '11px Inter', color: '#8b949e' });
  drawLabel(ctx, 20, H - 20, 'Gray = original token embedding, Color = after RoPE rotation', { font: '10px Inter', color: '#484f58' });
}

function drawScore(cx, cy, scale) {
  drawAxes(ctx, cx, cy, scale + 20, 'Phase Alignment Scoring');
  drawUnitCircle(ctx, cx, cy, scale);

  drawComplexArrow(ctx, cx, cy, vecA[0], vecA[1], scale, '#e06c75');
  drawComplexArrow(ctx, cx, cy, vecB[0], vecB[1], scale, '#61afef');
  drawDotLabel(cx + vecA[0] * scale, cy - vecA[1] * scale, 'Q', '#e06c75');
  drawDotLabel(cx + vecB[0] * scale, cy - vecB[1] * scale, 'K', '#61afef');

  const [conjR, conjI] = [vecB[0], -vecB[1]]; // K*
  const [prodR, prodI] = cmul(vecA[0], vecA[1], conjR, conjI);
  const score = prodR; // Re(Q · K*)
  const magQ = cabs(vecA[0], vecA[1]), magK = cabs(vecB[0], vecB[1]);
  const phQ = cphase(vecA[0], vecA[1]), phK = cphase(vecB[0], vecB[1]);
  const angleDiff = phQ - phK;

  // Score bar
  const barW = 200, barH = 20;
  const barX = W / 2 - barW / 2, barY = 50;
  drawBox(ctx, barX, barY, barW, barH, '#21262d', '#30363d');
  const norm = magQ * magK;
  const fill = norm > 0 ? (score / norm + 1) / 2 : 0.5;
  const fillColor = score > 0 ? '#2ecc71' : '#e74c3c';
  ctx.fillStyle = fillColor;
  ctx.fillRect(barX + barW / 2, barY + 2, (fill - 0.5) * barW, barH - 4);
  drawLabel(ctx, W / 2, barY - 4, `Score: Re(Q·K*) = ${score.toFixed(3)}`, { font: '12px "Fira Code"', color: '#c9d1d9', align: 'center' });

  // Angle arc
  ctx.save();
  ctx.beginPath();
  const arcR = scale * 0.25;
  ctx.arc(cx, cy, arcR, -phQ, -phK, phK > phQ);
  ctx.strokeStyle = '#f0883e';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.restore();

  drawLabel(ctx, 20, H - 60, `Re(Q · K*) = |Q||K|cos(θ_Q - θ_K) = ${magQ.toFixed(2)}·${magK.toFixed(2)}·cos(${(angleDiff * 180 / Math.PI).toFixed(0)}°)`, { font: '12px "Fira Code"', color: '#c9d1d9' });
  drawLabel(ctx, 20, H - 40, 'Aligned phases → high score. Opposing phases → negative score.', { font: '11px Inter', color: '#8b949e' });
  drawLabel(ctx, W / 2, H - 12, 'Drag Q (red) and K (blue) to see how phase alignment affects score', { font: '10px Inter', color: '#484f58', align: 'center' });
}

function drawDotLabel(x, y, text, color) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, 4, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.fillStyle = '#fff';
  ctx.font = 'bold 10px Inter';
  ctx.textAlign = 'center';
  ctx.fillText(text, x, y - 8);
  ctx.restore();
}
