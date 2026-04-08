// CGU (Complex Gated Unit) Detail View - Canvas 2D animation
// Visualizes: gated = sigmoid(|gate|) · (gate_phase ⊙ up)
import {
  cmul, cabs, cphase, generateRandomComplex, complexToColor,
  drawArrow, drawComplexArrow, drawAxes, drawUnitCircle, drawLabel,
  drawBox, clearCanvas, fitCanvas, magPhaseColor,
} from './canvas-utils.js';

let canvas, ctx, W, H;
let playing = false, animStep = 0, speed = 1, animId = null;
let frameCount = 0;
const TOTAL_ANIM_STEPS = 5;
let inputZ = [], gateOut = [], upOut = [], activated = [], gated = [], finalOut = [];
let expand = 3, dim = 4;

export function init(canvasEl) {
  canvas = canvasEl;
  generateData();
}

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function modReLU(r, i, bias) {
  const mag = cabs(r, i);
  const newMag = Math.max(0, mag + bias);
  if (mag < 1e-8) return [0, 0];
  return [r * newMag / mag, i * newMag / mag];
}

function generateData() {
  const d = dim;
  const hiddenD = d * expand;
  inputZ = generateRandomComplex(d, 1.0);
  gateOut = generateRandomComplex(hiddenD, 0.8);
  upOut = generateRandomComplex(hiddenD, 0.8);
  activated = upOut.map(([r, i]) => modReLU(r, i, -0.1));
  gated = [];
  for (let j = 0; j < hiddenD; j++) {
    const [gr, gi] = gateOut[j];
    const gateMag = sigmoid(cabs(gr, gi));
    const gateAngle = cphase(gr, gi);
    const gatePhaseR = Math.cos(gateAngle);
    const gatePhaseI = Math.sin(gateAngle);
    const [pr, pi] = cmul(gatePhaseR, gatePhaseI, activated[j][0], activated[j][1]);
    gated.push([pr * gateMag, pi * gateMag]);
  }
  finalOut = gated.slice(0, d);
}

export function update(cfg) {
  expand = cfg.expand || cfg.bank_expand || 3;
  dim = Math.min(cfg.dim || 384, 6);
  generateData();
}

export function play() { playing = true; }
export function pause() { playing = false; }
export function reset() { animStep = 0; frameCount = 0; generateData(); }
export function setSpeed(s) { speed = s; }

export function stepForward() {
  if (animStep < TOTAL_ANIM_STEPS - 1) animStep++;
}
export function stepBack() {
  if (animStep > 0) animStep--;
}

export function start() {
  if (animId) return;
  function loop() {
    animId = requestAnimationFrame(loop);
    frameCount++;
    if (playing && frameCount % Math.max(1, Math.round(40 / speed)) === 0) {
      animStep++;
      if (animStep >= TOTAL_ANIM_STEPS) { animStep = TOTAL_ANIM_STEPS - 1; playing = false; }
    }
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

  drawLabel(ctx, W / 2, 12, `CGU (Complex Gated Unit) — SwiGLU-style Complex Gating`, {
    font: 'bold 14px Inter', color: '#58a6ff', align: 'center',
  });

  const panelW = Math.min((W - 80) / 3, 260);
  const panelH = Math.min(H - 120, 340);
  const planeR = Math.min(panelW, panelH) / 2 - 30;
  const startX = 30;
  const midY = 80 + panelH / 2;

  // Step labels
  const stepLabels = [
    '① Input z (complex vector)',
    '② gate = gate_proj(z)',
    '③ up = act(up_proj(z))  [ModReLU]',
    '④ gated = σ(|gate|) · (gate_phase ⊙ up)',
    '⑤ out = down_proj(gated)',
  ];

  drawBox(ctx, startX - 4, 55, W - startX * 2 + 8, 24, '#161b22', '#30363d');
  drawLabel(ctx, startX, 60, stepLabels[animStep], { font: 'bold 12px Inter', color: '#7ee787' });

  if (animStep >= 0) drawInputPanel(startX, 90, panelW, panelH, planeR);
  if (animStep >= 1) drawGatePanel(startX + panelW + 20, 90, panelW, panelH, planeR);
  if (animStep >= 3) drawOutputPanel(startX + (panelW + 20) * 2, 90, panelW, panelH, planeR);

  if (animStep >= 2) drawActivationPanel(startX + panelW + 20, 90 + panelH / 2 + 10, panelW, panelH / 2 - 10);

  // Gate dial metaphor at bottom
  if (animStep >= 3) drawGateDial(W / 2, H - 65);

  // Equation
  drawLabel(ctx, 30, H - 22, 'fused_cgu_gate(gate, up) = σ(|gate|) · (e^{iφ_gate} ⊙ act(up))', {
    font: '11px "Fira Code", monospace', color: '#c9d1d9',
  });
}

function drawInputPanel(x, y, w, h, r) {
  drawBox(ctx, x, y, w, h / 2, null, '#30363d');
  const cx = x + w / 2, cy = y + h / 4;
  drawAxes(ctx, cx, cy, r, 'Input z');
  drawUnitCircle(ctx, cx, cy, r * 0.7);
  const scale = r * 0.7;
  for (let i = 0; i < inputZ.length; i++) {
    drawComplexArrow(ctx, cx, cy, inputZ[i][0], inputZ[i][1], scale, null);
  }
  drawLabel(ctx, x + 4, y + 4, `dim=${dim}`, { font: '10px Inter', color: '#484f58' });
}

function drawGatePanel(x, y, w, h, r) {
  const halfH = h / 2;
  drawBox(ctx, x, y, w, halfH, null, '#30363d');
  const cx = x + w / 2, cy = y + halfH / 2;
  drawAxes(ctx, cx, cy, r * 0.8, 'gate_proj(z)');
  drawUnitCircle(ctx, cx, cy, r * 0.55);
  const scale = r * 0.6;
  const showN = Math.min(gateOut.length, 8);
  for (let i = 0; i < showN; i++) {
    const mag = cabs(gateOut[i][0], gateOut[i][1]);
    const gateMag = sigmoid(mag);
    const alpha = 0.3 + gateMag * 0.7;
    const color = `rgba(74,144,217,${alpha})`;
    drawComplexArrow(ctx, cx, cy, gateOut[i][0], gateOut[i][1], scale, color);
  }

  // Show sigmoid(|gate|) values
  drawLabel(ctx, x + 4, y + halfH - 16, 'σ(|gate|) controls how much signal passes:', { font: '9px Inter', color: '#8b949e' });
}

function drawActivationPanel(x, y, w, h) {
  drawBox(ctx, x, y, w, h, null, '#30363d');
  drawLabel(ctx, x + 4, y + 4, 'ModReLU activation on up:', { font: '10px Inter', color: '#e5c07b' });
  const barH = h - 30;
  const n = Math.min(activated.length, 12);
  const barW = Math.min((w - 10) / n - 2, 16);
  for (let i = 0; i < n; i++) {
    const mag = cabs(upOut[i][0], upOut[i][1]);
    const actMag = cabs(activated[i][0], activated[i][1]);
    const bx = x + 5 + i * (barW + 2);
    const by = y + 20;
    // Before activation (dimmed)
    ctx.fillStyle = 'rgba(100,100,100,0.3)';
    const h1 = (mag / 2) * barH;
    ctx.fillRect(bx, by + barH - h1, barW / 2, h1);
    // After activation
    const phase = cphase(activated[i][0], activated[i][1]);
    ctx.fillStyle = magPhaseColor(actMag, phase, 1.5);
    const h2 = (actMag / 2) * barH;
    ctx.fillRect(bx + barW / 2, by + barH - h2, barW / 2, h2);
  }
  drawLabel(ctx, x + 4, y + h - 12, 'Gray=before  Color=after (phase preserved, mag thresholded)', { font: '8px Inter', color: '#484f58' });
}

function drawOutputPanel(x, y, w, h, r) {
  drawBox(ctx, x, y, w, h, null, '#30363d');
  const cx = x + w / 2, cy = y + h / 2;
  drawAxes(ctx, cx, cy, r * 0.9, 'Output');
  drawUnitCircle(ctx, cx, cy, r * 0.6);
  const scale = r * 0.6;
  const showN = Math.min(gated.length, 8);
  for (let i = 0; i < showN; i++) {
    drawComplexArrow(ctx, cx, cy, gated[i][0], gated[i][1], scale, null);
  }

  if (animStep >= 4) {
    drawLabel(ctx, x + 4, y + h - 24, '→ down_proj maps back to dim', { font: '10px Inter', color: '#98c379' });
    for (let i = 0; i < Math.min(finalOut.length, dim); i++) {
      const color = complexToColor(finalOut[i][0], finalOut[i][1]);
      const bx = x + 10 + i * 18;
      ctx.fillStyle = color;
      ctx.fillRect(bx, y + h - 14, 14, 10);
    }
  }
}

function drawGateDial(cx, cy) {
  ctx.save();
  drawLabel(ctx, cx, cy - 40, 'Gate Dial Metaphor', { font: '11px Inter', color: '#8b949e', align: 'center' });

  // Draw dial
  const r = 28;
  ctx.beginPath();
  ctx.arc(cx - 80, cy, r, 0, Math.PI * 2);
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 2;
  ctx.stroke();

  // Gate opening (magnitude = how much)
  const mag = sigmoid(cabs(gateOut[0][0], gateOut[0][1]));
  const openAngle = mag * Math.PI * 2;
  ctx.beginPath();
  ctx.moveTo(cx - 80, cy);
  ctx.arc(cx - 80, cy, r, -Math.PI / 2, -Math.PI / 2 + openAngle);
  ctx.closePath();
  ctx.fillStyle = `rgba(46,204,113,${mag})`;
  ctx.fill();
  drawLabel(ctx, cx - 80, cy + r + 6, `Opening: ${(mag * 100).toFixed(0)}%`, { font: '10px Inter', color: '#8b949e', align: 'center' });
  drawLabel(ctx, cx - 80, cy - r - 12, '|gate| → σ → amount', { font: '9px Inter', color: '#8b949e', align: 'center' });

  // Phase rotation (direction)
  const phase = cphase(gateOut[0][0], gateOut[0][1]);
  ctx.beginPath();
  ctx.arc(cx + 80, cy, r, 0, Math.PI * 2);
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 2;
  ctx.stroke();
  const px = cx + 80 + Math.cos(phase) * r;
  const py = cy - Math.sin(phase) * r;
  drawArrow(ctx, cx + 80, cy, px, py, magPhaseColor(1, phase, 1), 2);
  drawLabel(ctx, cx + 80, cy + r + 6, `Rotation: ${(phase * 180 / Math.PI).toFixed(0)}°`, { font: '10px Inter', color: '#8b949e', align: 'center' });
  drawLabel(ctx, cx + 80, cy - r - 12, 'arg(gate) → direction', { font: '9px Inter', color: '#8b949e', align: 'center' });

  ctx.restore();
}
