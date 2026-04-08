// Dual Form vs Recurrent Form Side-by-Side Visualization
// Shows mathematical equivalence of both PAM computation forms
import {
  cmul, cabs, cconj, cphase, generateRandomComplex, complexToColor,
  drawComplexHeatmap, drawVectorBars, drawArrow, drawLabel,
  drawBox, clearCanvas, fitCanvas, magPhaseColor,
} from './canvas-utils.js';

const D = 4; // head_dim for visualization
const T = 6; // sequence length
let canvas, ctx, W, H;
let playing = false, stepIdx = 0, speed = 1, animId = null, frameCount = 0;
let Qs = [], Ks = [], Vs = [], gammas = [];
let recurrentStates = [];
let recurrentOutputs = [];
let dualOutput = [];
let decayMatrix = [];
let dualAttnMatrix = [];
let showHierarchical = false;
let dtBias = -4.0;

export function init(canvasEl) {
  canvas = canvasEl;
  generateSequence();
}

function generateSequence() {
  Qs = []; Ks = []; Vs = []; gammas = [];
  for (let t = 0; t < T; t++) {
    Qs.push(generateRandomComplex(D, 0.7));
    Ks.push(generateRandomComplex(D, 0.7));
    Vs.push(generateRandomComplex(D, 0.7));
    gammas.push(sigmoid(dtBias + (Math.random() - 0.5) * 0.5));
  }
  computeAll();
}

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function computeAll() {
  computeRecurrent();
  computeDual();
}

function computeRecurrent() {
  recurrentStates = [];
  recurrentOutputs = [];
  let S = makeZero(D, D);
  for (let t = 0; t < T; t++) {
    const Kc = Ks[t].map(([r, i]) => cconj(r, i));
    const gamma = gammas[t];
    const newS = makeZero(D, D);
    for (let r = 0; r < D; r++) {
      for (let c = 0; c < D; c++) {
        const dec = [S[r][c][0] * gamma, S[r][c][1] * gamma];
        const outer = cmul(Vs[t][r][0], Vs[t][r][1], Kc[c][0], Kc[c][1]);
        newS[r][c] = [dec[0] + outer[0], dec[1] + outer[1]];
      }
    }
    S = newS;
    recurrentStates.push(cloneMatrix(S));
    // Y = S · Q
    const Y = [];
    for (let r = 0; r < D; r++) {
      let sr = 0, si = 0;
      for (let c = 0; c < D; c++) {
        const [pr, pi] = cmul(S[r][c][0], S[r][c][1], Qs[t][c][0], Qs[t][c][1]);
        sr += pr; si += pi;
      }
      Y.push([sr, si]);
    }
    recurrentOutputs.push(Y);
  }
}

function computeDual() {
  // Build decay matrix D[i,j] = product of gammas from j+1 to i (for i>j), 1 for i==j, 0 for i<j
  decayMatrix = [];
  for (let i = 0; i < T; i++) {
    decayMatrix.push([]);
    for (let j = 0; j < T; j++) {
      if (j > i) { decayMatrix[i].push(0); continue; }
      let prod = 1;
      for (let k = j + 1; k <= i; k++) prod *= gammas[k];
      decayMatrix[i].push(prod);
    }
  }

  // Build Q @ K^T attention-like matrix (T×T), then multiply by D, then apply to V
  dualAttnMatrix = [];
  for (let i = 0; i < T; i++) {
    dualAttnMatrix.push([]);
    for (let j = 0; j < T; j++) {
      let sumR = 0, sumI = 0;
      for (let d = 0; d < D; d++) {
        const [kr, ki] = cconj(Ks[j][d][0], Ks[j][d][1]);
        const [pr, pi] = cmul(Qs[i][d][0], Qs[i][d][1], kr, ki);
        sumR += pr; sumI += pi;
      }
      const decay = decayMatrix[i][j];
      dualAttnMatrix[i].push([sumR * decay, sumI * decay]);
    }
  }

  // Y = attnMatrix @ V
  dualOutput = [];
  for (let t = 0; t < T; t++) {
    const Y = [];
    for (let d = 0; d < D; d++) {
      let sr = 0, si = 0;
      for (let j = 0; j < T; j++) {
        const [ar, ai] = dualAttnMatrix[t][j];
        const [pr, pi] = cmul(ar, ai, Vs[j][d][0], Vs[j][d][1]);
        sr += pr; si += pi;
      }
      Y.push([sr, si]);
    }
    dualOutput.push(Y);
  }
}

function makeZero(r, c) {
  return Array.from({ length: r }, () => Array.from({ length: c }, () => [0, 0]));
}

function cloneMatrix(m) { return m.map(row => row.map(([r, i]) => [r, i])); }

export function update(cfg) {
  dtBias = -4.0;
  if (cfg.hierarchical_dt && cfg.dt_bias_schedule) {
    dtBias = cfg.dt_bias_schedule[0] || -4.0;
  }
  generateSequence();
}

export function play() { playing = true; }
export function pause() { playing = false; }
export function reset() { stepIdx = 0; frameCount = 0; generateSequence(); }
export function setSpeed(s) { speed = s; }
export function toggleHierarchical() { showHierarchical = !showHierarchical; generateSequence(); }
export function stepForward() { if (stepIdx < T - 1) { stepIdx++; } }
export function stepBack() { if (stepIdx > 0) { stepIdx--; } }

export function start() {
  if (animId) return;
  function loop() {
    animId = requestAnimationFrame(loop);
    frameCount++;
    if (playing && frameCount % Math.max(1, Math.round(30 / speed)) === 0) {
      stepIdx++;
      if (stepIdx >= T) { stepIdx = T - 1; playing = false; }
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

  drawLabel(ctx, W / 2, 12, `Dual Form vs Recurrent Form — T=${T}, d=${D}  Step ${stepIdx + 1}/${T}`, {
    font: 'bold 14px Inter', color: '#58a6ff', align: 'center',
  });

  const halfW = (W - 30) / 2;
  const topY = 42;

  drawRecurrentPanel(10, topY, halfW, H - topY - 40);
  drawDualPanel(halfW + 20, topY, halfW, H - topY - 40);

  // Output comparison at bottom
  drawOutputComparison(10, H - 36);
}

function drawRecurrentPanel(x, y, w, h) {
  drawBox(ctx, x, y, w, h, '#0d1117', '#30363d');
  drawLabel(ctx, x + w / 2, y + 6, 'Recurrent Form', { font: 'bold 12px Inter', color: '#e06c75', align: 'center' });
  drawLabel(ctx, x + 4, y + 22, 'S_t = γ_t·S_{t-1} + V_t⊗K_t*', { font: '11px "Fira Code"', color: '#c9d1d9' });

  // State matrix heatmap
  const matSize = Math.min(w - 20, h / 2 - 50);
  const cellS = matSize / D;
  const mx = x + (w - matSize) / 2;
  const my = y + 44;
  if (recurrentStates[stepIdx]) {
    drawLabel(ctx, mx, my - 2, `S_${stepIdx + 1}`, { font: '10px Inter', color: '#8b949e' });
    drawBox(ctx, mx - 1, my + 10, matSize + 2, matSize + 2, null, '#21262d');
    drawComplexHeatmap(ctx, mx, my + 11, cellS, cellS, recurrentStates[stepIdx]);
  }

  // Output vector
  const outY = my + matSize + 24;
  if (recurrentOutputs[stepIdx]) {
    drawLabel(ctx, mx, outY, `Y_${stepIdx + 1} = S·Q (recurrent)`, { font: '10px Inter', color: '#8b949e' });
    drawVectorBars(ctx, mx, outY + 14, matSize, 50, recurrentOutputs[stepIdx]);
  }

  // Step dots
  const dotY = outY + 74;
  for (let t = 0; t < T; t++) {
    ctx.beginPath();
    ctx.arc(mx + t * 18 + 5, dotY, 5, 0, Math.PI * 2);
    ctx.fillStyle = t <= stepIdx ? '#e06c75' : '#30363d';
    ctx.fill();
    if (t === stepIdx) {
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  // Gamma values
  const gY = dotY + 18;
  drawLabel(ctx, mx, gY, `γ: [${gammas.slice(0, T).map(g => g.toFixed(2)).join(', ')}]`, { font: '9px "Fira Code"', color: '#bd93f9' });
}

function drawDualPanel(x, y, w, h) {
  drawBox(ctx, x, y, w, h, '#0d1117', '#30363d');
  drawLabel(ctx, x + w / 2, y + 6, 'Dual Form', { font: 'bold 12px Inter', color: '#61afef', align: 'center' });
  drawLabel(ctx, x + 4, y + 22, 'Y = (D ⊙ QK^T) V', { font: '11px "Fira Code"', color: '#c9d1d9' });

  const matSize = Math.min((w - 30) / 2, (h - 100) / 2);
  const cellS = matSize / T;

  // Decay matrix D
  const mx1 = x + 8;
  const my = y + 44;
  drawLabel(ctx, mx1, my - 2, 'Decay D (T×T)', { font: '10px Inter', color: '#8b949e' });
  drawBox(ctx, mx1 - 1, my + 10, matSize + 2, matSize + 2, null, '#21262d');
  for (let i = 0; i < T; i++) {
    for (let j = 0; j < T; j++) {
      const v = decayMatrix[i][j];
      const bright = Math.round(v * 200);
      ctx.fillStyle = `rgb(${bright * 0.4},${bright * 0.6},${bright})`;
      ctx.fillRect(mx1 + j * cellS, my + 11 + i * cellS, cellS - 1, cellS - 1);
    }
  }

  // Highlight current row
  ctx.save();
  ctx.strokeStyle = '#f0883e';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(mx1, my + 11 + stepIdx * cellS, matSize, cellS);
  ctx.restore();

  // Attention-like matrix (D ⊙ QK^T)
  const mx2 = mx1 + matSize + 14;
  drawLabel(ctx, mx2, my - 2, 'D⊙QK^T (T×T)', { font: '10px Inter', color: '#8b949e' });
  drawBox(ctx, mx2 - 1, my + 10, matSize + 2, matSize + 2, null, '#21262d');
  let maxAttn = 0;
  for (const row of dualAttnMatrix) for (const [r, i] of row) maxAttn = Math.max(maxAttn, cabs(r, i));
  maxAttn = maxAttn || 1;
  const attnAsComplex = dualAttnMatrix.map(row => row.map(([r, i]) => [r, i]));
  drawComplexHeatmap(ctx, mx2, my + 11, cellS, cellS, attnAsComplex, maxAttn);

  ctx.save();
  ctx.strokeStyle = '#f0883e';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(mx2, my + 11 + stepIdx * cellS, matSize, cellS);
  ctx.restore();

  // Output vector
  const outY = my + matSize + 24;
  if (dualOutput[stepIdx]) {
    drawLabel(ctx, mx1, outY, `Y_${stepIdx + 1} (dual)`, { font: '10px Inter', color: '#8b949e' });
    drawVectorBars(ctx, mx1, outY + 14, matSize * 2 + 14, 50, dualOutput[stepIdx]);
  }

  // Explanation
  const eY = outY + 74;
  drawLabel(ctx, mx1, eY, 'D is lower-triangular (causal) with exponential decay.', { font: '9px Inter', color: '#484f58' });
  drawLabel(ctx, mx1, eY + 14, 'Bright = high weight (recent). Dark = decayed (old).', { font: '9px Inter', color: '#484f58' });
}

function drawOutputComparison(x, y) {
  if (!recurrentOutputs[stepIdx] || !dualOutput[stepIdx]) return;
  let maxDiff = 0;
  for (let d = 0; d < D; d++) {
    const dr = Math.abs(recurrentOutputs[stepIdx][d][0] - dualOutput[stepIdx][d][0]);
    const di = Math.abs(recurrentOutputs[stepIdx][d][1] - dualOutput[stepIdx][d][1]);
    maxDiff = Math.max(maxDiff, dr, di);
  }
  const match = maxDiff < 1e-6;
  const color = match ? '#2ecc71' : '#f0883e';
  drawLabel(ctx, W / 2, y, `Output match: max|diff| = ${maxDiff.toExponential(2)} ${match ? '✓ Identical' : '≈ Close'}`, {
    font: '11px "Fira Code"', color, align: 'center',
  });
}
