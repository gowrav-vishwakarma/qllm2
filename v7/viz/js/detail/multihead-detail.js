// Multi-Head PAM Detail View — shows H independent heads processing in parallel
// Each head has its own S_h state matrix; Q/K/V are split across heads then recombined
import {
  cmul, cabs, cconj, cphase, generateRandomComplex, complexToColor,
  drawComplexHeatmap, drawVectorBars, drawArrow, drawLabel,
  drawBox, clearCanvas, fitCanvas, magPhaseColor,
} from './canvas-utils.js';

const MAX_STEPS = 10;
let canvas, ctx, W, H;
let playing = false, stepIdx = 0, speed = 1, animId = null, frameCount = 0;

let nHeads = 6, headDim = 4;
let vizHeads = 6, vizDim = 4;
let headStates = [];   // [H] arrays of d×d complex matrices
let headOutputs = [];  // [H][step] arrays of d-dim complex vectors
let steps = [];        // per-step Q,K,V,gamma for each head
let hierarchicalDt = false;
let dtBiases = [];

export function init(canvasEl) {
  canvas = canvasEl;
  generateData();
}

function generateData() {
  vizHeads = Math.min(Math.max(nHeads, 2), 8);
  vizDim = Math.min(Math.max(headDim, 2), 6);
  dtBiases = [];
  for (let h = 0; h < vizHeads; h++) {
    dtBiases.push(hierarchicalDt ? -6.91 + 6.91 * h / Math.max(vizHeads - 1, 1) : -4.0);
  }
  headStates = [];
  headOutputs = [];
  steps = [];
  for (let h = 0; h < vizHeads; h++) {
    headStates.push(makeZero(vizDim, vizDim));
    headOutputs.push([]);
  }
  for (let t = 0; t < MAX_STEPS; t++) {
    const stepData = [];
    for (let h = 0; h < vizHeads; h++) {
      const baseGamma = 1 / (1 + Math.exp(-dtBiases[h]));
      stepData.push({
        Q: generateRandomComplex(vizDim, 0.7),
        K: generateRandomComplex(vizDim, 0.7),
        V: generateRandomComplex(vizDim, 0.7),
        gamma: baseGamma + (Math.random() - 0.5) * 0.05,
      });
    }
    steps.push(stepData);
  }
  stepIdx = 0;
  recomputeAll();
}

function makeZero(r, c) {
  return Array.from({ length: r }, () => Array.from({ length: c }, () => [0, 0]));
}

function recomputeAll() {
  for (let h = 0; h < vizHeads; h++) {
    headStates[h] = makeZero(vizDim, vizDim);
    headOutputs[h] = [];
  }
  for (let t = 0; t <= stepIdx; t++) applyStepAll(t);
}

function applyStepAll(t) {
  if (t >= MAX_STEPS) return;
  for (let h = 0; h < vizHeads; h++) {
    const { K, V, gamma } = steps[t][h];
    const Kc = K.map(([r, i]) => cconj(r, i));
    const S = headStates[h];
    for (let r = 0; r < vizDim; r++) {
      for (let c = 0; c < vizDim; c++) {
        const dec = [S[r][c][0] * gamma, S[r][c][1] * gamma];
        const outer = cmul(V[r][0], V[r][1], Kc[c][0], Kc[c][1]);
        S[r][c] = [dec[0] + outer[0], dec[1] + outer[1]];
      }
    }
    const Q = steps[t][h].Q;
    const Y = [];
    for (let r = 0; r < vizDim; r++) {
      let sr = 0, si = 0;
      for (let c = 0; c < vizDim; c++) {
        const [pr, pi] = cmul(S[r][c][0], S[r][c][1], Q[c][0], Q[c][1]);
        sr += pr; si += pi;
      }
      Y.push([sr, si]);
    }
    headOutputs[h].push(Y);
  }
}

export function update(cfg) {
  nHeads = cfg.n_heads || cfg.pam_num_heads || 6;
  headDim = cfg.head_dim || cfg.pam_head_dim || 64;
  hierarchicalDt = cfg.hierarchical_dt || false;
  generateData();
}

export function play() { playing = true; }
export function pause() { playing = false; }
export function reset() { generateData(); }
export function setSpeed(s) { speed = s; }
export function stepForward() { if (stepIdx < MAX_STEPS - 1) { stepIdx++; recomputeAll(); } }
export function stepBack() { if (stepIdx > 0) { stepIdx--; recomputeAll(); } }

export function start() {
  if (animId) return;
  function loop() {
    animId = requestAnimationFrame(loop);
    frameCount++;
    if (playing && frameCount % Math.max(1, Math.round(30 / speed)) === 0) {
      stepIdx++;
      if (stepIdx >= MAX_STEPS) { stepIdx = MAX_STEPS - 1; playing = false; }
      recomputeAll();
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

  drawLabel(ctx, W / 2, 8, `Multi-Head PAM — ${vizHeads} heads × ${vizDim}d   Step ${stepIdx + 1}/${MAX_STEPS}`, {
    font: 'bold 14px Inter', color: '#58a6ff', align: 'center',
  });

  // Layout: grid of head state matrices
  const cols = vizHeads <= 4 ? vizHeads : Math.ceil(vizHeads / 2);
  const rows = vizHeads <= 4 ? 1 : 2;
  const pad = 12;
  const topY = 34;
  const splitBarH = 40;
  const compH = 70;
  const availH = H - topY - splitBarH - compH - 20;
  const availW = W - 20;
  const cellW = Math.floor((availW - pad * (cols + 1)) / cols);
  const cellH = Math.floor((availH - pad * (rows + 1)) / rows);
  const matSize = Math.min(cellW - 8, cellH - 40);
  const heatCellS = matSize / vizDim;

  // QKV split bar
  drawQKVSplitBar(10, topY, availW, splitBarH);

  const gridY = topY + splitBarH + 4;

  for (let h = 0; h < vizHeads; h++) {
    const col = h % cols;
    const row = Math.floor(h / cols);
    const bx = 10 + pad + col * (cellW + pad);
    const by = gridY + pad + row * (cellH + pad);

    const headColor = `hsl(${(h / vizHeads) * 300}, 65%, 55%)`;
    drawBox(ctx, bx, by, cellW, cellH, '#0d1117', '#21262d');

    // Head label with decay
    const gamma = steps[stepIdx]?.[h]?.gamma ?? 0.9;
    drawLabel(ctx, bx + 4, by + 2, `Head ${h}`, { font: 'bold 11px Inter', color: headColor });
    drawLabel(ctx, bx + cellW - 4, by + 2, `γ=${gamma.toFixed(3)}`, {
      font: '9px Fira Code', color: '#bd93f9', align: 'right',
    });
    if (hierarchicalDt) {
      drawLabel(ctx, bx + cellW - 4, by + 13, `dt_bias=${dtBiases[h].toFixed(1)}`, {
        font: '8px Fira Code', color: '#484f58', align: 'right',
      });
    }

    // State heatmap
    const mx = bx + (cellW - matSize) / 2;
    const my = by + 24;
    drawComplexHeatmap(ctx, mx, my, heatCellS, heatCellS, headStates[h]);

    // Output vector bars below heatmap
    const outY = my + matSize + 4;
    const outVec = headOutputs[h]?.[stepIdx];
    if (outVec) {
      drawVectorBars(ctx, mx, outY, matSize, Math.min(cellH - matSize - 32, 30), outVec);
    }
    drawLabel(ctx, mx, outY + Math.min(cellH - matSize - 32, 30) + 2, `Y_h${h}`, {
      font: '8px Fira Code', color: '#8b949e',
    });
  }

  // Recombine bar
  const recombY = gridY + pad + rows * (cellH + pad) + 4;
  drawRecombineBar(10, recombY, availW);

  // Head comparison chart
  drawHeadComparison(10, recombY + 30, availW, compH - 30);
}

function drawQKVSplitBar(x, y, w, h) {
  drawBox(ctx, x, y, w, h, '#161b22', '#30363d');
  drawLabel(ctx, x + 6, y + 3, 'Fused QKV Projection: x → [Q₀..Q_H | K₀..K_H | V₀..V_H]', {
    font: '11px Fira Code', color: '#c9d1d9',
  });

  const barY = y + 18;
  const barH = h - 22;
  const segW = (w - 20) / (vizHeads * 3);

  for (let h = 0; h < vizHeads; h++) {
    const hue = (h / vizHeads) * 300;
    // Q segment
    ctx.fillStyle = `hsla(${hue}, 60%, 45%, 0.7)`;
    ctx.fillRect(x + 8 + h * segW, barY, segW - 1, barH);
    // K segment
    ctx.fillStyle = `hsla(${hue}, 60%, 35%, 0.7)`;
    ctx.fillRect(x + 8 + (vizHeads + h) * segW, barY, segW - 1, barH);
    // V segment
    ctx.fillStyle = `hsla(${hue}, 60%, 25%, 0.7)`;
    ctx.fillRect(x + 8 + (vizHeads * 2 + h) * segW, barY, segW - 1, barH);
  }
  const lblY = barY + barH / 2 + 3;
  drawLabel(ctx, x + 8 + vizHeads * segW / 2, lblY, 'Q', { font: '9px Inter', color: '#fff', align: 'center' });
  drawLabel(ctx, x + 8 + vizHeads * segW * 1.5, lblY, 'K', { font: '9px Inter', color: '#fff', align: 'center' });
  drawLabel(ctx, x + 8 + vizHeads * segW * 2.5, lblY, 'V', { font: '9px Inter', color: '#fff', align: 'center' });
}

function drawRecombineBar(x, y, w) {
  drawBox(ctx, x, y, w, 24, '#161b22', '#30363d');
  drawLabel(ctx, x + 6, y + 5, `Concatenate [Y_h0 .. Y_h${vizHeads-1}] → o_proj → output [dim×2]`, {
    font: '11px Fira Code', color: '#98c379',
  });
  for (let h = 0; h < vizHeads; h++) {
    const hue = (h / vizHeads) * 300;
    const segW = (w - 20) / vizHeads;
    ctx.fillStyle = `hsla(${hue}, 60%, 45%, 0.5)`;
    ctx.fillRect(x + 8 + h * segW, y + 18, segW - 1, 4);
  }
}

function drawHeadComparison(x, y, w, h) {
  drawLabel(ctx, x + 6, y, 'Head Magnitude Comparison', { font: '10px Inter', color: '#8b949e' });
  const barAreaY = y + 14;
  const barAreaH = h - 16;
  const barW = Math.min((w - 20) / vizHeads - 4, 40);

  let maxMag = 0;
  const mags = [];
  for (let hd = 0; hd < vizHeads; hd++) {
    const S = headStates[hd];
    let sum = 0;
    for (let r = 0; r < vizDim; r++)
      for (let c = 0; c < vizDim; c++)
        sum += cabs(S[r][c][0], S[r][c][1]);
    const avg = sum / (vizDim * vizDim);
    mags.push(avg);
    maxMag = Math.max(maxMag, avg);
  }
  maxMag = maxMag || 1;

  for (let hd = 0; hd < vizHeads; hd++) {
    const bx = x + 10 + hd * (barW + 4);
    const bh = (mags[hd] / maxMag) * barAreaH;
    const hue = (hd / vizHeads) * 300;
    ctx.fillStyle = `hsl(${hue}, 60%, 50%)`;
    ctx.fillRect(bx, barAreaY + barAreaH - bh, barW, bh);
    ctx.strokeStyle = '#30363d';
    ctx.strokeRect(bx, barAreaY, barW, barAreaH);
    drawLabel(ctx, bx + barW / 2, barAreaY + barAreaH + 2, `H${hd}`, {
      font: '8px Inter', color: '#8b949e', align: 'center',
    });
  }

  drawLabel(ctx, x + w - 6, y, hierarchicalDt ? 'Hierarchical dt: different decay per head' : 'Flat dt: same decay for all heads', {
    font: '9px Inter', color: hierarchicalDt ? '#bd93f9' : '#484f58', align: 'right',
  });
}
