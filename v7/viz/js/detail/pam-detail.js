// PAM Recurrence Detail View - Canvas 2D animation
// Visualizes: S_t = γ_t · S_{t-1} + V_t ⊗ K_t*  and  Y_t = S_t · Q_t
import {
  cmul, cabs, cconj, cphase, generateRandomComplex, complexToColor,
  drawComplexHeatmap, drawVectorBars, drawArrow, drawAxes, drawLabel,
  drawBox, clearCanvas, fitCanvas, magPhaseColor,
} from './canvas-utils.js';

const D = 4; // visualization dimension (small for clarity)
const MAX_STEPS = 12;
let canvas, ctx, W, H;
let playing = false, stepIdx = 0, speed = 1, frameCount = 0;
let S = []; // d×d complex state matrix
let steps = []; // pre-generated Q, K, V, gamma per step
let animId = null;
let transitionProgress = 1; // 0→1 animation within a step
let subStep = 0; // 0: show K,V  1: conjugate  2: outer product  3: decay  4: add  5: query  6: output
const SUB_STEP_COUNT = 7;
let headDim = D, nHeads = 4;
let currentPhase = '';

export function init(canvasEl) {
  canvas = canvasEl;
  resetState();
}

function resetState() {
  S = makeZeroMatrix(D, D);
  steps = [];
  for (let t = 0; t < MAX_STEPS; t++) {
    steps.push({
      Q: generateRandomComplex(D, 0.8),
      K: generateRandomComplex(D, 0.8),
      V: generateRandomComplex(D, 0.8),
      gamma: 0.85 + Math.random() * 0.12, // data-dependent decay
    });
  }
  stepIdx = 0;
  subStep = 0;
  transitionProgress = 1;
  S = makeZeroMatrix(D, D);
}

function makeZeroMatrix(r, c) {
  return Array.from({ length: r }, () => Array.from({ length: c }, () => [0, 0]));
}

function cloneMatrix(m) {
  return m.map(row => row.map(([r, i]) => [r, i]));
}

export function update(cfg) {
  headDim = Math.min(cfg.head_dim || cfg.pam_head_dim || 64, 8);
  nHeads = cfg.n_heads || cfg.pam_num_heads || 6;
  resetState();
}

export function play() { playing = true; }
export function pause() { playing = false; }
export function reset() { resetState(); }

export function stepForward() {
  if (stepIdx >= MAX_STEPS - 1 && subStep >= SUB_STEP_COUNT - 1) return;
  subStep++;
  if (subStep >= SUB_STEP_COUNT) {
    applyStep(stepIdx);
    stepIdx++;
    subStep = 0;
  }
  transitionProgress = 0;
}

export function stepBack() {
  if (stepIdx <= 0 && subStep <= 0) return;
  subStep--;
  if (subStep < 0) {
    stepIdx = Math.max(0, stepIdx - 1);
    subStep = SUB_STEP_COUNT - 1;
    recomputeState(stepIdx);
  }
  transitionProgress = 0;
}

function applyStep(t) {
  if (t >= MAX_STEPS) return;
  const { K, V, gamma } = steps[t];
  const Kconj = K.map(([r, i]) => cconj(r, i));
  for (let r = 0; r < D; r++) {
    for (let c = 0; c < D; c++) {
      const [sr, si] = S[r][c];
      const decayed = [sr * gamma, si * gamma];
      const outer = cmul(V[r][0], V[r][1], Kconj[c][0], Kconj[c][1]);
      S[r][c] = [decayed[0] + outer[0], decayed[1] + outer[1]];
    }
  }
}

function recomputeState(upTo) {
  S = makeZeroMatrix(D, D);
  for (let t = 0; t < upTo; t++) applyStep(t);
}

export function setSpeed(s) { speed = s; }

export function start() {
  if (animId) return;
  function loop() {
    animId = requestAnimationFrame(loop);
    frameCount++;
    if (playing && frameCount % Math.max(1, Math.round(12 / speed)) === 0) {
      transitionProgress += 0.15;
      if (transitionProgress >= 1) {
        transitionProgress = 1;
        subStep++;
        if (subStep >= SUB_STEP_COUNT) {
          applyStep(stepIdx);
          stepIdx++;
          subStep = 0;
          if (stepIdx >= MAX_STEPS) { stepIdx = MAX_STEPS - 1; subStep = SUB_STEP_COUNT - 1; playing = false; }
        }
        transitionProgress = 0;
      }
    } else {
      transitionProgress = Math.min(1, transitionProgress + 0.05);
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

  const step = steps[stepIdx] || steps[MAX_STEPS - 1];
  const { Q, K, V, gamma } = step;
  const Kconj = K.map(([r, i]) => cconj(r, i));

  // Title
  drawLabel(ctx, W / 2, 12, `PAM Recurrence — Step ${stepIdx + 1}/${MAX_STEPS}  (${headDim}d × ${nHeads} heads)`, {
    font: 'bold 14px Inter, sans-serif', color: '#58a6ff', align: 'center',
  });

  // Layout
  const matSize = Math.min(180, (H - 100) / 2);
  const cellW = matSize / D, cellH = matSize / D;
  const mx = 30, my = 50;
  const vecW = D * 18 + 10;
  const vecH = 60;

  // State matrix S
  drawLabel(ctx, mx, my - 4, `State Matrix S (${D}×${D})`, { font: '12px Inter', color: '#7ee787' });
  drawBox(ctx, mx - 2, my + 12, matSize + 4, matSize + 4, '#0d1117', '#30363d');
  drawComplexHeatmap(ctx, mx, my + 14, cellW, cellH, S);
  drawLabel(ctx, mx, my + matSize + 22, 'Color = phase, Brightness = magnitude', { font: '9px Inter', color: '#484f58' });

  // Equation display
  const eqY = my + matSize + 42;
  drawEquationState(eqY, gamma);

  // Right side: vectors and animation
  const rx = mx + matSize + 50;
  const vw = Math.min(W - rx - 20, 250);

  // Sub-step visualization
  currentPhase = '';
  if (subStep === 0) {
    currentPhase = '① Show K and V vectors';
    drawLabel(ctx, rx, my, 'K (Key Vector)', { font: '11px Inter', color: '#e5c07b' });
    drawVectorBars(ctx, rx, my + 14, vw, vecH, K, null);
    drawLabel(ctx, rx, my + vecH + 24, 'V (Value Vector)', { font: '11px Inter', color: '#e06c75' });
    drawVectorBars(ctx, rx, my + vecH + 38, vw, vecH, V, null);
  } else if (subStep === 1) {
    currentPhase = '② Conjugate K → K*';
    drawLabel(ctx, rx, my, 'K (original)', { font: '11px Inter', color: '#e5c07b' });
    drawVectorBars(ctx, rx, my + 14, vw, vecH * 0.8, K, null);
    drawLabel(ctx, rx, my + vecH + 10, 'K* (conjugated — flip imag)', { font: '11px Inter', color: '#d19a66' });
    drawVectorBars(ctx, rx, my + vecH + 24, vw, vecH * 0.8, Kconj, null);
    drawConjugateAnimation(rx + vw + 10, my + vecH / 2);
  } else if (subStep === 2) {
    currentPhase = '③ Outer Product: V ⊗ K*';
    const outer = makeZeroMatrix(D, D);
    for (let r = 0; r < D; r++)
      for (let c = 0; c < D; c++)
        outer[r][c] = cmul(V[r][0], V[r][1], Kconj[c][0], Kconj[c][1]);
    drawLabel(ctx, rx, my, 'Outer Product V ⊗ K*', { font: '11px Inter', color: '#c678dd' });
    const outerSize = Math.min(matSize, vw);
    const oCellW = outerSize / D, oCellH = outerSize / D;
    drawBox(ctx, rx - 2, my + 16, outerSize + 4, outerSize + 4, '#0d1117', '#30363d');
    drawComplexHeatmap(ctx, rx, my + 18, oCellW, oCellH, outer);
    drawLabel(ctx, rx, my + outerSize + 28, '"Memory write" — storing V keyed by K', { font: '10px Inter', color: '#484f58' });
  } else if (subStep === 3) {
    currentPhase = `④ Decay: γ = ${gamma.toFixed(3)}`;
    drawLabel(ctx, rx, my, `Decay old state: γ_t · S_{t-1}`, { font: '11px Inter', color: '#e5c07b' });
    drawDecayVisualization(rx, my + 20, vw, gamma);
  } else if (subStep === 4) {
    currentPhase = '⑤ Update: S_t = γ·S_{t-1} + V⊗K*';
    drawLabel(ctx, rx, my, 'Updated State S_t', { font: '11px Inter', color: '#7ee787' });
    const newS = computeNewState(S, V, Kconj, gamma);
    const sz = Math.min(matSize, vw);
    drawBox(ctx, rx - 2, my + 16, sz + 4, sz + 4, '#0d1117', '#30363d');
    drawComplexHeatmap(ctx, rx, my + 18, sz / D, sz / D, newS);
    drawLabel(ctx, rx, my + sz + 28, 'Decayed old + new write', { font: '10px Inter', color: '#484f58' });
  } else if (subStep === 5) {
    currentPhase = '⑥ Query arrives: Q_t';
    drawLabel(ctx, rx, my, 'Q (Query Vector)', { font: '11px Inter', color: '#61afef' });
    drawVectorBars(ctx, rx, my + 14, vw, vecH, Q, null);
    drawLabel(ctx, rx, my + vecH + 30, '"Memory read" — retrieve by similarity', { font: '10px Inter', color: '#484f58' });
    drawQueryArrows(rx + vw / 2, my + vecH + 50, Q);
  } else if (subStep === 6) {
    currentPhase = '⑦ Output: Y_t = S_t · Q_t';
    const Y = matVecMul(S, Q);
    drawLabel(ctx, rx, my, 'Output Y = S · Q', { font: '11px Inter', color: '#98c379' });
    drawVectorBars(ctx, rx, my + 14, vw, vecH, Y, null);
    drawLabel(ctx, rx, my + vecH + 30, 'Y combines all stored associations weighted by Q', { font: '10px Inter', color: '#484f58' });
  }

  // Phase label
  if (currentPhase) {
    drawBox(ctx, rx - 4, H - 50, vw + 8, 28, '#161b22', '#30363d');
    drawLabel(ctx, rx, H - 45, currentPhase, { font: 'bold 12px Inter', color: '#58a6ff' });
  }

  // GSP note + multi-head hint
  const gspY = H - 32;
  drawLabel(ctx, 30, gspY, 'GSP (Gated State Protection) gates how much new info can overwrite state — like a forget gate.', {
    font: '9px Inter', color: '#484f58',
  });
  drawLabel(ctx, 30, gspY + 12, 'This shows a single head. See the "Multi-Head" tab for all heads processing in parallel.', {
    font: '9px Inter', color: '#58a6ff',
  });

  // Step indicator dots
  const dotY = my + matSize + 80;
  for (let t = 0; t < MAX_STEPS; t++) {
    const dx = mx + t * 14;
    ctx.beginPath();
    ctx.arc(dx + 5, dotY, 4, 0, Math.PI * 2);
    ctx.fillStyle = t < stepIdx ? '#238636' : t === stepIdx ? '#58a6ff' : '#30363d';
    ctx.fill();
  }
  drawLabel(ctx, mx, dotY + 12, `Step ${stepIdx + 1}`, { font: '10px Inter', color: '#8b949e' });
}

function drawEquationState(y, gamma) {
  const eq = `S_t = γ_t · S_{t-1} + V_t ⊗ K_t*    |    Y_t = S_t · Q_t`;
  drawLabel(ctx, 30, y, eq, { font: '13px "Fira Code", monospace', color: '#c9d1d9' });
  drawLabel(ctx, 30, y + 18, `γ_t = ${gamma.toFixed(3)}  (data-dependent decay)`, {
    font: '11px "Fira Code", monospace', color: '#bd93f9',
  });
}

function drawConjugateAnimation(x, y) {
  ctx.save();
  ctx.fillStyle = '#484f58';
  ctx.font = '10px Inter';
  ctx.textAlign = 'left';
  ctx.fillText('Conjugate:', x, y - 10);
  ctx.fillText('flip imag →', x, y + 5);
  ctx.fillText('a+bi → a-bi', x, y + 20);
  ctx.restore();
}

function drawDecayVisualization(x, y, w, gamma) {
  const barW = 30;
  const barH = 80;
  ctx.save();
  for (let i = 0; i < 5; i++) {
    const decayed = Math.pow(gamma, i + 1);
    const bx = x + i * (barW + 8);
    const h = barH * decayed;
    ctx.fillStyle = `hsla(160, 60%, 50%, ${0.3 + decayed * 0.7})`;
    ctx.fillRect(bx, y + barH - h, barW, h);
    ctx.strokeStyle = '#30363d';
    ctx.strokeRect(bx, y, barW, barH);
    ctx.fillStyle = '#8b949e';
    ctx.font = '9px Inter';
    ctx.textAlign = 'center';
    ctx.fillText(`t-${i + 1}`, bx + barW / 2, y + barH + 12);
    ctx.fillText(`${(decayed * 100).toFixed(0)}%`, bx + barW / 2, y + barH - h - 4);
  }
  ctx.restore();
  drawLabel(ctx, x, y + barH + 28, `Older tokens fade exponentially: γ^n`, { font: '10px Inter', color: '#484f58' });
}

function drawQueryArrows(cx, cy, Q) {
  ctx.save();
  ctx.strokeStyle = '#61afef';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([3, 3]);
  for (let i = 0; i < D; i++) {
    const angle = (i / D) * Math.PI * 2 - Math.PI / 2;
    const r = 25;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(angle) * r, cy + Math.sin(angle) * r);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.restore();
}

function computeNewState(S, V, Kconj, gamma) {
  const newS = makeZeroMatrix(D, D);
  for (let r = 0; r < D; r++) {
    for (let c = 0; c < D; c++) {
      const [sr, si] = S[r][c];
      const decayed = [sr * gamma, si * gamma];
      const outer = cmul(V[r][0], V[r][1], Kconj[c][0], Kconj[c][1]);
      newS[r][c] = [decayed[0] + outer[0], decayed[1] + outer[1]];
    }
  }
  return newS;
}

function matVecMul(M, v) {
  const result = [];
  for (let r = 0; r < M.length; r++) {
    let sumR = 0, sumI = 0;
    for (let c = 0; c < v.length; c++) {
      const [pr, pi] = cmul(M[r][c][0], M[r][c][1], v[c][0], v[c][1]);
      sumR += pr; sumI += pi;
    }
    result.push([sumR, sumI]);
  }
  return result;
}
