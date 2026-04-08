// Memory Detail View — Working Memory, Internal Memory, Episodic Memory, MemoryFusion
// Visualizes slot-based read/write mechanics for V6 memory systems
import {
  cmul, cabs, cconj, cphase, generateRandomComplex, complexToColor,
  drawVectorBars, drawArrow, drawLabel, drawBox, clearCanvas, fitCanvas, magPhaseColor,
} from './canvas-utils.js';

const NUM_VIZ_SLOTS = 8;
const SEQ_LEN = 6;
let canvas, ctx, W, H;
let playing = false, stepIdx = 0, speed = 1, animId = null, frameCount = 0;
let dim = 4;
let wmEnabled = false, imEnabled = false, emEnabled = false;

// Working Memory state
let wmSlots = [], wmMask = [], wmWritePtr = 0;
let wmWriteGates = [], wmReadScores = [], wmReadOutput = [];

// Internal Memory state
let imSlots = [], imReadScores = [], imReadOutput = [];

// Episodic Memory state
let emSlots = [], emMask = [], emWritePtr = 0;
let salienceValues = [], eventSpans = [], emReadOutput = [];

// Input sequence
let inputSeq = [];

// Fusion
let fusionWeights = [0, 0, 0];

export function init(canvasEl) {
  canvas = canvasEl;
  generateData();
}

function generateData() {
  inputSeq = [];
  for (let t = 0; t < SEQ_LEN; t++) inputSeq.push(generateRandomComplex(dim, 0.8));

  // Working memory
  wmSlots = Array.from({ length: NUM_VIZ_SLOTS }, () => generateRandomComplex(dim, 0.3));
  wmMask = Array.from({ length: NUM_VIZ_SLOTS }, () => Math.random() * 0.3);
  wmWritePtr = 0;
  wmWriteGates = Array.from({ length: SEQ_LEN }, () => Math.random() * 0.8 + 0.1);
  wmReadScores = Array.from({ length: NUM_VIZ_SLOTS }, () => Math.random());
  normalizeScores(wmReadScores);
  wmReadOutput = generateRandomComplex(dim, 0.5);

  // Internal memory (learned, fixed)
  imSlots = Array.from({ length: NUM_VIZ_SLOTS }, () => generateRandomComplex(dim, 0.6));
  imReadScores = Array.from({ length: NUM_VIZ_SLOTS }, () => Math.random());
  normalizeScores(imReadScores);
  imReadOutput = generateRandomComplex(dim, 0.4);

  // Episodic memory
  salienceValues = Array.from({ length: SEQ_LEN }, () => Math.random());
  eventSpans = detectEvents(salienceValues, 0.5);
  emSlots = Array.from({ length: NUM_VIZ_SLOTS }, () => generateRandomComplex(dim, 0.2));
  emMask = Array.from({ length: NUM_VIZ_SLOTS }, () => 0);
  emWritePtr = 0;
  emReadOutput = generateRandomComplex(dim, 0.3);

  // Fusion weights
  const active = [wmEnabled, imEnabled, emEnabled].filter(Boolean).length;
  if (active > 0) {
    fusionWeights = [wmEnabled ? 1/active : 0, imEnabled ? 1/active : 0, emEnabled ? 1/active : 0];
  } else {
    fusionWeights = [0, 0, 0];
  }

  stepIdx = 0;
}

function normalizeScores(scores) {
  let sum = 0;
  for (const s of scores) sum += s;
  if (sum > 0) for (let i = 0; i < scores.length; i++) scores[i] /= sum;
}

function detectEvents(salience, threshold) {
  const spans = [];
  let inSpan = false, start = 0;
  for (let i = 0; i < salience.length; i++) {
    if (salience[i] > threshold && !inSpan) { inSpan = true; start = i; }
    else if (salience[i] <= threshold && inSpan) { spans.push([start, i - 1]); inSpan = false; }
  }
  if (inSpan) spans.push([start, salience.length - 1]);
  return spans;
}

export function update(cfg) {
  dim = Math.min(cfg.dim || 384, 6);
  generateData();
}

export function updateToggles(tog) {
  wmEnabled = tog.working_memory || false;
  imEnabled = tog.internal_memory || false;
  emEnabled = tog.episodic_memory || false;
  const active = [wmEnabled, imEnabled, emEnabled].filter(Boolean).length;
  if (active > 0) {
    fusionWeights = [wmEnabled ? 1/active : 0, imEnabled ? 1/active : 0, emEnabled ? 1/active : 0];
  }
}

export function play() { playing = true; }
export function pause() { playing = false; }
export function reset() { generateData(); }
export function setSpeed(s) { speed = s; }
export function stepForward() {
  if (stepIdx < SEQ_LEN - 1) {
    stepIdx++;
    simulateStep();
  }
}
export function stepBack() { if (stepIdx > 0) stepIdx--; }

function simulateStep() {
  if (wmEnabled) {
    // Write: top gate position writes to slot
    if (wmWriteGates[stepIdx] > 0.5) {
      const g = wmWriteGates[stepIdx];
      const slot = wmWritePtr % NUM_VIZ_SLOTS;
      for (let d = 0; d < dim; d++) {
        wmSlots[slot][d][0] = g * inputSeq[stepIdx][d][0] + (1-g) * wmSlots[slot][d][0];
        wmSlots[slot][d][1] = g * inputSeq[stepIdx][d][1] + (1-g) * wmSlots[slot][d][1];
      }
      wmMask[slot] = Math.min(1, wmMask[slot] + g);
      wmWritePtr++;
    }
    // Mask decay
    for (let i = 0; i < NUM_VIZ_SLOTS; i++) wmMask[i] *= 0.95;
    // Read scores
    for (let i = 0; i < NUM_VIZ_SLOTS; i++) {
      let score = 0;
      for (let d = 0; d < dim; d++) {
        const [qr, qi] = inputSeq[stepIdx][d];
        const [kr, ki] = wmSlots[i][d];
        score += qr * kr + qi * ki;
      }
      wmReadScores[i] = Math.max(0, score) * wmMask[i];
    }
    normalizeScores(wmReadScores);
  }

  if (emEnabled) {
    // Write events above salience threshold
    for (const [s, e] of eventSpans) {
      if (s <= stepIdx && e >= stepIdx) {
        const slot = emWritePtr % NUM_VIZ_SLOTS;
        emSlots[slot] = inputSeq[stepIdx].map(([r, i]) => [r * salienceValues[stepIdx], i * salienceValues[stepIdx]]);
        emMask[slot] = salienceValues[stepIdx];
        emWritePtr++;
      }
    }
  }
}

export function start() {
  if (animId) return;
  function loop() {
    animId = requestAnimationFrame(loop);
    frameCount++;
    if (playing && frameCount % Math.max(1, Math.round(30 / speed)) === 0) {
      stepIdx++;
      if (stepIdx >= SEQ_LEN) { stepIdx = SEQ_LEN - 1; playing = false; }
      else simulateStep();
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

  const anyEnabled = wmEnabled || imEnabled || emEnabled;
  if (!anyEnabled) {
    drawLabel(ctx, W / 2, H / 2 - 20, 'No Memory Modules Enabled', {
      font: 'bold 16px Inter', color: '#58a6ff', align: 'center',
    });
    drawLabel(ctx, W / 2, H / 2 + 10, 'Switch to V6 and enable Working Memory, Internal Memory,', {
      font: '12px Inter', color: '#8b949e', align: 'center',
    });
    drawLabel(ctx, W / 2, H / 2 + 28, 'or Episodic Memory in the left panel to visualize slot mechanics.', {
      font: '12px Inter', color: '#8b949e', align: 'center',
    });
    return;
  }

  drawLabel(ctx, W / 2, 8, `Memory Systems — Step ${stepIdx + 1}/${SEQ_LEN}`, {
    font: 'bold 14px Inter', color: '#58a6ff', align: 'center',
  });

  const enabledCount = [wmEnabled, imEnabled, emEnabled].filter(Boolean).length;
  const panelH = Math.floor((H - 80) / (enabledCount + (enabledCount > 1 ? 1 : 0)));
  let curY = 32;

  if (wmEnabled) {
    drawWMPanel(10, curY, W - 20, panelH);
    curY += panelH + 4;
  }
  if (imEnabled) {
    drawIMPanel(10, curY, W - 20, panelH);
    curY += panelH + 4;
  }
  if (emEnabled) {
    drawEMPanel(10, curY, W - 20, panelH);
    curY += panelH + 4;
  }
  if (enabledCount > 1) {
    drawFusionBar(10, curY, W - 20, Math.min(panelH, 50));
  }
}

function drawSlotGrid(x, y, w, h, slots, mask, scores, writePtr, label, color) {
  drawLabel(ctx, x + 4, y + 2, label, { font: 'bold 11px Inter', color });
  const slotW = Math.min((w - 16) / NUM_VIZ_SLOTS - 2, 50);
  const slotH = Math.min(h - 50, 40);
  const gridY = y + 18;

  for (let i = 0; i < NUM_VIZ_SLOTS; i++) {
    const sx = x + 8 + i * (slotW + 2);
    const maskAlpha = mask ? mask[i] : 1;
    ctx.globalAlpha = 0.2 + maskAlpha * 0.8;

    // Slot content as colored cells
    const cellW = slotW / dim;
    for (let d = 0; d < dim; d++) {
      const [r, im] = slots[i][d];
      ctx.fillStyle = complexToColor(r, im);
      ctx.fillRect(sx + d * cellW, gridY, cellW - 1, slotH);
    }
    ctx.globalAlpha = 1;

    ctx.strokeStyle = (writePtr !== undefined && i === writePtr % NUM_VIZ_SLOTS) ? '#f0883e' : '#30363d';
    ctx.lineWidth = (writePtr !== undefined && i === writePtr % NUM_VIZ_SLOTS) ? 2 : 1;
    ctx.strokeRect(sx, gridY, slotW, slotH);

    // Score bar
    if (scores) {
      const barH = scores[i] * 20;
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.6;
      ctx.fillRect(sx, gridY + slotH + 2, slotW, barH);
      ctx.globalAlpha = 1;
    }

    // Write pointer indicator
    if (writePtr !== undefined && i === writePtr % NUM_VIZ_SLOTS) {
      drawLabel(ctx, sx + slotW / 2, gridY - 8, '▼', {
        font: '10px Inter', color: '#f0883e', align: 'center',
      });
    }
  }

  // Labels
  drawLabel(ctx, x + 4, gridY + slotH + 24, scores ? 'Score bars show read attention per slot' : '', {
    font: '8px Inter', color: '#484f58',
  });
}

function drawWMPanel(x, y, w, h) {
  drawBox(ctx, x, y, w, h, '#0d1117', '#21262d');
  drawLabel(ctx, x + w - 8, y + 4, 'Read+Write / Runtime slots', {
    font: '9px Inter', color: '#484f58', align: 'right',
  });

  const halfW = w / 2 - 8;

  // Write side
  drawLabel(ctx, x + 8, y + 18, 'Write (gate-controlled):', { font: '10px Inter', color: '#e5c07b' });
  const gateY = y + 32;
  const gateH = 16;
  for (let t = 0; t < SEQ_LEN; t++) {
    const bx = x + 8 + t * 30;
    const g = wmWriteGates[t];
    ctx.fillStyle = g > 0.5 ? `rgba(46,204,113,${g})` : `rgba(150,150,150,${0.3})`;
    ctx.fillRect(bx, gateY, 26, gateH);
    drawLabel(ctx, bx + 13, gateY + gateH + 2, `t${t}`, {
      font: '8px Inter', color: t === stepIdx ? '#fff' : '#484f58', align: 'center',
    });
    if (t === stepIdx) {
      ctx.strokeStyle = '#58a6ff';
      ctx.lineWidth = 2;
      ctx.strokeRect(bx, gateY, 26, gateH);
    }
  }
  drawLabel(ctx, x + 8, gateY + gateH + 14, 'Gate > 0.5 → write to circular buffer', {
    font: '8px Inter', color: '#484f58',
  });

  // Slot grid with read scores
  drawSlotGrid(x, y + h / 2 - 10, w, h / 2, wmSlots, wmMask, wmReadScores, wmWritePtr,
    'Working Memory Slots', '#9b59b6');
}

function drawIMPanel(x, y, w, h) {
  drawBox(ctx, x, y, w, h, '#0d1117', '#21262d');
  drawLabel(ctx, x + w - 8, y + 4, 'Read-Only / Trained nn.Parameter slots', {
    font: '9px Inter', color: '#484f58', align: 'right',
  });

  drawSlotGrid(x, y + 2, w, h - 16, imSlots, null, imReadScores, undefined,
    'Internal Memory (learned knowledge)', '#3498db');

  drawLabel(ctx, x + 8, y + h - 14, 'No writes during inference — slots are trained parameters that store general knowledge', {
    font: '9px Inter', color: '#484f58',
  });
}

function drawEMPanel(x, y, w, h) {
  drawBox(ctx, x, y, w, h, '#0d1117', '#21262d');
  drawLabel(ctx, x + w - 8, y + 4, 'Event-based / Salience-gated writes', {
    font: '9px Inter', color: '#484f58', align: 'right',
  });

  // Salience bar
  drawLabel(ctx, x + 8, y + 18, 'Token salience (EventSalienceHead):', { font: '10px Inter', color: '#e06c75' });
  const salY = y + 32;
  const salH = 24;
  for (let t = 0; t < SEQ_LEN; t++) {
    const bx = x + 8 + t * 36;
    const s = salienceValues[t];
    ctx.fillStyle = s > 0.5 ? `rgba(231,76,60,${0.3 + s * 0.7})` : `rgba(100,100,100,0.3)`;
    ctx.fillRect(bx, salY + salH * (1 - s), 32, salH * s);
    ctx.strokeStyle = '#30363d';
    ctx.strokeRect(bx, salY, 32, salH);
    drawLabel(ctx, bx + 16, salY + salH + 2, `t${t}`, {
      font: '8px Inter', color: t === stepIdx ? '#fff' : '#484f58', align: 'center',
    });
  }

  // Event spans
  const spanY = salY + salH + 14;
  if (eventSpans.length) {
    drawLabel(ctx, x + 8, spanY, 'Detected events:', { font: '9px Inter', color: '#e06c75' });
    for (let i = 0; i < eventSpans.length; i++) {
      const [s, e] = eventSpans[i];
      const sx1 = x + 8 + s * 36;
      const sx2 = x + 8 + e * 36 + 32;
      ctx.strokeStyle = '#e74c3c';
      ctx.lineWidth = 2;
      ctx.strokeRect(sx1, spanY + 12, sx2 - sx1, 10);
      drawLabel(ctx, (sx1 + sx2) / 2, spanY + 14, `span ${s}-${e}`, {
        font: '8px Fira Code', color: '#e74c3c', align: 'center',
      });
    }
  }

  // Slot grid
  drawSlotGrid(x, y + h / 2 + 4, w, h / 2 - 8, emSlots, emMask, null, emWritePtr,
    'Episodic Memory Slots', '#e74c3c');
}

function drawFusionBar(x, y, w, h) {
  drawBox(ctx, x, y, w, h, '#161b22', '#30363d');
  drawLabel(ctx, x + 8, y + 4, 'MemoryFusion (magnitude-based softmax gate):', {
    font: '10px Inter', color: '#c678dd',
  });
  const labels = ['WM', 'IM', 'EM'];
  const colors = ['#9b59b6', '#3498db', '#e74c3c'];
  const barW = (w - 30) / 3;
  for (let i = 0; i < 3; i++) {
    const bx = x + 10 + i * (barW + 5);
    const fw = fusionWeights[i];
    ctx.fillStyle = colors[i];
    ctx.globalAlpha = 0.3 + fw * 0.7;
    ctx.fillRect(bx, y + 20, barW * fw * 2, h - 26);
    ctx.globalAlpha = 1;
    drawLabel(ctx, bx + 4, y + 22, `${labels[i]}: ${(fw*100).toFixed(0)}%`, {
      font: '10px Fira Code', color: '#fff',
    });
  }
}
