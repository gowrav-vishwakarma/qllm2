// Bank Detail View — Semantic + Context banks + PhaseInterferenceCoupler
// V6 dual-bank architecture: two parallel CGU pipelines fused by a learned coupler
import {
  cmul, cabs, cconj, cphase, generateRandomComplex, complexToColor,
  drawComplexArrow, drawVectorBars, drawArrow, drawAxes, drawUnitCircle,
  drawLabel, drawBox, clearCanvas, fitCanvas, magPhaseColor,
} from './canvas-utils.js';

let canvas, ctx, W, H;
let playing = false, animStep = 0, speed = 1, animId = null, frameCount = 0;
const TOTAL_STEPS = 7;
let singleBank = false;

let dim = 6;
let expand = 3;
let inputZ = [];
let semGateOut = [], semUpOut = [], semActivated = [], semOutput = [];
let ctxGateOut = [], ctxUpOut = [], ctxActivated = [], ctxOutput = [];
let routingWeights = [0.55, 0.45];
let phaseRot = [[1, 0], [0, 1]]; // unit complex rotations
let rotatedSem = [], rotatedCtx = [];
let couplerOutput = [];

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
  inputZ = generateRandomComplex(dim, 1.0);
  const hidden = dim * expand;

  semGateOut = generateRandomComplex(hidden, 0.8);
  semUpOut = generateRandomComplex(hidden, 0.8);
  semActivated = semUpOut.map(([r, i]) => modReLU(r, i, -0.1));
  semOutput = fusedGate(semGateOut, semActivated).slice(0, dim);

  ctxGateOut = generateRandomComplex(hidden, 0.7);
  ctxUpOut = generateRandomComplex(hidden, 0.7);
  ctxActivated = ctxUpOut.map(([r, i]) => modReLU(r, i, -0.1));
  ctxOutput = fusedGate(ctxGateOut, ctxActivated).slice(0, dim);

  // Coupler: compute routing weights from magnitudes
  let semMags = 0, ctxMags = 0;
  for (let i = 0; i < dim; i++) {
    semMags += cabs(semOutput[i][0], semOutput[i][1]);
    ctxMags += cabs(ctxOutput[i][0], ctxOutput[i][1]);
  }
  const total = semMags + ctxMags + 1e-8;
  routingWeights = [semMags / total, ctxMags / total];

  // Learned phase rotations (unit complex)
  const a1 = Math.random() * Math.PI * 2;
  const a2 = a1 + Math.PI * 0.7 + Math.random() * 0.6; // pushed apart by diversity loss
  phaseRot = [[Math.cos(a1), Math.sin(a1)], [Math.cos(a2), Math.sin(a2)]];

  rotatedSem = semOutput.map(([r, i]) => cmul(r, i, phaseRot[0][0], phaseRot[0][1]));
  rotatedCtx = ctxOutput.map(([r, i]) => cmul(r, i, phaseRot[1][0], phaseRot[1][1]));

  couplerOutput = [];
  for (let i = 0; i < dim; i++) {
    const wr = routingWeights[0] * rotatedSem[i][0] + routingWeights[1] * rotatedCtx[i][0];
    const wi = routingWeights[0] * rotatedSem[i][1] + routingWeights[1] * rotatedCtx[i][1];
    couplerOutput.push([wr, wi]);
  }
}

function fusedGate(gate, activated) {
  const out = [];
  for (let j = 0; j < gate.length; j++) {
    const [gr, gi] = gate[j];
    const gateMag = sigmoid(cabs(gr, gi));
    const gateAngle = cphase(gr, gi);
    const phR = Math.cos(gateAngle), phI = Math.sin(gateAngle);
    const [pr, pi] = cmul(phR, phI, activated[j][0], activated[j][1]);
    out.push([pr * gateMag, pi * gateMag]);
  }
  return out;
}

export function update(cfg) {
  dim = Math.min(cfg.dim || 384, 8);
  expand = cfg.expand || cfg.bank_expand || 3;
  singleBank = cfg.single_bank || false;
  generateData();
}

export function updateToggles(tog) {
  singleBank = !tog.dual_banks;
}

export function play() { playing = true; }
export function pause() { playing = false; }
export function reset() { animStep = 0; frameCount = 0; generateData(); }
export function setSpeed(s) { speed = s; }
export function stepForward() { if (animStep < TOTAL_STEPS - 1) animStep++; }
export function stepBack() { if (animStep > 0) animStep--; }

export function start() {
  if (animId) return;
  function loop() {
    animId = requestAnimationFrame(loop);
    frameCount++;
    if (playing && frameCount % Math.max(1, Math.round(40 / speed)) === 0) {
      animStep++;
      if (animStep >= TOTAL_STEPS) { animStep = TOTAL_STEPS - 1; playing = false; }
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

  if (singleBank) {
    drawSingleBankMode();
    return;
  }

  drawLabel(ctx, W / 2, 8, 'Dual Banks + PhaseInterferenceCoupler', {
    font: 'bold 14px Inter', color: '#58a6ff', align: 'center',
  });

  const stepLabels = [
    '① Input z → both banks',
    '② Semantic Bank: ComplexNorm → CGU',
    '③ Context Bank: ComplexNorm → CGU',
    '④ Coupler: compute routing weights from |outputs|',
    '⑤ Coupler: apply learned phase rotations',
    '⑥ Coupler: weighted sum of rotated outputs',
    '⑦ out_proj + ComplexNorm → coupled output',
  ];

  drawBox(ctx, 10, 28, W - 20, 22, '#161b22', '#30363d');
  drawLabel(ctx, 16, 32, stepLabels[animStep], { font: 'bold 11px Inter', color: '#7ee787' });

  const panelW = (W - 50) / 3;
  const panelH = H - 100;
  const topY = 58;

  // Left: Semantic Bank
  drawBankPanel(10, topY, panelW, panelH, 'Semantic Bank', '#5dade2',
    inputZ, semGateOut, semActivated, semOutput, animStep >= 1);

  // Center: Context Bank
  drawBankPanel(20 + panelW, topY, panelW, panelH, 'Context Bank', '#2e86c1',
    inputZ, ctxGateOut, ctxActivated, ctxOutput, animStep >= 2);

  // Right: Coupler
  drawCouplerPanel(30 + panelW * 2, topY, panelW, panelH);

  // Equation
  drawLabel(ctx, 10, H - 18, 'output = norm(proj(Σᵢ routeᵢ · rotate(sourceᵢ, phaseᵢ)))', {
    font: '11px Fira Code', color: '#c9d1d9',
  });
}

function drawBankPanel(x, y, w, h, title, color, input, gate, activated, output, active) {
  drawBox(ctx, x, y, w, h, active ? '#0d1117' : '#0a0e14', '#21262d');
  drawLabel(ctx, x + w / 2, y + 4, title, { font: 'bold 12px Inter', color, align: 'center' });

  if (!active) {
    drawLabel(ctx, x + w / 2, y + h / 2, '(waiting...)', {
      font: '11px Inter', color: '#484f58', align: 'center',
    });
    return;
  }

  const innerW = w - 16;
  const secH = (h - 30) / 4;

  // Input
  drawLabel(ctx, x + 8, y + 24, 'Input z:', { font: '10px Inter', color: '#8b949e' });
  drawVectorBars(ctx, x + 8, y + 36, innerW, secH - 20, input);

  // CGU gate
  const gY = y + secH + 20;
  drawLabel(ctx, x + 8, gY, 'CGU gate (σ(|gate|)):', { font: '10px Inter', color: '#e5c07b' });
  const showN = Math.min(gate.length, 10);
  const barW = innerW / showN;
  for (let i = 0; i < showN; i++) {
    const mag = sigmoid(cabs(gate[i][0], gate[i][1]));
    ctx.fillStyle = `rgba(229,192,123,${0.3 + mag * 0.7})`;
    const bh = mag * (secH - 24);
    ctx.fillRect(x + 8 + i * barW, gY + 14 + (secH - 24 - bh), barW - 1, bh);
  }

  // Activation
  const aY = y + secH * 2 + 10;
  drawLabel(ctx, x + 8, aY, 'ModReLU(up):', { font: '10px Inter', color: '#98c379' });
  drawVectorBars(ctx, x + 8, aY + 14, innerW, secH - 20, activated.slice(0, dim));

  // Output
  const oY = y + secH * 3;
  drawLabel(ctx, x + 8, oY, `${title.split(' ')[0]} output:`, { font: '10px Inter', color });
  drawVectorBars(ctx, x + 8, oY + 14, innerW, secH - 20, output);
}

function drawCouplerPanel(x, y, w, h) {
  const active = animStep >= 3;
  drawBox(ctx, x, y, w, h, active ? '#0d1117' : '#0a0e14', '#21262d');
  drawLabel(ctx, x + w / 2, y + 4, 'PhaseInterferenceCoupler', {
    font: 'bold 12px Inter', color: '#8e44ad', align: 'center',
  });

  if (!active) {
    drawLabel(ctx, x + w / 2, y + h / 2, '(waiting for banks...)', {
      font: '11px Inter', color: '#484f58', align: 'center',
    });
    return;
  }

  const innerW = w - 16;
  const cx = x + w / 2;

  // Routing weights (step 4)
  if (animStep >= 3) {
    const rY = y + 24;
    drawLabel(ctx, x + 8, rY, 'Routing weights (softmax):', { font: '10px Inter', color: '#8b949e' });
    const bw = innerW / 2 - 4;
    const bh = 30;
    ctx.fillStyle = '#5dade2';
    ctx.fillRect(x + 8, rY + 14, bw * routingWeights[0] / 0.5, bh);
    drawLabel(ctx, x + 8 + bw * routingWeights[0] / 1, rY + 22, `Sem: ${(routingWeights[0]*100).toFixed(0)}%`, {
      font: '10px Fira Code', color: '#fff', align: 'center',
    });
    ctx.fillStyle = '#2e86c1';
    ctx.fillRect(x + 8 + bw + 8, rY + 14, bw * routingWeights[1] / 0.5, bh);
    drawLabel(ctx, x + 8 + bw + 8 + bw * routingWeights[1] / 1, rY + 22, `Ctx: ${(routingWeights[1]*100).toFixed(0)}%`, {
      font: '10px Fira Code', color: '#fff', align: 'center',
    });
  }

  // Phase rotations (step 5)
  if (animStep >= 4) {
    const pY = y + 80;
    drawLabel(ctx, x + 8, pY, 'Learned phase rotations:', { font: '10px Inter', color: '#8b949e' });
    const planeR = Math.min(innerW / 4 - 8, 40);
    const p1cx = x + w / 4, p2cx = x + 3 * w / 4;
    const pcy = pY + 20 + planeR;
    drawAxes(ctx, p1cx, pcy, planeR);
    drawUnitCircle(ctx, p1cx, pcy, planeR);
    drawComplexArrow(ctx, p1cx, pcy, phaseRot[0][0], phaseRot[0][1], planeR, '#5dade2');
    drawLabel(ctx, p1cx, pcy + planeR + 6, `φ₀=${(cphase(phaseRot[0][0], phaseRot[0][1])*180/Math.PI).toFixed(0)}°`, {
      font: '9px Fira Code', color: '#5dade2', align: 'center',
    });

    drawAxes(ctx, p2cx, pcy, planeR);
    drawUnitCircle(ctx, p2cx, pcy, planeR);
    drawComplexArrow(ctx, p2cx, pcy, phaseRot[1][0], phaseRot[1][1], planeR, '#2e86c1');
    drawLabel(ctx, p2cx, pcy + planeR + 6, `φ₁=${(cphase(phaseRot[1][0], phaseRot[1][1])*180/Math.PI).toFixed(0)}°`, {
      font: '9px Fira Code', color: '#2e86c1', align: 'center',
    });

    // Diversity angle
    const angleDiff = Math.abs(cphase(phaseRot[0][0], phaseRot[0][1]) - cphase(phaseRot[1][0], phaseRot[1][1]));
    drawLabel(ctx, cx, pcy + planeR + 20, `Diversity: Δφ = ${(angleDiff*180/Math.PI).toFixed(0)}° (training pushes apart)`, {
      font: '9px Inter', color: '#bd93f9', align: 'center',
    });
  }

  // Weighted sum (step 6)
  if (animStep >= 5) {
    const sY = y + h - 100;
    drawLabel(ctx, x + 8, sY, 'Weighted rotated sum:', { font: '10px Inter', color: '#c678dd' });
    drawVectorBars(ctx, x + 8, sY + 14, innerW, 30, couplerOutput);
  }

  // Final output (step 7)
  if (animStep >= 6) {
    const fY = y + h - 44;
    drawLabel(ctx, x + 8, fY, 'out_proj → ComplexNorm → final:', { font: '10px Inter', color: '#7ee787' });
    drawVectorBars(ctx, x + 8, fY + 14, innerW, 24, couplerOutput);
  }
}

function drawSingleBankMode() {
  drawLabel(ctx, W / 2, H / 2 - 30, 'Single Bank Mode (V7)', {
    font: 'bold 16px Inter', color: '#58a6ff', align: 'center',
  });
  drawLabel(ctx, W / 2, H / 2, 'V7 uses a single CGU per block — no dual banks or coupler.', {
    font: '12px Inter', color: '#8b949e', align: 'center',
  });
  drawLabel(ctx, W / 2, H / 2 + 20, 'Switch to V6 and enable "Dual Banks" to see the full bank architecture.', {
    font: '12px Inter', color: '#8b949e', align: 'center',
  });
  drawLabel(ctx, W / 2, H / 2 + 50, 'See the "CGU Inside" tab for the single-bank CGU detail.', {
    font: '11px Inter', color: '#484f58', align: 'center',
  });
}
