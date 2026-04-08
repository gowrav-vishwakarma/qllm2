// Main entry point — tab switching, shared config, control wiring
import { V7_PRESETS, V6_PRESETS, fmtNum, computeTotalParams } from './presets.js';
import { initScene, resizeScene, startLoop, stopLoop, setParticlesEnabled } from './arch/arch-scene.js';
import { buildModel } from './arch/arch-builder.js';
import { updateInfoPanel } from './arch/arch-interaction.js';
import * as PAM from './detail/pam-detail.js';
import * as CGU from './detail/cgu-detail.js';
import * as CMP from './detail/complex-detail.js';
import * as DUAL from './detail/dual-recurrent.js';
import * as MH from './detail/multihead-detail.js';
import * as BANK from './detail/bank-detail.js';
import * as MEM from './detail/memory-detail.js';

let currentTab = 'arch';
let currentFamily = 'v7';
let _lastPresetValues = {};

// ---- DOM refs ----
const presetSel = document.getElementById('preset-sel');
const familySel = document.getElementById('family-sel');
const tabBtns = document.querySelectorAll('.tab-btn');
const views = document.querySelectorAll('.view');
const infoEl = document.getElementById('info-content');
const viewport = document.getElementById('viewport');

// Toggles
const togIds = [
  'use_rope','use_gsp','hierarchical_dt','cross_level','fused_qkv','use_reverse_assoc',
  'multi_scale_loss','dual_banks','sequential_pam','attention',
  'working_memory','internal_memory','episodic_memory',
];
const v6Only = new Set(['dual_banks','sequential_pam','attention','working_memory','internal_memory','episodic_memory']);

// Numeric inputs
const numFields = [
  { id:'inp-layers',  key:'n_layers', alt:'num_layers' },
  { id:'inp-dim',     key:'dim' },
  { id:'inp-heads',   key:'n_heads', alt:'pam_num_heads' },
  { id:'inp-headdim', key:'head_dim', alt:'pam_head_dim' },
  { id:'inp-expand',  key:'expand', alt:'bank_expand' },
  { id:'inp-vocab',   key:'vocab_size' },
  { id:'inp-maxseq',  key:'max_seq_len' },
];

// ---- Init ----
function main() {
  populatePresets();
  initScene(viewport, onBlockClick);

  // Tab buttons
  tabBtns.forEach(btn => btn.addEventListener('click', () => switchTab(btn.dataset.tab)));

  // Family / preset changes
  familySel.addEventListener('change', () => {
    currentFamily = familySel.value;
    populatePresets();
    syncUI();
    rebuild();
  });
  presetSel.addEventListener('change', () => { syncUI(); rebuild(); });

  // Toggle changes
  togIds.forEach(id => {
    const el = document.getElementById('tog-' + id);
    if (el) el.addEventListener('change', rebuild);
  });

  // Numeric input changes
  numFields.forEach(f => {
    const el = document.getElementById(f.id);
    if (el) el.addEventListener('change', () => { checkCustomIndicator(); rebuild(); });
  });

  // Detail controls
  wireDetailControls();

  syncUI();
  rebuild();
  startLoop(viewport);

  window.addEventListener('resize', () => {
    if (currentTab === 'arch') resizeScene(viewport);
  });
}

// ---- Presets ----
function populatePresets() {
  const presets = currentFamily === 'v7' ? V7_PRESETS : V6_PRESETS;
  presetSel.innerHTML = '';
  for (const k of Object.keys(presets)) {
    const opt = document.createElement('option');
    opt.value = k; opt.textContent = k;
    presetSel.appendChild(opt);
  }
}

function getPresetConfig() {
  const presets = currentFamily === 'v7' ? V7_PRESETS : V6_PRESETS;
  return presets[presetSel.value] || Object.values(presets)[0];
}

function syncUI() {
  const cfg = getPresetConfig();
  fillInputs(cfg);
  syncToggles(cfg);
  _lastPresetValues = {};
  numFields.forEach(f => {
    const el = document.getElementById(f.id);
    if (el) _lastPresetValues[f.id] = el.value;
  });
  checkCustomIndicator();
}

function fillInputs(cfg) {
  numFields.forEach(f => {
    const el = document.getElementById(f.id);
    if (el) el.value = cfg[f.key] ?? cfg[f.alt] ?? '';
  });
}

function syncToggles(cfg) {
  const isV7 = currentFamily === 'v7';
  togIds.forEach(id => {
    const el = document.getElementById('tog-' + id);
    if (!el) return;
    el.disabled = isV7 && v6Only.has(id);
    if (id === 'dual_banks') el.checked = !isV7 && !cfg.single_bank;
    else if (id === 'sequential_pam') el.checked = !isV7 && !cfg.interleave_pam;
    else if (id === 'attention') el.checked = cfg.use_attention || false;
    else if (id === 'working_memory') el.checked = (cfg.num_wm_slots || 0) > 0;
    else if (id === 'internal_memory') el.checked = (cfg.num_im_slots || 0) > 0;
    else if (id === 'episodic_memory') el.checked = (cfg.num_episodic_slots || 0) > 0;
    else el.checked = cfg[id] !== false && cfg[id] !== undefined;
  });
}

function checkCustomIndicator() {
  const ind = document.getElementById('custom-indicator');
  if (!ind) return;
  let custom = false;
  numFields.forEach(f => {
    const el = document.getElementById(f.id);
    if (el && _lastPresetValues[f.id] !== undefined && el.value !== _lastPresetValues[f.id]) custom = true;
  });
  ind.style.display = custom ? 'block' : 'none';
}

// ---- Config ----
function getActiveConfig() {
  const base = { ...getPresetConfig() };
  numFields.forEach(f => {
    const el = document.getElementById(f.id);
    if (el && el.value !== '') {
      const v = parseInt(el.value, 10);
      if (!isNaN(v)) { base[f.key] = v; if (f.alt) base[f.alt] = v; }
    }
  });
  return base;
}

function getToggles() {
  const tog = {};
  togIds.forEach(id => {
    const el = document.getElementById('tog-' + id);
    tog[id] = el ? el.checked : false;
  });
  return tog;
}

// ---- Build ----
function rebuild() {
  const cfg = getActiveConfig();
  const tog = getToggles();
  if (currentTab === 'arch') {
    buildModel(cfg, tog, (c, t) => updateInfoPanel(c, t, infoEl));
  }
  PAM.update(cfg);
  CGU.update(cfg);
  CMP.update(cfg);
  DUAL.update(cfg);
  MH.update(cfg);
  BANK.update(cfg);
  BANK.updateToggles(tog);
  MEM.update(cfg);
  MEM.updateToggles(tog);
  updateInfoPanel(cfg, tog, infoEl);

  // Data flow toggle
  const dfEl = document.getElementById('tog-dataflow');
  if (dfEl) setParticlesEnabled(dfEl.checked);
}

// ---- Tabs ----
function switchTab(tab) {
  currentTab = tab;
  tabBtns.forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  views.forEach(v => v.classList.toggle('active', v.id === 'view-' + tab));

  if (tab === 'arch') { resizeScene(viewport); startLoop(viewport); }
  else { stopLoop(); }

  const detailModules = { pam: PAM, cgu: CGU, complex: CMP, dual: DUAL, multihead: MH, banks: BANK, memory: MEM };
  for (const [key, mod] of Object.entries(detailModules)) {
    if (tab === key) mod.start();
    else mod.stop();
  }
}

function onBlockClick(data) {
  if (data.type === 'pam') switchTab('pam');
  else if (data.type === 'cgu') switchTab('cgu');
  else if (data.type === 'bank') switchTab('banks');
  else if (data.type === 'memory') switchTab('memory');
  else if (data.type === 'attn') switchTab('complex');
}

// ---- Detail controls ----
function wireDetailControls() {
  // PAM
  PAM.init(document.getElementById('pam-canvas'));
  wire('pam-play', () => PAM.play());
  wire('pam-pause', () => PAM.pause());
  wire('pam-step-fwd', () => PAM.stepForward());
  wire('pam-step-back', () => PAM.stepBack());
  wire('pam-reset', () => PAM.reset());
  wireRange('pam-speed', v => PAM.setSpeed(v));

  // CGU
  CGU.init(document.getElementById('cgu-canvas'));
  wire('cgu-play', () => CGU.play());
  wire('cgu-pause', () => CGU.pause());
  wire('cgu-step-fwd', () => CGU.stepForward());
  wire('cgu-step-back', () => CGU.stepBack());
  wire('cgu-reset', () => CGU.reset());
  wireRange('cgu-speed', v => CGU.setSpeed(v));

  // Complex
  CMP.init(document.getElementById('complex-canvas'));
  const modeSel = document.getElementById('complex-mode');
  if (modeSel) modeSel.addEventListener('change', () => CMP.setMode(modeSel.value));

  // Dual
  DUAL.init(document.getElementById('dual-canvas'));
  wire('dual-play', () => DUAL.play());
  wire('dual-pause', () => DUAL.pause());
  wire('dual-step-fwd', () => DUAL.stepForward());
  wire('dual-step-back', () => DUAL.stepBack());
  wire('dual-reset', () => DUAL.reset());
  wireRange('dual-speed', v => DUAL.setSpeed(v));

  // Multi-Head PAM
  MH.init(document.getElementById('multihead-canvas'));
  wire('mh-play', () => MH.play());
  wire('mh-pause', () => MH.pause());
  wire('mh-step-fwd', () => MH.stepForward());
  wire('mh-step-back', () => MH.stepBack());
  wire('mh-reset', () => MH.reset());
  wireRange('mh-speed', v => MH.setSpeed(v));

  // Banks
  BANK.init(document.getElementById('bank-canvas'));
  wire('bank-play', () => BANK.play());
  wire('bank-pause', () => BANK.pause());
  wire('bank-step-fwd', () => BANK.stepForward());
  wire('bank-step-back', () => BANK.stepBack());
  wire('bank-reset', () => BANK.reset());
  wireRange('bank-speed', v => BANK.setSpeed(v));

  // Memory
  MEM.init(document.getElementById('mem-canvas'));
  wire('mem-play', () => MEM.play());
  wire('mem-pause', () => MEM.pause());
  wire('mem-step-fwd', () => MEM.stepForward());
  wire('mem-step-back', () => MEM.stepBack());
  wire('mem-reset', () => MEM.reset());
  wireRange('mem-speed', v => MEM.setSpeed(v));

  // Data flow toggle
  const dfEl = document.getElementById('tog-dataflow');
  if (dfEl) dfEl.addEventListener('change', () => setParticlesEnabled(dfEl.checked));
}

function wire(id, fn) {
  const el = document.getElementById(id);
  if (el) el.addEventListener('click', fn);
}

function wireRange(id, fn) {
  const el = document.getElementById(id);
  if (el) el.addEventListener('input', () => fn(parseFloat(el.value)));
}

// Boot
main();
