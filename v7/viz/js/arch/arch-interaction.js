// Architecture interaction helpers (hover tooltips, click-to-detail routing)
// Most interaction is handled directly in arch-scene.js (raycaster, hover).
// This module provides the info panel update logic.
import { fmtNum, computeBlockParams, computeTotalParams, getDtSchedule } from '../presets.js';

export function updateInfoPanel(cfg, tog, infoEl) {
  const dim = cfg.dim;
  const nLayers = cfg.n_layers || cfg.num_layers;
  const heads = cfg.n_heads || cfg.pam_num_heads || 6;
  const headDim = cfg.head_dim || cfg.pam_head_dim || 64;
  const expand = cfg.expand || cfg.bank_expand || 3;

  const totals = computeTotalParams(cfg, tog);
  const block0 = computeBlockParams(cfg, tog, 0);
  const dtSchedule = getDtSchedule(cfg);

  let html = `<div class="pt">${fmtNum(totals.total)}<span>Total Parameters</span></div>`;
  html += `<div class="is"><h3 style="font-size:12px;color:#7ee787;margin-bottom:6px">Architecture</h3>`;
  html += row('Dimension', dim);
  html += row('Layers', nLayers);
  html += row('Heads', heads);
  html += row('Head Dim', headDim);
  html += row('CGU Expand', expand + '×');
  html += row('Vocab', fmtNum(cfg.vocab_size || 50257));
  html += row('Max Seq', fmtNum(cfg.max_seq_len || 2048));
  html += row('Activation', cfg.activation || 'modrelu');
  html += `</div>`;

  html += `<div class="is"><h3 style="font-size:12px;color:#7ee787;margin-bottom:6px">Features</h3>`;
  html += row('RoPE', tog.use_rope ? '✓' : '✗');
  html += row('GSP', tog.use_gsp ? '✓' : '✗');
  html += row('Hierarchical dt', tog.hierarchical_dt ? '✓' : '✗');
  html += row('Cross-level', tog.cross_level ? '✓' : '✗');
  html += row('Fused QKV', tog.fused_qkv ? '✓' : '✗');
  html += row('Reverse Assoc', tog.use_reverse_assoc ? '✓' : '✗');
  if (tog.attention) html += row('Attention', `every ${cfg.attn_every}`);
  if (tog.dual_banks) html += row('Banks', 'Dual');
  if (tog.working_memory) html += row('Working Mem', '✓');
  if (tog.internal_memory) html += row('Internal Mem', '✓');
  if (tog.episodic_memory) html += row('Episodic Mem', '✓');
  html += `</div>`;

  html += `<div class="is"><h3 style="font-size:12px;color:#7ee787;margin-bottom:6px">Param Breakdown</h3>`;
  html += row('Embed', fmtNum(totals.embed));
  html += row('Blocks', fmtNum(totals.blocks));
  html += row('  CGU/block', fmtNum(block0.cgu));
  html += row('  PAM/block', fmtNum(block0.pam));
  if (block0.attn) html += row('  Attn/block', fmtNum(block0.attn));
  if (block0.memory) html += row('  Mem/block', fmtNum(block0.memory));
  html += row('Head', fmtNum(totals.head));
  if (totals.aux) html += row('Aux Heads', fmtNum(totals.aux));
  html += `</div>`;

  html += `<div class="is"><h3 style="font-size:12px;color:#7ee787;margin-bottom:6px">dt Schedule</h3>`;
  for (let i = 0; i < Math.min(dtSchedule.length, 8); i++) {
    html += row(`Layer ${i}`, dtSchedule[i].toFixed(2));
  }
  if (dtSchedule.length > 8) html += `<div class="ir"><span class="ik">...</span></div>`;
  html += `</div>`;

  html += `<div class="is"><h3 style="font-size:12px;color:#7ee787;margin-bottom:6px">Legend</h3>`;
  html += legend(0xf0c674, 'ComplexEmbed');
  html += legend(0x4a90d9, 'CGU');
  html += legend(0x2ecc71, 'PAM');
  if (tog.attention) html += legend(0x00bcd4, 'Attention');
  if (tog.dual_banks) { html += legend(0x5dade2, 'SemanticBank'); html += legend(0x2e86c1, 'ContextBank'); }
  if (tog.working_memory || tog.internal_memory || tog.episodic_memory) html += legend(0x9b59b6, 'Memory');
  html += legend(0xe67e22, 'Output Head');
  if (tog.cross_level) html += legend(0xe91e63, 'Cross-level');
  html += `</div>`;

  html += `<div class="hint">Click a CGU or PAM block to explore its internals in detail tabs →</div>`;
  infoEl.innerHTML = html;
}

function row(k, v) { return `<div class="ir"><span class="ik">${k}</span><span class="iv">${v}</span></div>`; }
function legend(hex, name) {
  const c = '#' + hex.toString(16).padStart(6, '0');
  return `<div class="li"><div class="lc" style="background:${c}"></div>${name}</div>`;
}
