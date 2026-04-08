// Builds the 3D architecture model from config + toggles
import * as THREE from 'three';
import {
  COLORS, dimScale, makeBox, makeLabel, makeLine, makeCurve,
  addClickable, addHeadStripes, dtColor, getSceneRefs, clearModel,
  setFlowWaypoints,
} from './arch-scene.js';
import { getDtSchedule, fmtNum, computeBlockParams, computeTotalParams } from '../presets.js';

const GAP = 0.3, LAYER_GAP = 0.5, NORM_H = 0.12;

export function buildModel(cfg, tog, infoCallback) {
  clearModel();
  const { modelGroup, camera, controls } = getSceneRefs();
  const nLayers = cfg.n_layers || cfg.num_layers;
  const S = dimScale(cfg);
  const BW = S.BW, BD = S.BD, SUB_H = S.SUB_H;
  const heads = cfg.n_heads || cfg.pam_num_heads || 6;
  const dtSchedule = getDtSchedule(cfg);
  let y = 0;

  // Complex Embed
  const embedH = 0.4;
  const embed = makeBox(BW, embedH, BD, COLORS.embed);
  embed.position.set(0, y + embedH / 2, 0);
  modelGroup.add(embed);
  const eLbl = makeLabel('ComplexEmbed');
  eLbl.position.set(0, embedH + 0.15, 0);
  modelGroup.add(eLbl);
  const dLbl = makeLabel(`${cfg.dim}d  v=${cfg.vocab_size || 50257}`, 'label3d dim');
  dLbl.position.set(0, embedH + 0.35, 0);
  modelGroup.add(dLbl);
  y += embedH + GAP;

  if (tog.sequential_pam) {
    // V6 sequential mode
    const cguBlockH = (nLayers * (SUB_H + NORM_H + GAP)) + GAP;
    const cguContainer = makeBox(BW + 0.4, cguBlockH, BD + 0.4, COLORS.container, 0.15);
    cguContainer.position.set(0, y + cguBlockH / 2, 0);
    modelGroup.add(cguContainer);
    const cguTitle = makeLabel('CGU Blocks');
    cguTitle.position.set(0, y + cguBlockH + 0.2, 0);
    modelGroup.add(cguTitle);

    let cy = y + GAP / 2;
    for (let i = 0; i < nLayers; i++) {
      const norm = makeBox(BW, NORM_H, BD, COLORS.norm);
      norm.position.set(0, cy + NORM_H / 2, 0);
      modelGroup.add(norm);
      cy += NORM_H + 0.05;

      if (tog.dual_banks) {
        const bw = BW * 0.45;
        const sem = makeBox(bw, SUB_H, BD * S.CGU_D_MULT, COLORS.semBank);
        sem.position.set(-bw / 2 - 0.1, cy + SUB_H / 2, 0);
        modelGroup.add(sem);
        addClickable(sem, { type:'cgu', layer:i });
        const ctx = makeBox(bw, SUB_H, BD * S.CGU_D_MULT, COLORS.ctxBank);
        ctx.position.set(bw / 2 + 0.1, cy + SUB_H / 2, 0);
        modelGroup.add(ctx);
        addClickable(ctx, { type:'cgu', layer:i });
        if (i === 0) {
          const sl = makeLabel('Semantic', 'label3d dim');
          sl.position.set(-bw / 2 - 0.1, cy + SUB_H + 0.15, 0);
          modelGroup.add(sl);
          const cl = makeLabel('Context', 'label3d dim');
          cl.position.set(bw / 2 + 0.1, cy + SUB_H + 0.15, 0);
          modelGroup.add(cl);
        }
      } else {
        const cgu = makeBox(BW, SUB_H, BD * S.CGU_D_MULT, COLORS.cgu);
        cgu.position.set(0, cy + SUB_H / 2, 0);
        modelGroup.add(cgu);
        addClickable(cgu, { type:'cgu', layer:i });
        if (i === 0) {
          const l = makeLabel('CGU', 'label3d');
          l.position.set(0, cy + SUB_H + 0.15, 0);
          modelGroup.add(l);
        }
      }
      cy += SUB_H + GAP;
    }
    y += cguBlockH + GAP;

    const pamBlockH = (nLayers * (SUB_H + NORM_H + GAP)) + GAP;
    const pamContainer = makeBox(BW + 0.4, pamBlockH, BD + 0.4, COLORS.container, 0.15);
    pamContainer.position.set(0, y + pamBlockH / 2, 0);
    modelGroup.add(pamContainer);
    const pamTitle = makeLabel('PAM Blocks');
    pamTitle.position.set(0, y + pamBlockH + 0.2, 0);
    modelGroup.add(pamTitle);

    let py = y + GAP / 2;
    for (let i = 0; i < nLayers; i++) {
      const norm = makeBox(BW, NORM_H, BD, COLORS.norm);
      norm.position.set(0, py + NORM_H / 2, 0);
      modelGroup.add(norm);
      py += NORM_H + 0.05;
      const pam = makeBox(BW, SUB_H, BD, COLORS.pam);
      pam.position.set(0, py + SUB_H / 2, 0);
      modelGroup.add(pam);
      addClickable(pam, { type:'pam', layer:i });
      addHeadStripes(pam, BW, SUB_H, BD, heads);
      if (i === 0) {
        const l = makeLabel('PAM', 'label3d');
        l.position.set(0, py + SUB_H + 0.15, 0);
        modelGroup.add(l);
      }
      const dtLbl = makeLabel(`dt=${dtSchedule[i]?.toFixed(1)}`, 'label3d dt');
      dtLbl.position.set(BW / 2 + 0.3, py + SUB_H / 2, 0);
      modelGroup.add(dtLbl);
      py += SUB_H + GAP;
    }
    y += pamBlockH + GAP;
  } else {
    // Interleaved (default V7)
    const blockH = nLayers * (NORM_H + SUB_H + 0.05 + NORM_H + SUB_H + GAP) + GAP;
    for (let i = 0; i < nLayers; i++) {
      const slotH = NORM_H + SUB_H + 0.05 + NORM_H + SUB_H;
      const blockContainer = makeBox(BW + 0.6, slotH + 0.3, BD + 0.6, COLORS.container, 0.1);
      blockContainer.position.set(0, y + (slotH + 0.3) / 2, 0);
      modelGroup.add(blockContainer);

      const lLbl = makeLabel(`Layer ${i}`, 'label3d layer');
      lLbl.position.set(-BW / 2 - 0.5, y + slotH / 2, 0);
      modelGroup.add(lLbl);

      let ly = y + 0.15;
      // Pre-norm + CGU
      const n1 = makeBox(BW, NORM_H, BD, COLORS.norm);
      n1.position.set(0, ly + NORM_H / 2, 0);
      modelGroup.add(n1);
      ly += NORM_H + 0.05;

      if (tog.dual_banks) {
        const bw = BW * 0.45;
        const sem = makeBox(bw, SUB_H, BD * S.CGU_D_MULT, COLORS.semBank);
        sem.position.set(-bw / 2 - 0.1, ly + SUB_H / 2, 0);
        modelGroup.add(sem);
        addClickable(sem, { type:'bank', layer:i });
        const ctxMesh = makeBox(bw, SUB_H, BD * S.CGU_D_MULT, COLORS.ctxBank);
        ctxMesh.position.set(bw / 2 + 0.1, ly + SUB_H / 2, 0);
        modelGroup.add(ctxMesh);
        addClickable(ctxMesh, { type:'bank', layer:i });
        if (i === 0) {
          const sl = makeLabel('Sem', 'label3d dim');
          sl.position.set(-bw / 2 - 0.1, ly - 0.1, 0);
          modelGroup.add(sl);
          const cl = makeLabel('Ctx', 'label3d dim');
          cl.position.set(bw / 2 + 0.1, ly - 0.1, 0);
          modelGroup.add(cl);
        }
      } else {
        const cgu = makeBox(BW, SUB_H, BD * S.CGU_D_MULT, COLORS.cgu);
        cgu.position.set(0, ly + SUB_H / 2, 0);
        modelGroup.add(cgu);
        addClickable(cgu, { type:'cgu', layer:i });
        if (i === 0) {
          const l = makeLabel('CGU', 'label3d');
          l.position.set(0, ly + SUB_H + 0.1, 0);
          modelGroup.add(l);
        }
      }
      ly += SUB_H + 0.05;

      // Pre-norm + PAM
      const n2 = makeBox(BW, NORM_H, BD, COLORS.norm);
      n2.position.set(0, ly + NORM_H / 2, 0);
      modelGroup.add(n2);
      ly += NORM_H + 0.05;

      const pam = makeBox(BW, SUB_H, BD, COLORS.pam);
      pam.position.set(0, ly + SUB_H / 2, 0);
      modelGroup.add(pam);
      addClickable(pam, { type:'pam', layer:i });
      addHeadStripes(pam, BW, SUB_H, BD, heads);
      if (i === 0) {
        const l = makeLabel('PAM', 'label3d');
        l.position.set(0, ly + SUB_H + 0.1, 0);
        modelGroup.add(l);
      }
      const dtLbl = makeLabel(`dt=${dtSchedule[i]?.toFixed(1)}`, 'label3d dt');
      dtLbl.position.set(BW / 2 + 0.4, ly + SUB_H / 2, 0);
      modelGroup.add(dtLbl);

      // Residual arrow
      const arrowStart = new THREE.Vector3(BW / 2 + 0.8, y, 0);
      const arrowEnd = new THREE.Vector3(BW / 2 + 0.8, y + slotH + 0.3, 0);
      modelGroup.add(makeLine([arrowStart, arrowEnd], COLORS.residual));

      // Attention
      if (tog.attention && cfg.attn_every && ((i + 1) % cfg.attn_every === 0)) {
        const attn = makeBox(BW * 0.5, SUB_H * 0.6, BD * 0.5, COLORS.attn, 0.85);
        attn.position.set(BW / 2 + 1.5, y + slotH / 2, 0);
        modelGroup.add(attn);
        addClickable(attn, { type:'attn', layer:i });
        const al = makeLabel('Attn', 'label3d');
        al.position.set(BW / 2 + 1.5, y + slotH / 2 + SUB_H * 0.4, 0);
        modelGroup.add(al);
      }

      // Memory (clickable -> memory tab)
      let mx = -BW / 2 - 1.3;
      if (tog.working_memory) {
        const wm = makeBox(0.5, SUB_H * 0.5, BD * 0.3, COLORS.memory, 0.75);
        wm.position.set(mx, y + slotH / 2, 0);
        modelGroup.add(wm);
        addClickable(wm, { type:'memory', layer:i });
        if (i === 0) {
          const l = makeLabel('WM', 'label3d dim');
          l.position.set(mx, y + slotH / 2 + 0.35, 0);
          modelGroup.add(l);
        }
        mx -= 0.8;
      }
      if (tog.internal_memory) {
        const imBox = makeBox(0.5, SUB_H * 0.5, BD * 0.3, COLORS.memory, 0.7);
        imBox.position.set(mx, y + slotH / 2, 0);
        modelGroup.add(imBox);
        addClickable(imBox, { type:'memory', layer:i });
        if (i === 0) {
          const l = makeLabel('IM', 'label3d dim');
          l.position.set(mx, y + slotH / 2 + 0.35, 0);
          modelGroup.add(l);
        }
        mx -= 0.8;
      }
      if (tog.episodic_memory) {
        const emBox = makeBox(0.5, SUB_H * 0.5, BD * 0.3, COLORS.memory, 0.65);
        emBox.position.set(mx, y + slotH / 2, 0);
        modelGroup.add(emBox);
        addClickable(emBox, { type:'memory', layer:i });
        if (i === 0) {
          const l = makeLabel('EM', 'label3d dim');
          l.position.set(mx, y + slotH / 2 + 0.35, 0);
          modelGroup.add(l);
        }
      }

      // Cross-level drift
      if (tog.cross_level && i > 0) {
        const from = new THREE.Vector3(-BW / 2 - 0.3, y - LAYER_GAP * 0.5, BD / 2 + 0.2);
        const to = new THREE.Vector3(-BW / 2 - 0.3, y + 0.3, BD / 2 + 0.2);
        modelGroup.add(makeCurve(from, to, COLORS.drift, { x: -0.6, y: 0, z: 0.3 }));
        if (i === 1) {
          const dl = makeLabel('cross-level', 'label3d dim');
          dl.position.set(-BW / 2 - 1.0, y + 0.15, BD / 2 + 0.5);
          modelGroup.add(dl);
        }
      }

      // Aux head
      if (tog.multi_scale_loss) {
        const stride = cfg.aux_layer_stride || 3;
        if (i % stride === 0 && i < nLayers - 1) {
          const aux = makeBox(0.4, 0.2, 0.4, COLORS.aux, 0.75);
          aux.position.set(BW / 2 + 2.4, y + slotH / 2, 0);
          modelGroup.add(aux);
          if (i === 0) {
            const l = makeLabel('Aux', 'label3d dim');
            l.position.set(BW / 2 + 2.4, y + slotH / 2 + 0.2, 0);
            modelGroup.add(l);
          }
        }
      }

      y += slotH + 0.3 + LAYER_GAP;
    }
  }

  // Output head
  const outH = 0.5;
  const head = makeBox(BW, outH, BD, COLORS.output);
  head.position.set(0, y + outH / 2, 0);
  modelGroup.add(head);
  const oLbl = makeLabel('Output Head');
  oLbl.position.set(0, y + outH + 0.15, 0);
  modelGroup.add(oLbl);
  y += outH + GAP;

  // Center camera
  const totalH = y;
  camera.position.set(BW * 2 + 6, totalH * 0.55, BW * 2 + 6);
  controls.target.set(0, totalH * 0.45, 0);
  controls.update();

  // Build data-flow waypoints for particle system
  const wps = [{ y: 0.2, color: 0xf0c674, x: 0, z: 0 }];
  const nL = nLayers;
  const stepH = (totalH - 0.9) / nL;
  for (let i = 0; i < nL; i++) {
    const baseY = 0.7 + i * stepH;
    wps.push({ y: baseY, color: 0x4a90d9, x: 0, z: 0 });
    wps.push({ y: baseY + stepH * 0.5, color: 0x2ecc71, x: 0, z: 0 });
  }
  wps.push({ y: totalH - 0.3, color: 0xe67e22, x: 0, z: 0 });
  setFlowWaypoints(wps);

  if (infoCallback) infoCallback(cfg, tog, totalH);
}
