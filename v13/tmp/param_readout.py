#!/usr/bin/env python3
"""CPU-only readout of where V13's LEARNED parameters went on a trained ckpt.

Answers "what maths needs to improve" by reporting, per layer, the values the
training actually pushed (not the init): GSP protect-gate bias (over-protect?),
delta write/erase gains (beta_w/beta_e — is the cap binding?), phase router
(K=3 specialization / collapse?), decay horizon (dt_bias + state_dt_offset),
and a few weight-drift magnitudes. All on CPU; does NOT touch the GPU run.

Usage: .venv/bin/python -m v13.tmp.param_readout [checkpoint.pt]
"""
from __future__ import annotations
import sys
import torch
import torch.nn.functional as F

REPO = '/home/gowrav/Development/qllm2'
sys.path.insert(0, REPO)

CKPT = sys.argv[1] if len(sys.argv) > 1 else f'{REPO}/checkpoints_v13/500m_v13_r1recipe/latest.pt'


def cfg_get(cfg, name, default=None):
    """config may be a V13Config dataclass or a plain dict."""
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def summarize(sd: dict, cfg) -> None:
    H = cfg_get(cfg, 'n_heads', 16)
    K = cfg_get(cfg, 'n_states', 3)
    pg_keys = [k for k in sd if k.startswith('blocks.') and k.endswith('pam.protect_gate.bias')]
    L = len(set(k.split('.')[1] for k in pg_keys))
    print(f"checkpoint: {CKPT}")
    print(f"layers={L} heads={H} K={K} vault={cfg_get(cfg,'vault_state')}@idx{cfg_get(cfg,'vault_state_idx')} "
          f"erase_gate={cfg_get(cfg,'delta_erase_gate')} key_norm={cfg_get(cfg,'delta_key_norm')} "
          f"erase_cap={cfg_get(cfg,'delta_erase_beta_cap')}")

    def per_layer(suffix):
        return [sd[f'blocks.{i}.pam.{suffix}'] for i in range(L)]

    def row(name, suffix, init):
        bs = per_layer(suffix)
        d = [float((b - init).mean()) for b in bs]
        print(f"  {name:26s} init={init:+.3f}  meanΔ={sum(d)/len(d):+.4f}  "
              f"L0={float((bs[0]-init).mean()):+.4f}  "
              f"mid={float((bs[L//2]-init).mean()):+.4f}  "
              f"L{L-1}={float((bs[-1]-init).mean()):+.4f}")

    print("\n[1] GSP protect-gate bias (init -3.0; +Δ = gate opens = LESS protection / more overwrite)")
    row('protect_gate.bias', 'protect_gate.bias', -3.0)
    pgw = per_layer('protect_gate.weight')
    print(f"    protect_gate.weight L2: L0={float(pgw[0].norm()):.4f} L{L-1}={float(pgw[-1].norm()):.4f} "
          f"(0 = gate ignores content)")

    print("\n[2] delta write/erase gains (sigmoid -> beta_w/beta_e in (0,1); cap binds at 0.95)")
    if cfg_get(cfg, 'delta_erase_gate'):
        row('erase_beta_proj.bias', 'erase_beta_proj.bias', -3.0)
        ebw = per_layer('erase_beta_proj.weight')
        print(f"    erase_beta_proj.weight L2: L0={float(ebw[0].norm()):.4f} L{L-1}={float(ebw[-1].norm()):.4f}")
        eff = [float(F.sigmoid(b).mean()) for b in per_layer('erase_beta_proj.bias')]
        print(f"    mean sigmoid(erase) per layer: L0={eff[0]:.3f} mid={eff[L//2]:.3f} L{L-1}={eff[-1]:.3f}")
        print(f"    cap binding? any>0.95: {any(e>0.95 for e in eff)}  (post-cap max={max(min(e,0.95) for e in eff):.3f})")
    row('beta_proj.bias (write)', 'beta_proj.bias', 0.0)
    bw = per_layer('beta_proj.weight')
    print(f"    beta_proj.weight L2: L0={float(bw[0].norm()):.4f} L{L-1}={float(bw[-1].norm()):.4f}")
    weff = [float(F.sigmoid(b).mean()) for b in per_layer('beta_proj.bias')]
    print(f"    mean sigmoid(write): L0={weff[0]:.3f} mid={weff[L//2]:.3f} L{L-1}={weff[-1]:.3f}")

    print("\n[3] E3 phase router (K=3 specialization; 0 = states degenerate -> collapse)")
    phw = per_layer('phase_proj.weight')
    print(f"    phase_proj.weight L2: L0={float(phw[0].norm()):.4f} mid={float(phw[L//2].norm()):.4f} "
          f"L{L-1}={float(phw[-1].norm()):.4f}")
    phb = per_layer('phase_proj.bias')
    print(f"    phase_proj.bias L2:   L0={float(phb[0].norm()):.4f} mid={float(phb[L//2].norm()):.4f} "
          f"L{L-1}={float(phb[-1].norm()):.4f}")

    print("\n[4] state_dt_offset (learned [K] decay offsets; +offset = FASTER decay; vault idx pinned gamma=1)")
    sdo = per_layer('state_dt_offset')
    for i in range(L):
        v = "  ".join(f"s{j}={float(x):+.3f}" for j, x in enumerate(sdo[i]))
        print(f"    L{i}: {v}")

    print("\n[5] base decay horizon (dt_bias learned; + = slower decay = LONGER memory horizon)")
    dtb = per_layer('dt_bias')
    for i in range(0, L, max(1, L // 8)):
        print(f"    L{i}: " + "  ".join(f"s{j}={float(x):+.4f}" for j, x in enumerate(dtb[i])))
    dtp = per_layer('dt_proj.weight')
    print(f"    dt_proj.weight L2: L0={float(dtp[0].norm()):.4f} L{L-1}={float(dtp[-1].norm()):.4f}")

    print("\n[6] weight-drift magnitude (are layers differentiating?)")
    for prefix in ['pam.qkv_proj.weight_real', 'cgu.up_proj.weight_real', 'cgu.down_proj.weight_real']:
        vals = [float(sd[f'blocks.{i}.{prefix}'].norm()) for i in range(L)]
        print(f"    {prefix}: L0={vals[0]:.1f} mid={vals[L//2]:.1f} L{L-1}={vals[-1]:.1f}")


def main():
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    sd = ck['model_state_dict']
    cfg = ck['config']
    print(f"global_step={ck.get('global_step')} global_tokens={ck.get('global_tokens')} "
          f"best_val_ppl={ck.get('best_val_ppl')} best_wiki_val_ppl={ck.get('best_wiki_val_ppl')}")
    summarize(sd, cfg)


if __name__ == '__main__':
    main()
