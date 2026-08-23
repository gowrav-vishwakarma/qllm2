#!/usr/bin/env python3
"""Diff the checkpoint frame's saved-stream vs recomputed-stream, with the
PYTHON CODE LOCATION of every save.

Patches _default_metadata_fn (called exactly once per saved tensor, 1:1 with
the frame's weak_holders, in both passes) to also capture the Python stack,
tagged fwd/bwd. On CheckpointError, prints the mismatched positions with the
forward-save and recompute-save stacks — naming the exact function.

Also: per-block parameter grad-norm report (multi-layer run) to verify the
"only the last two layers received gradient" observation.

Usage: .venv/bin/python -m v13.tmp.dbg_ckpt_probe2 [n_layers]
"""
from __future__ import annotations
import sys
import traceback

import torch
import torch.utils.checkpoint as ckpt

sys.path.insert(0, '/home/gowrav/Development/qllm2')
from v13.model import V13Config, V13LM

N_LAYERS = int(sys.argv[1]) if len(sys.argv) > 1 else 1

_orig_meta = ckpt._allowed_determinism_checks_to_fns['default']
_orig_check = ckpt._CheckpointFrame.check_recomputed_tensors_match
FWD = []
BWD = []
REGION = {'cur': None}


def _meta_with_stack(x):
    meta = _orig_meta(x)
    rec = {
        'shape': tuple(x.shape),
        'dtype': str(x.dtype),
        'req_grad': bool(x.requires_grad),
        'grad_en': torch.is_grad_enabled(),
        'inf_en': torch.is_inference_mode_enabled(),
        'stack': traceback.format_stack()[-8:-1],
    }
    if REGION['cur'] == 'fwd':
        FWD.append((meta, rec))
    elif REGION['cur'] == 'bwd':
        BWD.append((meta, rec))
    return meta


def _check_with_report(self, gid):
    # snapshot recomputed stacks BEFORE the original check raises
    rstacks = list(BWD)
    try:
        return _orig_check(self, gid)
    except ckpt.CheckpointError:
        n_holders = len(self.weak_holders)
        print(f"\n{'='*90}\nCHECKPOINT MISMATCH: holders(fwd saves)={n_holders}, "
              f"recomp_counter={self.recomp_counter[gid]}, fwd_records={len(FWD)}, bwd_records={len(BWD)}\n{'='*90}")
        for idx, wh in enumerate(self.weak_holders):
            holder = wh()
            if holder is None or gid not in holder.handles or holder.handles[gid] is None:
                continue
            rx = self.recomputed[gid].get(holder.handles[gid])
            if rx is None:
                continue
            sm = self.x_metadatas[idx]
            rm = _orig_meta(rx)
            if sm != rm:
                print(f"\n--- position {idx}: saved={sm}  recomputed={rm}")
                if idx < len(FWD):
                    r = FWD[idx][1]
                    print(f"  FORWARD save: shape={r['shape']} dtype={r['dtype']} req_grad={r['req_grad']} "
                          f"grad_en={r['grad_en']} inf_en={r['inf_en']}")
                    for ln in r['stack'][-5:]:
                        print("   |", ln.strip().replace('\n', ''))
                if idx < len(rstacks):
                    r = rstacks[idx][1]
                    print(f"  RECOMPUTE save: shape={r['shape']} dtype={r['dtype']} req_grad={r['req_grad']} "
                          f"grad_en={r['grad_en']} inf_en={r['inf_en']}")
                    for ln in r['stack'][-5:]:
                        print("   |", ln.strip().replace('\n', ''))
        raise


ckpt._allowed_determinism_checks_to_fns['default'] = _meta_with_stack
ckpt._CheckpointFrame.check_recomputed_tensors_match = _check_with_report

# ── model ────────────────────────────────────────────────────────────────────
cfg = V13Config(vocab_size=50257, dim=64, n_heads=2, head_dim=32, n_layers=N_LAYERS,
    expand=2, dropout=0.0, max_seq_len=256, chunk_size=64,
    gradient_checkpointing=True, n_states=3, state_dt_spread=2.0,
    write_mode='delta', delta_chunk=32, delta_erase_gate=True,
    gate_content_aware=True, vault_state=True, vault_state_idx=0,
    write_phase_address=True, fused_e3=True, gate_surprisal_lambda=0.1,
    delta_key_norm=True, delta_erase_beta_cap=0.95)
torch.manual_seed(0)
m = V13LM(cfg).cuda()
m.train()
B, T = 2, 128
ids = torch.randint(0, 50257, (B, T), device='cuda')
lab = torch.randint(0, 50257, (B, T), device='cuda')

CRASH = None
try:
    REGION['cur'] = 'fwd'
    lm, _, _ = m._hidden_to_lm(ids)
    REGION['cur'] = None
    loss = m.ce_from_lm(lm, lab, chunk=4096)
    REGION['cur'] = 'bwd'
    loss.backward()
    torch.cuda.synchronize()
except Exception as e:
    CRASH = type(e).__name__
    print(f"backward raised {CRASH}")
finally:
    REGION['cur'] = None

# ── grad-flow probe ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("GRAD-FLOW PROBE: per-block parameter grad L2 norm after backward")
if CRASH is None:
    for i, block in enumerate(m.blocks):
        norms = [float(p.grad.norm()) if p.grad is not None else float('nan')
                 for p in block.parameters(recurse=True)]
        nz = sum(1 for x in norms if x == x and x > 0)
        s = sum(x for x in norms if x == x)
        print(f"  block {i}: params={len(norms)} nonzero_grad={nz} grad_norm_sum={s:.6f}")
    for name, p in m.named_parameters():
        if name.startswith('blocks'):
            continue
        g = float(p.grad.norm()) if p.grad is not None else float('nan')
        flag = "  <== DEAD (no grad)" if (g != g or g == 0.0) else ""
        print(f"  {name:28s} grad_norm={g if g == g else float('nan'):.6f}{flag}")
else:
    print("  (skipped — backward crashed)")
