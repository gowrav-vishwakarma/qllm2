#!/usr/bin/env python3
"""Decisive view of the checkpoint saved-vs-recomputed mismatch: dump an
aligned window (positions around the first mismatch) showing, for EACH
position, the SAVED tensor (shape + value checksum) and the RECOMPUTED tensor
(shape + value checksum). This answers: pure order-swap (same values reversed)
or genuine value difference?

Also logs, per saved position: requires_grad, is_grad_enabled, is_inference —
the "what did the step receive" context.

Usage: .venv/bin/python -m v13.tmp.dbg_ckpt_probe3 [n_layers] [window]
"""
from __future__ import annotations
import sys
import torch
import torch.utils.checkpoint as ckpt

sys.path.insert(0, '/home/gowrav/Development/qllm2')
from v13.model import V13Config, V13LM

N_LAYERS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
WIN = int(sys.argv[2]) if len(sys.argv) > 2 else 6

_orig_meta = ckpt._allowed_determinism_checks_to_fns['default']
_orig_check = ckpt._CheckpointFrame.check_recomputed_tensors_match

FWD = []   # list of (shape, cksum, req_grad, grad_en, inf_en)
REGION = {'cur': None}


def _cksum(x):
    try:
        if x.is_floating_point() and x.numel() <= (1 << 22):
            v = x.detach().float().to('cpu')
            return (round(float(v.mean()), 6), round(float(v.abs().mean()), 6))
    except Exception:
        pass
    return None


def _meta_with_rec(x):
    meta = _orig_meta(x)
    if REGION['cur'] == 'fwd':
        FWD.append((tuple(x.shape), _cksum(x), bool(x.requires_grad),
                    torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
    return meta


def _check_with_dump(self, gid):
    try:
        return _orig_check(self, gid)
    except ckpt.CheckpointError:
        # find first mismatch position
        first = None
        rows = []
        for idx, wh in enumerate(self.weak_holders):
            holder = wh()
            if holder is None or gid not in holder.handles or holder.handles[gid] is None:
                continue
            rx = self.recomputed[gid].get(holder.handles[gid])
            if rx is None:
                continue
            sm = self.x_metadatas[idx]
            rm = _orig_meta(rx)
            mismatch = (sm != rm)
            if mismatch and first is None:
                first = idx
            s = FWD[idx] if idx < len(FWD) else None
            rows.append((idx, sm, s, rx, mismatch))
        lo, hi = max(0, first - WIN), min(len(rows), first + WIN + 1)
        print(f"\n{'='*100}\nMISMATCH WINDOW (first mismatch at {first}); "
              f"fwd_saves={len(FWD)}, holders={len(self.weak_holders)}, recomp={self.recomp_counter[gid]}")
        print(f"{'idx':>4} {'MATCH':6} | {'SAVED shape':22} {'SAVED cksum':22} {'rg/ge/ie':10} | "
              f"{'RECOMP shape':22} {'RECOMP cksum'}")
        for (idx, sm, s, rx, mismatch) in rows[lo:hi]:
            if first is not None and lo <= idx < hi:
                sshape = s[0] if s else '?'
                sck = s[1] if s else None
                flag = f"rg={s[2]}/ge={s[3]}/ie={s[4]}" if s else ''
                rshape = tuple(rx.shape)
                rck = _cksum(rx)
                tag = 'DIFF' if mismatch else '  ok'
                print(f"{idx:>4} {tag:6} | {str(sshape):22} {str(sck):22} {flag:10} | "
                      f"{str(rshape):22} {rck}")
        raise


ckpt._allowed_determinism_checks_to_fns['default'] = _meta_with_rec
ckpt._CheckpointFrame.check_recomputed_tensors_match = _check_with_dump

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

REGION['cur'] = 'fwd'
lm, _, _ = m._hidden_to_lm(ids)
REGION['cur'] = None
loss = m.ce_from_lm(lm, lab, chunk=4096)
try:
    REGION['cur'] = 'bwd'
    loss.backward()
    torch.cuda.synchronize()
    print("NO CRASH")
except Exception as e:
    print(f"backward raised {type(e).__name__}")
finally:
    REGION['cur'] = None
