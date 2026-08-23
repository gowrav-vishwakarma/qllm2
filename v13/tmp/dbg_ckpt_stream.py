#!/usr/bin/env python3
"""Capture the FULL save-order stream (all tensors, shapes) in the checkpoint
forward AND the recompute, then diff them to find the exact extent of any
reordering. Loops fresh runs until a crash occurs (the mismatch is flaky),
then dumps the aligned divergence window.

This tells us: is it a local 2-swap (div operands) or a larger reordering?
What ops sit around the divergence?

Usage: .venv/bin/python -m v13.tmp.dbg_ckpt_stream [max_runs]
"""
from __future__ import annotations
import sys
import torch
import torch.utils.checkpoint as ckpt

sys.path.insert(0, '/home/gowrav/Development/qllm2')
from v13.model import V13Config, V13LM

MAX_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 30

_orig_meta = ckpt._allowed_determinism_checks_to_fns['default']
FWD = []       # list of shape (forward save order)
REC = []       # list of shape (recompute save order)
REGION = {'cur': None}
CRASH = {'hit': False, 'idx': None}


def _meta_rec(x):
    meta = _orig_meta(x)
    if REGION['cur'] == 'fwd':
        FWD.append(tuple(x.shape))
    return meta


def _check_with_dump(self, gid):
    # build recompute stream from self.recomputed in weak_holder order
    CRASH['hit'] = True
    CRASH['idx'] = len(FWD)
    return True  # swallow the crash so we can dump; caller re-raises if needed


ckpt._allowed_determinism_checks_to_fns['default'] = _meta_rec
# Patch check to record + swallow so we can inspect
_orig_check = ckpt._CheckpointFrame.check_recomputed_tensors_match
def _check_record(self, gid):
    # fill REC in the order the recompute saved (weak_holder order)
    REC.clear()
    for wh in self.weak_holders:
        holder = wh()
        if holder is None or holder.handles.get(gid) is None:
            continue
        rx = self.recomputed[gid].get(holder.handles[gid])
        if rx is not None:
            REC.append(tuple(rx.shape))
    try:
        return _orig_check(self, gid)
    except ckpt.CheckpointError:
        CRASH['hit'] = True
        return True


def build():
    torch.manual_seed(0)
    cfg = V13Config(vocab_size=50257, dim=64, n_heads=2, head_dim=32, n_layers=1,
        expand=2, dropout=0.0, max_seq_len=256, chunk_size=64,
        gradient_checkpointing=True, n_states=3, state_dt_spread=2.0,
        write_mode='delta', delta_chunk=32, delta_erase_gate=True,
        gate_content_aware=True, vault_state=True, vault_state_idx=0,
        write_phase_address=True, fused_e3=True, gate_surprisal_lambda=0.1,
        delta_key_norm=True, delta_erase_beta_cap=0.95)
    m = V13LM(cfg).cuda()
    m.train()
    B, T = 2, 128
    ids = torch.randint(0, 50257, (B, T), device='cuda')
    lab = torch.randint(0, 50257, (B, T), device='cuda')
    return m, ids, lab


ckpt._CheckpointFrame.check_recomputed_tensors_match = _check_record

for run in range(MAX_RUNS):
    FWD.clear()
    REC.clear()
    m, ids, lab = build()
    try:
        REGION['cur'] = 'fwd'
        lm, _, _ = m._hidden_to_lm(ids)
        REGION['cur'] = None
        loss = m.ce_from_lm(lm, lab, chunk=4096)
        loss.backward()
        torch.cuda.synchronize()
        crashed = False
    except ckpt.CheckpointError:
        crashed = True
    except Exception as e:
        print(f"run {run}: unexpected {type(e).__name__}: {str(e)[:60]}")
        crashed = False
    finally:
        del m
        torch.cuda.empty_cache()

    if not crashed:
        # still compare streams even without crash (should be identical)
        diff = [i for i in range(min(len(FWD), len(REC))) if FWD[i] != REC[i]]
        print(f"run {run:2d}: PASS   fwd={len(FWD)} rec={len(REC)} first_diff_at={diff[0] if diff else '-'}")
        continue

    # crashed: find first divergence
    diff = [i for i in range(min(len(FWD), len(REC))) if FWD[i] != REC[i]]
    first = diff[0] if diff else -1
    print(f"\nrun {run}: CRASH  fwd={len(FWD)} rec={len(REC)} first_diff_at={first}")
    if first >= 0:
        lo, hi = max(0, first - 5), min(min(len(FWD), len(REC)), first + 6)
        print(f"{'idx':>4} {'FWD':>16} {'REC':>16}  match")
        for i in range(lo, hi):
            mark = 'OK' if i not in diff else 'DIFF'
            print(f"{i:>4} {str(FWD[i]):>16} {str(REC[i]):>16}  {mark}")
    break
