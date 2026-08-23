#!/usr/bin/env python3
"""Observe (not guess) where the checkpoint original-forward vs recompute
diverge: forward hooks on every submodule log (module, pass, in/out shapes),
pass-tagged by grad mode (original fwd runs under no_grad, recompute under grad).

Prints, per module, the orig trace and the recomp trace side by side; the first
module whose in/out shapes differ between passes is the divergence.

Usage: .venv/bin/python -m v13.tmp.dbg_ckpt_trace
"""
from __future__ import annotations
import sys
import torch

sys.path.insert(0, '/home/gowrav/Development/qllm2')
from v13.model import V13Config, V13LM

cfg = V13Config(vocab_size=50257, dim=64, n_heads=2, head_dim=32, n_layers=1,
    expand=2, dropout=0.0, max_seq_len=256, chunk_size=64,
    gradient_checkpointing=True, n_states=3, state_dt_spread=2.0,
    write_mode='delta', delta_chunk=32, delta_erase_gate=True,
    gate_content_aware=True, vault_state=True, vault_state_idx=0,
    write_phase_address=True, fused_e3=True, gate_surprisal_lambda=0.1,
    delta_key_norm=True, delta_erase_beta_cap=0.95)
torch.manual_seed(0)
m = V13LM(cfg).cuda()
m.train()

LOG = []   # (pass_tag, module_name, in_shapes, out_shapes)

def shapes(t):
    if isinstance(t, torch.Tensor):
        return [tuple(t.shape)]
    if isinstance(t, (tuple, list)):
        return [tuple(x.shape) for x in t if isinstance(x, torch.Tensor)]
    return []

def make_hook(name):
    def hook(mod, args, kwargs, output):
        tag = 'orig' if not torch.is_grad_enabled() else 'recomp'
        ins = shapes(args) if args else shapes(kwargs.get('x'))
        LOG.append((tag, name, ins, shapes(output)))
    return hook

for name, mod in m.named_modules():
    if name:
        mod.register_forward_hook(make_hook(name), with_kwargs=True)

B, T = 2, 128
ids = torch.randint(0, 50257, (B, T), device='cuda')
lab = torch.randint(0, 50257, (B, T), device='cuda')
try:
    lm, _, _ = m._hidden_to_lm(ids)
    loss = m.ce_from_lm(lm, lab, chunk=4096)
    loss.backward()
    torch.cuda.synchronize()
    print("NO CRASH")
except Exception as e:
    print("CRASH:", str(e).splitlines()[0])

# Diff per module: orig vs recomp, first divergence wins.
from collections import defaultdict
by_mod = defaultdict(dict)
for tag, name, ins, outs in LOG:
    # a module may run multiple times (e.g. twice: orig + recomp). keep list.
    by_mod[name].setdefault(tag, []).append((ins, outs))

print(f"\ntotal hook calls: {len(LOG)}  (orig={sum(1 for l in LOG if l[0]=='orig')}, recomp={sum(1 for l in LOG if l[0]=='recomp')})")
print("="*100)
for name in by_mod:
    o = by_mod[name].get('orig', [])
    r = by_mod[name].get('recomp', [])
    # compare element-wise up to min length
    diff = ''
    for i in range(max(len(o), len(r))):
        oo = o[i] if i < len(o) else None
        rr = r[i] if i < len(r) else None
        if oo != rr:
            diff = f'  <<< DIFF call#{i}: orig={oo} recomp={rr}'
            break
    if diff:
        print(f"{name}\n{diff}")
    else:
        # print a compact match line only for the block/pam internals of interest
        if any(k in name for k in ('pam', 'norm', 'cgu', 'gate', 'proj')):
            print(f"{name:40s} match  orig={len(o)} recomp={len(r)}  in={o[0][0] if o else '-'} out={o[0][1] if o else '-'}")
