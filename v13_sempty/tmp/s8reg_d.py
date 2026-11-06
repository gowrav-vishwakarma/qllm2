"""s8reg arm D (EXPERIMENTS_SEMPY "s8reg", Lead + outside reviewer 2026-11-06): weight-space points between M9 (stable, s8 0.770)
and its 1B decay (s8 0.231, benchmarks up). Closest prior: WiSE-FT / model soups / checkpoint averaging; release engineering, no claim.
  build: theta = (1 - l) M9 + l decay for l in --lams, and a graft = M9's notebook / reader / echo params (blocks.*.pam*, blocks.*.reader,
         echo_*) + the decay's everything else (MLP / embeddings / norms / head). Weights-only files (the c3b_*_weights.pt format).
  nll:   held-out NLL (pretrain_holdout_v1 cache, 24 x 2048 tokens, rng 0: the clk3 prose guard) for a list of checkpoints.
    python v13_sempty/tmp/s8reg_d.py build --m9 <full> --decay <full> --out_dir <dir> [--lams 0.3,0.5,0.7]
    python v13_sempty/tmp/s8reg_d.py nll --ckpts a.pt,b.pt --out <json>
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

M9_PARTS = ('.pam.', '.pam_scale', '.reader.')        # blocks.<i>.<part>; echo_* is top-level


def is_mind(k: str) -> bool:
    return k.startswith('echo_') or (k.startswith('blocks.') and any(p in k for p in M9_PARTS))


def build(a):
    m9 = torch.load(a.m9, map_location='cpu', weights_only=False, mmap=True)
    dec = torch.load(a.decay, map_location='cpu', weights_only=False, mmap=True)
    A, B = m9['model_state_dict'], dec['model_state_dict']
    assert A.keys() == B.keys(), 'param names differ'
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    meta = {k: dec[k] for k in ('global_step', 'global_tokens', 'best_val_loss', 'config', 'args') if k in dec}
    pts = {f'lerp{int(round(float(l) * 100)):02d}': float(l) for l in a.lams.split(',')}
    for name, l in pts.items():
        sd = {k: ((1 - l) * A[k].float() + l * B[k].float()).to(B[k].dtype) if B[k].is_floating_point() else B[k] for k in B}
        torch.save({'model_state_dict': sd, **meta, 's8reg': f'(1-{l}) M9 + {l} decay'}, out / f'{name}_weights.pt')
        print(f'[build] {name} l={l}', flush=True)
    n_m = sum(A[k].numel() for k in A if is_mind(k)); n = sum(v.numel() for v in A.values())
    sd = {k: (A[k] if is_mind(k) else B[k]) for k in B}
    torch.save({'model_state_dict': sd, **meta, 's8reg': 'graft: M9 pam/reader/echo + decay rest'}, out / 'graft_weights.pt')
    print(f'[build] graft: {n_m:,} of {n:,} params from M9 ({100 * n_m / n:.1f} %)', flush=True)


def nll(a):
    from run_memory_behavioral import _load_v13_sempty
    from memory_probes.spread import filler_tokens
    dev = torch.device('cuda'); filler = filler_tokens(a.filler)
    rng = np.random.default_rng(0)
    seqs = [filler[s:s + 2048] for s in rng.integers(0, len(filler) - 2048, size=24)]
    res = {}
    for c in a.ckpts.split(','):
        t0 = time.time(); model, _, _ = _load_v13_sempty(Path(c), 'baseline_real_pm', dev); model.eval(); v = []
        with torch.no_grad():
            for sq in seqs:
                ids = torch.tensor([sq.tolist()], device=dev); lg, _, _ = model(ids)
                v.append(float(torch.nn.functional.cross_entropy(lg[0, :-1].float(), ids[0, 1:])))
        res[c] = float(np.mean(v)); print(f'[nll] {c}: {res[c]:.4f} ({time.time() - t0:.0f} s)', flush=True)
        del model; torch.cuda.empty_cache()
    json.dump(res, open(a.out, 'w'), indent=1)


GROUPS = {'embed_head': lambda k: k.startswith(('embed', 'lm_head')), 'cgu': lambda k: '.cgu.' in k,
          'pam': lambda k: '.pam.' in k, 'reader_echo': lambda k: '.reader.' in k or k.startswith('echo_')}


def wdelta(a):
    """||theta_t - theta_ref|| / ||theta_ref|| per group (embed + lm_head, MLP .cgu., notebook .pam., reader + echo), per checkpoint."""
    ld = lambda f: (lambda c: c.get('model_state_dict', c))(torch.load(f, map_location='cpu', weights_only=False, mmap=True))
    R = ld(a.ref); res = {}
    for c in a.ckpts.split(','):
        T = ld(c); row = {}
        for g, f in GROUPS.items():
            ks = [k for k in R if f(k) and R[k].is_floating_point()]
            num = sum(float((T[k].float() - R[k].float()).pow(2).sum()) for k in ks); den = sum(float(R[k].float().pow(2).sum()) for k in ks)
            row[g] = round((num / den) ** 0.5, 6)
        res[c] = row; print(f'[wdelta] {Path(c).name}: {row}', flush=True)
    json.dump({'ref': a.ref, 'rows': res}, open(a.out, 'w'), indent=1)


def main():
    p = argparse.ArgumentParser(); p.add_argument('step', choices=['build', 'nll', 'wdelta'])
    p.add_argument('--m9'); p.add_argument('--decay'); p.add_argument('--out_dir'); p.add_argument('--lams', default='0.3,0.5,0.7')
    p.add_argument('--ckpts'); p.add_argument('--out'); p.add_argument('--ref')
    p.add_argument('--filler', default='.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt')
    a = p.parse_args(); {'build': build, 'nll': nll, 'wdelta': wdelta}[a.step](a)


if __name__ == '__main__':
    main()
