"""GATE-FREE told read, offline (PRE-REGISTERED 2026-11-17 05:38 IST, EXPERIMENTS_V14 "GATE-FREE told read"). No GPU.
From told_margin jsonl at 8k: twin(alpha) gate = h_g + a(h_g - n_g) (exact); identity ~ h_i + a(h_i - n_i) (approx, checked at a 4
against the stored twin_m_id); contrast gate(tau): emit iff C_doc = h_g - n_g > tau, unit = twin(4) identity.
Predicted told 8k = frac(identity > 0 and gate > 0). Selection on saves <= 41B only; the others are held out.
    python v14/gate_free_offline.py [--extra name=path.jsonl ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as S
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
SEL = ['m11', 'm26', 'm27', 'm22best', 'm30', 'm39', 'm40', 'm405', 'm41']          # pre-registered selection saves
MEASURED = dict(m11=.10, m26=.03, m27=.45, m22best=.53, m30=.31, m39=.06, m40=.37, m405=.12, m41=.35, m42=.09)  # FAR-TOLD told 8k
ALPHAS = (4, 8, 16)
TAUS = [round(0.1 * k, 1) for k in range(11)]


def rows(path):
    return [r for r in map(json.loads, open(path)) if r['L'] == 8192]


def reads(R):
    out = {}
    for a in ALPHAS:
        out[f'twin{a}'] = sum((r['held_m_id'] + a * (r['held_m_id'] - r['noprot_m_id']) > 0) and
                              (r['held_m_gate'] + a * (r['held_m_gate'] - r['noprot_m_gate']) > 0) for r in R) / len(R)
    for t in TAUS:
        out[f'tau{t}'] = sum((r['twin_m_id'] > 0) and (r['held_m_gate'] - r['noprot_m_gate'] > t) for r in R) / len(R)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--extra', nargs='*', default=[]); a = ap.parse_args()
    saves = {}
    for f in sorted(glob.glob(str(ROOT / 'logs/v14_told_margin_m*.jsonl'))):
        saves[Path(f).name.split('_')[3]] = f
    for e in a.extra:
        k, v = e.split('='); saves[k] = v
    D = {k: rows(v) for k, v in saves.items()}
    err = [abs(r['held_m_id'] + 4 * (r['held_m_id'] - r['noprot_m_id']) - r['twin_m_id']) for R in D.values() for r in R]
    print(f'identity approx at alpha 4 vs stored twin_m_id: mean |err| {S.mean(err):.4f}, max {max(err):.4f} (n {len(err)})')
    P = {k: reads(R) for k, R in D.items()}
    B = {k: S.mean(r['noprot_m_gate'] for r in R) for k, R in D.items()}
    cal = [k for k in P if k in MEASURED]
    print(f"calibration twin4 predicted vs measured told 8k: Spearman {spearmanr([P[k]['twin4'] for k in cal], [MEASURED[k] for k in cal])[0]:.2f}, "
          f"mean gap {S.mean(P[k]['twin4'] - MEASURED[k] for k in cal):+.2f} (n {len(cal)})")
    names = list(P[SEL[0]])
    score = {n: (S.mean(P[k][n] for k in SEL), S.pstdev(P[k][n] for k in SEL)) for n in names}
    best = max(names, key=lambda n: (score[n][0], -score[n][1]))
    print('selection (saves <= 41B): ' + ' '.join(f'{n} {m:.2f}/{sd:.2f}' for n, (m, sd) in score.items()))
    print(f'CHOSEN {best}')
    order = SEL + [k for k in P if k not in SEL]
    print('save     B      twin4  ' + f'{best:>7s}  measured  held_out')
    for k in order:
        print(f"{k:8s} {B[k]:+.2f}  {P[k]['twin4']:.2f}   {P[k][best]:.2f}     {MEASURED.get(k, float('nan')):.2f}      {k not in SEL}")
    allk = list(P)
    sd4, sdb = S.pstdev(P[k]['twin4'] for k in allk), S.pstdev(P[k][best] for k in allk)
    rho = spearmanr([P[k][best] for k in allk], [B[k] for k in allk])[0]
    rho4 = spearmanr([P[k]['twin4'] for k in allk], [B[k] for k in allk])[0]
    ho = [k for k in allk if k not in SEL]
    c1 = all(P[k][best] >= 0.45 for k in ho if k == 'm42')
    print(f'held-out: ' + ' '.join(f'{k} {P[k][best]:.2f}' for k in ho))
    print(f'(1) 42B >= 0.45: {c1} | (2) SD all saves {sdb:.3f} vs twin4 {sd4:.3f} (<= half: {sdb <= sd4 / 2}) | '
          f'(3) Spearman with B {rho:.2f} (twin4 {rho4:.2f}; <= 0.5: {rho <= 0.5})')
    print('VERDICT offline: ' + ('MET' if c1 and sdb <= sd4 / 2 and rho <= 0.5 else 'NOT met'))


if __name__ == '__main__':
    main()
