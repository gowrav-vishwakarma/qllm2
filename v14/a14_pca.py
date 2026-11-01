"""A14: the shipped PCA-64 basis for the surprise-keyed address (fit ONCE on held-out FineWeb, not the eval prose).

b2p2's block-8 states over 65,536 tokens of the v14 train pool (8 chunks of 8,192, each centred by its own mean),
top-64 principal directions -> /media/gowrav/d_drive/qllm_ckpt/v14/a14_pca64_b2p2.pt (588 x 64 fp16 + variance).

    python v14/a14_pca.py --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from v16.bench import Reader  # noqa: E402
from v14.a6_reread import CKPT, Mind  # noqa: E402
from v14.a13_manykey import capture  # noqa: E402

POOL = '/home/gowrav/Development/qllm2/.cache/v14_train_pool_fineweb014.pt'
OUT = '/media/gowrav/d_drive/qllm_ckpt/v14/a14_pca64_b2p2.pt'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda'); ap.add_argument('--tokens', type=int, default=65536)
    ap.add_argument('--chunk', type=int, default=8192); ap.add_argument('--k', type=int, default=64)
    a = ap.parse_args()
    mind = Mind(CKPT, torch.device(a.device)); R = Reader(mind.m, 'pam')
    toks = torch.load(POOL, map_location='cpu', weights_only=False)['tokens']
    toks = [int(t) for t in toks[:a.tokens]]
    X = []
    for s in range(0, len(toks), a.chunk):
        got, _ = capture(R, toks[s:s + a.chunk], (8,))
        v = got[8]
        X.append((v - v.mean(0)).cpu())
    X = torch.cat(X).double()
    U, S, Vh = torch.linalg.svd(X - X.mean(0), full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    basis = Vh[:a.k].T.contiguous()                                   # [588, k]
    torch.save({'basis': basis.half(), 'explained': float(var[:a.k].sum()), 'tokens': len(toks),
                'source': POOL, 'layer': 8}, OUT)
    print(f'basis {tuple(basis.shape)} explains {float(var[:a.k].sum()):.3f} of the centred variance; saved {OUT}')


if __name__ == '__main__':
    main()
