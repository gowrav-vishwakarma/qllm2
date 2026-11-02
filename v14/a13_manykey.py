"""A13 / D4 (pre-registered 2026-11-01 05:12 IST): a question is many keys, chosen by the mind's own surprise.

Per doc (bench's syn / synneg / natq / natqneg): one b2p2 pass captures the frozen per-token states at blocks 2 and 8
and the mind's surprisal; rows = the doc's sentences (split at '.', 4-64 tokens); fetch arms score rows by max cosine
over their tokens (centred by the doc mean): F2 = the 2 most surprising question tokens (after the first 2) at block
8, F3 = the same at block 2, F1 = the key's last token at block 8 (reference), MiniLM-L6 over sentences. Reads
(warm + no-row contrast a = 4, answer tokens only): F2's top-1 row, F2's top-5 rows together (line-up, doc order),
the oracle fact row. Key-swap rel (A11) on F2's top-1.

    python v14/a13_manykey.py --seed 33000 --docs 40 --out logs/v14_a13_manykey_<hash>_<stamp>.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from memory_probes.spread import filler_tokens  # noqa: E402
from v16.bench import Reader, build, natural_key  # noqa: E402
from v14.a6_reread import CKPT, FILL, Mind, sent_ok  # noqa: E402
from v14.a11_keyswap import K_SWAP, read as swap_read, swap_keys  # noqa: E402
from v14.shared_fetch import minilm_embed  # noqa: E402

DOT, LAYERS, ALPHA = 13, (2, 8), 4.0


@torch.no_grad()
def capture(R, ids, layers):
    """frozen per-token states after the given blocks [T, D] each, and the pre-logit vectors (for surprisal)."""
    m = R.m
    x = torch.tensor([ids], device=next(m.parameters()).device)
    z, b, t = m._stem(x, 0)
    share = getattr(m.config, 'share_reads', False); shared = None; got = {}
    for i, block in enumerate(m.blocks):                       # = bench Reader.vectors (incl. M2 shared reads)
        if share:
            z, _ = block(z, pam_state=None, step_offset=0, shared=shared)
            if block.pam is not None:
                shared = block.pam._shared
        else:
            z, _ = block(z, pam_state=None, step_offset=0)
        if m.cond_mem_layers and i in m.cond_mem_layers:
            z = m.cond_mem[str(i)](z, x)
        if i in layers:
            got[i] = z.raw(b, t, m.model_dim)[0].float().clone()
    lm, _ = m._lm_vector(z, x, b, t)
    return got, lm.raw(b, t, m.model_dim)[0].float()


def sentence_rows(ids, end):
    rows, s0 = [], 0
    from v14.package_fetch import is_end
    for j in range(end):
        if is_end(ids, j, end) or j == end - 1:
            a, e = s0, j + 1
            while e - a > 64:                                   # long sentences: 64-token pieces from the END
                rows.append((e - 64, e)); e -= 64              # (an inserted fact ends its sentence: keep it whole)
            if e - a >= 4:
                rows.append((a, e))
            s0 = j + 1
    return rows


def maxsim_scores(V, keys, rows):
    """V [T, D] normalised states, keys [k, D] normalised -> per-row sum over keys of the max cosine."""
    sims = V @ keys.T                                           # [T, k]
    return torch.stack([sims[a:b].max(0).values.sum() for a, b in rows]).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=33000); ap.add_argument('--L', type=int, default=32768)
    ap.add_argument('--docs', type=int, default=40); ap.add_argument('--needles', type=int, default=8)
    ap.add_argument('--tasks', default='syn,synneg,natq,natqneg'); ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = torch.device(a.device)
    mind = Mind(CKPT, dev); tok = mind.tok
    R = Reader(mind.m, 'pam')
    fill = filler_tokens(FILL)
    filler = fill if a.L + 2 < len(fill) else np.concatenate([fill] * (a.L // len(fill) + 2))
    outf = open(a.out, 'a')
    for task in a.tasks.split(','):
        keys_kind = 'natural'
        rng = random.Random(a.seed * 1000 + a.L + zlib.crc32(task.encode()) % 997)   # = bench.py main()
        for d in range(a.docs):
            t0 = time.time()
            ids, ans, qlen, meta = build(rng, tok, filler, a.L, task, a.needles, keys=keys_kind)
            end = len(ids) - qlen; q = ids[end:]
            neg = bool(meta.get('negative'))
            rows = sentence_rows(ids, end)
            texts = [tok.decode(ids[x:y]) for x, y in rows]
            key, val = meta['key'], str(meta['val'])
            holds = lambda s: (not neg) and key in s and re.search(r'\b' + re.escape(val) + r'\b', s) is not None
            fact = [i for i, s in enumerate(texts) if holds(s)]
            got, pre = capture(R, ids, LAYERS)
            surp = R.surprisal(ids, pre.cpu())                        # nats per token
            qs = surp[end:].clone(); qs[:2] = -1.0                     # skip the question's first 2 tokens
            sal = [end + int(i) for i in qs.topk(2).indices]
            rec = dict(task=task, keys=keys_kind, seed=a.seed, doc=d, L=a.L, negative=int(neg), n_rows=len(rows),
                       fact_rows=fact, salient=[tok.decode([ids[p]]) for p in sal])
            ranks = {}
            for name, layer, pos in (('F2', 8, sal), ('F3', 2, sal)):
                Vr = got[layer]; mu = Vr[:end].mean(0)
                V = torch.nn.functional.normalize(Vr - mu, dim=-1)
                sc = maxsim_scores(V[:end], V[pos], rows)
                ranks[name] = sc.argsort(descending=True).tolist()
                rec[name + '_top1_score'] = float(sc.max())
            qsrc = meta.get('query_key') or key
            kid = tok.encode(f" {qsrc}", add_special_tokens=False)
            if kid[-1] in q:                                          # F1: the key's last token (reference arm)
                Vr = got[8]; V = torch.nn.functional.normalize(Vr - Vr[:end].mean(0), dim=-1)
                sc = maxsim_scores(V[:end], V[[end + q.index(kid[-1])]], rows)
                ranks['F1'] = sc.argsort(descending=True).tolist(); rec['F1_top1_score'] = float(sc.max())
            E = minilm_embed(texts + [tok.decode(q).strip()])
            msc = E[:-1] @ E[-1]
            ranks['MiniLM'] = np.argsort(-msc).tolist(); rec['MiniLM_top1_score'] = float(msc.max())
            for name, rk in ranks.items():
                pos_ = [rk.index(i) for i in fact] if fact else []
                rec[name + '_rank'] = min(pos_) if pos_ else None
            # reads: warm + no-row contrast from the doc's state
            _, st, off = mind.feed(ids[:end])
            def warmc(span):
                g = mind.greedy_contrast(span + q, copy.deepcopy(st), off, q, copy.deepcopy(st), off, ALPHA)
                txt = tok.decode(g); return txt, sent_ok(txt, val)
            top1 = rows[ranks['F2'][0]]
            rec['read_top1_text'], rec['read_top1'] = warmc(ids[top1[0]:top1[1]])
            five = sorted(ranks['F2'][:5])
            rec['read_lineup_text'], rec['read_lineup'] = warmc([t for i in five for t in ids[rows[i][0]:rows[i][1]]])
            if fact:
                fr = rows[fact[-1]]
                rec['read_oracle_text'], rec['read_oracle'] = warmc(ids[fr[0]:fr[1]])
            side = random.Random(a.seed * 7919 + d)
            swaps = [tok.encode(f" Query: {k} refers to" if task.startswith('syn') else f" Question: what does {k} stand for? Answer:",
                                add_special_tokens=False) for k in swap_keys(side, tok, filler, tok.decode(ids[:end]), K_SWAP)]
            v, lp1, lp2 = swap_read(mind, ids[top1[0]:top1[1]], q, swaps)
            rec.update(swap_rel=lp1 - lp2, swap_v=tok.decode([v]), sec=round(time.time() - t0, 1))
            outf.write(json.dumps(rec) + '\n'); outf.flush()
            print(task, d, {k: rec.get(k + '_rank') for k in ('F2', 'F3', 'F1', 'MiniLM')}, rec['read_top1'], rec['read_lineup'],
                  rec.get('read_oracle'), rec['salient'], rec['sec'], flush=True)


if __name__ == '__main__':
    main()
