"""nkr (EXPERIMENTS_SEMPY "nkr"): notebook-key retrieval for ARC. Rerank BM25's top-100 passages per question by the trained
mind's OWN notebook match and prepend the new top-3 exactly as BENCH-LIB's library arm; scoring is Library's unmodified
`v14/benchlib_eval.py score` on a library dir holding our retrieved file.

  step top100 (CPU): BM25 top-100 per item (the BENCH-LIB library and query)        -> <out>/top100_<task>.jsonl
  step rerank (GPU): score(p) = sum over notebook layers and heads of max over p's tokens of cos(q_head, k_head,t), with q the
                     notebook query at the question prompt's last token and k the key projections over p, both from the qkv
                     projection BEFORE the Chrono rotation (position is meaningless across documents)
                     -> <out>/nk/retrieved_<task>.jsonl and <out>/rand/retrieved_<task>.jsonl (3 random of the top-100, seed 0)
  then:  python v14/benchlib_eval.py score --lib <out>/nk --tasks arc_challenge --ckpt <weights> --tag c3b_nk --out ...
    python v13_sempty/tmp/nkr_arc.py top100 --lib <lib> --out <out> [--tasks arc_challenge] [--limit N]
    python v13_sempty/tmp/nkr_arc.py rerank --out <out> --ckpt <weights> [--tasks arc_challenge] [--k 100] [--max_tok 160]
"""
from __future__ import annotations

import argparse, json, random, shutil, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]


def top100(a):
    from mc_eval import load_task
    from v14.arclib_index import BM25
    from v14.benchlib_eval import query_text
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); bm = BM25(a.lib); print(f'[top100] index loaded ({time.time() - t0:.0f} s)', flush=True)
    for name in ('nk', 'rand'):
        (out / name).mkdir(exist_ok=True); shutil.copy(Path(a.lib) / 'meta.json', out / name / 'meta.json')
    for task in a.tasks.split(','):
        with open(out / f'top100_{task}.jsonl', 'w') as f:
            for i, (prompt, choices, gold) in enumerate(load_task(task)):
                if a.limit and i >= a.limit:
                    break
                top = bm.top(query_text(prompt, choices), a.k)
                f.write(json.dumps(dict(i=i, prompt=prompt, passages=[t for _, _, t in top], ids=[j for j, _, _ in top],
                                        source=[int(bm.src[j]) for j, _, _ in top], score=[round(s, 2) for _, s, _ in top])) + '\n')
        print(f'[top100] {task} done ({time.time() - t0:.0f} s)', flush=True)


def rerank(a):
    import torch
    from run_memory_behavioral import _load_v13_sempty
    dev = torch.device('cuda')
    model, tok, _ = _load_v13_sempty(Path(a.ckpt), 'baseline_real_pm', dev); model.eval()
    layers = [b.pam for b in model.blocks if getattr(b, 'pam', None) is not None]
    H, K = layers[0].heads.size, layers[0].head_dim
    cap = {}
    for li, L in enumerate(layers):
        L.qkv_proj.register_forward_hook(lambda m, i, o, li=li: cap.__setitem__(li, o.data))   # [B, T, 3*H*K] (slot, head, feat)
    def qk(ids_list):
        T = max(len(x) for x in ids_list)
        ids = torch.zeros(len(ids_list), T, dtype=torch.long, device=dev); mask = torch.zeros(len(ids_list), T, device=dev)
        for b, x in enumerate(ids_list):
            ids[b, :len(x)] = torch.tensor(x, device=dev); mask[b, :len(x)] = 1
        cap.clear()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            model._hidden_to_lm(ids)
        out = [cap[li].float().reshape(len(ids_list), T, 3, H, K) for li in range(len(layers))]
        return out, mask
    out = Path(a.out); rng = random.Random(0); t0 = time.time()
    for task in a.tasks.split(','):
        rows = [json.loads(l) for l in open(out / f'top100_{task}.jsonl')]
        fn = open(out / 'nk' / f'retrieved_{task}.jsonl', 'w'); fr = open(out / 'rand' / f'retrieved_{task}.jsonl', 'w')
        for n, r in enumerate(rows):
            qo, _ = qk([tok.encode(r['prompt'], add_special_tokens=False)])
            q = [torch.nn.functional.normalize(x[0, -1, 0], dim=-1) for x in qo]                 # per layer [H, K]
            P = [tok.encode(p, add_special_tokens=False)[:a.max_tok] or [0] for p in r['passages']]
            ko, mask = qk(P)
            s = torch.zeros(len(P), device=dev)
            for li, x in enumerate(ko):
                k = torch.nn.functional.normalize(x[:, :, 1], dim=-1)                           # [N, T, H, K]
                cs = torch.einsum('nthk,hk->nth', k, q[li])                                     # cos per token / head
                cs = cs.masked_fill(mask[:, :, None] == 0, -2.0)
                s += cs.max(dim=1).values.sum(-1)
            order = torch.argsort(-s).tolist()
            pick = lambda idx: dict(i=r['i'], passages=[r['passages'][j] for j in idx], ids=[r['ids'][j] for j in idx],
                                    source=[r['source'][j] for j in idx], score=[round(float(s[j]), 3) for j in idx])
            fn.write(json.dumps(pick(order[:3])) + '\n')
            fr.write(json.dumps(pick(rng.sample(range(len(P)), min(3, len(P))))) + '\n')
            if n % 100 == 0:
                print(f'[rerank] {task} {n}/{len(rows)} ({time.time() - t0:.0f} s)', flush=True)
        fn.close(); fr.close()
    print('[rerank] DONE', flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('step', choices=['top100', 'rerank'])
    ap.add_argument('--lib'); ap.add_argument('--out', required=True); ap.add_argument('--tasks', default='arc_challenge')
    ap.add_argument('--limit', type=int, default=None); ap.add_argument('--k', type=int, default=100)
    ap.add_argument('--ckpt'); ap.add_argument('--max_tok', type=int, default=160)
    a = ap.parse_args(); {'top100': top100, 'rerank': rerank}[a.step](a)


if __name__ == '__main__':
    main()
