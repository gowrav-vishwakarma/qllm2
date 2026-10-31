"""L1 training/validation data for the library key head (v14/EXPERIMENTS_V14.md section "L").

Documents of L tokens of prose from a POOL (training: the FineWeb pool built by
v14.train_pool; never the held-out prose), with one query whose answer is a token position
inside an earlier "event":

  synthetic : n needles "<K> is defined as <V>." (8 templates, none of them the harness's
              "Definition: K refers to V." / "Query: K refers to") + a query sentence at the end.
              K = a pseudo-word or a real two-word phrase from the pool. Target = the first
              token of K inside the needle. Query position = the first token of K in the query.
  natural   : a rare word (>= 5 letters) that occurs twice in the window, >= 400 tokens apart;
              the query position is the later mention, the target the earlier one.

The frozen mind (a v13 PAM checkpoint) reads each document once; we cache its pre-logit vectors
(fp16), the surprisal of every token, the query position and the target position. The gate
(bursts) is computed later from the cached surprisal, so tau can change without re-reading.

    .venv/bin/python -m v14.l1_data --ckpt <best_model.pt> --pool .cache/v14_train_pool_fineweb014.pt \
        --out <dir> --docs 1500 --length 4096 --seed 1
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from memory_probes.spread import filler_tokens  # noqa: E402
from run_memory_behavioral import _load_v13_sempty  # noqa: E402
from v7.data import _RECALL_VALUE_NOUNS, _recall_key  # noqa: E402

# (needle, query); {K} {V}. The harness's own template is deliberately absent (held-out template).
TEMPLATES = [
    (" {K} is defined as {V}.", " What is {K}? It is defined as"),
    (" Note that {K} means {V}.", " Recall that {K} means"),
    (" By convention, {K} stands for {V}.", " Earlier we said {K} stands for"),
    (" Our code name for {K} is {V}.", " The code name for {K} is"),
    (" {K} always corresponds to {V}.", " Question: {K} corresponds to"),
    (" In this report, {K} denotes {V}.", " Therefore {K} denotes"),
    (" Remember: {K} equals {V}.", " So {K} equals"),
    (" The label of {K} is {V}.", " The label of {K} is"),
]
VALUES_TRAIN = ('river', 'stone', 'lantern', 'bridge', 'meadow', 'harbor', 'violet', 'copper', 'falcon', 'ember',
                'orchid', 'glacier', 'compass', 'velvet', 'thunder', 'pebble', 'tundra', 'canyon', 'mosaic', 'prairie')
HARNESS_DEF, HARNESS_QUERY = " Definition: {K} refers to {V}.", " Query: {K} refers to"


def _find(hay: list[int], needle: list[int], start: int = 0) -> int | None:
    n = len(needle)
    for i in range(start, len(hay) - n + 1):
        if hay[i:i + n] == needle:
            return i
    return None


def natural_phrase(rng, tok, pool, avoid):
    for _ in range(1000):
        i = rng.randrange(0, len(pool) - 2)
        if avoid[0] <= i < avoid[1]:
            continue
        txt = tok.decode(pool[i:i + 2].tolist())
        w = txt.strip().split()
        if len(w) == 2 and all(x.isalpha() and len(x) >= 4 for x in w) and txt.startswith(' '):
            return txt.strip()
    return _recall_key(rng)


def make_synthetic(rng, tok, pool, L, templates, values, keys='mixed', n_lo=6, n_hi=24, harness=False):
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    start = rng.randrange(0, len(pool) - L - 1)
    body = pool[start:start + L].tolist()
    n = rng.randint(n_lo, n_hi)
    use_nat = (keys == 'natural') or (keys == 'mixed' and rng.random() < 0.5)
    ks, seen = [], set()
    while len(ks) < n:
        k = natural_phrase(rng, tok, pool, (start, start + L)) if use_nat else _recall_key(rng)
        if k not in seen:
            seen.add(k); ks.append(k)
    vs = [rng.choice(values) for _ in ks]
    tpl = (HARNESS_DEF, HARNESS_QUERY) if harness else rng.choice(templates)
    pos = sorted(rng.sample(range(0, L - 64), n))
    out, cur, spans = [], 0, []
    for p, k, v in zip(pos, ks, vs):
        out += body[cur:p]; cur = p
        ins = enc(tpl[0].format(K=k, V=v))
        spans.append((len(out), len(out) + len(ins)))
        out += ins
    out += body[cur:]
    qi = rng.randrange(n)
    q_ids = enc(tpl[1].format(K=ks[qi]))
    key_ids = enc(' ' + ks[qi])
    # position of the key inside the needle and inside the query
    ins_ids = out[spans[qi][0]:spans[qi][1]]
    tpos = _find(ins_ids, key_ids)
    qpos = _find(q_ids, key_ids)
    if tpos is None or qpos is None:
        return None
    ids = out + q_ids
    return dict(ids=ids, qpos=len(out) + qpos, target=spans[qi][0] + tpos, tlen=len(key_ids),
                kind='synthetic', answer=enc(' ' + vs[qi])[0])


def make_natural(rng, tok, pool, L, counts):
    for _ in range(40):
        start = rng.randrange(0, len(pool) - L - 1)
        w = pool[start:start + L].tolist()
        first = {}
        cands = []
        for i, t in enumerate(w):
            if t in first:
                if i - first[t] >= 400 and counts[t] <= 3000:
                    cands.append((first[t], i, t))
            else:
                first[t] = i
        good = []
        for f, i, t in cands:
            s = tok.decode([t])
            if s.startswith(' ') and s.strip().isalpha() and len(s.strip()) >= 5:
                good.append((f, i, t))
        if good:
            f, i, t = rng.choice(good)
            return dict(ids=w[:i + 1], qpos=i, target=f, tlen=1, kind='natural', answer=-1)
    return None


@torch.no_grad()
def read_doc(model, ids: list[int], device):
    """-> vecs [T, D] fp16 (the actual output vector, echo included), nll [T] fp16 (nats)."""
    x = torch.tensor([ids], device=device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        z, b, t = model._stem(x, 0)
        z, _ = model._run_blocks(z, b, t, None, 0, x)
        lm, _ = model._lm_vector(z, x, b, t)
        v = lm.raw(b, t, model.model_dim)[0]
    E = model.embed.embed.weight.float()
    nll = torch.zeros(len(ids), device=device)
    for s in range(0, len(ids) - 1, 1024):
        lg = v[s:min(s + 1024, len(ids) - 1)].float() @ E.t()
        nll[s + 1:s + 1 + lg.shape[0]] = torch.logsumexp(lg, -1) - lg.gather(-1, x[0, s + 1:s + 1 + lg.shape[0], None]).squeeze(-1)
    return v.half().cpu(), nll.half().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--pool', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--docs', type=int, default=1500)
    ap.add_argument('--length', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--frac_natural', type=float, default=0.5)
    ap.add_argument('--harness_template', action='store_true', help='validation: the harness template (held out of training)')
    ap.add_argument('--mem_frac', type=float, default=0.0)
    args = ap.parse_args()
    if args.mem_frac > 0:
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')
    device = torch.device('cuda')
    model, _, _ = _load_v13_sempty(args.ckpt, 'baseline_real_pm', device)
    model.eval()
    pool = filler_tokens(args.pool)
    counts = np.bincount(pool, minlength=50257)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    values = VALUES_TRAIN if not args.harness_template else tuple(_RECALL_VALUE_NOUNS)
    made, tries = 0, 0
    while made < args.docs and tries < args.docs * 3:
        tries += 1
        if rng.random() < args.frac_natural and not args.harness_template:
            d = make_natural(rng, tok, pool, args.length, counts)
        else:
            d = make_synthetic(rng, tok, pool, args.length, TEMPLATES, values, harness=args.harness_template)
        if d is None:
            continue
        vecs, nll = read_doc(model, d['ids'], device)
        d.update(vecs=vecs, nll=nll)
        torch.save(d, out / f'doc_{made:05d}.pt')
        made += 1
        if made % 50 == 0:
            print(f"  {made}/{args.docs} docs", flush=True)
    print(f"saved {made} docs to {out}", flush=True)


if __name__ == '__main__':
    main()
