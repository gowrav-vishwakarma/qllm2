"""Curriculum cells on the BENCH vocabulary (mind side), for the cu1 pre-registration.

  fsq  : 4 re-read definitions " Definition: KEY refers to VALUE." (bench keys and
         _RECALL_VALUE_NOUNS, never in the curriculum's own vocabulary), then
         " Query: KEY refers to". Half the queries ask a stored key (answer = its value),
         half a key that was NOT stored (answer " unknown"). Candidates: the 20 bench value
         nouns + " unknown" (first tokens). Scores: present accuracy, abstain rate on the
         absent half, false abstain on the present half.
  hop  : bench hop2 records with no filler: " Record: K1 maps to K2." / " Record: K2 maps
         to V." + a distractor chain, shuffled, " Query: K1 maps to" -> V; candidates the
         20 nouns.
One JSON line per document (arch, ckpt, task, doc, correct, ...).
"""
from __future__ import annotations

import argparse, json, random, sys, time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from run_memory_behavioral import _load_v13_sempty  # noqa: E402
from run_skillpack_probe import Body  # noqa: E402
from v7.data import _RECALL_VALUE_NOUNS, _recall_key  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--docs', type=int, default=200)
    p.add_argument('--seed', type=int, default=9000)
    p.add_argument('--device', default='cpu')
    p.add_argument('--mem_frac', type=float, default=0.0)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    dev = torch.device(a.device)
    if a.mem_frac and dev.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    model, tok, _ = _load_v13_sempty(a.checkpoint, 'baseline_real_pm', dev)
    model.eval(); body = Body(model, dev)
    first = lambda w: tok.encode(' ' + w, add_special_tokens=False)[0]  # noqa: E731
    nouns = list(_RECALL_VALUE_NOUNS); cand = [first(w) for w in nouns]; unk = first('unknown')
    out = open(a.out, 'a')
    for d in range(a.docs):
        rng = random.Random(a.seed * 1000 + d)
        t0 = time.time()
        keys = []
        while len(keys) < 5:
            k = _recall_key(rng)
            if k not in keys:
                keys.append(k)
        # fsq
        vals = [rng.choice(nouns) for _ in range(4)]
        spans = ''.join(f' Definition: {k} refers to {v}.' for k, v in zip(keys[:4], vals))
        absent = d % 2 == 1
        q = keys[4] if absent else keys[rng.randrange(4)]
        ans = unk if absent else first(vals[keys.index(q)])
        lg = body.last_logits(tok.encode(spans + f' Query: {q} refers to', add_special_tokens=False))
        cs = cand + [unk]; pred = cs[int(lg[cs].argmax())]
        out.write(json.dumps({'arch': body.arch, 'ckpt': str(a.checkpoint), 'task': 'fsq', 'doc': d,
                              'absent': absent, 'correct': int(pred == ans), 'abstained': int(pred == unk),
                              'sec': round(time.time() - t0, 3)}) + '\n')
        # hop
        k1, k2, d1, d2 = keys[:4]
        v, dv = rng.sample(nouns, 2)
        recs = [f' Record: {k1} maps to {k2}.', f' Record: {k2} maps to {v}.',
                f' Record: {d1} maps to {d2}.', f' Record: {d2} maps to {dv}.']
        rng.shuffle(recs)
        lg = body.last_logits(tok.encode(''.join(recs) + f' Query: {k1} maps to', add_special_tokens=False))
        out.write(json.dumps({'arch': body.arch, 'ckpt': str(a.checkpoint), 'task': 'hop', 'doc': d,
                              'correct': int(cand[int(lg[cand].argmax())] == first(v))}) + '\n')
        out.flush()


if __name__ == '__main__':
    main()
