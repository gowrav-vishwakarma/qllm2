"""Session-index probe (organiser, CPU): frozen model + non-learned session index,
360 view. Any checkpoint loadable by scripts/run_memory_behavioral._load_v13_sempty
(PAM, Transformer++, Mamba-2). Writes one JSON line per document to --out (survives kills).

Tasks
  kv     : n definitions "Definition: KEY refers to VALUE." at random positions in L tokens of
           held-out prose, query one at the end; candidates = 20 value nouns (chance 0.05).
  kv1    : same with a single definition.
  kvupd  : kv, but the queried key is re-defined later with a new value (latest wins).
  hop2   : "Record: K1 maps to K2." and "Record: K2 maps to V." far apart; query "K1 maps to".
           The planner chain = two reads of the n-gram organ (K1 -> K2, then K2 -> V).
Read paths
  A  state only (any model).  B  n-gram organ (model-free; fallback A).
  C2 snapshot index: key = the model's pre-logit vector at the query's KEY token, matched
     against every document position; re-read a short raw span + query (PAM only).
  C3 same, but the index keeps only sentence-boundary snapshots ('.' tokens): the cheap one.
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]      # scripts/ -> repo root
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from run_memory_behavioral import _load_v13_sempty  # noqa: E402
from memory_probes.spread import filler_tokens  # noqa: E402
from v7.data import _RECALL_VALUE_NOUNS, _recall_key  # noqa: E402

torch.set_num_threads(int(os.environ.get('THREADS', '8')))
DOT = 13  # GPT-2 '.'


class Reader:
    """Last-position candidate scores + (PAM only) per-token pre-logit vectors."""
    def __init__(self, model, arch):
        self.m, self.arch = model, arch

    @torch.no_grad()
    def vectors(self, ids: list[int]) -> torch.Tensor | None:
        if self.arch != 'pam':
            return None
        x = torch.tensor([ids])
        z, b, t = self.m._stem(x, 0)
        z, _ = self.m._run_blocks(z, b, t, None, 0, x)
        lm, _ = self.m._lm_vector(z, x, b, t)
        return lm.raw(b, t, self.m.model_dim)[0].float()

    @torch.no_grad()
    def last_logits(self, ids: list[int], vecs=None) -> torch.Tensor:
        if self.arch == 'pam':
            v = vecs[-1] if vecs is not None else self.vectors(ids)[-1]
            return v @ self.m.embed.embed.weight.float().T
        x = torch.tensor([ids])
        if self.arch == 'transformer':
            # window = the whole sequence (RoPE); attention cost grows with length
            out = self.m(x)
        else:
            out = self.m(x)
        logits = out[0] if isinstance(out, tuple) else out
        return logits[0, -1].float()


def build(rng, tok, filler, L, task, n):
    def enc(s): return tok.encode(s, add_special_tokens=False)
    start = rng.randrange(0, len(filler) - L - 1)
    body = filler[start:start + L].tolist()
    inserts = []  # (pos, ids)
    if task in ('kv', 'kv1', 'kvupd'):
        n = 1 if task == 'kv1' else n
        keys = [_recall_key(rng) for _ in range(n)]
        vals = rng.sample(_RECALL_VALUE_NOUNS, n + 1)
        for k, v in zip(keys, vals[:n]):
            inserts.append(enc(f" Definition: {k} refers to {v}."))
        qi = rng.randrange(n); key, val = keys[qi], vals[qi]
        if task == 'kvupd':
            val = vals[n]; inserts.append(enc(f" Definition: {key} refers to {val}."))  # later = newer
        query = enc(f" Query: {key} refers to")
        pos = sorted(rng.sample(range(0, max(1, L - 64)), len(inserts)))
        if task == 'kvupd':  # the update must come after the original
            pos = sorted(pos); pos_map = dict(zip(range(len(inserts) - 1), pos[:-1])); pos_map[len(inserts) - 1] = pos[-1]
            order = [i for i, _ in sorted(pos_map.items(), key=lambda kv: kv[1])]
            inserts = [inserts[i] for i in order]
        meta = dict(key=key, val=val, hop_key=None)
    else:  # hop2
        k1, k2 = _recall_key(rng), _recall_key(rng)
        dk1, dk2 = _recall_key(rng), _recall_key(rng)   # a distractor chain
        v, dv = rng.sample(_RECALL_VALUE_NOUNS, 2)
        inserts = [enc(f" Record: {k1} maps to {k2}."), enc(f" Record: {k2} maps to {v}."),
                   enc(f" Record: {dk1} maps to {dk2}."), enc(f" Record: {dk2} maps to {dv}.")]
        pos = sorted(rng.sample(range(0, max(1, L - 64)), 4))
        order = [0, 1, 2, 3]; rng.shuffle(order); inserts = [inserts[i] for i in order]
        query = enc(f" Query: {k1} maps to")
        key, val = k1, v
        meta = dict(key=k1, val=v, hop_key=k2)
    out, cur = [], 0
    for p, d in zip(pos, inserts):
        out += body[cur:p]; out += d; cur = p
    out += body[cur:]
    ids = out + query
    ans = tok.encode(f" {val}", add_special_tokens=False)[0]
    return ids, ans, len(query), meta


def ngram_table(ids, K, upto):
    t = {}
    for i in range(K, upto):
        t[tuple(ids[i - K:i])] = i          # position of the token that followed (latest wins)
    return t


def read_until_dot(ids, i, upto, maxlen=12):
    out = []
    while i < upto and ids[i] != DOT and len(out) < maxlen:
        out.append(ids[i]); i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--arch', default='pam', choices=['pam', 'transformer', 'mamba2'])
    ap.add_argument('--task', default='kv'); ap.add_argument('--lengths', default='2048,8192,32768')
    ap.add_argument('--docs', type=int, default=8); ap.add_argument('--needles', type=int, default=8)
    ap.add_argument('--ngram', type=int, default=4); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    model, tok, cfg = _load_v13_sempty(Path(a.ckpt), 'baseline_real_pm', torch.device('cpu')); model.eval()
    R = Reader(model, a.arch)
    fill = filler_tokens(str(ROOT / '.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt'))
    cand = [tok.encode(f" {v}", add_special_tokens=False)[0] for v in _RECALL_VALUE_NOUNS]
    outf = open(a.out, 'a')
    for L in [int(x) for x in a.lengths.split(',')]:
        filler = fill if L + 2 < len(fill) else np.concatenate([fill] * (L // len(fill) + 2))
        rng = random.Random(a.seed * 1000 + L + hash(a.task) % 997)
        for d in range(a.docs):
            t0 = time.time()
            ids, ans, qlen, meta = build(rng, tok, filler, L, a.task, a.needles)
            T = len(ids); doc_end = T - qlen
            rec = dict(arch=a.arch, ckpt=a.ckpt, task=a.task, L=L, doc=d, T=T)
            try:
                vecs = R.vectors(ids)
                lg = R.last_logits(ids, vecs)
                pa = cand[int(lg[cand].argmax())]
                rec['A_state'] = int(pa == ans)
                if a.task == 'hop2':  # one-hop exact: next token == first token of K2
                    k2_first = tok.encode(f" {meta['hop_key']}", add_special_tokens=False)[0]
                    rec['A_hop1_exact'] = int(int(lg.argmax()) == k2_first)
                # B: n-gram organ
                K = a.ngram; table = ngram_table(ids, K, doc_end)
                hit = table.get(tuple(ids[T - K:T]))
                rec['B_hit'] = int(hit is not None)
                rec['B_ngram'] = int((ids[hit] if hit is not None else pa) == ans)
                if a.task == 'hop2':  # planner: organ hop 1 -> K2 text -> organ hop 2
                    k2_ids = read_until_dot(ids, hit, doc_end) if hit is not None else []
                    q2 = tok.encode(' Query:', add_special_tokens=False) + k2_ids + tok.encode(' maps to', add_special_tokens=False)
                    hit2 = table.get(tuple(q2[-K:]))
                    rec['B_hop2_chain'] = int(hit2 is not None and ids[hit2] == ans)
                    if vecs is not None and hit is not None:  # hop 2 by the model re-reading the retrieved span
                        span = ids[max(0, hit - 16):min(doc_end, hit + 12)]
                        if hit2 is not None:
                            span2 = ids[max(0, hit2 - 16):min(doc_end, hit2 + 8)]
                            lg2 = R.last_logits(span + span2 + q2)
                            rec['C_hop2_reread'] = int(cand[int(lg2[cand].argmax())] == ans)
                # C2 / C3 (PAM): snapshot index keyed by the model's own vectors
                if vecs is not None and a.task != 'hop2':
                    key_ids = tok.encode(f" {meta['key']}", add_special_tokens=False)
                    qk = doc_end + ids[doc_end:].index(key_ids[-1])
                    q = torch.nn.functional.normalize(vecs[qk], dim=-1)
                    docv = torch.nn.functional.normalize(vecs[:doc_end], dim=-1)
                    def in_right(pp):
                        return any(ids[j] == ans for j in range(pp + 1, min(doc_end, pp + 6)))
                    sims = docv @ q; top = sims.topk(5).indices.tolist(); p = top[0]
                    rec['C2_hit'] = int(in_right(p)); rec['C2_top5'] = int(any(in_right(pp) for pp in top))
                    lg2 = R.last_logits(ids[max(0, p - 8):min(doc_end, p + 8)] + ids[doc_end:])
                    rec['C2_snap'] = int(cand[int(lg2[cand].argmax())] == ans)
                    bpos = [i for i in range(doc_end) if ids[i] == DOT]
                    if bpos:
                        bv = docv[bpos]; bs = bv @ q; btop = [bpos[i] for i in bs.topk(min(5, len(bpos))).indices.tolist()]
                        def in_right_b(pp):  # boundary = end of the definition sentence
                            return any(ids[j] == ans for j in range(max(0, pp - 3), pp))
                        rec['C3_hit'] = int(in_right_b(btop[0])); rec['C3_top5'] = int(any(in_right_b(pp) for pp in btop))
                        pb = btop[0]; lg3 = R.last_logits(ids[max(0, pb - 14):pb + 1] + ids[doc_end:])
                        rec['C3_snap'] = int(cand[int(lg3[cand].argmax())] == ans)
                        rec['C3_index_rows'] = len(bpos)
            except Exception as e:  # keep going; record the failure
                rec['error'] = f'{type(e).__name__}: {str(e)[:200]}'
            rec['sec'] = round(time.time() - t0, 1)
            outf.write(json.dumps(rec) + '\n'); outf.flush()
            print(rec, flush=True)
            del vecs


if __name__ == '__main__':
    main()
