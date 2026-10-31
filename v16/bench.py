"""v16/bench.py: the library-half benchmark harness (MLB-1 section 2). Session-index probe (organiser): frozen model + non-learned session index,
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
ROOT = Path(__file__).resolve().parents[1]      # v16/ -> repo root
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
    def surprisal(self, ids: list[int], vecs: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
        """nats of each token given the prefix (PAM only), from the pre-logit vectors."""
        W = self.m.embed.embed.weight.float()
        out = torch.empty(len(ids))
        out[0] = 0.0
        tgt = torch.tensor(ids[1:]); pv = vecs[:-1]          # position t predicts ids[t+1]
        for s0 in range(0, len(ids) - 1, chunk):
            lg = pv[s0:s0 + chunk] @ W.T
            out[s0 + 1:s0 + 1 + lg.shape[0]] = torch.nn.functional.cross_entropy(lg, tgt[s0:s0 + lg.shape[0]], reduction='none')
        return out

    @torch.no_grad()
    def last_logits(self, ids: list[int], vecs=None) -> torch.Tensor:
        if self.arch == 'pam':
            v = vecs[-1] if vecs is not None else self.vectors(ids)[-1]
            return v @ self.m.embed.embed.weight.float().T
        dev = next(self.m.parameters()).device
        x = torch.tensor([ids], device=dev)
        if self.arch == 'mamba2':
            # last-position logits only (the full [T, V] head would not fit beside other jobs)
            hidden = self.m.net.backbone(x)[0, -1]
            return (hidden.float() @ self.m._head_weight().float().T).cpu()
        # transformer: window = the whole sequence (RoPE); attention cost grows with length
        out = self.m(x)
        logits = out[0] if isinstance(out, tuple) else out
        return logits[0, -1].float().cpu()


def natural_key(rng, tok, filler, avoid_lo, avoid_hi):
    """A real two-word phrase taken from the held-out prose OUTSIDE the document's own
    slice (so the organ and the index meet look-alike text): two consecutive tokens
    that decode to two alphabetic words of >= 4 letters."""
    for _ in range(1000):
        i = rng.randrange(0, len(filler) - 2)
        if avoid_lo <= i < avoid_hi:
            continue
        txt = tok.decode(filler[i:i + 2].tolist())
        w = txt.strip().split()
        if len(w) == 2 and all(x.isalpha() and len(x) >= 4 for x in w) and txt.startswith(' '):
            return txt.strip()
    return _recall_key(rng)


def build(rng, tok, filler, L, task, n, keys='synthetic'):
    def enc(s): return tok.encode(s, add_special_tokens=False)
    start = rng.randrange(0, len(filler) - L - 1)
    body = filler[start:start + L].tolist()
    def mk():
        return natural_key(rng, tok, filler, start, start + L) if keys == 'natural' else _recall_key(rng)
    inserts = []  # (pos, ids)
    if task in ('kv', 'kv1', 'kvupd'):
        n = 1 if task == 'kv1' else n
        keys_ = [mk() for _ in range(n)]
        # values may repeat when n > 20; the queried needle's value is still one of 20
        vals = [rng.choice(_RECALL_VALUE_NOUNS) for _ in range(n)] + [rng.choice(_RECALL_VALUE_NOUNS)]
        keys = keys_
        for k, v in zip(keys, vals[:n]):
            inserts.append(enc(f" Definition: {k} refers to {v}."))
        qi = rng.randrange(n); key, val = keys[qi], vals[qi]
        if task == 'kvupd':
            val = rng.choice([v for v in _RECALL_VALUE_NOUNS if v != vals[qi]])
            inserts.append(enc(f" Definition: {key} refers to {val}."))  # later = newer
        query = enc(f" Query: {key} refers to")
        pos = sorted(rng.sample(range(0, max(1, L - 64)), len(inserts)))
        if task == 'kvupd':  # the update must come after the original
            pos = sorted(pos); pos_map = dict(zip(range(len(inserts) - 1), pos[:-1])); pos_map[len(inserts) - 1] = pos[-1]
            order = [i for i, _ in sorted(pos_map.items(), key=lambda kv: kv[1])]
            inserts = [inserts[i] for i in order]
        meta = dict(key=key, val=val, hop_key=None)
    elif task == 'instr':  # a rule stated ONCE in words at the start; "x ->" asked after L filler
        from v14.rules_ood import SHIFT_NEW
        k = rng.choice(SHIFT_NEW)                      # held-out shift value (never trained)
        letters = 'abcdefghijklmnopqrstuvwxyz'
        rule = enc(f" Rule: shift each letter forward by {k}.")
        x = rng.choice(letters[:26 - k]); y = letters[(letters.index(x) + k) % 26]
        query = enc(f" {x} ->")
        ids = rule + body[:L] + query
        ans = tok.encode(f" {y}", add_special_tokens=False)[0]
        meta = dict(key=None, val=y, hop_key=None, rule_len=len(rule), rule_text=f" Rule: shift each letter forward by {k}.")
        return ids, ans, len(query), meta
    elif task == 'neg':   # n needles stored, but the queried key was NEVER stored: abstain is right
        keys_ = [mk() for _ in range(n)]
        for kk in keys_:
            inserts.append(enc(f" Definition: {kk} refers to {rng.choice(_RECALL_VALUE_NOUNS)}."))
        key = mk(); val = rng.choice(_RECALL_VALUE_NOUNS)   # a 'right answer' only for bookkeeping
        query = enc(f" Query: {key} refers to")
        pos = sorted(rng.sample(range(0, max(1, L - 64)), len(inserts)))
        meta = dict(key=key, val=val, hop_key=None, negative=True)
    elif task == 'para':  # key stored one way, asked in a paraphrase (inflection / case); organ misses by design
        keys_ = [mk() for _ in range(n)]
        vals = [rng.choice(_RECALL_VALUE_NOUNS) for _ in range(n)]
        for kk, v in zip(keys_, vals):
            inserts.append(enc(f" Definition: {kk} refers to {v}."))
        qi = rng.randrange(n); key, val = keys_[qi], vals[qi]
        w = key.split()
        forms = [key.title(), key.upper(), (w[0] + ' ' + (w[1][:-1] if w[1].endswith('s') else w[1] + 's')), 'the ' + key]
        qkey = rng.choice(forms)
        query = enc(f" Query: {qkey} refers to")
        pos = sorted(rng.sample(range(0, max(1, L - 64)), len(inserts)))
        meta = dict(key=key, val=val, hop_key=None, query_key=qkey)
    elif task == 'tool':  # a tool call's result block mid-stream; one field asked later
        job = mk(); fields = ['status', 'owner', 'region', 'label']
        vals = [rng.choice(_RECALL_VALUE_NOUNS) for _ in fields]
        inserts = [enc(f" Result: job: {job}; " + "; ".join(f"{f}: {v}" for f, v in zip(fields, vals)) + ".")]
        for _ in range(n - 1):   # distractor result blocks for other jobs
            inserts.append(enc(f" Result: job: {mk()}; " + "; ".join(f"{f}: {rng.choice(_RECALL_VALUE_NOUNS)}" for f in fields) + "."))
        pos = sorted(rng.sample(range(0, max(1, L - 64)), len(inserts)))
        order = list(range(len(inserts))); rng.shuffle(order); inserts = [inserts[i] for i in order]
        fi = rng.randrange(len(fields)); key, val = job, vals[fi]
        query = enc(f" Query: job: {job}; {fields[fi]}:")
        meta = dict(key=job, val=val, hop_key=None)
    else:  # hop2
        k1, k2 = mk(), mk()
        dk1, dk2 = mk(), mk()   # a distractor chain
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


def mem_bytes(arch, cfg, L, n_events, index_rows_all):
    """Per-row memory accounting (bytes), from the checkpoint config.
    mind_state: what the O(1) body carries between tokens.  library_*: the session
    index.  tpp_kv: a Transformer++ KV cache for the same width/depth at length L
    (bf16, K and V, every layer).  mamba2_state: SSM + conv state per layer."""
    out = {'raw_text': 2 * L}
    if arch == 'pam':
        d = cfg.dim; hd = cfg.head_dim; H = cfg.n_heads; P = len(cfg.pam_layers)
        out['mind_state'] = 4 * (P * H * hd * hd + (H * hd * hd + d if getattr(cfg, 'echo', False) else 0))
        out['library_index_all_tokens'] = 4 * index_rows_all * d
        out['library_index_per_event'] = 4 * n_events * d
        out['library_272B_per_event'] = 272 * n_events + 2 * L
        out['tpp_kv_same_shape'] = 2 * cfg.n_layers * 2 * d * L
    elif arch == 'transformer':
        d = cfg.d_model; out['kv_cache'] = 2 * cfg.n_layer * 2 * d * L
    elif arch == 'mamba2':
        d_in = cfg.d_model * cfg.expand; nh = d_in // cfg.headdim
        out['mind_state'] = 4 * cfg.n_layer * (nh * cfg.headdim * cfg.d_state + cfg.d_conv * (d_in + 2 * cfg.ngroups * cfg.d_state))
    return out


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
    ap.add_argument('--keys', default='synthetic', choices=['synthetic', 'natural'])
    ap.add_argument('--gate_frac', type=float, default=0.02, help='C5: fraction of most-surprising tokens that get an index row')
    ap.add_argument('--gate_tau', type=float, default=None, help='C5: surprisal threshold in nats (overrides --gate_frac)')
    ap.add_argument('--filler', default=None, help='token cache for the filler (default: the held-out prose cache = the EVAL pool; pass a pretrain-mix cache for TRAINING docs)')
    ap.add_argument('--key_fn', default=None, help='module:function(vecs[T,D], ids) -> keys[T,d]; replaces the frozen address in C2/C5 (v14 learned fetch)')
    ap.add_argument('--dump_rows', default=None, help='dir: per doc, save ids, candidate row positions/kinds, their vectors, query pos and the answer span (v14 training/eval interface)')
    ap.add_argument('--out', required=True); ap.add_argument('--device', default='cpu', help='cpu, or cuda (required for mamba2: Triton kernels)')
    a = ap.parse_args()
    model, tok, cfg = _load_v13_sempty(Path(a.ckpt), 'baseline_real_pm', torch.device(a.device)); model.eval()
    R = Reader(model, a.arch)
    fill = filler_tokens(a.filler or str(ROOT / '.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt'))
    key_fn = None
    if a.key_fn:
        import importlib; mod, fn = a.key_fn.split(':'); key_fn = getattr(importlib.import_module(mod), fn)
    if a.dump_rows:
        Path(a.dump_rows).mkdir(parents=True, exist_ok=True)
    cand = [tok.encode(f" {v}", add_special_tokens=False)[0] for v in _RECALL_VALUE_NOUNS]
    outf = open(a.out, 'a')
    for L in [int(x) for x in a.lengths.split(',')]:
        filler = fill if L + 2 < len(fill) else np.concatenate([fill] * (L // len(fill) + 2))
        rng = random.Random(a.seed * 1000 + L + hash(a.task) % 997)
        for d in range(a.docs):
            t0 = time.time()
            ids, ans, qlen, meta = build(rng, tok, filler, L, a.task, a.needles, keys=a.keys)
            T = len(ids); doc_end = T - qlen
            rec = dict(arch=a.arch, ckpt=a.ckpt, task=a.task, keys=a.keys, needles=a.needles, L=L, doc=d, T=T)
            n_ev = 4 if a.task == 'hop2' else 1 if a.task in ('kv1', 'instr') else a.needles + (1 if a.task == 'kvupd' else 0)
            try:
                rec['mem'] = mem_bytes(a.arch, cfg, L, n_ev, doc_end)
            except Exception as e:
                rec['mem'] = f'n/a: {e}'
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
                # oracle fetch: the stored sentence put right in front of the query (upper bound of any fetch)
                if vecs is not None and a.task in ('instr', 'kv', 'kv1', 'para', 'tool'):
                    if a.task == 'instr':
                        span_o = ids[:meta['rule_len']]
                    else:
                        tgt_ids = tok.encode(f" {meta['key']}", add_special_tokens=False)
                        pk = next((i for i in range(doc_end) if ids[i:i + len(tgt_ids)] == tgt_ids), None)
                        span_o = ids[max(0, pk - 4):min(doc_end, pk + 40)] if pk is not None else []
                    if span_o:
                        lgo = R.last_logits(span_o + ids[doc_end:])
                        rec['O_reread'] = int(cand[int(lgo[cand].argmax())] == ans) if a.task != 'instr' else int(int(lgo.argmax()) == ans)
                if a.task == 'instr':
                    rec['A_state'] = int(int(lg.argmax()) == ans)      # letters: exact next token
                    rec['B_hit'] = rec['B_ngram'] = 0                  # "x ->" shares no 4-gram with the rule
                # C2 / C3 (PAM): snapshot index keyed by the model's own vectors
                if vecs is not None and a.task not in ('hop2', 'instr'):
                    qsrc = meta.get('query_key') or meta['key']
                    key_ids = tok.encode(f" {qsrc}", add_special_tokens=False)
                    qk = doc_end + ids[doc_end:].index(key_ids[-1])
                    K_all = key_fn(vecs, ids) if key_fn else vecs
                    q = torch.nn.functional.normalize(K_all[qk], dim=-1)
                    docv = torch.nn.functional.normalize(K_all[:doc_end], dim=-1)
                    span_r = 24 if a.task == 'tool' else 6
                    def in_right(pp):
                        return any(ids[j] == ans for j in range(pp + 1, min(doc_end, pp + span_r)))
                    sims = docv @ q; top = sims.topk(5).indices.tolist(); p = top[0]
                    rec['C2_maxsim'] = round(float(sims[p]), 4)
                    rec['C2_hit'] = int(in_right(p)); rec['C2_top5'] = int(any(in_right(pp) for pp in top))
                    if meta.get('negative'):   # nothing stored for this key: any confident fetch is wrong
                        rec['C2_hit'] = rec['C2_top5'] = 0
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
                        # C4: one row per sentence, key = mean of the sentence's vectors
                        # (a non-learned 'write with hindsight at the unit boundary')
                        starts = [0] + [b + 1 for b in bpos[:-1]]
                        sv = torch.stack([docv[s:b + 1].mean(0) for s, b in zip(starts, bpos)])
                        sv = torch.nn.functional.normalize(sv, dim=-1)
                        ss = sv @ q; stop = ss.topk(min(5, len(bpos))).indices.tolist()
                        def in_right_s(si):
                            return any(ids[j] == ans for j in range(starts[si], bpos[si] + 1))
                        rec['C4_hit'] = int(in_right_s(stop[0])); rec['C4_top5'] = int(any(in_right_s(si) for si in stop))
                        si = stop[0]; lg4 = R.last_logits(ids[starts[si]:bpos[si] + 1] + ids[doc_end:])
                        rec['C4_snap'] = int(cand[int(lg4[cand].argmax())] == ans)
                    # C5: surprise-gated rows (N12 as a LIBRARY write gate, no training): keep only
                    # tokens whose surprisal is in the top --gate_frac of the document; key = the
                    # vector at that token. Reports rows kept, whether the queried needle got a row,
                    # and the hit when the search is restricted to those rows.
                    nll = R.surprisal(ids, vecs)[:doc_end]
                    if a.gate_tau is not None:
                        gpos = (nll > a.gate_tau).nonzero().flatten().tolist() or [int(nll.argmax())]
                    else:
                        kkeep = max(1, int(a.gate_frac * doc_end))
                        gpos = nll.topk(kkeep).indices.sort().values.tolist()
                    ndef = len(tok.encode(f" Definition: {meta['key']} refers to", add_special_tokens=False))
                    # the queried needle's own token span (its key last token is at p_true for in_right)
                    gv = docv[gpos]; gs = gv @ q; gtop = [gpos[i] for i in gs.topk(min(5, len(gpos))).indices.tolist()]
                    rec['C5_rows'] = len(gpos); rec['C5_rows_per_event'] = round(len(gpos) / max(1, n_ev), 1)
                    rec['C5_hit'] = int(in_right(gtop[0])); rec['C5_top5'] = int(any(in_right(pp) for pp in gtop))
                    rec['C5_event_has_row'] = int(any(in_right(pp) for pp in gpos))
                    pg = gtop[0]; lg5 = R.last_logits(ids[max(0, pg - 8):min(doc_end, pg + 8)] + ids[doc_end:])
                    rec['C5_snap'] = int(cand[int(lg5[cand].argmax())] == ans)
                    rec['C5_maxsim'] = round(float(gs.max()), 4)
                    rec['C5_rows_per_16tok'] = round(16 * len(gpos) / max(1, doc_end), 2)
                    rec['C5_index_bytes'] = 4 * len(gpos) * cfg.dim
                    rec['C5_library_bytes_272'] = 272 * len(gpos) + 2 * doc_end   # v14 row format + raw text
                    if a.dump_rows:   # v14 interface: candidate rows (surprise + boundary), vectors, query, answer span
                        torch.save({'ids': ids, 'doc_end': doc_end, 'query_pos': qk, 'answer_token': ans,
                                    'needle_key': meta.get('key'), 'rows_surprise': gpos, 'rows_boundary': bpos,
                                    'vecs_surprise': vecs[gpos].half(), 'vecs_boundary': vecs[bpos].half(),
                                    'nll': nll.half(), 'task': a.task, 'L': L, 'doc': d},
                                   Path(a.dump_rows) / f"{a.task}_{a.keys}_L{L}_d{d}.pt")
            except Exception as e:  # keep going; record the failure
                rec['error'] = f'{type(e).__name__}: {str(e)[:200]}'
            rec['sec'] = round(time.time() - t0, 1)
            outf.write(json.dumps(rec) + '\n'); outf.flush()
            print(rec, flush=True)
            del vecs


if __name__ == '__main__':
    main()
