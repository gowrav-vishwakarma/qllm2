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
import argparse, json, os, random, re, sys, time, zlib
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

    vec_layer = None    # None = pre-logit vector; k = residual after block k (Library: mid-layer address)
    centre = False      # subtract the document mean from the keys before the cosine
    last_keys = None    # key vectors of the last vectors() call (PAM only; None for other bodies)

    @torch.no_grad()
    def vectors(self, ids: list[int]) -> torch.Tensor | None:
        """Pre-logit vectors [T, D] (always, for surprisal and logits); with vec_layer set the
        KEY vectors are the residual after that block instead, kept on self.last_keys."""
        if self.arch != 'pam':
            return None
        x = torch.tensor([ids], device=next(self.m.parameters()).device)   # PAM on cuda too (package table)
        z, b, t = self.m._stem(x, 0)
        if self.vec_layer is None:
            z, _ = self.m._run_blocks(z, b, t, None, 0, x)
            keys = None
        else:
            keys = None; share = getattr(self.m.config, 'share_reads', False); shared = None
            for i, block in enumerate(self.m.blocks):       # mirrors LM._run_blocks (incl. M2 shared reads)
                if share:
                    z, _ = block(z, pam_state=None, step_offset=0, shared=shared)
                    if block.pam is not None:
                        shared = block.pam._shared
                else:
                    z, _ = block(z, pam_state=None, step_offset=0)
                if self.m.cond_mem_layers and i in self.m.cond_mem_layers:
                    z = self.m.cond_mem[str(i)](z, x)
                if i == self.vec_layer:
                    keys = z.raw(b, t, self.m.model_dim)[0].float().clone()
            assert keys is not None, f'vec_layer {self.vec_layer} out of range'
        lm, _ = self.m._lm_vector(z, x, b, t)
        out = lm.raw(b, t, self.m.model_dim)[0].float().cpu()
        self.last_keys = (keys if keys is not None else out).cpu()
        return out

    @torch.no_grad()
    def surprisal(self, ids: list[int], vecs: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
        """nats of each token given the prefix (PAM only), from the pre-logit vectors."""
        W = self.m.embed.embed.weight.float(); dev = W.device
        out = torch.empty(len(ids))
        out[0] = 0.0
        tgt = torch.tensor(ids[1:], device=dev); pv = vecs[:-1].to(dev)          # position t predicts ids[t+1]
        for s0 in range(0, len(ids) - 1, chunk):
            lg = pv[s0:s0 + chunk] @ W.T
            out[s0 + 1:s0 + 1 + lg.shape[0]] = torch.nn.functional.cross_entropy(lg, tgt[s0:s0 + lg.shape[0]], reduction='none').cpu()
        return out

    @torch.no_grad()
    @torch.no_grad()
    def greedy(self, ids: list[int], max_new: int = 16, stops=(13, 198)) -> list[int]:
        """Free generation (user, Nov 1): greedy tokens after `ids` until '.' / newline or max_new.
        Re-runs the short context each step (the re-read is span + query, not the document)."""
        out = []
        for _ in range(max_new):
            t = int(self.last_logits(ids + out).argmax())
            out.append(t)
            if t in stops:
                break
        return out

    def last_logits(self, ids: list[int], vecs=None) -> torch.Tensor:
        if self.arch == 'pam':
            v = vecs[-1] if vecs is not None else self.vectors(ids)[-1]
            return (v.to(self.m.embed.embed.weight.device) @ self.m.embed.embed.weight.float().T).cpu()
        dev = next(self.m.parameters()).device
        x = torch.tensor([ids], device=dev)
        if self.arch == 'mamba2':
            # last-position logits only (the full [T, V] head would not fit beside other jobs)
            hidden = self.m.net.backbone(x)[0, -1]
            return (hidden.float() @ self.m._head_weight().float().T).cpu()
        # transformer: window = the whole sequence (RoPE); attention cost grows with length.
        # Last-position logits only: the full [T, V] head (6.6 GB fp32 at 32k) OOMed the 4090 on natq.
        # bf16 autocast so SDPA takes the flash / memory-efficient kernel (fp32 at 32k fell back to the
        # quadratic math kernel and OOMed the 4090); the state-only argmax is read in fp32 from the last hidden.
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(x.device.type == 'cuda')):
            hidden, _ = self.m._hidden_to_lm(x)
        return (hidden[0, -1].float() @ self.m._head_weight().float().T).cpu()


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


def build(rng, tok, filler, L, task, n, keys='synthetic', shots=0):
    def enc(s): return tok.encode(s, add_special_tokens=False)
    if task in ('syn', 'synneg'):   # A5 (Library): synonym cell, docs built by v14/syn_cell (draws its own start)
        from v14.syn_cell import build_syn
        return build_syn(rng, tok, filler, L, n, neg=(task == 'synneg'))
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
        picks = rng.sample(letters[:26 - k], shots + 1)
        x = picks[-1]; y = letters[letters.index(x) + k]
        demo = ''.join(f'{a} -> {letters[letters.index(a) + k]}\n' for a in picks[:-1])   # trained demo lines
        query = enc(f"Letter rule.\n{demo}{x} ->")       # the trained header + optional shots + query line
        ids = rule + body[:L] + query
        ans = tok.encode(f" {y}", add_special_tokens=False)[0]
        meta = dict(key=None, val=y, hop_key=None, rule_len=len(rule), rule_text=f" Rule: shift each letter forward by {k}.",
                    letter_cands=[tok.encode(f" {c}", add_special_tokens=False)[0] for c in letters])
        return ids, ans, len(query), meta
    elif task == 'neg':   # n needles stored, but the queried key was NEVER stored: abstain is right
        keys_ = [mk() for _ in range(n)]
        for kk in keys_:
            inserts.append(enc(f" Definition: {kk} refers to {rng.choice(_RECALL_VALUE_NOUNS)}."))
        key = mk(); val = rng.choice(_RECALL_VALUE_NOUNS)   # a 'right answer' only for bookkeeping
        query = enc(f" Query: {key} refers to")
        pos = sorted(rng.sample(range(0, max(1, L - 64)), len(inserts)))
        meta = dict(key=key, val=val, hop_key=None, negative=True)
    elif task == 'natq':  # NATURAL QUESTION (user, Nov 1: generation counts): fact in prose, plain question, free answer
        keys_ = [mk() for _ in range(n)]
        vals = [rng.choice(_RECALL_VALUE_NOUNS) for _ in range(n)]
        state_t = [" By the way, {k} is just another name for {v}.", " People around here call {v} by the name {k}.",
                   " It turns out that {k} is what everyone means when they say {v}.", " In the old records, {k} stands for {v}."]
        for kk, v in zip(keys_, vals):
            inserts.append(enc(rng.choice(state_t).format(k=kk, v=v)))
        qi = rng.randrange(n); key, val = keys_[qi], vals[qi]
        query = enc(f" Question: what does {key} stand for? Answer:")
        pos = sorted(rng.sample(range(0, max(1, L - 64)), len(inserts)))
        meta = dict(key=key, val=val, hop_key=None)
    elif task == 'para':  # key stored one way, asked in a paraphrase (inflection / case); organ misses by design
        keys_ = [mk() for _ in range(n)]
        vals = [rng.choice(_RECALL_VALUE_NOUNS) for _ in range(n)]
        for kk, v in zip(keys_, vals):
            inserts.append(enc(f" Definition: {kk} refers to {v}."))
        qi = rng.randrange(n); key, val = keys_[qi], vals[qi]
        w = key.split(); last = w[-1]
        plural = ' '.join(w[:-1] + [last[:-1] if last.endswith('s') else last + 's'])
        forms = [key.title(), key.upper(), plural, 'the ' + key]
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
    out = {'raw_text': 2 * L, 'install_cost_mind_passes': 1, 'install_cost_tokens': L}
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
    ap.add_argument('--shots', type=int, default=0, help='instr: demonstration lines placed right before the query (control that the skill is present)')
    ap.add_argument('--gate_frac', type=float, default=0.02, help='C5: fraction of most-surprising tokens that get an index row')
    ap.add_argument('--gate_tau', type=float, default=None, help='C5: surprisal threshold in nats (overrides --gate_frac)')
    ap.add_argument('--vec_layer', type=int, default=None, help='key vectors = residual after block k (Library: mid-layer address); surprisal still from the output')
    ap.add_argument('--centre', action='store_true', help='subtract the document mean from the key vectors before the cosine')
    ap.add_argument('--centre_ref', default=None, help='with --centre: subtract a FIXED reference mean [D] (torch .pt) instead of the document mean; the deployed condition, where library and session come from different texts')
    ap.add_argument('--filler', default=None, help='token cache for the filler (default: the held-out prose cache = the EVAL pool; pass a pretrain-mix cache for TRAINING docs)')
    ap.add_argument('--key_fn', default=None, help='module:function(vecs[T,D], ids) -> keys[T,d]; replaces the frozen address in C2/C5 (v14 learned fetch)')
    ap.add_argument('--dump_rows', default=None, help='dir: per doc, save ids, candidate row positions/kinds, their vectors, query pos and the answer span (v14 training/eval interface)')
    ap.add_argument('--out', required=True); ap.add_argument('--device', default='cpu', help='cpu, or cuda (required for mamba2: Triton kernels)')
    a = ap.parse_args()
    model, tok, cfg = _load_v13_sempty(Path(a.ckpt), 'baseline_real_pm', torch.device(a.device)); model.eval()
    R = Reader(model, a.arch); R.vec_layer = a.vec_layer; R.centre = a.centre
    centre_ref = None
    if a.centre_ref:
        centre_ref = torch.as_tensor(torch.load(a.centre_ref, map_location='cpu')).float().reshape(1, -1)
    fill = filler_tokens(a.filler or str(ROOT / '.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt'))
    key_fn = None; key_fn_bursts = False
    if a.key_fn:
        import importlib, inspect; mod, fn = a.key_fn.split(':'); key_fn = getattr(importlib.import_module(mod), fn)
        key_fn_bursts = len(inspect.signature(key_fn).parameters) >= 3   # fn(keys, ids, bursts) -> one key per burst
    if a.dump_rows:
        Path(a.dump_rows).mkdir(parents=True, exist_ok=True)
    cand = [tok.encode(f" {v}", add_special_tokens=False)[0] for v in _RECALL_VALUE_NOUNS]
    outf = open(a.out, 'a')
    for L in [int(x) for x in a.lengths.split(',')]:
        filler = fill if L + 2 < len(fill) else np.concatenate([fill] * (L // len(fill) + 2))
        rng = random.Random(a.seed * 1000 + L + zlib.crc32(a.task.encode()) % 997)   # stable across processes (A4; was hash(), randomized per process)
        for d in range(a.docs):
            t0 = time.time()
            ids, ans, qlen, meta = build(rng, tok, filler, L, a.task, a.needles, keys=a.keys, shots=a.shots)
            T = len(ids); doc_end = T - qlen
            rec = dict(arch=a.arch, ckpt=a.ckpt, task=a.task, keys=a.keys, needles=a.needles, L=L, doc=d, T=T,
                       vec_layer=a.vec_layer, centre=a.centre, centre_ref=a.centre_ref)
            n_ev = 4 if a.task == 'hop2' else 1 if a.task in ('kv1', 'instr') else a.needles + (1 if a.task == 'kvupd' else 0)
            try:
                rec['mem'] = mem_bytes(a.arch, cfg, L, n_ev, doc_end)
            except Exception as e:
                rec['mem'] = f'n/a: {e}'
            try:
                vecs = R.vectors(ids)
                keys_doc = R.last_keys          # captured now: later short re-reads overwrite R.last_keys
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
                if vecs is not None and a.task in ('instr', 'kv', 'kv1', 'para', 'tool', 'natq', 'syn'):
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
                            lc = meta['letter_cands']; rec['O_letters'] = int(lc[int(lgo[lc].argmax())] == ans)
                if a.task == 'instr':
                    rec['A_state'] = int(int(lg.argmax()) == ans)      # letters: exact next token
                    lc = meta['letter_cands']; rec['A_letters'] = int(lc[int(lg[lc].argmax())] == ans)
                    rec['B_hit'] = rec['B_ngram'] = 0                  # "x ->" shares no 4-gram with the rule
                    rec['shots'] = a.shots
                # C2 / C3 (PAM): snapshot index keyed by the model's own vectors
                if vecs is not None and a.task not in ('hop2', 'instr'):
                    qsrc = meta.get('query_key') or meta['key']
                    key_ids = tok.encode(f" {qsrc}", add_special_tokens=False)
                    qk = doc_end + ids[doc_end:].index(key_ids[-1])
                    # key_fn receives the key vectors the built-in path uses (layer-k residual with
                    # --vec_layer, else the pre-logit vector); 2-arg form -> per-token keys [T, d]
                    K_all = key_fn(keys_doc, ids) if (key_fn and not key_fn_bursts) else keys_doc
                    if a.centre:
                        mu = centre_ref if centre_ref is not None else K_all[:doc_end].mean(0, keepdim=True)
                        K_all = K_all - mu
                    q = torch.nn.functional.normalize(K_all[qk], dim=-1)
                    docv = torch.nn.functional.normalize(K_all[:doc_end], dim=-1)
                    span_r = 24 if a.task in ('tool', 'natq') else 6
                    def in_right(pp):
                        return any(ids[j] == ans for j in range(pp + 1, min(doc_end, pp + span_r)))
                    sims = docv @ q; top = sims.topk(5).indices.tolist(); p = top[0]
                    rec['C2_maxsim'] = round(float(sims[p]), 4)
                    rec['C2_hit'] = int(in_right(p)); rec['C2_top5'] = int(any(in_right(pp) for pp in top))
                    if meta.get('negative'):   # nothing stored for this key: any confident fetch is wrong
                        rec['C2_hit'] = rec['C2_top5'] = 0
                    lg2 = R.last_logits(ids[max(0, p - 8):min(doc_end, p + 8)] + ids[doc_end:])
                    rec['C2_snap'] = int(cand[int(lg2[cand].argmax())] == ans)
                    # planner v0 on the top-5 fetched spans: concat-all vs each-alone-max-margin
                    spans = [ids[max(0, pp - 8):min(doc_end, pp + span_r + 2)] for pp in top]
                    lgc = R.last_logits([t for sp in spans for t in sp] + ids[doc_end:])
                    rec['P_concat'] = int(cand[int(lgc[cand].argmax())] == ans)
                    best = None
                    for sp in spans:
                        lgs = R.last_logits(sp + ids[doc_end:])[cand]
                        v2, i2 = lgs.topk(2); margin = float(v2[0] - v2[1])
                        if best is None or margin > best[0]:
                            best = (margin, cand[int(i2[0])])
                    rec['P_margin'] = int(best[1] == ans)
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
                    bursts = []            # [start, end) of consecutive firings (gap <= 2)
                    for gp in gpos:
                        if bursts and gp - bursts[-1][1] <= 2:
                            bursts[-1][1] = gp + 1
                        else:
                            bursts.append([gp, gp + 1])
                    bstart = [b[0] for b in bursts]
                    rec['C5_bursts'] = len(bursts); rec['C5_bursts_per_16tok'] = round(16 * len(bursts) / max(1, doc_end), 3)
                    rec['C5_library_bytes_272_bursts'] = 272 * len(bursts) + 2 * doc_end
                    nk = tok.encode(f" {meta['key']}", add_special_tokens=False) if meta.get('key') else []
                    pkn = next((i for i in range(doc_end) if ids[i:i + len(nk)] == nk), None) if nk else None
                    if pkn is not None:   # needle span = the inserted sentence around the key
                        rec['C5_burst_in_needle'] = int(any(pkn - 4 <= b0 <= pkn + len(nk) + 6 for b0 in bstart))
                    row_spans = None              # [start, end) per row; default = burst start + 24 tokens
                    if key_fn and key_fn_bursts:   # learned fetch: one key per burst row + its own query key
                        kf_info = {'bursts': bursts, 'query_pos': qk, 'doc_end': doc_end}
                        out_kf = key_fn(keys_doc, ids, kf_info)
                        kb, kq = out_kf[0], out_kf[1]
                        for fld in ('organ_hit', 'abstain', 'best_sim'):   # package fetch flags (Library)
                            if fld in kf_info:
                                v = kf_info[fld]; rec['F_' + fld] = round(float(v), 4) if fld == 'best_sim' else int(bool(v))
                        if len(out_kf) > 2 and out_kf[2] is not None:   # optional: the row's own span (field-aware rows)
                            row_spans = [(int(a0), min(doc_end, int(b1))) for a0, b1 in out_kf[2]]
                        bvec = torch.nn.functional.normalize(torch.as_tensor(kb).float(), dim=-1)
                        bsim = bvec @ torch.nn.functional.normalize(torch.as_tensor(kq).float(), dim=-1)
                    else:
                        bvec = torch.stack([docv[b0:b1].mean(0) for b0, b1 in bursts]); bvec = torch.nn.functional.normalize(bvec, dim=-1)
                        bsim = bvec @ q
                    border = bsim.topk(min(5, len(bursts))).indices.tolist(); btop_b = [bursts[i] for i in border]
                    rec['C5_burst_hit'] = int(any(in_right(pp) for pp in range(btop_b[0][0] - 1, btop_b[0][1])))   # old rule (continuity)
                    # span rule (A4): a hit = the answer token lies inside the stored row span [f, f + 24),
                    # i.e. what the re-read actually gets; C2 keeps its 6-token rule
                    def span_of(i):                # row i's stored span
                        return row_spans[i] if row_spans is not None else (bursts[i][0], min(doc_end, bursts[i][0] + 24))
                    def in_span_i(i):
                        a0, b1 = span_of(i); return any(ids[j] == ans for j in range(a0, b1))
                    def in_span(b0):               # kept for the per-firing rows below
                        return any(ids[j] == ans for j in range(b0, min(doc_end, b0 + 24)))
                    if rec.get('F_abstain'):
                        rec['F_abstain_correct'] = int(bool(meta.get('negative')))   # abstain is right only when nothing is stored
                    rec['C5_burst_hit_span'] = int(in_span_i(border[0]))
                    rec['C5_burst_top5_span'] = int(any(in_span_i(i) for i in border))
                    rec['C5_row_span_mode'] = 'key_fn' if row_spans is not None else 'fixed24'
                    if row_spans is not None:
                        rec['C5_row_len_mean'] = round(sum(b1 - a0 for a0, b1 in row_spans) / max(1, len(row_spans)), 1)
                        rec['C5_library_bytes_272_rows'] = 272 * len(row_spans) + 2 * doc_end
                    rec['C5_burst_maxsim'] = round(float(bsim.max()), 4)
                    # planner v0 on the top-5 BURST rows (L1 + planner): re-read each stored span [f, f+24)
                    # alone in front of the query, pick the answer with the largest top1-top2 margin
                    bspans = [ids[span_of(i)[0]:span_of(i)[1]] for i in border]
                    bbest = None; pc = []
                    lp0 = torch.log_softmax(lg[cand].float(), -1)        # state-only log p over the candidates (no row)
                    for rank, (sp, bi) in enumerate(zip(bspans, border)):
                        lgbf = R.last_logits(sp + ids[doc_end:]); lgb = lgbf[cand]
                        vb, ib = lgb.topk(2); mb = float(vb[0] - vb[1]); pa = cand[int(ib[0])]
                        free_tok = int(lgbf.argmax())                     # FREE generation: greedy next token over the whole vocab
                        lpb = torch.log_softmax(lgb.float(), -1)
                        gain = float(lpb[int(ib[0])] - lp0[int(ib[0])])   # Library's idea: gain of the row for ITS answer, prior cancelled
                        # per-candidate record for offline selectors (L1+P1): key sim, re-read answer,
                        # margin, whether that answer is right, whether the row truly holds the answer
                        pc.append({'rank': rank, 'sim': round(float(bsim[bi]), 5), 'answer': int(pa),
                                   'margin': round(mb, 4), 'gain': round(gain, 4), 'correct': int(pa == ans),
                                   'free': int(free_tok == ans), 'free_tok': free_tok,
                                   'row_has_answer': int(in_span_i(bi))})
                        if bbest is None or mb > bbest[0]:
                            bbest = (mb, pa)
                    rec['P_cands'] = pc
                    rec['P_free'] = pc[0]['free'] if pc else 0           # user (Nov 1): the answer must be GENERATED, not picked among candidates
                    if bspans:                                           # sentence-level free generation on the top row (Library's scorer)
                        gen = R.greedy(bspans[0] + ids[doc_end:])
                        txt = tok.decode(gen)
                        rec['P_free_text'] = txt
                        rec['P_free_sent'] = int(re.search(r'\b' + re.escape(str(meta['val'])) + r'\b', txt, re.I) is not None)
                    rec['A_free'] = int(int(lg.argmax()) == ans)        # state only, free
                    rec['P_margin_burst'] = int(bbest[1] == ans)
                    lgcb = R.last_logits([t for sp in bspans for t in sp] + ids[doc_end:])
                    rec['P_concat_burst'] = int(cand[int(lgcb[cand].argmax())] == ans)
                    rec['P_rereads_per_query'] = len(bspans)
                    gv = docv[gpos]; gs = gv @ q; gtop = [gpos[i] for i in gs.topk(min(5, len(gpos))).indices.tolist()]
                    rec['C5_rows'] = len(gpos); rec['C5_rows_per_event'] = round(len(gpos) / max(1, n_ev), 1)
                    rec['C5_hit'] = int(in_right(gtop[0])); rec['C5_top5'] = int(any(in_right(pp) for pp in gtop))
                    rec['C5_hit_span'] = int(any(ids[j] == ans for j in range(gtop[0], min(doc_end, gtop[0] + 24))))
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
            if a.device == 'cuda' and a.arch != 'pam':
                torch.cuda.empty_cache()                      # T++ at 32k fills the 4090; release between docs
            outf.write(json.dumps(rec) + '\n'); outf.flush()
            print(rec, flush=True)
            vecs = None


if __name__ == '__main__':
    main()
