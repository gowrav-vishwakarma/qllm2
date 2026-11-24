"""The mind + library PACKAGE for SCI-1, standalone (torch + transformers + modeling_pam2.py only).

A port of the research repo's v16/package.py (v0.3.1), the `answer()` path only. Every function below is a copy of the
repo function named in its docstring; the answers are checked equal to the repo's on the same weights (PACKAGE.md).

    from modeling_pam2 import load_model, load_tokenizer
    from package_lite import Package
    pkg = Package(load_model('model.pt'), load_tokenizer())
    text, info = pkg.answer(session_text, "\\n\\nQuestion: Where did Mary go?\\nAnswer:")    # the package
    text = pkg.bare(session_text, "\\n\\nQuestion: Where did Mary go?\\nAnswer:")           # the model alone

What the package does:
  1. READ the session once. The mind streams it into its fixed state (O(1) in length: 3.67 MB for SCI-1, whatever the
     length). The library keeps the session's token text (2 B per token) and, per token, the mind's hidden vector after
     block n_layers // 2 (the "address", used only by the F2 step below).
  2. FETCH one sentence ("row") for the question, by text first:
       token organ: the question's last 4 tokens, latest exact match in the session;
       word organ: normalised words (lower case, a / an / the dropped, one plural "s" stripped): the longest question
       suffix n-gram (6..3 words), then any question n-gram, then (step 4) a question word seen 1-2 times in the session
       AND <= 3 times in 4.2M words of background English (bg_wordcounts_fineweb014.json: a name or a made-up key);
     then, in mode 'lenient' only, F2: the 2 question tokens the mind finds most surprising are matched by max cosine
     against each session sentence's tokens in the address. mode 'strict' abstains when the text organs miss.
     A record row "Result: job: K; f1: v1; ..." is cut to the field the question names.
  3. ANSWER: the fetched row, then the question, are streamed on a COPY of the session state, and the mind answers
     greedily (stop at a sentence end, a newline or <|im_end|>; a "." between digits does not stop).
  BARE (the model alone) = the session state + the question, the same greedy decode, no fetch.

Not ported (they stay in the research repo): R x K (a rule held in state applied to a fetched value), the rule twin /
state fork, BABILong's multi-row hops and the TEXT planner chain.
"""
from __future__ import annotations

import bisect
import json
import os
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
BG_PATH = HERE / 'bg_wordcounts_fineweb014.json'
NGRAM, MAX_ROW = 4, 64                 # token organ n-gram; longest row in tokens
NMIN, NMAX, RARE = 3, 6, 2             # word organ: n-gram range; "rare in the session" = seen <= 2 times
BG_MAX = 3                             # step 4: "rare in English" = seen <= 3 times in the background counts
CHUNK = 2048                           # the session is streamed in 2,048-token chunks (state carried)
ART = {'a', 'an', 'the'}
WORD = re.compile(r'[A-Za-z0-9]+')
_TOOL_ROW = re.compile(r'Result: job: (.+?); (.+?)\.')


# ── text organs (repo v14/norm_organ.py and v14/package_fetch.py) ────────────────────────────────────────────────────


def norm(w: str):
    """norm_organ.norm: lower case, articles dropped, one trailing "s" stripped from words of >= 4 letters."""
    w = w.lower()
    if w in ART:
        return None
    if len(w) >= 4 and w.endswith('s'):
        w = w[:-1]
    return w


def words(strs, ids, lo, hi):
    """norm_organ.words: [(normalised word, first token, last token)] for tokens [lo, hi)."""
    offs, parts, n = [], [], 0
    for j in range(lo, hi):
        s = strs[int(ids[j])]
        offs.append(n); parts.append(s); n += len(s)
    out = []
    for m in WORD.finditer(''.join(parts)):
        w = norm(m.group())
        if w is not None:
            out.append((w, lo + bisect.bisect_right(offs, m.start()) - 1, lo + bisect.bisect_right(offs, m.end() - 1) - 1))
    return out


def _latest(sw, grams, n):
    best = None
    for i in range(len(sw) - n + 1):
        g = tuple(sw[i:i + n])
        if g in grams:
            best = (i, g)
    return best


def find(stream, query, bg=None, bg_max=BG_MAX, bigram=True):
    """norm_organ.find: (step, n, first token, last token + 1) of the latest word-organ match, or None."""
    sw = [w for w, _, _ in stream]
    qw = [w for w, _, _ in query]
    hit = None
    for n in range(NMAX, NMIN - 1, -1):                          # (1) suffix
        if len(qw) >= n:
            r = _latest(sw, {tuple(qw[-n:])}, n)
            if r:
                hit = (1, n, r[0]); break
    if hit is None:
        for n in range(NMAX, NMIN - 1, -1):                      # (2) any position
            r = _latest(sw, {tuple(qw[i:i + n]) for i in range(len(qw) - n + 1)}, n)
            if r:
                hit = (2, n, r[0]); break
    if hit is None and bg is not None:                           # (4) rare in the session AND rare in English
        cnt = Counter(sw)
        cands = [(bg.get(w, 0), cnt[w], -i, w) for i, w in enumerate(qw) if 1 <= cnt[w] <= RARE and bg.get(w, 0) <= bg_max]
        if cands:
            w = min(cands)[3]
            last = max(i for i, x in enumerate(sw) if x == w)
            hit = (4, 1, last)
    if hit is None and bigram:                                   # (3) rare-word bigram (off in the package)
        cnt = Counter(sw)
        grams = {tuple(qw[i:i + 2]) for i in range(len(qw) - 1)}
        grams = {g for g in grams if min(cnt[g[0]], cnt[g[1]]) <= RARE}
        r = _latest(sw, grams, 2)
        if r:
            hit = (3, 2, r[0])
    if hit is None:
        return None
    step, n, i = hit
    return step, n, stream[i][1], stream[i + n - 1][2] + 1


class TextLibrary:
    """the text side of the package for one tokenizer (repo v14/package_fetch.py: is_end, sentence_end, row_around,
    organ, norm_hit, text_fetch, field_cut; v14/a13_manykey.sentence_rows)."""

    def __init__(self, tok, bg_path=BG_PATH):
        self.tok = tok
        self.strs = [tok.decode([i]) for i in range(len(tok))]     # norm_organ.tok_strings
        self.nl = frozenset(i for i, t in enumerate(self.strs) if '\n' in t)
        self.dot = tok.encode('.', add_special_tokens=False)[0]
        self._bg_path = bg_path
        self._bg = None

    def bg(self):
        if self._bg is None:
            with open(os.environ.get('PACKAGE_BG', self._bg_path)) as f:
                self._bg = json.load(f)
        return self._bg

    def is_end(self, ids, j, end) -> bool:
        """a newline token ends a row; a '.' ends a sentence unless it is a decimal point ('12' '.' '3')."""
        if ids[j] in self.nl:
            return True
        if ids[j] != self.dot:
            return False
        if j + 1 >= end or j == 0:
            return True
        a, b = ids[j - 1], ids[j + 1]
        s = self.strs
        return not (a < len(s) and b < len(s) and s[a][-1:].isdigit() and s[b][:1].isdigit())

    def sentence_end(self, ids, start, end):
        for j in range(start, min(end, start + MAX_ROW)):
            if self.is_end(ids, j, end):
                return j + 1
        return min(end, start + MAX_ROW)

    def row_around(self, ids, a, b, end):
        """the sentence holding tokens [a, b): from after the previous end (<= 64 back) to the next end at or after b."""
        s0 = max(0, a - 16)
        for j in range(a - 1, max(0, a - 64), -1):
            if self.is_end(ids, j, end):
                s0 = j + 1
                break
        return s0, max(a + 1, self.sentence_end(ids, b, end))

    def organ(self, ids, doc_end, qend):
        """the token after the LATEST exact match of the question's last NGRAM tokens."""
        g = tuple(ids[qend - NGRAM:qend])
        hit = None
        for i in range(NGRAM, doc_end):
            if tuple(ids[i - NGRAM:i]) == g:
                hit = i
        return hit

    def norm_hit(self, ids, end, q0, step4=False, bigram=True):
        h = find(words(self.strs, ids, 0, end), words(self.strs, ids, q0, len(ids)),
                 bg=self.bg() if step4 else None, bigram=bigram)
        if h is None:
            return None
        step, n, a, b = h
        return (step, n) + self.row_around(ids, a, b, end)

    def text_fetch(self, ids, doc_end, step4=True, bigram=False):
        """ids = session + question, doc_end = where the question starts -> (row (start, end) or None, info)."""
        end = int(doc_end)
        hit = self.organ(ids, end, len(ids))
        if hit is not None:
            return self.row_around(ids, hit, hit, end), dict(organ_hit=1, step=0, abstain=0)
        nh = self.norm_hit(ids, end, end, step4=step4, bigram=bigram)
        if nh:
            return tuple(nh[2:]), dict(organ_hit=1, step=int(nh[0]), abstain=0)
        return None, dict(organ_hit=0, step=None, abstain=1)

    def sentence_rows(self, ids, end):
        """every sentence of the session as a row (long sentences: 64-token pieces from the end; rows < 4 tokens dropped)."""
        rows, s0 = [], 0
        for j in range(end):
            if self.is_end(ids, j, end) or j == end - 1:
                a, e = s0, j + 1
                while e - a > 64:
                    rows.append((e - 64, e)); e -= 64
                if e - a >= 4:
                    rows.append((a, e))
                s0 = j + 1
        return rows


def field_cut(row_text, q_text):
    """a tool row "Result: job: K; f1: v1; ...; fn: vn." -> " Result: job: K; f: v." for the field the question names."""
    m = _TOOL_ROW.search(row_text)
    if not m:
        return None
    fields = [f.split(': ', 1) for f in m.group(2).split('; ') if ': ' in f]
    qwords = re.findall(r"[A-Za-z]+", q_text.lower())
    hits = [(qwords.index(f.lower()) if f.lower() in qwords else -1, f, v) for f, v in fields]
    hits = [h for h in hits if h[0] >= 0]
    if not hits:
        return None
    _, f, v = max(hits)
    return f" Result: job: {m.group(1)}; {f}: {v}."


# ── the package (repo v16/package.py, v0.3.1) ───────────────────────────────────────────────────────────────────────


class Package:
    def __init__(self, model, tok, max_new: int = 40, address_layer: int | None = None, bg_path=BG_PATH):
        self.m, self.tok = model, tok
        self.dev = next(model.parameters()).device
        self.lib = TextLibrary(tok, bg_path)
        self.address_layer = len(model.blocks) // 2 if address_layer is None else address_layer
        self.max_new = max_new
        self.dot = self.lib.dot
        nl = tok.encode('\n', add_special_tokens=False)[0]
        im_end = tok.convert_tokens_to_ids('<|im_end|>') if '<|im_end|>' in tok.get_vocab() else None
        self.stops = (self.dot, nl) + ((im_end,) if im_end is not None else ())
        self._digit_start = [s[:1].isdigit() for s in self.lib.strs]
        self._digit_end = [s[-1:].isdigit() for s in self.lib.strs]
        self._session = None

    # -- the mind ------------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def _run(self, ids, state, capture=None):
        """modeling_pam2.PAM2LM._run (chunked form) with the residual after block `address_layer` kept in `capture`."""
        m = self.m; cfg = m.config
        x = m.embed_norm(m.embed(ids))
        keep = (ids != cfg.reset_token_id).float() if cfg.reset_at_eos else None
        new_layers, shared = [], None
        for i, blk in enumerate(m.blocks):
            x = x + blk.cgu(blk.norm1(x)) * blk.cgu_scale
            if blk.pam is not None:
                out, shared, st_new = blk.pam.forward(blk.norm2(x), keep, state.layers[i] if state is not None else None)
                x = x + out * blk.pam_scale
                new_layers.append(st_new)
            else:
                if blk.reader is not None:
                    x = x + blk.reader(blk.norm2(x), shared)
                new_layers.append(None)
            if capture is not None and i == self.address_layer:
                capture.append(x[0])
        hidden = m.output_norm(x)
        lm = m.lm_head_norm(m.lm_head_proj(hidden))
        echo_out, echo_state = m._echo(hidden, ids, state.echo if state is not None else None, False)
        logits = F.linear(lm + echo_out, m.embed.embed.weight)
        n0 = state.n_tokens if state is not None else 0
        from modeling_pam2 import PAM2State
        return logits, PAM2State(new_layers, echo_state, n0 + ids.shape[1])

    @torch.no_grad()
    def feed(self, ids, state=None, chunk: int = CHUNK):
        """stream ids on a state -> (last logits fp32, new state). The state passed in is not changed."""
        lg = None
        for s in range(0, len(ids), chunk):
            logits, state = self.m(torch.tensor([ids[s:s + chunk]], device=self.dev), state)
            lg = logits[0, -1].float()
        return lg, state

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False) if isinstance(text, str) else list(text)

    @torch.no_grad()
    def read(self, session, chunk: int = CHUNK):
        """stream the session into the state once (cached per session); keep its tokens, rows, address and surprisal."""
        key = session if isinstance(session, str) else tuple(session)
        if self._session is not None and self._session['key'] == key:
            return self._session
        ids = self.encode(session)
        caps, surp, prev, st = [], [], None, None
        for c0 in range(0, len(ids), chunk):
            x = torch.tensor([ids[c0:c0 + chunk]], device=self.dev)
            cap = []
            logits, st = self._run(x, st, cap)
            caps.append(cap[0].half())
            lp = torch.log_softmax(logits[0].float(), -1)
            tgt = x[0]
            s_chunk = torch.empty(len(tgt), device=self.dev)
            s_chunk[0] = -prev[tgt[0]] if prev is not None else 0.0
            if len(tgt) > 1:
                s_chunk[1:] = -lp[:-1].gather(1, tgt[1:, None])[:, 0]
            surp.append(s_chunk); prev = lp[-1]
            del logits, lp
        V = torch.cat(caps).float(); mu = V.mean(0)
        self._session = dict(key=key, ids=ids, state=st, state_bytes=st.nbytes(), last_lp=prev,
                             rows=self.lib.sentence_rows(ids, len(ids)), V=V.half(), mu=mu, surp=torch.cat(surp))
        return self._session

    @torch.no_grad()
    def _gen(self, la, sa, max_new):
        """greedy from logits la / state sa; stops at '.', newline or <|im_end|>; a '.' right after a digit is held one
        token: if the next token starts with a digit the number goes on, else the look-ahead token is dropped."""
        out, held = [], False
        for _ in range(max_new):
            t = int(la.argmax())
            if held:
                if not (t < len(self._digit_start) and self._digit_start[t]):
                    break
                held = False
            out.append(t)
            if t in self.stops:
                if t == self.dot and len(out) >= 2 and out[-2] < len(self._digit_end) and self._digit_end[out[-2]]:
                    held = True
                else:
                    break
            la, sa = self.feed([t], sa)
        return out

    @torch.no_grad()
    def _f2(self, q):
        """F2: the question's 2 most surprising tokens (after its first 2), matched by max cosine against each session
        sentence's tokens in the address (the residual after block address_layer, centred on the session mean)."""
        s = self._session
        cap = []
        x = torch.tensor([q], device=self.dev)
        logits, _ = self._run(x, s['state'].clone(), cap)
        lp = torch.log_softmax(logits[0].float(), -1)
        qs = torch.empty(len(q), device=self.dev)
        qs[0] = -s['last_lp'][q[0]]
        if len(q) > 1:
            qs[1:] = -lp[:-1].gather(1, x[0, 1:, None])[:, 0]
        qs[:2] = -1.0
        sal = qs.topk(min(2, len(q))).indices.tolist()
        Q = cap[0].float()[sal] - s['mu']
        V = s['V'].float() - s['mu']
        sims = F.normalize(V, dim=-1) @ F.normalize(Q, dim=-1).T
        sc = torch.stack([sims[a:b].max(0).values.sum() for a, b in s['rows']])
        return s['rows'][int(sc.argmax())], [self.tok.decode([q[i]]) for i in sal]

    # -- the two entry points ------------------------------------------------------------------------------------------

    def bytes_held(self, n_tokens: int) -> dict:
        D = self.m.config.dim
        return dict(mind_state=self._session['state_bytes'], library_text=2 * n_tokens,
                    word_token_indexes_est=int(9.4 * n_tokens), address_int8=n_tokens * (D + 4))

    @torch.no_grad()
    def bare(self, session, question_text: str, max_new: int | None = None):
        """the model alone: the session state + the question, greedy; no fetch."""
        s = self.read(session)
        la, sa = self.feed(self.encode(question_text), s['state'].clone())
        return self.tok.decode(self._gen(la, sa, max_new or self.max_new))

    @torch.no_grad()
    def answer(self, session, question_text: str, mode: str = 'lenient', cut_fields: bool = True):
        """the package: fetch a row for the question, stream row + question on a copy of the session state, answer.
        Returns (text, info); info = {path: text0 / text1 / text2 / text4 / F2 / None, row, row_text, abstain, bytes_held}."""
        s = self.read(session)
        base, end = s['ids'], len(s['ids'])
        q = self.encode(question_text)
        row, finfo = self.lib.text_fetch(base + q, end)
        path, salient = (f"text{finfo['step']}" if row is not None else None), None
        if row is None and mode == 'lenient':
            row, salient = self._f2(q); path = 'F2'
        info = dict(path=path, row=tuple(row) if row else None, abstain=int(row is None), salient=salient,
                    bytes_held=self.bytes_held(end))
        if row is None:
            return '', dict(info, row_text=None)
        span = base[row[0]:row[1]]
        cut = field_cut(self.tok.decode(span), question_text) if cut_fields else None
        if cut is not None:
            span = self.encode(cut)
        la, sa = self.feed(span + q, s['state'].clone())
        text = self.tok.decode(self._gen(la, sa, self.max_new))
        return text, dict(info, row_text=self.tok.decode(span), field_cut=int(cut is not None))
