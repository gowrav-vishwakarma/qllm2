"""The mind + library PACKAGE, one entry point (user, 2026-11-01: "v16 is a set of model and library; use the set").

    from v16.package import Package
    pkg = Package('<mind best_model.pt>', device='cuda')
    text, info = pkg.answer(session_text, question_text, mode='lenient')     # or mode='strict'

What it does (v14/PACKAGE.md, each step measured in v14/EXPERIMENTS_V14.md):
  1. READ the session once: the mind streams it into its fixed state (O(1) in length: 1.15 MB for b2p2); the library
     keeps the raw text (2 B / token). Cached per session text.
  2. FETCH a row for the question:
       text organs: token organ (exact last 4 tokens) -> word organ steps 1-2 (normalised n-grams) -> step 4 (a
       question word rare in the session AND in English);
       then (lenient only) F2: the 2 question tokens the mind finds most surprising, matched by max cosine against
       each session sentence's tokens in the mind's block-8 states.
     strict: abstain when the text organs miss (never-stored questions are then safe; words that differ are missed).
  3. ANSWER (v0.3, WRITE-IN 2026-11-05): the fetched row is streamed through the mind's own delta write right before the
     question and read PLAINLY (warm: no contrast); greedy, stop at "." / newline / end-of-turn, at most max_new tokens.
     The no-row contrast of v0-v0.2 (l + 4 (l - l_without_row), "A9") is retired: it cancels facts the state already
     holds (needle 0.998 -> 0.927; warm 1.000). A rule HELD in the state keeps its rule twin: apply / apply_value
     (R x K, the no-pack twin, alpha 4) and the protocol twin / fork (v14/a28_twin.py, a31_fork.py). v0.3.1: apply_value
     reads a pack placed ADJACENT to the query (A29 rewrite) warm too; the rule twin only for a pack held further back.
     A record row ("Result: job: K; f1: v1; ...") is first cut to the asked field's clause (A19, labelled
     "field-cut" in info; Mind's w1s showed a body window reader does not fix field selection).
info = {path: text0 / text1 / text2 / text4 / F2 / None, row: (start, end), row_text, abstain, field_cut, bytes_held}.

The question text is appended verbatim (the caller chooses the form, e.g. "\\n\\nQuestion: ...\\nAnswer:" or ChatML).
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'v16')):
    if p not in sys.path:
        sys.path.insert(0, p)
from bench import Reader  # noqa: E402
from v14.a6_reread import CKPT as B2P2, Mind  # noqa: E402
from v14.a13_manykey import capture, maxsim_scores, sentence_rows  # noqa: E402
from v14.package_fetch import field_cut, row_value, text_fetch  # noqa: E402

PACK_ADJ = 8                                 # apply_value: a pack ending <= 8 tokens before the query is "in view" (warm read)
CONTENT_BG = 3000                            # fetch_rows: a question word is CONTENT if seen <= 3,000 times in 4.2M background words


def _nbytes(st) -> int:
    """bytes held by a mind state (any nesting of lists / tuples / dicts of tensors), measured, not a per-body constant."""
    if torch.is_tensor(st):
        return st.numel() * st.element_size()
    if isinstance(st, dict):
        return sum(_nbytes(v) for v in st.values())
    if isinstance(st, (list, tuple)):
        return sum(_nbytes(v) for v in st)
    return 0


class Package:
    def __init__(self, ckpt: str = B2P2, device: str = 'cuda', max_new: int = 40, address_layer: int | None = None):
        self.mind = Mind(ckpt, torch.device(device))
        self.tok = self.mind.tok
        # v0 interface rules (no b2p2 constants): the address is the residual after block n_blocks // 2 (b2p2: 8 of 16);
        # the sentence end / stops come from the tokenizer; state bytes are measured from the state itself
        self.address_layer = len(self.mind.m.blocks) // 2 if address_layer is None else address_layer
        dot, nl = self.tok.encode('.', add_special_tokens=False)[0], self.tok.encode('\n', add_special_tokens=False)[0]
        import v14.a13_manykey as _a13
        import v14.package_fetch as _pf
        _a13.DOT = _pf.DOT = dot
        from v14.norm_organ import tok_strings
        _pf._S['strs'] = strs = tok_strings(self.tok)               # token texts from THIS tokenizer (row ends, stops)
        _pf._S.pop('nl', None)                                         # newline-token set rebuilt for this tokenizer
        self.dot = dot
        self._digit_start = [s_[:1].isdigit() for s_ in strs]         # a digit right after '.' continues a number
        self._digit_end = [s_[-1:].isdigit() for s_ in strs]
        self.R = Reader(self.mind.m, 'pam')
        self.max_new = max_new
        im_end = self.tok.convert_tokens_to_ids('<|im_end|>') if '<|im_end|>' in self.tok.get_vocab() else None
        self.stops = (dot, nl) + ((im_end,) if im_end is not None else ())
        self._session = None

    def address_bytes(self, n_tokens: int, form: str = 'int8', keep_frac: float = 1 / 16) -> int:
        """stored size of the F2 address for n_tokens, from the address width D (no body constant):
        int8 = D bytes + a fp32 scale per token; sal1 = sign bits (ceil(D / 8) B) + a uint32 position per KEPT token."""
        D = self._addr_dim()
        if form == 'int8':
            return n_tokens * (D + 4)
        return int(n_tokens * keep_frac * (-(-D // 8) + 4))

    def _addr_dim(self) -> int:
        if getattr(self, '_D', None) is None:
            caps = []
            h = self.mind.m.blocks[self.address_layer].register_forward_hook(
                lambda mod, i, o: caps.append((o[0] if isinstance(o, tuple) else o).data[0].shape[-1]))
            try:
                with torch.no_grad():
                    self.mind.m(torch.tensor([[self.stops[0]]], device=self.mind.dev))
            finally:
                h.remove()
            self._D = int(caps[0])
        return self._D

    def read(self, session_text, chunk: int = 2048):
        """stream the session into the mind's state once (chunks, state carried: O(1) memory in length). The library
        keeps its tokens and sentence rows; the ADDRESS (F2) keeps each token's block-8 state (fp16 on the device,
        D + 4 B / token at int8 when stored: 592 for b2p2) and its surprisal (for the sign-bit form on the surprising tokens)."""
        key = session_text if isinstance(session_text, str) else tuple(session_text)
        if self._session is not None and self._session['key'] == key:
            return self._session
        ids = self.tok.encode(session_text, add_special_tokens=False) if isinstance(session_text, str) else list(session_text)
        m = self.mind.m; dev = self.mind.dev
        caps, surp, prev = [], [], None
        hook = m.blocks[self.address_layer].register_forward_hook(lambda mod, i, o: caps.append((o[0] if isinstance(o, tuple) else o).data[0].half()))
        st, off = None, 0
        try:
            with torch.no_grad():
                for c0 in range(0, len(ids), chunk):
                    x = torch.tensor([ids[c0:c0 + chunk]], device=dev)
                    logits, st, _ = m(x, states=st, step_offset=c0)
                    lp = torch.log_softmax(logits[0].float(), -1)
                    tgt = x[0]
                    s_chunk = torch.empty(len(tgt), device=dev)
                    s_chunk[0] = -prev[tgt[0]] if prev is not None else 0.0
                    if len(tgt) > 1:
                        s_chunk[1:] = -lp[:-1].gather(1, tgt[1:, None])[:, 0]
                    surp.append(s_chunk); prev = lp[-1]
                    del logits, lp
        finally:
            hook.remove()
        off = len(ids)
        V = torch.cat(caps).float(); mu = V.mean(0)
        state_bytes = _nbytes(st)
        self._session = dict(key=key, text=session_text if isinstance(session_text, str) else None, ids=ids, state=st, state_bytes=state_bytes,
                             off=off, last_lp=prev, rows=sentence_rows(ids, len(ids)), V=V.half(), mu=mu,
                             surp=torch.cat(surp))
        return self._session

    @torch.no_grad()
    def _state(self, ids, dil=None):
        """the mind's state after ids, in ONE pass (so a dilation span (start, end, s, where) is in absolute positions)."""
        self.mind.m._dilate = dil
        try:
            # one pass only when a dilation span is set (it uses absolute positions); else 2k chunks with the state carried
            # (parity ~1e-6): a one-pass 32k session does not fit a 386M body on a 24 GB card
            _, st, off = self.mind.feed(ids, chunk=max(1, len(ids)) if dil is not None else 2048)
        finally:
            self.mind.m._dilate = None
        return st, off

    def apply(self, session, key_question: str, skill: dict, alpha: float | None = None, dilate: float | None = None,
              max_new: int = 4):
        """R x K (A23): apply a SKILL held in the session (its demonstrations, read earlier) to a FACT the library
        fetches. The planner fetches the key's row, takes the value out (row_value) and writes the skill's trained
        query line skill['query'].format(v=value); the mind answers with the SKILL contrast: the twin is the same
        session WITHOUT the skill's pack (a second full stream; 1.15 MB, 2x read compute). dilate = s: echo
        retention ** (1/s) from the pack's end to the session's end, in both streams. session: text or token ids;
        skill: {'pack': text or ids of the demonstrations as they appear in the session, 'query': e.g. "\\n{v} ->"}."""
        enc = lambda x: self.tok.encode(x, add_special_tokens=False) if isinstance(x, str) else list(x)  # noqa: E731
        S, pack = enc(session), enc(skill['pack'])
        p0 = next((i for i in range(len(S) - len(pack) + 1) if S[i:i + len(pack)] == pack), None)
        info = dict(skill_found=int(p0 is not None))
        kq = enc(key_question)
        row, finfo = text_fetch(S + kq, len(S), step4=True, bigram=False)
        info.update(path=f"text{finfo['step']}" if row is not None else None, row=tuple(row) if row else None)
        if row is None or p0 is None:
            return '', dict(info, abstain=1, value=None)
        row_text = self.tok.decode(S[row[0]:row[1]])
        v = row_value(row_text, self.tok.decode(kq))
        info.update(row_text=row_text, value=v)
        if v is None:
            return '', dict(info, abstain=1)
        text, vinfo = self.apply_value(S, skill, v, alpha=alpha, dilate=dilate, max_new=max_new)
        return text, dict(info, **vinfo)

    def apply_value(self, session, skill: dict, v: str, alpha: float | None = None, dilate: float | None = None, max_new: int = 4):
        """the second half of apply: the skill's query line for a GIVEN value. The read follows where the pack sits (v0.3.1,
        WRITE-IN): a pack ADJACENT to the query (at most PACK_ADJ tokens after it) is evidence in view -> the warm read
        (alpha 0); a pack held further back -> the rule twin (the same session without the pack, alpha 4). alpha given =
        forced."""
        enc = lambda x: self.tok.encode(x, add_special_tokens=False) if isinstance(x, str) else list(x)  # noqa: E731
        if isinstance(session, str) and isinstance(skill['pack'], str) and skill['pack'] in session:
            i = session.index(skill['pack'])                  # tokenise around the pack so its tokens match exactly
            pre, pack, post = enc(session[:i]), enc(skill['pack']), enc(session[i + len(skill['pack']):])
            S, p0 = pre + pack + post, len(pre)
        else:
            S, pack = enc(session), enc(skill['pack'])
            p0 = next((i for i in range(len(S) - len(pack) + 1) if S[i:i + len(pack)] == pack), None)
        if p0 is None:
            return '', dict(abstain=1, skill_found=0)
        info = {}
        q = enc(skill['query'].format(v=v))
        T = S[:p0] + S[p0 + len(pack):]                                       # the session without the skill pack
        dS = (p0 + len(pack), len(S), dilate, 'echo') if dilate else None
        dT = (p0, len(T), dilate, 'echo') if dilate else None
        gap = len(S) - (p0 + len(pack))
        if alpha is None:
            alpha = 0.0 if gap <= PACK_ADJ else 4.0
        stS, offS = self._state(S, dS)
        la, sa, oa = self.mind.feed(q, stS, offS)
        if alpha == 0.0:                                                      # warm: no twin needed
            out = self._gen(la, sa, oa, None, None, None, 0.0, max_new); twin_bytes = 0
        else:
            stT, offT = self._state(T, dT); lb, sb, ob = self.mind.feed(q, stT, offT); twin_bytes = _nbytes(stT)
            out = self._gen(la, sa, oa, lb, sb, ob, alpha, max_new)
        info.update(abstain=0, query_text=self.tok.decode(q), first_token=out[0] if out else None, pack_gap=gap,
                    read='warm' if alpha == 0.0 else 'rule twin',
                    bytes_held=dict(mind_state=_nbytes(stS), no_pack_twin_state=twin_bytes, library_text=2 * len(S)))
        return self.tok.decode(out), info

    @torch.no_grad()
    def _f2(self, q, form='full', keep_frac=1 / 16):
        """F2 from the stored address: the question continues the session state (its block-8 states and surprisal
        come from that pass); its 2 most surprising tokens (after the first 2) are matched by max cosine against each
        session sentence's tokens. form 'full' = every token; 'sal1' = sign bits of the top keep_frac surprising
        session tokens only (A15's 4.9 MB / 1M form)."""
        s = self._session; m = self.mind.m; dev = self.mind.dev
        caps = []
        hook = m.blocks[self.address_layer].register_forward_hook(lambda mod, i, o: caps.append((o[0] if isinstance(o, tuple) else o).data[0].float()))
        try:
            x = torch.tensor([q], device=dev)
            logits, _, _ = m(x, states=copy.deepcopy(s['state']), step_offset=s['off'])
        finally:
            hook.remove()
        lp = torch.log_softmax(logits[0].float(), -1)
        qs = torch.empty(len(q), device=dev)
        qs[0] = -s['last_lp'][q[0]]
        if len(q) > 1:
            qs[1:] = -lp[:-1].gather(1, x[0, 1:, None])[:, 0]
        qs[:2] = -1.0
        sal = qs.topk(min(2, len(q))).indices.tolist()
        Q = caps[0][sal] - s['mu']
        V = s['V'].float() - s['mu']
        rows = s['rows']
        if form == 'sal1':
            k = max(1, int(len(s['ids']) * keep_frac))
            keep = torch.zeros(len(s['ids']), dtype=torch.bool, device=dev); keep[s['surp'].topk(k).indices] = True
            Vn = torch.nn.functional.normalize(torch.sign(V), dim=-1); Qn = torch.nn.functional.normalize(torch.sign(Q), dim=-1)
            sims = Vn @ Qn.T
            sc = torch.stack([sims[a:b][keep[a:b]].max(0).values.sum() if bool(keep[a:b].any()) else torch.tensor(-1e9, device=dev)
                              for a, b in rows])
        else:
            sims = torch.nn.functional.normalize(V, dim=-1) @ torch.nn.functional.normalize(Q, dim=-1).T
            sc = torch.stack([sims[a:b].max(0).values.sum() for a, b in rows])
        return rows[int(sc.argmax())], [self.tok.decode([q[i]]) for i in sal]

    @torch.no_grad()
    def _gen(self, la, sa, oa, lb, sb, ob, alpha, max_new):
        """greedy over l_a + alpha (l_a - l_b), both streams fed the output; lb = None is the plain read of stream a
        alone (the warm read). Stops at '.', newline or end of turn; a '.' right after a digit is held one token: if the
        next token starts with a digit ('12' '.' '3') the number goes on, else the '.' was the sentence end and the
        look-ahead token is dropped."""
        out, held = [], False
        for _ in range(max_new):
            t = int((la if lb is None else la + alpha * (la - lb)).argmax())
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
            la, sa, oa = self.mind.feed([t], sa, oa)
            if lb is not None:
                lb, sb, ob = self.mind.feed([t], sb, ob)
        return out

    @torch.no_grad()
    def _decode(self, prefix):
        """the warm read: row + question streamed on the session state, greedy, no contrast (v0.3)."""
        s = self._session
        la, sa, oa = self.mind.feed(prefix, copy.deepcopy(s['state']), s['off'])
        return self._gen(la, sa, oa, None, None, None, 0.0, self.max_new)

    def fetch_rows(self, session_text, question_text: str, k: int = 3, mode: str = 'lenient', address: str = 'full'):
        """Top-k rows for a question, each with its fetch path (BABILong QA2 / QA3 need 2-3 facts). Row 1: the usual
        chain (token organ -> word organ steps 1-2 -> step 4 -> F2 in lenient mode). Then HOPS by name (library-side
        multi-hop retrieval, plumbing): every capitalised name in a fetched row that is not in the question brings the
        LATEST session sentence mentioning it ("John took the milk." -> "John went to the kitchen."). Returns rows in
        session order: [(start, end, path, text)]."""
        import re
        s = self.read(session_text)
        base, end = s['ids'], len(s['ids'])
        q = self.tok.encode(question_text, add_special_tokens=False)
        row, finfo = text_fetch(base + q, end, step4=True, bigram=False)
        path = f"text{finfo['step']}" if row is not None else None
        texts = s.setdefault('row_texts', [self.tok.decode(base[a:b]) for a, b in s['rows']])
        if row is None:                                   # latest mention of a NAME in the question (mid-sentence capital)
            names = [m.group(0) for m in re.finditer(r'(?<![.?!:]\s)(?<!^)\b[A-Z][a-z]+\b', question_text.strip())]
            hits = [i for i, t in enumerate(texts) if any(re.search(r'\b' + n + r'\b', t) for n in names)]
            if hits:
                row, path = s['rows'][hits[-1]], 'name'
        if row is None:                                   # latest mention of the question's rarest content word
            from v14.package_fetch import _bg
            from v14.norm_organ import norm
            bg = _bg()
            words = [w for w in re.findall(r'[A-Za-z]+', question_text) if norm(w) and bg.get(norm(w), 0) <= CONTENT_BG]
            words = sorted(set(words), key=lambda w: bg.get(norm(w), 0))
            for w in words:
                hits = [i for i, t in enumerate(texts) if re.search(r'\b' + re.escape(w) + r'\b', t, re.I)]
                if hits:
                    row, path = s['rows'][hits[-1]], f'latest:{w}'
                    break
        if row is None and mode == 'lenient':
            row, _ = self._f2(q, form=address); path = 'F2'
        if row is None:
            return []
        got = [(row[0], row[1], path, self.tok.decode(base[row[0]:row[1]]))]
        qnames = set(re.findall(r'\b[A-Z][a-z]+\b', question_text))
        frontier = [w for w in re.findall(r'\b[A-Z][a-z]+\b', got[0][3]) if w not in qnames]
        seen = set(qnames)
        while frontier and len(got) < k:
            name = frontier.pop(0)
            if name in seen:
                continue
            seen.add(name)
            hits = [i for i, t in enumerate(texts) if re.search(r'\b' + name + r'\b', t)
                    and not any(s['rows'][i][0] < g[1] and g[0] < s['rows'][i][1] for g in got)]
            if hits:
                a, b = s['rows'][hits[-1]]
                t = texts[hits[-1]]
                got.append((a, b, f'hop:{name}', t))
                frontier += [w for w in re.findall(r'\b[A-Z][a-z]+\b', t) if w not in seen]
        return sorted(got)

    def answer_chain(self, session_text, question: str, prompt: str = "\n\nQuestion: {q}\nAnswer:", k: int = 3,
                     mode: str = 'lenient'):
        """PLANNER CHAIN (labelled TEXT: the composition is the library's, not the mind's; R x K's substitution rule).
        If the first fetched row was found by the question's object word ("latest:milk") and names a holder that is not
        in the question ("John took the milk there."), the planner asks the SUB-question with the holder in place of
        the object ("Where is John?") and the mind answers it from the holder's latest row; otherwise the plain
        answer_rows. Returns (text, info) with info['chain'] = {holder, object, sub_question} or None."""
        import re
        rows = self.fetch_rows(session_text, question, k=k, mode=mode)
        first = next((r for r in rows if not r[2].startswith('hop:')), None)
        chain = None
        if first is not None and first[2].startswith('latest:'):
            w = first[2].split(':', 1)[1]
            qnames = set(re.findall(r'\b[A-Z][a-z]+\b', question))
            holders = [n for n in re.findall(r'\b[A-Z][a-z]+\b', first[3]) if n not in qnames]
            if holders:
                sub = re.sub(r'\b(the\s+)?' + re.escape(w) + r'\b', holders[0], question, count=1, flags=re.I)
                chain = dict(holder=holders[0], object=w, sub_question=sub)
        if chain is None:
            text, info = self.answer_rows(session_text, prompt.format(q=question), k=k, mode=mode)
            return text, dict(info, chain=None)
        text, info = self.answer_rows(session_text, prompt.format(q=chain['sub_question']), k=1, mode=mode)
        return text, dict(info, chain=chain)

    def answer_rows(self, session_text, question_text: str, k: int = 3, mode: str = 'lenient'):
        """read the top-k rows (session order) + the question on the session state, with the no-row contrast."""
        rows = self.fetch_rows(session_text, question_text, k=k, mode=mode)
        if not rows:
            return '', dict(rows=[], abstain=1)
        s = self._session
        q = self.tok.encode(question_text, add_special_tokens=False)
        span = [t for a, b, _, _ in rows for t in s['ids'][a:b]]
        text = self.tok.decode(self._decode(span + q))
        return text, dict(rows=[(a, b, p, t[-90:]) for a, b, p, t in rows], abstain=0)

    def answer(self, session_text, question_text: str, mode: str = 'lenient', cut_fields: bool = True,
               address: str = 'full'):
        s = self.read(session_text)
        base, end = s['ids'], len(s['ids'])
        q = self.tok.encode(question_text, add_special_tokens=False)
        ids = base + q
        row, finfo = text_fetch(ids, end, step4=True, bigram=False)
        path, salient = (f"text{finfo['step']}" if row is not None else None), None
        if row is None and mode == 'lenient':
            row, salient = self._f2(q, form=address); path = 'F2'
        info = dict(path=path, row=tuple(row) if row else None, abstain=int(row is None), salient=salient,
                    bytes_held=dict(mind_state=s['state_bytes'], library_text=2 * end,
                                    word_token_indexes_est=int(9.4 * end),
                                    address_int8_all=self.address_bytes(end, 'int8'), address_sal1=self.address_bytes(end, 'sal1')))
        if row is None:
            return '', dict(info, row_text=None)
        span = base[row[0]:row[1]]
        cut = field_cut(self.tok.decode(span), question_text) if cut_fields else None   # A19: a record row -> the asked field
        if cut is not None:
            span = self.tok.encode(cut, add_special_tokens=False)
        text = self.tok.decode(self._decode(span + q))
        return text, dict(info, row_text=self.tok.decode(span), field_cut=int(cut is not None))


def _story(argv=None):
    """run the package on v16/story_eval.py's story prompts and questions; markdown report in logs/."""
    import argparse
    import subprocess
    import time
    from story_eval import IM_END, IM_START, QUESTIONS, build_prompt
    from memory_probes.spread import filler_tokens
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=B2P2); ap.add_argument('--lengths', default='2048,4096,8192')
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--device', default='cuda')
    ap.add_argument('--forms', default='chat,plain'); ap.add_argument('--mode', default='lenient')
    a = ap.parse_args(argv)
    pkg = Package(a.ckpt, a.device)
    fill = filler_tokens(str(ROOT / '.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt'))
    name = Path(a.ckpt).parent.name
    h = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    rep = Path('/home/gowrav/Development/qllm2/logs') / f"v16_story_package_{name}_{h}_{time.strftime('%Y%m%d_%H%M')}.md"
    out = [f"# Story comprehension: the PACKAGE ({name}, mode {a.mode}), seed {a.seed}\n",
           "v16/package.py: read the session into the mind's state; fetch by text (token / word organ / rare word) then "
           "the mind's surprise keys (F2); answer = the mind reading row + question on its state, no-row contrast. "
           "Keyword scorer as v16/story_eval.py.\n"]
    for L in [int(x) for x in a.lengths.split(',')]:
        pf = ROOT / 'v16' / 'prompts' / f'story_L{L}_s{a.seed}.txt'
        text = pf.read_text() if pf.exists() else build_prompt(pkg.tok, fill, L, a.seed)
        out.append(f"\n## L = {L} ({len(pkg.tok.encode(text, add_special_tokens=False))} tokens, {pf.name})\n\n"
                   "| question | form | expected | path | fetched row | package answer |\n|---|---|---|---|---|---|")
        hits = 0
        for qname, q, exp in QUESTIONS:
            for form in a.forms.split(','):
                qt = f"\n\n{IM_START}user\n{q}{IM_END}\n{IM_START}assistant\n" if form == 'chat' else f"\n\nQuestion: {q}\nAnswer:"
                ans, info = pkg.answer(text, qt, mode=a.mode)
                ok = exp.lower() in ans.lower(); hits += int(ok)
                row = (info['row_text'] or '').replace('\n', ' ').replace('|', '/').strip()[-80:]
                out.append(f"| {qname} | {form} | {exp} | {info['path']} | `{row}` | {'OK' if ok else '--'} "
                           f"`{ans.replace(chr(10), ' / ').replace('|', '/').strip()[:90]}` |")
                print(f"{name} L={L} {qname:13s} {form:5s} {str(info['path']):5s} {'OK' if ok else '--'} {ans[:70]!r}", flush=True)
        n = len(QUESTIONS) * len(a.forms.split(','))
        out.append(f"\nkeyword hits: {hits} / {n}; bytes held: mind state 1.15 MB + library text {2 * len(pkg.read(text)['ids']) / 1e3:.0f} kB")
        print(f"{name} L={L} package hits {hits} / {n}", flush=True)
        rep.write_text('\n'.join(out) + '\n')
    print('report', rep)


if __name__ == '__main__':
    _story()
