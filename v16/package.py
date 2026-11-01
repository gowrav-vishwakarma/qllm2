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
  3. ANSWER: the mind reads row + question on its own session state, with the no-row contrast on the answer tokens
     only (l + 4 (l - l_without_row)); greedy, stop at "." / newline / end-of-turn, at most max_new tokens.
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

STOP_TOKENS = (13, 198)                       # "." and newline
STATE_BYTES = {'b2p2': 1154832}               # the mind's fixed state, fp32 at decode (M2 body)


class Package:
    def __init__(self, ckpt: str = B2P2, device: str = 'cuda', alpha: float = 4.0, max_new: int = 40):
        self.mind = Mind(ckpt, torch.device(device))
        self.tok = self.mind.tok
        self.R = Reader(self.mind.m, 'pam')
        self.alpha, self.max_new = alpha, max_new
        im_end = self.tok.convert_tokens_to_ids('<|im_end|>') if '<|im_end|>' in self.tok.get_vocab() else None
        self.stops = STOP_TOKENS + ((im_end,) if im_end is not None else ())
        self._session = None

    def read(self, session_text, chunk: int = 2048):
        """stream the session into the mind's state once (chunks, state carried: O(1) memory in length). The library
        keeps its tokens and sentence rows; the ADDRESS (F2) keeps each token's block-8 state (fp16 on the device,
        592 B / token at int8 when stored) and its surprisal (for the sign-bit form on the surprising tokens)."""
        key = session_text if isinstance(session_text, str) else tuple(session_text)
        if self._session is not None and self._session['key'] == key:
            return self._session
        ids = self.tok.encode(session_text, add_special_tokens=False) if isinstance(session_text, str) else list(session_text)
        m = self.mind.m; dev = self.mind.dev
        caps, surp, prev = [], [], None
        hook = m.blocks[8].register_forward_hook(lambda mod, i, o: caps.append((o[0] if isinstance(o, tuple) else o).data[0].half()))
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
        self._session = dict(key=key, text=session_text if isinstance(session_text, str) else None, ids=ids, state=st,
                             off=off, last_lp=prev, rows=sentence_rows(ids, len(ids)), V=V.half(), mu=mu,
                             surp=torch.cat(surp))
        return self._session

    @torch.no_grad()
    def _state(self, ids, dil=None):
        """the mind's state after ids, in ONE pass (so a dilation span (start, end, s, where) is in absolute positions)."""
        self.mind.m._dilate = dil
        try:
            _, st, off = self.mind.feed(ids, chunk=max(1, len(ids)))
        finally:
            self.mind.m._dilate = None
        return st, off

    def apply(self, session, key_question: str, skill: dict, alpha: float = 4.0, dilate: float | None = None,
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

    def apply_value(self, session, skill: dict, v: str, alpha: float = 4.0, dilate: float | None = None, max_new: int = 4):
        """the second half of apply: the skill's query line for a GIVEN value, answered with the skill contrast."""
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
        stS, offS = self._state(S, dS); stT, offT = self._state(T, dT)
        la, sa, oa = self.mind.feed(q, stS, offS); lb, sb, ob = self.mind.feed(q, stT, offT)
        out = []
        for _ in range(max_new):
            t = int((la + alpha * (la - lb)).argmax()); out.append(t)
            if t in self.stops:
                break
            la, sa, oa = self.mind.feed([t], sa, oa); lb, sb, ob = self.mind.feed([t], sb, ob)
        info.update(abstain=0, query_text=self.tok.decode(q), first_token=out[0],
                    bytes_held=dict(mind_state=STATE_BYTES['b2p2'], no_pack_twin_state=STATE_BYTES['b2p2'], library_text=2 * len(S)))
        return self.tok.decode(out), info

    @torch.no_grad()
    def _f2(self, q, form='full', keep_frac=1 / 16):
        """F2 from the stored address: the question continues the session state (its block-8 states and surprisal
        come from that pass); its 2 most surprising tokens (after the first 2) are matched by max cosine against each
        session sentence's tokens. form 'full' = every token; 'sal1' = sign bits of the top keep_frac surprising
        session tokens only (A15's 4.9 MB / 1M form)."""
        s = self._session; m = self.mind.m; dev = self.mind.dev
        caps = []
        hook = m.blocks[8].register_forward_hook(lambda mod, i, o: caps.append((o[0] if isinstance(o, tuple) else o).data[0].float()))
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
    def _decode(self, prefix, twin):
        s = self._session
        la, sa, oa = self.mind.feed(prefix, copy.deepcopy(s['state']), s['off'])
        lb, sb, ob = self.mind.feed(twin, copy.deepcopy(s['state']), s['off'])
        out = []
        for _ in range(self.max_new):
            t = int((la + self.alpha * (la - lb)).argmax()); out.append(t)
            if t in self.stops:
                break
            la, sa, oa = self.mind.feed([t], sa, oa)
            lb, sb, ob = self.mind.feed([t], sb, ob)
        return out

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
                    bytes_held=dict(mind_state=STATE_BYTES.get('b2p2'), library_text=2 * end,
                                    word_token_indexes_est=int(9.4 * end),
                                    address_int8_all=592 * end, address_sal1=int(78 * end / 16)))
        if row is None:
            return '', dict(info, row_text=None)
        span = base[row[0]:row[1]]
        cut = field_cut(self.tok.decode(span), question_text) if cut_fields else None   # A19: a record row -> the asked field
        if cut is not None:
            span = self.tok.encode(cut, add_special_tokens=False)
        text = self.tok.decode(self._decode(span + q, q))
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
