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
from v14.package_fetch import field_cut, text_fetch  # noqa: E402

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

    def read(self, session_text: str):
        """stream the session into the mind's state once; the library keeps its tokens and sentence rows."""
        if self._session is None or self._session['text'] != session_text:
            ids = self.tok.encode(session_text, add_special_tokens=False)
            _, st, off = self.mind.feed(ids)
            self._session = dict(text=session_text, ids=ids, state=st, off=off, rows=sentence_rows(ids, len(ids)))
        return self._session

    def _f2(self, ids, end, rows):
        got, pre = capture(self.R, ids, (8,))
        surp = self.R.surprisal(ids, pre.cpu())
        qs = surp[end:].clone(); qs[:2] = -1.0
        sal = [end + int(i) for i in qs.topk(min(2, len(qs))).indices]
        v = got[8]; V = torch.nn.functional.normalize(v - v[:end].mean(0), dim=-1)
        return rows[int(maxsim_scores(V[:end], V[sal], rows).argmax())], [self.tok.decode([ids[p]]) for p in sal]

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

    def answer(self, session_text: str, question_text: str, mode: str = 'lenient', cut_fields: bool = True):
        s = self.read(session_text)
        base, end = s['ids'], len(s['ids'])
        q = self.tok.encode(question_text, add_special_tokens=False)
        ids = base + q
        row, finfo = text_fetch(ids, end, step4=True, bigram=False)
        path, salient = (f"text{finfo['step']}" if row is not None else None), None
        if row is None and mode == 'lenient':
            row, salient = self._f2(ids, end, s['rows']); path = 'F2'
        info = dict(path=path, row=tuple(row) if row else None, abstain=int(row is None), salient=salient,
                    bytes_held=dict(mind_state=STATE_BYTES.get('b2p2'), library_text=2 * end,
                                    address_if_stored_int8=592 * end if path == 'F2' else 0))
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
