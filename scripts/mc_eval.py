"""Standalone zero-shot multiple-choice scorer (SCI-1 milestone CHECK): HellaSwag / ARC-Easy / PIQA exactly as lm-eval-harness
0.4.9.1 builds them, with no lm_eval dependency (it does not import under transformers 5 on the 6000).

Copied from lm-eval 0.4.9.1:
- prompts: arc "Question: {q}\\nAnswer:"; piqa "Question: {goal}\\nAnswer:"; hellaswag utils.preprocess(activity_label + ": " +
  ctx_a + " " + ctx_b.capitalize());
- each choice is the continuation " " + choice, encoded as the HF path does (trailing context whitespace moves to the
  continuation; continuation = whole[len(context):]);
- acc = argmax summed log-likelihood; acc_norm = argmax ll / len(choice) (characters).
Validated against lm-eval on b2p2 (kb1): see the commit message.

  python scripts/mc_eval.py --ckpt <best_model.pt> --out <json> [--tasks hellaswag,arc_easy,piqa] [--limit N]
"""
from __future__ import annotations

import argparse, json, re, sys, time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))


def _hs_pre(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")


def load_task(name: str):
    from datasets import load_dataset
    if name == 'hellaswag':
        for d in load_dataset('Rowan/hellaswag', split='validation'):
            ctx = d['ctx_a'] + ' ' + d['ctx_b'].capitalize()
            yield _hs_pre(d['activity_label'] + ': ' + ctx), [_hs_pre(e) for e in d['endings']], int(d['label'])
    elif name in ('arc_easy', 'arc_challenge'):
        for d in load_dataset('allenai/ai2_arc', 'ARC-Easy' if name == 'arc_easy' else 'ARC-Challenge', split='test'):
            yield f"Question: {d['question']}\nAnswer:", list(d['choices']['text']), d['choices']['label'].index(d['answerKey'])
    elif name == 'piqa':
        for d in load_dataset('baber/piqa', split='validation'):
            yield f"Question: {d['goal']}\nAnswer:", [d['sol1'], d['sol2']], int(d['label'])
    else:
        raise ValueError(name)


class Scorer:
    def __init__(self, ckpt: str, device: str = 'cuda', batch_size: int = 32, max_len: int = 2048, fp32: bool = False,
                 hf: str = '', revision: str | None = None):
        self.dev = torch.device(device); self.hf = bool(hf)
        if hf:    # a HF reference (e.g. Pythia at a revision), fp32 unless asked otherwise; same prompts and scoring
            from run_memory_behavioral import _load_hf
            self.m, self.tok, _ = _load_hf(hf, self.dev, revision); self.bos = self.tok.eos_token_id
        else:
            from run_memory_behavioral import _load_v13_sempty
            self.m, self.tok, _ = _load_v13_sempty(Path(ckpt), 'baseline_real_pm', self.dev); self.bos = 50256
        self.m.eval(); self.bs, self.max_len, self.V = batch_size, max_len, len(self.tok)
        self.amp = self.dev.type == 'cuda' and not fp32

    def _enc(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    def _pair(self, ctx, cont):
        n = len(ctx) - len(ctx.rstrip())
        if n:
            cont, ctx = ctx[-n:] + cont, ctx[:-n]
        if not ctx:
            return [self.bos], self._enc(cont)
        whole, c = self._enc(ctx + cont), self._enc(ctx)
        return c, whole[len(c):]

    @torch.no_grad()
    def loglik(self, reqs):
        items = [((c + k)[-(self.max_len + 1):], len(k)) for c, k in (self._pair(a, b) for a, b in reqs)]
        order = sorted(range(len(items)), key=lambda j: -len(items[j][0])); out = [0.0] * len(items)
        for s in range(0, len(order), self.bs):
            idx = order[s:s + self.bs]; T = min(self.max_len, ((len(items[idx[0]][0]) - 1 + 63) // 64) * 64)
            x = torch.zeros((len(idx), T), dtype=torch.long)
            for b, j in enumerate(idx):
                x[b, :len(items[j][0]) - 1] = torch.tensor(items[j][0][:-1])
            with torch.autocast(device_type=self.dev.type, dtype=torch.bfloat16, enabled=self.amp):
                logits = self.m(x.to(self.dev)).logits if self.hf else self.m(x.to(self.dev))[0]
            for b, j in enumerate(idx):
                full, nk = items[j]; n = len(full) - 1
                lg = logits[b, n - nk:n, :self.V].float().log_softmax(-1)
                out[j] = float(lg.gather(1, torch.tensor(full[-nk:], device=self.dev)[:, None]).sum())
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--ckpt', default=''); p.add_argument('--out', required=True)
    p.add_argument('--hf', default='', help='HF model id instead of --ckpt'); p.add_argument('--revision', default=None)
    p.add_argument('--tasks', default='hellaswag,arc_easy,piqa'); p.add_argument('--device', default='cuda')
    p.add_argument('--batch_size', type=int, default=32); p.add_argument('--limit', type=int, default=0)
    p.add_argument('--fp32', action='store_true')
    a = p.parse_args(); t0 = time.time(); sc = Scorer(a.ckpt, a.device, a.batch_size, fp32=a.fp32, hf=a.hf, revision=a.revision); res = {}
    for task in a.tasks.split(','):
        docs = list(load_task(task)); docs = docs[:a.limit] if a.limit else docs
        reqs = [(q, ' ' + ch) for q, chs, _ in docs for ch in chs]
        ll = sc.loglik(reqs); i = acc = accn = 0
        for q, chs, gold in docs:
            l = ll[i:i + len(chs)]; i += len(chs)
            acc += int(max(range(len(chs)), key=lambda c: l[c]) == gold)
            accn += int(max(range(len(chs)), key=lambda c: l[c] / len(chs[c])) == gold)
        res[task] = {'acc': acc / len(docs), 'acc_norm': accn / len(docs), 'n': len(docs)}
        print(task, {k: round(v, 4) if isinstance(v, float) else v for k, v in res[task].items()}, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({'ckpt': a.ckpt or f'{a.hf}@{a.revision}', 'results': res, 'sec': round(time.time() - t0, 1), 'fp32': a.fp32}, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
