"""lm-eval-harness wrapper for the v13_sempty bodies (PAM, Transformer++, Mamba-2), as they are (kb1, EXPERIMENTS_SEMPY).

Log-likelihood requests only (the zero-shot multiple-choice suite and LAMBADA). Requests are encoded the way the harness's
HF path does it:
- trailing context whitespace moves to the continuation;
- continuation = whole[len(context):];
- an empty context becomes EOT;
- requests are left-truncated to 2048 tokens;
- no prefix is added.
bf16 autocast, batches sorted by length, right padding (every body is causal).

Run in the venv that has lm_eval 0.4.9.1 (+ mamba_ssm for Mamba-2):
  ~/.venvs/qllm311/bin/python scripts/lm_eval_bodies.py --ckpt <best_model.pt> --out logs/kb1/<name>_<hash>.json
  ~/.venvs/qllm311/bin/python scripts/lm_eval_bodies.py --hf HuggingFaceTB/SmolLM2-135M --out ...   (reference, stock hf model)
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from lm_eval import simple_evaluate  # noqa: E402
from lm_eval.api.model import LM  # noqa: E402

TASKS = 'hellaswag,arc_easy,arc_challenge,piqa,winogrande,openbookqa,lambada_openai'
EOT = 50256


class BodyLM(LM):
    def __init__(self, ckpt: str, device: str = 'cuda', batch_size: int = 32, max_len: int = 2048):
        super().__init__()
        from run_memory_behavioral import _load_v13_sempty
        self.dev = torch.device(device)
        self.m, self.tok, self.cfg = _load_v13_sempty(Path(ckpt), 'baseline_real_pm', self.dev)
        self.m.eval()
        self.bs, self.max_len, self.V = batch_size, max_len, len(self.tok)
        self.n_params = sum(p.numel() for p in self.m.parameters())

    def _enc(self, s: str):
        return self.tok.encode(s, add_special_tokens=False)

    def _pair(self, ctx: str, cont: str):
        n = len(ctx) - len(ctx.rstrip())
        if n:
            cont, ctx = ctx[-n:] + cont, ctx[:-n]
        if not ctx:
            return [EOT], self._enc(cont)
        whole, c = self._enc(ctx + cont), self._enc(ctx)
        return c, whole[len(c):]

    @torch.no_grad()
    def loglikelihood(self, requests):
        items = []
        for r in requests:
            c, k = self._pair(*r.args)
            items.append(((c + k)[-(self.max_len + 1):], len(k)))
        order = sorted(range(len(items)), key=lambda j: -len(items[j][0]))
        out = [None] * len(items)
        for s in range(0, len(order), self.bs):
            idx = order[s:s + self.bs]
            T = len(items[idx[0]][0]) - 1
            T = min(self.max_len, ((T + 63) // 64) * 64)          # pad to a multiple of 64: fewer recompiles
            x = torch.zeros((len(idx), T), dtype=torch.long)
            for b, j in enumerate(idx):
                full = items[j][0]
                x[b, :len(full) - 1] = torch.tensor(full[:-1])
            with torch.autocast(device_type=self.dev.type, dtype=torch.bfloat16, enabled=self.dev.type == 'cuda'):
                logits = self.m(x.to(self.dev))[0]
            for b, j in enumerate(idx):
                full, nk = items[j]; n = len(full) - 1
                lg = logits[b, n - nk:n, :self.V].float().log_softmax(-1)
                tgt = torch.tensor(full[-nk:], device=self.dev)
                out[j] = (float(lg.gather(1, tgt[:, None]).sum()), bool((lg.argmax(-1) == tgt).all()))
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError('kb1 uses log-likelihood tasks only')

    def generate_until(self, requests):
        raise NotImplementedError('kb1 uses log-likelihood tasks only')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--ckpt', help='a v13_sempty checkpoint (PAM / T++ / Mamba-2)')
    g.add_argument('--hf', help='a reference HF model id (stock lm-eval hf model)')
    p.add_argument('--tasks', default=TASKS)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--limit', type=int, default=None, help='smoke only; never for a reported number')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    t0 = time.time()
    if a.ckpt:
        model = BodyLM(a.ckpt, a.device, a.batch_size)
        meta = {'ckpt': a.ckpt, 'params': model.n_params, 'arch': type(model.m).__name__}
        res = simple_evaluate(model=model, tasks=a.tasks.split(','), num_fewshot=0, limit=a.limit, bootstrap_iters=1000)
    else:
        meta = {'hf': a.hf}
        res = simple_evaluate(model='hf', model_args=f'pretrained={a.hf},dtype=bfloat16', tasks=a.tasks.split(','), num_fewshot=0,
                              device=a.device, batch_size=a.batch_size, limit=a.limit, bootstrap_iters=1000)
    import lm_eval
    rec = {'meta': {**meta, 'lm_eval': lm_eval.__version__, 'limit': a.limit, 'sec': round(time.time() - t0, 1)},
           'results': res['results'], 'n_samples': res.get('n-samples')}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(a.out, 'w'), indent=1, default=str)
    for t, r in res['results'].items():
        print(t, {k: round(v, 4) for k, v in r.items() if isinstance(v, float) and 'stderr' not in k})


if __name__ == '__main__':
    main()
