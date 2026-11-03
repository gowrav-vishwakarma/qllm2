# Copy of scripts/think_eval.py (SCI-1 code) that saves the WHOLE answer + question (user 11-03: let thinking finish). Run with PYTHONPATH=<worktree>:<worktree>/scripts.
"""SCI-1 milestone CHECK for the thinking switch (user 11:30 IST via Lead): GSM8K TEST items (never trained; the think source
uses GSM8K train + MetaMathQA) in both modes, greedy:
  ON  = system 'Reasoning: on' + user question -> expect <think> ... </think> The answer is X.
  OFF = user question only                     -> expect 'The answer is X.' with NO trace.
Reports accuracy ON / OFF (the number after 'The answer is'), the trace-leak rate in OFF, and the trace rate in ON.
Pre-registered (EXPERIMENTS_SEMPY "SCI-1"): ON <= OFF + 0.02 at milestone 2, or OFF leak > 5 % -> drop thinking at the first continuation.

  python scripts/think_eval.py --ckpt <ckpt.pt> --out <json> [--n 200]
"""
from __future__ import annotations

import argparse, json, re, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from v7.sci_tools import THINK_ON, _turn  # noqa: E402

NUM = re.compile(r'-?\d[\d,]*\.?\d*')


def answer_of(text: str):
    m = re.search(r'The answer is\s*([^\n<]*)', text)
    if not m:
        return None
    n = NUM.findall(m.group(1))
    return n[0].replace(',', '').rstrip('.') if n else None


def main():
    p = argparse.ArgumentParser(); p.add_argument('--ckpt', required=True); p.add_argument('--out', required=True)
    p.add_argument('--n', type=int, default=200); p.add_argument('--device', default='cuda'); p.add_argument('--max_new', type=int, default=256)
    a = p.parse_args()
    from datasets import load_dataset
    from run_memory_behavioral import _load_v13_sempty
    dev = torch.device(a.device); m, tok, _ = _load_v13_sempty(Path(a.ckpt), 'baseline_real_pm', dev); m.eval()
    end = tok.convert_tokens_to_ids('<|im_end|>')
    items = list(load_dataset('openai/gsm8k', 'main', split='test'))[:a.n]
    res = {'on': {'right': 0, 'trace': 0}, 'off': {'right': 0, 'trace': 0}}; rows = []
    for it in items:
        gold = it['answer'].rsplit('####', 1)[1].strip().replace(',', '')
        for mode in ('on', 'off'):
            prompt = (_turn('system', THINK_ON) if mode == 'on' else '') + _turn('user', it['question']) + '<|im_start|>assistant\n'
            ids = torch.tensor([tok.encode(prompt)], device=dev)
            with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
                out = m.generate(ids, max_new_tokens=a.max_new, temperature=0.0, eos_token_id=end)
            txt = tok.decode(out[0, ids.shape[1]:].tolist())
            ok = answer_of(txt) == gold
            res[mode]['right'] += int(ok); res[mode]['trace'] += int('<think>' in txt)
            rows.append({'mode': mode, 'gold': gold, 'pred': answer_of(txt), 'ok': ok, 'trace': '<think>' in txt, 'text': txt, 'question': it['question'], 'closed': '</think>' in txt, 'ended': out.shape[1] - ids.shape[1] < a.max_new})
    n = len(items)
    summ = {'n': n, 'acc_on': res['on']['right'] / n, 'acc_off': res['off']['right'] / n,
            'trace_rate_on': res['on']['trace'] / n, 'leak_rate_off': res['off']['trace'] / n}
    print('think_eval', {k: round(v, 3) if isinstance(v, float) else v for k, v in summ.items()}, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({'ckpt': a.ckpt, 'summary': summ, 'rows': rows}, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
