"""BENCH-TOOL (PRE-REGISTERED 2026-11-05 14:40 IST, amended 14:50 before any number): GSM8K with a calculator tool, per item.

Items, plain arm, generation and scoring = the review page's GSM8K check (scripts/think_eval.py OFF mode: the first --n
TEST items, user turn only, greedy m.generate to <|im_end|>, max 256 new tokens, answer = the number after "The answer is").
Tool arm: the same, with a system turn offering one function in the format SCI-1 trained on (v7/sci_tools.py: glaive /
ToolACE rendered as <tool_call>{"name", "arguments"}</tool_call> and <|im_start|>tool <tool_response>...). When an assistant
turn holds a tool call, the sandboxed executor (v14/tool_exec.py: bwrap, arithmetic only) computes it, a tool turn with the
result is appended and the assistant continues; at most 6 calls and 512 new tokens in all.
Rows (pairs format, cell gsm8k): bare = plain, package = tool; calls, well_formed, exec_errors per item.
    python v14/benchtool_gsm8k.py --ckpt <weights> --tag c3b --out logs/v14_benchtool_gsm8k_<tag>_<hash>_<stamp>.jsonl [--suite_log ...]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)
from think_eval import answer_of  # noqa: E402
from v7.sci_tools import _turn  # noqa: E402
from v14.tool_exec import calculate  # noqa: E402

FN = {'name': 'calculate', 'description': 'Evaluate an arithmetic expression and return the number',
      'parameters': {'type': 'object', 'properties': {'expression': {'type': 'string', 'description': 'The arithmetic expression, e.g. (48 + 12) / 4'}},
                     'required': ['expression']}}
SYSTEM = ('You are a helpful assistant with access to the following functions. Use them if required -\n'
          + json.dumps(FN, indent=4) + '\nSolve the problem step by step. Use calculate for every arithmetic step. '
          'End with "The answer is <number>."')
CALL = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.S)


@torch.no_grad()
def gen(m, tok, text, end, max_new, dev):
    ids = torch.tensor([tok.encode(text)], device=dev)
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
        out = m.generate(ids, max_new_tokens=max_new, temperature=0.0, eos_token_id=end)
    new = out[0, ids.shape[1]:].tolist()
    return tok.decode([t for t in new if t != end]), len(new)


def tool_arm(m, tok, q, end, dev, max_calls=6, budget=512):
    text = _turn('system', SYSTEM) + _turn('user', q) + '<|im_start|>assistant\n'
    trace, calls, used = '', [], 0
    while used < budget:
        seg, n = gen(m, tok, text, end, min(256, budget - used), dev); used += n
        text += seg; trace += seg
        mm = CALL.search(seg)
        if not mm or len(calls) >= max_calls:
            break
        c = dict(raw=mm.group(1)[:300], well_formed=0, ok=0, result=None)
        try:
            j = json.loads(mm.group(1)); arg = j.get('arguments', {})
            arg = json.loads(arg) if isinstance(arg, str) else arg
            expr = arg.get('expression') if isinstance(arg, dict) else None
            if j.get('name') == 'calculate' and expr is not None:
                c['well_formed'] = 1; ok, val = calculate(expr); c.update(ok=int(ok), result=val)
            else:
                val = 'error: unknown function or missing expression'
        except (ValueError, AttributeError):
            val = 'error: the call is not valid JSON'
        calls.append(c)
        resp = '<tool_response>\n' + json.dumps({'name': 'calculate', 'content': val}) + '\n</tool_response>'
        add = '<|im_end|>\n' + _turn('tool', resp) + '<|im_start|>assistant\n'
        text += add; trace += '\n' + resp + '\n'
    return trace, calls, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--tag', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=200); ap.add_argument('--device', default='cuda'); ap.add_argument('--suite_log', default=None)
    a = ap.parse_args()
    import subprocess
    from datasets import load_dataset
    from run_memory_behavioral import _load_v13_sempty
    h = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    print('HEADER ' + json.dumps(dict(unit='BENCH-TOOL GSM8K', hash=h, args=vars(a), system=SYSTEM)), flush=True)
    dev = torch.device(a.device); m, tok, _ = _load_v13_sempty(Path(a.ckpt), 'baseline_real_pm', dev); m.eval()
    meta = torch.load(a.ckpt, map_location='cpu', weights_only=False, mmap=True)
    base = dict(tag=a.tag, step=meta.get('global_step'), tokens=meta.get('global_tokens'), hash=h)
    end = tok.convert_tokens_to_ids('<|im_end|>')
    items = list(load_dataset('openai/gsm8k', 'main', split='test'))[:a.n]
    out = open(a.out, 'w'); nb = nt = nc = nw = 0; t0 = time.time()
    for i, it in enumerate(items):
        gold = it['answer'].rsplit('####', 1)[1].strip().replace(',', '')
        plain, _ = gen(m, tok, _turn('user', it['question']) + '<|im_start|>assistant\n', end, 256, dev)
        tool, calls, used = tool_arm(m, tok, it['question'], end, dev)
        bp, tp = answer_of(plain), answer_of(tool)
        row = dict(base, cell='gsm8k', task='gsm8k', L=None, doc=i, context='', question=it['question'], gold=gold,
                   bare_answer=plain, bare_pred=bp, bare_ok=int(bp == gold), package_answer=tool, package_pred=tp,
                   package_ok=int(tp == gold), package_path='calculator tool (bwrap)', calls=len(calls),
                   well_formed=sum(c['well_formed'] for c in calls), exec_ok=sum(c['ok'] for c in calls), call_log=calls,
                   tokens_used=used)
        out.write(json.dumps(row, ensure_ascii=False) + '\n'); out.flush()
        nb += row['bare_ok']; nt += row['package_ok']; nc += bool(calls); nw += bool(row['well_formed'])
        if (i + 1) % 20 == 0:
            print(f'{i + 1}: plain {nb / (i + 1):.3f} tool {nt / (i + 1):.3f} any call {nc / (i + 1):.2f} '
                  f'well-formed {nw / (i + 1):.2f} ({time.time() - t0:.0f} s)', flush=True)
    out.close()
    if a.suite_log:
        from v14.pkg_pairs_more import merge
        more = a.suite_log.replace('.jsonl', '_pairsx.jsonl')
        rest = [l for l in open(more) if json.loads(l)['cell'] != 'gsm8k'] if Path(more).exists() else []
        open(more, 'w').writelines(rest + open(a.out).readlines())
        k, n = merge(a.suite_log.replace('.jsonl', '_pairs.jsonl'), more, ['gsm8k'])
        print(f'merged: {k} kept + {n} new rows -> pairs', flush=True)
    print(f'BENCH-TOOL DONE plain {nb / len(items):.3f} tool {nt / len(items):.3f} any call {nc / len(items):.2f} '
          f'well-formed {nw / len(items):.2f}', flush=True)


if __name__ == '__main__':
    main()
