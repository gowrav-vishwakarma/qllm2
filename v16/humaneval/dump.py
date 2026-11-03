"""Human-eval dump for one SCI-1 checkpoint: per-item prompts + answers for a reader.
Benchmarks (HellaSwag / ARC-e / PIQA, random fixed sample, scored exactly like scripts/mc_eval.py) and a
broad generation set (greedy + sampled). Writes one JSON.
  python dump.py <worktree> <ckpt.pt> <out.json> [n_per_bench]
"""
import json, random, sys, time
from pathlib import Path
W, CK, OUT = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 40
sys.path.insert(0, str(W)); sys.path.insert(0, str(W / 'scripts'))
import torch
from mc_eval import Scorer, load_task

U = lambda q: f'<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n'
P = []
def add(group, name, prompt, n=120): P.append(dict(group=group, name=name, prompt=prompt, n=n))
# science writing (plain text continuation)
for i, s in enumerate([
    'Photosynthesis converts light energy into chemical energy. In the light-dependent reactions,',
    'The second law of thermodynamics states that',
    'CRISPR-Cas9 is a genome editing tool that works by',
    'In a randomized controlled trial, the purpose of the control group is',
    'Abstract. We study the effect of temperature on enzyme activity. Our results show that',
    'Plate tectonics explains earthquakes because']): add('Science writing', f'sci{i+1}', s)
# chat Q&A
for i, q in enumerate([
    'What is the difference between a hypothesis and a theory?',
    'Explain what a p-value is in simple words.',
    'Why is the sky blue?',
    'What does DNA polymerase do?',
    'Give me three tips for designing a good experiment.',
    'What is the unit of electrical resistance?']): add('Chat Q&A', f'chat{i+1}', U(q))
# multi-turn
add('Multi-turn chat', 'mt1', '<|im_start|>user\nI am measuring the pH of rainwater.<|im_end|>\n<|im_start|>assistant\n'
    'Great. Rainwater is usually slightly acidic, around pH 5.6, because it absorbs carbon dioxide.<|im_end|>\n'
    '<|im_start|>user\nMy sample reads pH 4.2. What could cause that?<|im_end|>\n<|im_start|>assistant\n')
add('Multi-turn chat', 'mt2', '<|im_start|>user\nMy name is Ravi and I study bees.<|im_end|>\n<|im_start|>assistant\n'
    'Nice to meet you, Ravi! Bees are fascinating pollinators.<|im_end|>\n<|im_start|>user\nWhat is my name and what do I study?<|im_end|>\n<|im_start|>assistant\n')
add('Multi-turn chat', 'mt3', '<|im_start|>user\nConvert 2 km to meters.<|im_end|>\n<|im_start|>assistant\n2 km = 2000 m.<|im_end|>\n'
    '<|im_start|>user\nAnd 3.5 km?<|im_end|>\n<|im_start|>assistant\n')
# tools
SYS = ('<|im_start|>system\nYou can call tools. Available: library.fetch(query) returns stored records; '
       'python.run(code) runs Python. Call a tool as <tool_call>{"name": ..., "arguments": {...}}</tool_call>.<|im_end|>\n')
add('Tool use', 'tool_call1', SYS + U('Look up the melting point recorded for sample K3 in my notes.'))
add('Tool use', 'tool_call2', SYS + U('Compute the mean of 4.1, 4.5 and 3.9 using Python.'))
add('Tool use', 'tool_read1', U('Look up the boiling point recorded for sample Q7 in my notes.') +
    '<tool_call>\n{"name": "library.fetch", "arguments": {"query": "Q7 boiling point"}}\n</tool_call><|im_end|>\n'
    '<|im_start|>tool\n<tool_response>\n{"name": "library.fetch", "content": "Record: Q7 | boiling point: 78.4 C."}\n'
    '</tool_response><|im_end|>\n<|im_start|>assistant\n')
add('Tool use', 'tool_read2', U('What was the yield in run 12?') +
    '<tool_call>\n{"name": "library.fetch", "arguments": {"query": "run 12 yield"}}\n</tool_call><|im_end|>\n'
    '<|im_start|>tool\n<tool_response>\n{"name": "library.fetch", "content": "Run 12: yield 63 %, temperature 40 C."}\n'
    '</tool_response><|im_end|>\n<|im_start|>assistant\n')
# python for science
add('Python for science', 'py1', U('Write a Python function that converts Celsius to Kelvin.'))
add('Python for science', 'py2', U('Write Python code with numpy to compute the standard deviation of a list of measurements.'))
add('Python for science', 'py3', 'import numpy as np\nimport matplotlib.pyplot as plt\n\n# plot a damped sine wave\nt = np.linspace(0, 10, 500)\n')
add('Python for science', 'py4', 'def molarity(moles, volume_l):\n    """Return the concentration in mol/L."""\n')
# protocol / stated rules
add('Rules & protocols', 'proto1', 'Lab protocol: report every mass in grams, rounded to one decimal place.\n'
    'Sample A weighed 12.347 g and sample B weighed 8.912 g.\nReport: sample A =', 40)
add('Rules & protocols', 'proto2', 'Rule: always report temperatures in Kelvin.\nThe reading was 25 C.\nReport: temperature =', 30)
add('Rules & protocols', 'proto3', U('From now on answer only with YES or NO. Is water wet?'), 30)
# quantitative
for i, q in enumerate([
    'A solution has 0.25 mol of salt in 2.0 L of water. What is its concentration in mol/L?',
    'A car travels 150 km in 2 hours. What is its average speed in km/h?',
    'What is 15% of 80?',
    'If a cell divides every 20 minutes, how many cells are there after 1 hour starting from one cell?']):
    add('Quantitative', f'q{i+1}', U(q), 60)
# in-context rule learning (few-shot, unseen mapping)
add('Learning from examples', 'icl1', 'cat -> tac\ndog -> god\nbird -> drib\nfish -> hsif\nlamp ->', 10)
add('Learning from examples', 'icl2', '2 -> 7\n5 -> 16\n1 -> 4\n3 -> 10\n6 ->', 6)
add('Learning from examples', 'icl3', 'red -> RED!\nblue -> BLUE!\ngreen -> GREEN!\nyellow ->', 8)
add('Learning from examples', 'icl4', 'France: Paris\nJapan: Tokyo\nItaly: Rome\nIndia:', 6)

def main():
    t0 = time.time(); sc = Scorer(CK, 'cuda', 16); m, tok = sc.m, sc.tok
    ck = torch.load(CK, map_location='cpu', weights_only=False)
    meta = dict(ckpt=Path(CK).name, step=ck.get('global_step'), tokens=ck.get('global_tokens')); del ck
    out = dict(meta=meta, bench={}, gens=[])
    rng = random.Random(7)
    for task in ['hellaswag', 'arc_easy', 'piqa']:
        docs = list(load_task(task)); docs = rng.sample(docs, N) if N else docs
        ll = sc.loglik([(q, ' ' + c) for q, cs, _ in docs for c in cs]); i = 0; rows = []
        for q, cs, g in docs:
            l = ll[i:i + len(cs)]; i += len(cs)
            pn = max(range(len(cs)), key=lambda c: l[c] / len(cs[c]))
            rows.append(dict(q=q, ll=[round(x,2) for x in l], choices=cs, gold=g, pick=pn, ok=pn == g,
                             score=[round(l[c] / len(cs[c]), 3) for c in range(len(cs))]))
        out['bench'][task] = rows; print(task, sum(r['ok'] for r in rows), '/', len(rows), flush=True)
    dev = sc.dev
    for p in P:
        ids = torch.tensor([tok.encode(p['prompt'])], device=dev)
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            g = m.generate(ids, max_new_tokens=p['n'], temperature=0.0)
            torch.manual_seed(1234)
            s = m.generate(ids, max_new_tokens=p['n'], temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.2)
        p['greedy'] = tok.decode(g[0, ids.shape[1]:].tolist()); p['sampled'] = tok.decode(s[0, ids.shape[1]:].tolist())
        out['gens'].append(p); print('gen', p['name'], flush=True)
    out['sec'] = round(time.time() - t0); json.dump(out, open(OUT, 'w'), indent=1); print('done', out['sec'], 's')

main()
