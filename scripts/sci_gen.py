"""SCI-1 milestone generations (the user's rule: show prompt + output + checkpoint). Five fixed prompts (science, protocol
following, plain chat, a tool call + result, a short <think>), sampled with the trainer's settings (temperature 0.8, top-k 50, top-p 0.9, repetition 1.2) at a fixed
seed, plus a greedy line. Printed with the checkpoint, its step / tokens and the code hash.

  python scripts/sci_gen.py --ckpt <ckpt.pt> [--tokens 80]
"""
from __future__ import annotations

import argparse, subprocess, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
PROMPTS = {
    'science': 'Photosynthesis converts light energy into chemical energy. In the light-dependent reactions,',
    'protocol': ('Lab protocol: report every mass in grams, rounded to one decimal place.\n'
                 'Sample A weighed 12.347 g and sample B weighed 8.912 g.\nReport: sample A ='),
    'chat': '<|im_start|>user\nWhat is the difference between a hypothesis and a theory?<|im_end|>\n<|im_start|>assistant\n',
    # SCI-1 formats (v7/sci_tools.py): does it call the tool, and read the result like a fetched row?
    'tool': ('<|im_start|>user\nLook up the boiling point recorded for sample Q7 in my notes.<|im_end|>\n<|im_start|>assistant\n'
             '<tool_call>\n{"name": "library.fetch", "arguments": {"query": "Q7 boiling point"}}\n</tool_call><|im_end|>\n'
             '<|im_start|>tool\n<tool_response>\n{"name": "library.fetch", "content": "Record: Q7 | boiling point: 78.4 C."}\n'
             '</tool_response><|im_end|>\n<|im_start|>assistant\n'),
    'think': ('<|im_start|>system\nReasoning: on<|im_end|>\n'
              '<|im_start|>user\nA solution has 0.25 mol of salt in 2.0 L of water. What is its concentration in mol/L?<|im_end|>\n'
              '<|im_start|>assistant\n'),
}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--ckpt', required=True); p.add_argument('--tokens', type=int, default=80)
    p.add_argument('--device', default='cuda'); a = p.parse_args()
    from run_memory_behavioral import _load_v13_sempty
    dev = torch.device(a.device); m, tok, _ = _load_v13_sempty(Path(a.ckpt), 'baseline_real_pm', dev); m.eval()
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    h = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    print(f"=== generations: {a.ckpt} step {ck.get('global_step')} tokens {ck.get('global_tokens'):,} code {h}")
    for name, prompt in PROMPTS.items():
        ids = torch.tensor([tok.encode(prompt)], device=dev)
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
            torch.manual_seed(1234)
            s = m.generate(ids, max_new_tokens=a.tokens, temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.2)
            g = m.generate(ids, max_new_tokens=a.tokens // 2, temperature=0.0)
        print(f"--- [{name}] prompt: {prompt!r}")
        print(f"    sampled: {tok.decode(s[0, ids.shape[1]:].tolist())!r}")
        print(f"    greedy : {tok.decode(g[0, ids.shape[1]:].tolist())!r}")


if __name__ == '__main__':
    main()
