"""Long-prompt stress test (user, Nov 1): realistic prompts kept in files, fed to our best minds AND the baselines
(all trained at 2k), free generation, outputs shown. Not a benchmark cell: a demo with the prompt text on disk.

Builds v16/prompts/long_L{L}_s{seed}.txt: held-out prose of ~L tokens with (a) facts stated in natural sentences at
spread positions (one near the start, one near the end, one stated then UPDATED later), (b) a rule stated once with
three worked examples in the trained "Map rule." format and once in plain words, (c) a plain-language fact for a chat
question. Then asks questions one at a time (prompt + question -> greedy continuation):
  trained recall format, chat-format recall (ChatML, as in the smoltalk share of training), updated fact, the rule in
  the trained format, the rule in plain words (chat), and two world-knowledge questions (chat) to see what 600M tokens
  of web text taught a 103M body.
Usage: python v16/long_prompt_eval.py --ckpt <path> --arch pam|transformer|mamba2 --lengths 2048,4096,8192 --device cuda
Writes one markdown report per run to logs/ and prints every generation.
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'v16'))
from run_memory_behavioral import _load_v13_sempty  # noqa: E402
from memory_probes.spread import filler_tokens  # noqa: E402
from bench import Reader  # noqa: E402

DOT, NL = 13, 198
IM_START, IM_END = '<|im_start|>', '<|im_end|>'


def build_prompt(tok, filler, L, seed):
    rng = random.Random(seed)
    start = rng.randrange(0, len(filler) - L - 1)
    body = tok.decode(filler[start:start + L].tolist())
    paras = [p for p in body.split('\n') if p.strip()]
    facts = {
        'near_start': (" Definition: velmor refers to cobalt.", 'velmor', 'cobalt'),                       # the TRAINED fact format (control)
        'near_end':   (" One more thing to remember: the word tarnik refers to juniper.", 'tarnik', 'juniper'),   # natural sentence
        'updated_old': (" Definition: pelrov refers to amber.", 'pelrov', 'amber'),
        'updated_new': (" Correction: from here on, pelrov refers to indigo.", 'pelrov', 'indigo'),
        'chat_fact':  (" The backup key for the server room is kept in the blue cabinet next to the printer.", None, 'blue cabinet'),
        'rule_trained': (" Map rule. 3->10, 5->14, 8->20.", None, '26'),                                    # the TRAINED rule format (control)
        'rule_plain':  (" The shop's pricing rule is simple: the price is twice the cost plus four. Cost 3 gives price 10, cost 5 gives 14, cost 8 gives 20.", None, '26'),
    }
    n = len(paras)
    slots = {'near_start': 0.06, 'rule_trained': 0.15, 'updated_old': 0.25, 'chat_fact': 0.45, 'rule_plain': 0.6, 'updated_new': 0.75, 'near_end': 0.92}
    for name, frac in sorted(slots.items(), key=lambda kv: -kv[1]):   # insert from the back so indices stay valid
        i = min(n - 1, int(frac * n)); paras[i] = paras[i].rstrip() + facts[name][0]
    text = '\n'.join(paras)
    questions = [
        ('recall_trained_near_end', " Query: tarnik refers to", 'juniper'),
        ('recall_trained_near_start', " Query: velmor refers to", 'cobalt'),
        ('recall_updated_trained', " Query: pelrov refers to", 'indigo'),
        ('recall_chat', f"\n{IM_START}user\nIn the notes above, what does the word velmor refer to?{IM_END}\n{IM_START}assistant\n", 'cobalt'),
        ('chat_fact', f"\n{IM_START}user\nWhere is the backup key for the server room kept?{IM_END}\n{IM_START}assistant\n", 'blue cabinet'),
        ('rule_trained', " Map rule. 11 ->", '26'),
        ('rule_plain_chat', f"\n{IM_START}user\nUsing the shop's pricing rule from the notes, what is the price when the cost is 11?{IM_END}\n{IM_START}assistant\n", '26'),
        ('knowledge_capital', f"\n{IM_START}user\nWhat is the capital of France?{IM_END}\n{IM_START}assistant\n", 'Paris'),
        ('knowledge_author', f"\n{IM_START}user\nWho wrote the play Hamlet?{IM_END}\n{IM_START}assistant\n", 'Shakespeare'),
    ]
    return text, questions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--arch', default='pam', choices=['pam', 'transformer', 'mamba2'])
    ap.add_argument('--lengths', default='2048,4096,8192'); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--max_new', type=int, default=24)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    model, tok, cfg = _load_v13_sempty(Path(a.ckpt), 'baseline_real_pm', torch.device(a.device)); model.eval()
    R = Reader(model, a.arch)
    fill = filler_tokens(str(ROOT / '.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt'))
    im_end = tok.convert_tokens_to_ids(IM_END) if IM_END in tok.get_vocab() else None
    stops = tuple(t for t in (NL, im_end) if t is not None)
    name = Path(a.ckpt).parent.name; ts = time.strftime('%Y%m%d_%H%M')
    rep = ROOT / 'logs' / f'v16_longprompt_{name}{a.tag}_{ts}.md'
    out = [f"# Long-prompt stress test: {name} ({a.arch}), seed {a.seed}, greedy, max_new {a.max_new}\n"]
    for L in [int(x) for x in a.lengths.split(',')]:
        text, qs = build_prompt(tok, fill, L, a.seed)
        pf = ROOT / 'v16' / 'prompts' / f'long_L{L}_s{a.seed}.txt'; pf.parent.mkdir(exist_ok=True); pf.write_text(text)
        base_ids = tok.encode(text, add_special_tokens=False)
        out.append(f"\n## L = {L} (prompt {len(base_ids)} tokens, file {pf.relative_to(ROOT)})\n\n| question | expected | generated |\n|---|---|---|")
        for qname, q, exp in qs:
            ids = base_ids + tok.encode(q, add_special_tokens=False)
            t0 = time.time()
            gen = R.greedy(ids, max_new=a.max_new, stops=stops)
            txt = tok.decode(gen).replace('\n', ' / ').replace('|', '\\|')
            hit = exp.lower() in txt.lower()
            line = f"| {qname} | {exp} | {'OK' if hit else '--'} `{txt.strip()}` |"
            out.append(line); print(f"L={L} {qname:28s} exp={exp:12s} {'OK' if hit else '--'} {txt.strip()[:90]!r} ({time.time()-t0:.1f}s)", flush=True)
        rep.write_text('\n'.join(out) + '\n')
    print('report', rep)


if __name__ == '__main__':
    main()
