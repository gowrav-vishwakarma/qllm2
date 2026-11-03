#!/usr/bin/env python3
"""Score a trained LM on the in-context rule-learning tasks (memory_probes/rules.py).

Bar 3 of the project goal: learning beyond recall. Each prompt shows a few
examples of a hidden rule; the model must apply it to a new input (for
letter_shift / number_step the answer never appears in the context).
Contrastive scoring over 8 single-token candidates (chance 0.125).

Example (CPU, so a training run can keep the GPU):
  CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/run_learning_probe.py \
    --model-type v13_sempty --checkpoint checkpoints_v13_sempty/<run>/best_model.pt \
    --output logs/learning_probes/v13_sempty_<run>_rules.json

``--suite b2`` scores v14's "inference beats selection" tasks instead
(v14/rules_ood.py: shift_wide, step_wide, affine_map + their *_new
held-out-parameter variants; the nearest trained parameter's answer is always
among the decoys, so selection by a memorised neighbour scores ~0). For a model
with no B2 training, base and *_new should score alike (difficulty control).

``--spread 0,1024,2048,4096,8192`` spreads each prompt's examples across a long
document of held-out web text (memory_probes/spread.py; tasks become
``<task>@<length>``): does in-context learning hold past the trained context?
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory_probes.rules import TASKS, aggregate, build_rule_suite, score_candidate_logits
from run_memory_behavioral import _last_logits, _load_hf, _load_transformer, _load_v13_sempty


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model-type', choices=('v13_sempty', 'transformer', 'hf'), required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--model-id', default='state-spaces/mamba-130m-hf')
    p.add_argument('--revision', default=None, help='HF revision for --model-type hf (e.g. Pythia step2000)')
    p.add_argument('--preset', default='baseline_real_pm')
    p.add_argument('--suite', choices=('rules', 'b2', 'label_diag'), default='rules',
                   help='rules = memory_probes/rules.py; b2 = v14/rules_ood.py; label_diag = '
                        'label_category + label_exact (memory_probes/label_diag.py)')
    p.add_argument('--tasks', default='', help='comma list (default: every task of the suite)')
    p.add_argument('--shots', default='2,4,8')
    p.add_argument('--trials', type=int, default=100)
    p.add_argument('--seed', type=int, default=5000)
    p.add_argument('--spread_tail', type=int, default=0,
                   help='extra filler tokens between the last spread example and the query (carry the rule past a window)')
    p.add_argument('--spread', default='',
                   help='comma list of prompt lengths (0 = no filler) to spread the examples to')
    p.add_argument('--filler', default='.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt',
                   help='token cache the spread filler is drawn from (held-out web text)')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model_type == 'v13_sempty':
        model, tokenizer, _ = _load_v13_sempty(args.checkpoint, args.preset, device)
        identity = str(args.checkpoint)
    elif args.model_type == 'transformer':
        model, tokenizer, _ = _load_transformer(args.checkpoint, device)
        identity = str(args.checkpoint)
    else:
        model, tokenizer, _ = _load_hf(args.model_id, device, args.revision)
        identity = args.model_id + (f'@{args.revision}' if args.revision else '')
    model.eval()

    tasks = tuple(t.strip() for t in args.tasks.split(',') if t.strip())
    shots = tuple(int(s) for s in args.shots.split(','))
    seeds = tuple(range(args.seed, args.seed + args.trials))
    if args.suite == 'b2':
        from v14.rules_ood import B2_NEW_TASKS, B2_TASKS, build_example as b2_example
        tasks = tasks or B2_TASKS + B2_NEW_TASKS
        examples = [b2_example(tokenizer, t, n, sd) for t in tasks for n in shots for sd in seeds]
    elif args.suite == 'label_diag':
        from memory_probes.label_diag import exact_variant
        base = build_rule_suite(tokenizer, tasks=('label_category',), shots=shots, seeds=seeds)
        examples = base + [exact_variant(ex, tokenizer) for ex in base]
        tasks = ('label_category', 'label_exact')
    else:
        tasks = tasks or TASKS
        examples = build_rule_suite(tokenizer, tasks=tasks, shots=shots, seeds=seeds)
    lengths = tuple(int(x) for x in args.spread.split(',') if x.strip())
    if lengths:
        from memory_probes.spread import filler_tokens, spread_suite
        filler = filler_tokens(args.filler)
        if args.model_type == 'hf' and tokenizer.encode(' The cat sat.', add_special_tokens=False) != [383, 3797, 3332, 13]:
            from transformers import AutoTokenizer   # the cache is GPT-2 ids: re-tokenize the same text for this model's vocab
            g2 = AutoTokenizer.from_pretrained('gpt2')
            filler = np.asarray([t for k in range(0, len(filler), 2048)
                                 for t in tokenizer.encode(g2.decode(filler[k:k + 2048].tolist()), add_special_tokens=False)], dtype=np.int64)
        examples = spread_suite(examples, tokenizer, lengths, filler, args.spread_tail)
        tasks = tuple(dict.fromkeys(ex.task for ex in examples))
    rows = []
    with torch.no_grad():
        for i, ex in enumerate(examples, 1):
            ids = torch.tensor([ex.prompt_ids], dtype=torch.long, device=device)
            logits = _last_logits(args.model_type, model, ids)[0]
            cand = logits[ex.candidate_token_ids].float().cpu()
            if not torch.isfinite(cand).all():
                rows.append({**ex.to_dict(), 'skipped': True})
                continue
            rows.append({**ex.to_dict(), **score_candidate_logits(ex, cand.tolist()),
                         'free': int(int(logits.argmax()) == ex.target_token_id)})   # MLB-1 A7: greedy over the whole vocab
            if i % 250 == 0 or i == len(examples):
                print(f'  scored {i}/{len(examples)}', flush=True)

    agg = aggregate(rows)
    per_task = {t: float(np.mean([a['accuracy'] for a in agg if a['task'] == t])) for t in tasks}
    per_task_free = {t: float(np.mean([r['free'] for r in rows if r['task'] == t and 'free' in r])) for t in tasks}
    result = {
        'schema_version': 'learning-probe-b2/v1' if args.suite == 'b2' else 'learning-probe-rules/v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'model_type': args.model_type,
        'model_identity': identity,
        'device': str(device),
        'platform': platform.platform(),
        'parameter_count': sum(x.numel() for x in model.parameters()),
        'protocol': {'suite': args.suite, 'spread': list(lengths), 'spread_tail': args.spread_tail, 'tasks': list(tasks), 'shots': list(shots), 'seeds': [seeds[0], seeds[-1]],
                     'candidates': 8, 'chance': 0.125},
        'per_task_accuracy': per_task,
        'mean_accuracy': float(np.mean(list(per_task.values()))),
        'per_task_free': per_task_free,          # A7: argmax over the whole vocab == answer (primary from A7 on)
        'mean_free': float(np.mean(list(per_task_free.values()))),
        'aggregates': agg,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('per task:', {t: round(v, 3) for t, v in per_task.items()},
          ' mean', round(result['mean_accuracy'], 3))
    print('free (A7):', {t: round(v, 3) for t, v in per_task_free.items()}, ' mean', round(result['mean_free'], 3))
    print(f'saved {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
