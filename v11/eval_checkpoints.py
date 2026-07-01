"""
Evaluate V11 checkpoints on WikiText-103 val and a fixed DCLM-Edu holdout slice.

Usage:
    .venv/bin/python -m v11.eval_checkpoints \\
        --checkpoints checkpoints_v11_e3_k3/best_model.pt \\
                      checkpoints_v11_e3_k3_dclm/best_model.pt \\
        --labels wiki,dclm
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from v11.model import V11Config, V11LM
from v7.data import TextDataset, load_wikitext103_val, pretrain_holdout_bucket


def load_dclm_holdout_tokens(
    *,
    target_tokens: int = 500_000,
    edu_score_min: int = 3,
    holdout_pct: int = 5,
    use_cache: bool = True,
) -> torch.Tensor:
    from transformers import AutoTokenizer

    cache_path = (
        Path('.cache') / 'v7_tokens'
        / f'dclm_edu_holdout_v{_HOLDOUT_CACHE_VERSION}_t{target_tokens}_e{edu_score_min}_p{holdout_pct}.pt'
    )
    if use_cache and cache_path.exists():
        cached = torch.load(cache_path, weights_only=False)
        print(f"[cache] DCLM holdout: {cache_path} ({len(cached['tokens']):,} tokens)")
        return cached['tokens']

    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    print(
        f"Building DCLM holdout (~{holdout_pct}% hash bucket, "
        f"edu>={edu_score_min}, target>={target_tokens:,} tokens)..."
    )
    stream = load_dataset('HuggingFaceTB/dclm-edu', split='train', streaming=True)
    tokens: List[int] = []
    kept_docs = 0
    for row in stream:
        score = row.get('edu_int_score')
        if score is not None and score < edu_score_min:
            continue
        text = row.get('text') or row.get('content') or ''
        if not text.strip() or not pretrain_holdout_bucket(text, holdout_pct):
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        ids.append(tokenizer.eos_token_id)
        tokens.extend(ids)
        kept_docs += 1
        if len(tokens) >= target_tokens:
            break

    if len(tokens) < target_tokens // 4:
        raise RuntimeError(
            f"DCLM holdout too small ({len(tokens):,} tokens from {kept_docs} docs)"
        )

    out = torch.tensor(tokens[:target_tokens], dtype=torch.long)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'tokens': out,
            'kept_docs': kept_docs,
            'holdout_pct': holdout_pct,
            'edu_score_min': edu_score_min,
            'cache_version': _HOLDOUT_CACHE_VERSION,
        },
        cache_path,
    )
    print(f"  holdout: {kept_docs:,} docs, {len(out):,} tokens -> {cache_path}")
    return out


def load_model(checkpoint: str, device: torch.device) -> Tuple[V11LM, V11Config]:
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg_dict = ckpt.get('config')
    cfg = V11Config(**cfg_dict) if cfg_dict else V11Config()
    model = V11LM(cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model, cfg


@torch.no_grad()
def eval_ppl(model: V11LM, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    total_loss_w = 0.0
    total_tokens = 0
    for batch in loader:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        batch_tokens = input_ids.numel()
        logits, _, _ = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )
        total_loss_w += loss.item() * batch_tokens
        total_tokens += batch_tokens
    avg_loss = total_loss_w / max(total_tokens, 1)
    return {
        'loss': avg_loss,
        'ppl': math.exp(min(avg_loss, 20)),
        'tokens': total_tokens,
    }


def main():
    p = argparse.ArgumentParser(description='Evaluate V11 checkpoints on val sets')
    p.add_argument(
        '--checkpoints', nargs='+', required=True,
        help='Checkpoint paths (compare multiple)',
    )
    p.add_argument('--labels', type=str, default='wiki,dclm',
                   help='Comma-separated: wiki, dclm')
    p.add_argument('--batch_size', type=int, default=18)
    p.add_argument('--seq_len', type=int, default=2048)
    p.add_argument('--dclm_holdout_tokens', type=int, default=500_000)
    p.add_argument('--edu_score_min', type=int, default=3)
    p.add_argument('--holdout_pct', type=int, default=5)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_sets = [s.strip().lower() for s in args.labels.split(',') if s.strip()]

    datasets = {}
    if 'wiki' in eval_sets:
        wiki_val, _ = load_wikitext103_val(seq_len=args.seq_len)
        datasets['wiki'] = wiki_val
    if 'dclm' in eval_sets:
        holdout = load_dclm_holdout_tokens(
            target_tokens=args.dclm_holdout_tokens,
            edu_score_min=args.edu_score_min,
            holdout_pct=args.holdout_pct,
        )
        datasets['dclm'] = TextDataset(holdout, args.seq_len)

    print(f"\nEval sets: {list(datasets.keys())} | device={device}\n")
    print(f"{'checkpoint':<55} | {'set':<5} | {'loss':>8} | {'ppl':>8} | {'tokens':>10}")
    print('-' * 95)

    for ckpt_path in args.checkpoints:
        name = Path(ckpt_path).parent.name + '/' + Path(ckpt_path).name
        print(f"Loading {ckpt_path}...")
        model, cfg = load_model(ckpt_path, device)
        bs = args.batch_size
        for set_name, ds in datasets.items():
            loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
            metrics = eval_ppl(model, loader, device)
            print(
                f"{name:<55} | {set_name:<5} | "
                f"{metrics['loss']:8.4f} | {metrics['ppl']:8.2f} | {metrics['tokens']:10,}"
            )
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print()


if __name__ == '__main__':
    main()
