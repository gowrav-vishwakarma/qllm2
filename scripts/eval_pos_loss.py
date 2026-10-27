"""Per-position loss on long natural text: does later context make tokens easier?

The LLM-side version of the needle probe. For each of N long documents
(PG-19 test books by default) take one window of --context tokens, run the
model once, and average next-token loss over documents in position buckets
(0-64, 64-128, ..., up to --context). Raw position curves mix model
behaviour with document structure (front matter, chapter starts), so each
token is also scored with only --short..2*--short tokens of context
(overlapping windows, second half scored). ``far_gain`` = short-context loss
minus full-context loss on the SAME tokens: > 0 means context beyond --short
tokens helps; ~0 means the model does not use it (its horizon).

Usage:
  .venv/bin/python scripts/eval_pos_loss.py --model-type v13_sempty \
     --checkpoint checkpoints_v13_sempty/<run>/best_model.pt \
     --context 8192 --docs 50 --output logs/pos_loss/<name>.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from run_memory_behavioral import _load_transformer, _load_v13_sempty  # noqa: E402


def _buckets(context: int) -> list:
    edges, e = [0], 64
    while e < context:
        edges.append(e)
        e *= 2
    edges.append(context)
    return list(zip(edges[:-1], edges[1:]))


def _docs(tokenizer, n_docs: int, context: int, skip_tokens: int, source: str):
    from datasets import load_dataset
    if source == 'pg19':
        ds = load_dataset('emozilla/pg19', split='test', streaming=True)
        field = 'text'
    else:
        raise ValueError(source)
    out = []
    for row in ds:
        ids = tokenizer(row[field], add_special_tokens=False)['input_ids']
        need = skip_tokens + context + 1
        if len(ids) < need:
            continue
        out.append(torch.tensor(ids[skip_tokens:need], dtype=torch.long))
        if len(out) >= n_docs:
            break
    return out


@torch.no_grad()
def _token_losses(model, model_type: str, ids: torch.Tensor, device) -> torch.Tensor:
    x = ids[:-1].unsqueeze(0).to(device)
    y = ids[1:].to(device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == 'cuda'):
        out = model(x)
    logits = out[0] if isinstance(out, (tuple, list)) else getattr(out, 'logits', out)
    logits = logits[0]
    losses = []
    for s in range(0, logits.shape[0], 1024):  # chunked fp32 CE
        losses.append(F.cross_entropy(logits[s:s + 1024].float(), y[s:s + 1024],
                                      reduction='none'))
    return torch.cat(losses).cpu()


def _short_ctx_losses(model, model_type, ids, device, short: int) -> torch.Tensor:
    """Loss of every token when it may see only short..2*short past tokens.

    Window k covers [(k-1)*short, (k+1)*short) and scores its second half;
    the first `short` positions keep their (already short) full context.
    """
    n = ids.numel() - 1
    out = torch.full((n,), float('nan'))
    out[:short] = _token_losses(model, model_type, ids[:short + 1], device)
    k = 1
    while k * short < n:
        a, b = (k - 1) * short, min((k + 1) * short, n)
        win = _token_losses(model, model_type, ids[a:b + 1], device)
        out[k * short:b] = win[short:short + (b - k * short)]
        k += 1
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--model-type', choices=('v13_sempty', 'transformer'), required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--preset', default='baseline_real_pm')
    p.add_argument('--context', type=int, default=8192)
    p.add_argument('--docs', type=int, default=50)
    p.add_argument('--skip-tokens', type=int, default=1024,
                   help='skip each book\'s first tokens (Gutenberg header)')
    p.add_argument('--source', default='pg19')
    p.add_argument('--short', type=int, default=512,
                   help='context cap for the far-gain comparison (0 = off)')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model_type == 'v13_sempty':
        model, tokenizer, cfg = _load_v13_sempty(args.checkpoint, args.preset, device)
    else:
        model, tokenizer, cfg = _load_transformer(args.checkpoint, device)
        limit = getattr(cfg, 'max_position_embeddings', None) or getattr(cfg, 'max_seq_len', None)
        if limit and args.context > limit:
            print(f'transformer context limit {limit}: --context {args.context} -> {limit}')
            args.context = int(limit)
    docs = _docs(tokenizer, args.docs, args.context, args.skip_tokens, args.source)
    print(f'{len(docs)} docs x {args.context} tokens from {args.source}', flush=True)

    t0 = time.time()
    total = torch.zeros(args.context)
    total_short = torch.zeros(args.context)
    for i, ids in enumerate(docs):
        total += _token_losses(model, args.model_type, ids, device)
        if args.short:
            total_short += _short_ctx_losses(model, args.model_type, ids, device, args.short)
        if (i + 1) % 10 == 0:
            print(f'  {i + 1}/{len(docs)} docs ({time.time() - t0:.0f}s)', flush=True)
    mean = total / len(docs)
    mean_short = total_short / len(docs)
    rows = []
    for a, b in _buckets(args.context):
        loss = mean[a:b].mean().item()
        row = {'from': a, 'to': b, 'loss': loss, 'ppl': math.exp(loss)}
        msg = f'  [{a:>5},{b:>5})  loss {loss:.4f}  ppl {math.exp(loss):7.2f}'
        if args.short and a >= args.short:
            sl = mean_short[a:b].mean().item()
            row.update(short_loss=sl, far_gain=sl - loss)
            msg += f'   short-ctx {sl:.4f}  far_gain {sl - loss:+.4f}'
        rows.append(row)
        print(msg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump({'checkpoint': str(args.checkpoint), 'model_type': args.model_type,
               'source': args.source, 'docs': len(docs), 'context': args.context,
               'skip_tokens': args.skip_tokens, 'short': args.short,
               'buckets': rows},
              open(args.output, 'w'), indent=1)
    print(f'wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
