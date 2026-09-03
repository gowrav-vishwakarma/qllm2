#!/usr/bin/env python3
"""Train matched ~100M Transformer or Mamba on the recall-program data mix.

Same fineweb+recall blend / token budget as V11 from-scratch, so behavioral
comparisons are architecture claims rather than pretrained-vs-scratch.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

# Run-as-script: put repo root on sys.path so `v7`/`v6` import (like `python -m`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _build_transformer(vocab_size: int = 50257, size: str = '100m'):
    from v6.transformer_baseline import TransformerConfig, TransformerLM, TRANSFORMER_CONFIGS
    key = size if size in TRANSFORMER_CONFIGS else '100m'
    cfg = copy.deepcopy(TRANSFORMER_CONFIGS[key])
    cfg.vocab_size = vocab_size
    cfg.dropout = 0.1
    return TransformerLM(cfg), cfg


def _build_mamba(vocab_size: int = 50257, size: str = '100m'):
    """Mamba from scratch. size=100m ≈130M-class; size=tiny ≈10M for micro tests."""
    from transformers import MambaConfig, MambaForCausalLM
    if size == 'tiny':
        cfg = MambaConfig(
            vocab_size=vocab_size,
            hidden_size=256,
            num_hidden_layers=8,
            state_size=16,
            intermediate_size=512,
            time_step_rank=16,
            use_cache=False,
        )
    else:
        cfg = MambaConfig(
            vocab_size=vocab_size,
            hidden_size=768,
            num_hidden_layers=24,
            state_size=16,
            intermediate_size=1536,
            time_step_rank=48,
            use_cache=False,
        )
    model = MambaForCausalLM(cfg)
    return model, cfg


def _save_transformer(path: Path, model, cfg, step: int, tokens: int, nparams: int):
    from dataclasses import asdict
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': asdict(cfg),
        'step': step,
        'global_tokens': tokens,
        'arch': 'transformer',
        'parameter_count': nparams,
    }, path)


def _save_mamba_hf(dir_path: Path, model, step: int, tokens: int, nparams: int):
    """HF directory so run_memory_behavioral --model-type hf can load it."""
    dir_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dir_path)
    meta = {'step': step, 'global_tokens': tokens, 'arch': 'mamba', 'parameter_count': nparams}
    (dir_path / 'train_meta.json').write_text(json.dumps(meta, indent=2) + '\n')


def _tx_forward(model, input_ids, device):
    T = input_ids.shape[1]
    pos = torch.arange(T, device=device)
    h = model.drop(model.token_embed(input_ids) + model.pos_embed(pos))
    for block in model.blocks:
        h = block(h)
    return model.lm_head(model.ln_f(h))


@torch.no_grad()
def _tx_val_loss(model, val_loader, device):
    model.eval()
    tot, n = 0.0, 0
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda', dtype=torch.float32):
            logits = _tx_forward(model, input_ids, device)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                                   reduction='sum')
        tot += float(loss); n += labels.numel()
    model.train()
    return tot / max(1, n)


def run_wikitext(args, device) -> int:
    """Apples-to-apples WikiText-103 control (transformer only), optional recall
    mix, matching v13_sempty's tokenizer / val protocol (fp32 val head)."""
    assert args.arch == 'transformer', 'wikitext path is the transformer control'
    from v7.data import load_wikitext103
    train_ds, val_ds, tokenizer = load_wikitext103(max_samples=None, seq_len=args.seq_len)
    vocab_size = len(tokenizer)
    if args.recall_frac > 0.0:
        from v13_sempty.data_mix import RecallMixDataset
        train_ds = RecallMixDataset(train_ds, frac=args.recall_frac,
                                    seq_len=args.seq_len, tokenizer=tokenizer, seed=args.seed)
        print(f"recall mix: {args.recall_frac:.1%} of train samples (val unmixed)")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model, cfg = _build_transformer(vocab_size=vocab_size, size=args.size)
    model = model.to(device)
    nparams = sum(p.numel() for p in model.parameters())
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    print(f"[transformer/wikitext] params={nparams:,} chunks={len(train_ds)} "
          f"steps/epoch={steps_per_epoch} total_steps={total_steps} device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    step, tokens, t0 = 0, 0, time.time()
    best = float('inf')
    model.train()
    for epoch in range(args.epochs):
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            for g in opt.param_groups:
                g['lr'] = lr_at(step)
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda', dtype=torch.bfloat16):
                logits = _tx_forward(model, input_ids, device)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            tokens += input_ids.numel(); step += 1
            if step % args.log_every == 0:
                tok_s = tokens / max(1e-6, time.time() - t0)
                print(f"  [tx] ep{epoch} step={step}/{total_steps} loss={float(loss):.4f} "
                      f"ppl={math.exp(min(20, float(loss))):.1f} tok/s={tok_s:.0f} gtok={tokens}",
                      flush=True)
            if args.val_every_steps and step % args.val_every_steps == 0:
                vl = _tx_val_loss(model, val_loader, device)
                tag = ''
                if vl < best:
                    best = vl
                    _save_transformer(args.checkpoint_dir / 'best_model.pt', model, cfg,
                                      step, tokens, nparams)
                    tag = ' *best*'
                print(f"  [tx val] step={step} val_loss={vl:.4f} val_ppl={math.exp(min(20, vl)):.2f}"
                      f"{tag} (best {math.exp(min(20, best)):.2f})", flush=True)
        # end-of-epoch val
        vl = _tx_val_loss(model, val_loader, device)
        if vl < best:
            best = vl
            _save_transformer(args.checkpoint_dir / 'best_model.pt', model, cfg, step, tokens, nparams)
        print(f"  [tx val] end-epoch {epoch} val_ppl={math.exp(min(20, vl)):.2f} "
              f"(best {math.exp(min(20, best)):.2f})", flush=True)
    _save_transformer(args.checkpoint_dir / 'final_model.pt', model, cfg, step, tokens, nparams)
    meta = {'arch': 'transformer', 'dataset': 'wikitext103', 'params': nparams,
            'tokens': tokens, 'steps': step, 'epochs': args.epochs,
            'recall_frac': args.recall_frac, 'best_val_ppl': math.exp(min(20, best)),
            'wall_s': time.time() - t0}
    (args.checkpoint_dir / 'train_meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f"[transformer/wikitext] done best_val_ppl={math.exp(min(20, best)):.2f} "
          f"wall={meta['wall_s']/3600:.2f}h -> {args.checkpoint_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', choices=('transformer', 'mamba'), required=True)
    ap.add_argument('--size', choices=('100m', 'tiny', '10m', '5m', '50m'), default='100m',
                    help='Model size class (transformer: TRANSFORMER_CONFIGS key; mamba tiny≈10M)')
    ap.add_argument('--token_budget', type=int, default=1_000_000_000)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--seq_len', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--warmup_steps', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--pretrain_sources', default='fineweb,recall')
    ap.add_argument('--pretrain_weights', default='96,3')
    ap.add_argument('--fineweb_name', default='sample-10BT')
    ap.add_argument('--edu_score_min', type=int, default=3)
    ap.add_argument('--checkpoint_dir', type=Path, required=True)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--save_every_steps', type=int, default=2000)
    ap.add_argument('--chat_vocab', action='store_true',
                    help='Use ChatML+reasoning tokenizer (50261) like V13 recall presets')
    ap.add_argument('--dataset', choices=('mix', 'wikitext103'), default='mix',
                    help='mix = FineWeb+recall stream (default); wikitext103 = the '
                         'apples-to-apples WikiText LM control for v13_sempty')
    ap.add_argument('--recall_frac', type=float, default=0.0,
                    help='(wikitext103 only) fraction of train samples replaced by '
                         'synthetic recall docs; matches v13_sempty --recall_frac')
    ap.add_argument('--epochs', type=int, default=10,
                    help='(wikitext103 only) passes over the corpus')
    ap.add_argument('--val_every_steps', type=int, default=2000,
                    help='(wikitext103 only) steps between val PPL evals')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == 'wikitext103':
        return run_wikitext(args, device)

    from v7.data import load_pretrain_mix
    sources = tuple(s.strip() for s in args.pretrain_sources.split(',') if s.strip())
    weights = tuple(float(w) for w in args.pretrain_weights.split(','))
    train_ds, val_ds, tokenizer = load_pretrain_mix(
        seq_len=args.seq_len,
        edu_score_min=args.edu_score_min,
        token_budget=args.token_budget,
        sources=sources,
        weights=weights,
        chat_vocab=args.chat_vocab,
        fineweb_name=args.fineweb_name,
        holdout_pct=5,
        mix_seed=args.seed,
        blend_warmup_tokens=0,
    )
    loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0)

    vocab_size = int(getattr(tokenizer, 'vocab_size', 50257) or 50257)
    if args.arch == 'transformer':
        model, cfg = _build_transformer(vocab_size=vocab_size, size=args.size)
    else:
        model, cfg = _build_mamba(vocab_size=vocab_size, size=args.size)
    model = model.to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[{args.arch}] params={nparams:,} device={device} budget={args.token_budget:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        # cosine to 10% of peak
        progress = (step - args.warmup_steps) / max(1, args.token_budget // (args.batch_size * args.seq_len) - args.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    step = 0
    tokens = 0
    t0 = time.time()
    model.train()
    best_loss = float('inf')
    running = 0.0
    running_n = 0

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        for g in opt.param_groups:
            g['lr'] = lr_at(step)

        with torch.amp.autocast('cuda', enabled=device.type == 'cuda', dtype=torch.bfloat16):
            if args.arch == 'transformer':
                T = input_ids.shape[1]
                pos = torch.arange(T, device=device)
                h = model.drop(model.token_embed(input_ids) + model.pos_embed(pos))
                for block in model.blocks:
                    h = block(h)
                logits = model.lm_head(model.ln_f(h))
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )
            else:
                out = model(input_ids=input_ids, labels=labels)
                loss = out.loss

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        toks = input_ids.numel()
        tokens += toks
        step += 1
        running += float(loss.detach())
        running_n += 1

        if step % args.log_every == 0:
            avg = running / max(1, running_n)
            tok_s = tokens / max(1e-6, time.time() - t0)
            print(f"  [{args.arch}] step={step} loss={avg:.4f} ppl={math.exp(min(20, avg)):.1f} "
                  f"tok/s={tok_s:.0f} gtok={tokens}", flush=True)
            running = 0.0
            running_n = 0

        if step % args.save_every_steps == 0:
            if args.arch == 'transformer':
                _save_transformer(args.checkpoint_dir / 'latest.pt', model, cfg, step, tokens, nparams)
            else:
                _save_mamba_hf(args.checkpoint_dir / 'latest_hf', model, step, tokens, nparams)

        if tokens >= args.token_budget:
            break

    if args.arch == 'transformer':
        _save_transformer(args.checkpoint_dir / 'final_model.pt', model, cfg, step, tokens, nparams)
        _save_transformer(args.checkpoint_dir / 'best_model.pt', model, cfg, step, tokens, nparams)
    else:
        _save_mamba_hf(args.checkpoint_dir / 'best_hf', model, step, tokens, nparams)
        # also mirror tokenizer for HF load convenience
        try:
            tokenizer.save_pretrained(args.checkpoint_dir / 'best_hf')
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] tokenizer save: {e}")
    meta = {
        'arch': args.arch, 'params': nparams, 'tokens': tokens, 'steps': step,
        'wall_s': time.time() - t0,
    }
    (args.checkpoint_dir / 'train_meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f"[{args.arch}] done tokens={tokens:,} wall={meta['wall_s']/3600:.2f}h -> {args.checkpoint_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
