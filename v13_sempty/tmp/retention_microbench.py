"""Retention micro-bench: what lengthens the needle-at-START horizon under LM
pressure?  (R2 retention program, 2026-09-06.)

Context. mix3b_delta_answ100 (100M, 3B tok) reads a needle placed just before
the query perfectly at every length incl. 8192, but a needle at the START of
the context decays with distance (a1 pos0: 1.00 @512, 0.45 @1024, 0.10 @2048,
0 @8192).  Retrieval is length-invariant; RETENTION over distance is the
limiter.  `dt_bias` never leaves its -4 init and the realised per-token
retention is 0.64-0.97 (< the 0.982 the bias alone gives): the web LM loss
pushes toward forgetting and nothing pushes back.  The pure-recall
micro-bench cannot show this (no LM pressure -> the model learns to retain);
this bench adds it.

Data (per step, one length bucket, byte-identical across arms):
  * ``web_frac`` of the batch = WikiText-103 train windows (LM loss on every
    token, weight 1) -- the forgetting pressure;
  * the rest = invented-association recall docs (records, filler, query ->
    value) with the reference recipe's loss: every token weight 1, the answer
    token weight ``answer_w`` (=100).  Needle positions 0 / 0.5 / 1.0, 1-8
    associations, so long-gap needles are in-distribution up to bucket 4096.

Eval: position-RESOLVED probe (pos0 = needle at start, pos1 = needle just
before the query) at ctx 256..8192 for a1/a4, + the LM guard (WikiText val
PPL on 2048 windows) + realised per-layer retention + dt_bias drift.

Arms (``--arms``): ``delta`` (reference, base_dt_bias -4), ``delta@dt-6``,
``delta@dt-8`` (longer default retention -- the bias never learns, so set it),
``base`` (additive, no delta).  (A ``nodecay`` arm existed 2026-09-06: it won
this bench outright -- a4 chance -> 1.00 to 8192 -- and then FAILED at 100M/3B
tok on both PPL and recall; the knob was removed.  CAVEAT: this bench did not
predict 100M behaviour for a decay knob -- 25 % recall docs at 27M/6k steps
under-represents the web-token load on the state.)

Usage (RTX 6000, ~110K tok/s, ~15 min/arm):
  PYTHONPATH=. .venv/bin/python -m v13_sempty.tmp.retention_microbench \
      --steps 6000 --batch 16 --arms delta,delta@dt-6,delta@dt-8
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import torch

from v13_sempty.config import PAMConfig
from v13_sempty.model import LM, _retention_capture
from v13_sempty.triton_kernels import set_kernel_enabled
from memory_probes.behavioral import build_example, build_suite, score_candidate_logits

WEB_TRAIN = Path('.cache/v7_tokens/wikitext103_train_v2_full_sl2048.pt')
WEB_VAL = Path('.cache/v7_tokens/wikitext103_validation_v2_full_sl2048.pt')


def _make_cfg(vocab_size, max_seq_len, delta=True, base_dt_bias=-4.0,
              head_dim=64, n_layers=4) -> PAMConfig:
    """Same small real-arm model as recall_microbench (chrono+gate, 27M)."""
    return PAMConfig(
        vocab_size=vocab_size, dim=6 * head_dim, n_heads=6, head_dim=head_dim,
        n_layers=n_layers, expand=3, dropout=0.0, max_seq_len=max_seq_len,
        chunk_size=256, gradient_checkpointing=False, is_complex=False,
        chrono=True, out_gate=True, delta=delta, base_dt_bias=base_dt_bias,
    )


def _load_tokens(path: Path) -> torch.Tensor:
    d = torch.load(path, weights_only=False)
    t = d['tokens'] if isinstance(d, dict) else d
    return t.to(torch.long).flatten()


def _build_schedule(steps, batch, web_frac, length_buckets, bucket_weights, seed,
                    n_web_tokens):
    """Per-step plan: (L, web_offsets, recall_specs). Byte-identical across arms."""
    rng = random.Random(seed)
    positions = (0.0, 0.5, 1.0)
    assoc_choices = (1, 2, 4, 6, 8)
    n_web = int(round(batch * web_frac))
    plan = []
    ex_seed = 20_000_000
    for _ in range(steps):
        L = rng.choices(length_buckets, weights=bucket_weights, k=1)[0]
        offs = [rng.randrange(0, n_web_tokens - L - 2) for _ in range(n_web)]
        specs = []
        for _ in range(batch - n_web):
            specs.append((L, rng.choice(assoc_choices), rng.choice(positions), ex_seed))
            ex_seed += 1
        plan.append((L, offs, specs))
    return plan


def _batch(tokenizer, web_tokens, L, offs, specs, device, answer_w):
    """Equal-length batch: web windows (weight 1) + recall docs (weight 1,
    answer token ``answer_w``).  Recall docs are built at ctx=L, so their length
    is L+1 ids -> x/y of length L; web windows are cut to match."""
    seqs, weights = [], []
    for (ctx, assoc, pos, seed) in specs:
        ex = build_example(tokenizer, context_tokens=ctx, target_position=pos,
                           associations=assoc, seed=seed, candidate_count=8)
        ids = ex.prompt_ids + [ex.target_token_id]
        w = [1.0] * (len(ids) - 1)
        w[-1] = answer_w
        seqs.append(ids); weights.append(w)
    n = len(seqs[0]) if seqs else L + 1
    for o in offs:
        ids = web_tokens[o:o + n].tolist()
        seqs.append(ids); weights.append([1.0] * (n - 1))
    # recall docs from build_example can differ by a token or two in length;
    # trim every sequence to the shortest so the batch is rectangular.
    n = min(len(s) for s in seqs)
    seqs = [s[:n] for s in seqs]
    weights = [w[:n - 1] for w in weights]
    # keep the answer weight on the (possibly trimmed) recall docs' last label
    for i, (ctx, assoc, pos, seed) in enumerate(specs):
        weights[i][-1] = answer_w
    t = torch.tensor(seqs, dtype=torch.long, device=device)
    x, y = t[:, :-1].contiguous(), t[:, 1:].contiguous()
    mask = torch.tensor(weights, dtype=torch.float32, device=device)
    return x, y, mask


@torch.no_grad()
def _eval_probe(model, tokenizer, device, ctxs, assocs, positions, trials):
    """accuracy[(ctx, assoc, pos)] -- position-resolved."""
    model.eval()
    suite = build_suite(tokenizer, context_lengths=ctxs, positions=positions,
                        association_counts=assocs, seeds=tuple(range(trials)),
                        candidate_count=8)
    hit = {(c, a, p): [0, 0] for c in ctxs for a in assocs for p in positions}
    for ex in suite:
        x = torch.tensor([ex.prompt_ids], dtype=torch.long, device=device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _, _ = model(x)
        last = logits[0, -1].float()
        cand = [float(last[i]) for i in ex.candidate_token_ids]
        res = score_candidate_logits(ex, cand)
        rec = hit[(ex.context_tokens, ex.associations, ex.target_position)]
        rec[0] += int(res['correct']); rec[1] += 1
    model.train()
    return {k: v[0] / max(v[1], 1) for k, v in hit.items()}


@torch.no_grad()
def _eval_lm(model, val_tokens, device, n_windows=32, L=2048, seed=0):
    """LM guard: mean CE over fixed held-out WikiText windows -> PPL."""
    model.eval()
    rng = random.Random(seed)
    tot, cnt = 0.0, 0
    for _ in range(n_windows):
        o = rng.randrange(0, val_tokens.numel() - L - 2)
        t = val_tokens[o:o + L + 1].to(device).unsqueeze(0)
        x, y = t[:, :-1], t[:, 1:]
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            lm, _ = model._hidden_to_lm(x)
        loss = model.ce_from_lm(lm, y, chunk=4096, gemm_dtype=torch.float32)
        tot += float(loss); cnt += 1
    model.train()
    return math.exp(tot / cnt)


@torch.no_grad()
def _realised_retention(model, val_tokens, device, L=2048, seed=1):
    """Per-layer mean realised retention on one web window (+ dt_bias)."""
    model.eval()
    for blk in model.blocks:
        blk.pam.capture_decay = True
    _retention_capture.clear()
    o = random.Random(seed).randrange(0, val_tokens.numel() - L - 2)
    x = val_tokens[o:o + L].to(device).unsqueeze(0)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        model(x)
    per_layer = [float(r.mean()) for r in _retention_capture]
    for blk in model.blocks:
        blk.pam.capture_decay = False
    _retention_capture.clear()
    dtb = [float(blk.pam.dt_bias.mean()) for blk in model.blocks]
    model.train()
    return per_layer, dtb


def _train_arm(label, cfg, tokenizer, web_tokens, val_tokens, plan, device, lr,
               log_every, answer_w, ctxs, assocs, positions, trials):
    torch.manual_seed(0)
    model = LM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=len(plan), pct_start=0.05,
        anneal_strategy='cos', div_factor=10, final_div_factor=10)
    model.train()
    t0 = time.time(); toks = 0
    for step, (L, offs, specs) in enumerate(plan, 1):
        x, y, mask = _batch(tokenizer, web_tokens, L, offs, specs, device, answer_w)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            lm, _ = model._hidden_to_lm(x)
        loss = model.ce_from_lm(lm, y, loss_mask=mask, chunk=4096, gemm_dtype=torch.float32)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        toks += x.numel()
        if step % log_every == 0 or step == 1:
            tps = toks / (time.time() - t0)
            print(f"  [{label}] step {step:4d}/{len(plan)} L={L:5d} loss={loss.item():.4f} "
                  f"| {tps:,.0f} tok/s", flush=True)
    train_s = time.time() - t0
    ppl = _eval_lm(model, val_tokens, device)
    ret, dtb = _realised_retention(model, val_tokens, device)
    acc = _eval_probe(model, tokenizer, device, ctxs, assocs, positions, trials)
    print(f"  [{label}] {n_params/1e6:.1f}M params, trained {train_s:.0f}s | "
          f"wikitext val PPL {ppl:.2f} | realised retention/layer "
          f"{' '.join(f'{r:.3f}' for r in ret)} | dt_bias/layer "
          f"{' '.join(f'{b:.2f}' for b in dtb)}", flush=True)
    return {'acc': acc, 'ppl': ppl, 'ret': ret, 'dt_bias': dtb}


def _parse_arm(tok: str):
    kind, dt = tok.strip(), -4.0
    if '@' in kind:
        kind, mod = kind.split('@', 1)
        if mod.startswith('dt'):
            dt = float(mod[2:])
        else:
            raise SystemExit(f"unknown arm modifier '{mod}'")
    if kind not in ('base', 'delta'):
        raise SystemExit(f"unknown arm '{kind}'")
    label = 'chrono+gate' + ('+delta' if kind == 'delta' else '') + f' dt{dt:g}'
    return label, kind == 'delta', dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--web_frac', type=float, default=0.75)
    p.add_argument('--answer_w', type=float, default=100.0)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--trials', type=int, default=24)
    p.add_argument('--log_every', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--arms', default='delta,delta@dt-6,delta@dt-8')
    args = p.parse_args()

    set_kernel_enabled(True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab = len(tokenizer)
    web_tokens = _load_tokens(WEB_TRAIN)
    val_tokens = _load_tokens(WEB_VAL)

    length_buckets = [256, 512, 1024, 2048, 4096]
    bucket_weights = [0.30, 0.28, 0.22, 0.12, 0.08]
    max_seq_len = max(length_buckets + [8192])
    eval_ctxs = (256, 512, 1024, 2048, 4096, 8192)
    eval_assocs = (1, 4)
    eval_pos = (0.0, 1.0)

    plan = _build_schedule(args.steps, args.batch, args.web_frac, length_buckets,
                           bucket_weights, args.seed, web_tokens.numel())
    print(f"retention-bench: {args.steps} steps x B{args.batch}, web_frac {args.web_frac} "
          f"(WikiText-103 train, {web_tokens.numel()/1e6:.0f}M tok), answer_w {args.answer_w}, "
          f"buckets {length_buckets} w={bucket_weights}, eval ctx {eval_ctxs} "
          f"assoc {eval_assocs} pos {eval_pos}, {args.trials} trials, vocab {vocab}", flush=True)

    results = {}
    for tok in args.arms.split(','):
        label, delta, dt = _parse_arm(tok)
        print(f"\n--- arm: {label} (delta={delta}, base_dt_bias={dt}) ---", flush=True)
        cfg = _make_cfg(vocab, max_seq_len, delta=delta, base_dt_bias=dt)
        results[label] = _train_arm(label, cfg, tokenizer, web_tokens, val_tokens, plan,
                                    args.device, args.lr, args.log_every, args.answer_w,
                                    eval_ctxs, eval_assocs, eval_pos, args.trials)

    print("\n" + "=" * 72)
    print("LM guard (WikiText val PPL, 32x2048 windows):")
    for label, r in results.items():
        print(f"  {label.ljust(34)} {r['ppl']:8.2f}")
    for a in eval_assocs:
        for pos, tag in ((0.0, 'pos0 = needle at START (retention)'),
                         (1.0, 'pos1 = needle just before query (retrieval)')):
            print(f"\na{a} {tag}:")
            print("  arm".ljust(36) + "".join(f"{c:>7}" for c in eval_ctxs))
            for label, r in results.items():
                row = "  " + label.ljust(34)
                for c in eval_ctxs:
                    row += f"{r['acc'][(c, a, pos)]:7.2f}"
                print(row)


if __name__ == '__main__':
    main()
