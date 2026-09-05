"""Recall micro-bench: does the A3 delta rule de-interfere the read path?

Throwaway experiment tool (v*/tmp, no commit needed per AGENTS.md). Rationale
(2026-09-05): two 1B-token runs (mix-3B, L1) proved that big pretrains measure
PPL well and the recall HORIZON badly, and cost hours each. The L1 verdict was
that retention is not the bottleneck -- RETRIEVAL is (q.S cannot recover a
k v^T binding after other tokens are written; interference). The delta rule
(A3) erases the old value under key k before writing, which is exactly the
de-interference operator. Before spending another 1B run we test it here, in
minutes, on the pure task:

  * train a SMALL real-arm PAM on invented-association recall documents drawn
    from the SAME family as memory_probes.build_example (records "K means V",
    filler, "query: K means" -> V), lengths bucketed per step so there is no
    padding, associations 1-8 so interference is in-distribution;
  * score the behavioral probe (contrastive next-token over candidate values)
    at ctx 128..8192, associations 1/4/8;
  * run two arms on byte-identical data/seeds: chrono+gate (the reference) and
    chrono+gate+delta. Everything else equal.

Verdict rule: delta must lift a8 (8-way) accuracy clearly above chance (~0.12)
where the reference is stuck at chance, without wrecking a1, to justify a 1B
run. Otherwise A3 is removed like R1.

OUTCOME (2026-09-05): at 1500 steps NEITHER PAM nor a same-size transformer
(`xf` positive-control arm) gets a4/a8 above chance -- the bench has no ceiling
at that budget. At 6000 steps PAM chrono+gate+delta (27M, 4 layers) scores
1.00 on a1/a4/a8 at every ctx 128..8192 (2x its train window) while the
abs-pos transformer control stays at chance. See EXPERIMENTS_SEMPY.md
"Positive control". Always run >= 6000 steps to judge multi-way.

Usage (RTX 6000):
  .venv/bin/python -m v13_sempty.tmp.recall_microbench --steps 6000 --batch 16 \
      --arms xf,delta
"""
from __future__ import annotations

import argparse
import math
import random
import time

import torch

from v13_sempty.config import PAMConfig
from v13_sempty.model import LM
from v13_sempty.triton_kernels import set_kernel_enabled
from memory_probes.behavioral import (
    build_example, build_suite, score_candidate_logits,
)


def _make_cfg(vocab_size: int, delta: bool, max_seq_len: int,
              head_dim: int = 64, n_layers: int = 4) -> PAMConfig:
    """Small real-arm model: chrono + out_gate is the reference; delta optional.

    dim 384 / 6 heads / head_dim 64 => a 64x64 matrix memory per head. 4 layers
    keeps a step well under a second at ctx<=4096. ``head_dim`` was swept (no
    a8 gain -> multi-way is not a capacity knob). dim tracks 6*head_dim so the
    per-head width stays the driver. ``n_layers`` is the depth knob (recall
    circuits may need depth rather than width).
    """
    return PAMConfig(
        vocab_size=vocab_size, dim=6 * head_dim, n_heads=6, head_dim=head_dim,
        n_layers=n_layers, expand=3, dropout=0.0, max_seq_len=max_seq_len,
        chunk_size=256, gradient_checkpointing=False, is_complex=False,
        chrono=True, out_gate=True, delta=delta,
    )


class _XfArm(torch.nn.Module):
    """Positive-control arm: a plain causal transformer (v6 GPT-2-style, SDPA
    flash attention, learned absolute positions) sized like the PAM arm
    (d_model 384, 6 heads, d_ff 4x). Attention has an O(T) KV cache, so this is
    the ceiling for in-context exact recall INSIDE its window. Positions >= the
    longest train bucket (4096) are never trained, so the 8192 column is
    expected to fail -- that is the real transformer limitation, not a bug.

    Exposes the two entry points the bench uses: ``forward(x) -> (logits, None,
    None)`` and ``masked_ce(x, y, mask)``."""

    def __init__(self, vocab_size: int, max_seq_len: int, n_layers: int,
                 d_model: int = 384, n_heads: int = 6):
        super().__init__()
        from v6.transformer_baseline import TransformerConfig, TransformerLM
        self.net = TransformerLM(TransformerConfig(
            vocab_size=vocab_size, max_seq_len=max_seq_len, d_model=d_model,
            n_layers=n_layers, n_heads=n_heads, d_ff=4 * d_model, dropout=0.0,
            tie_weights=True))

    def forward(self, x):
        return self.net(x), None, None

    def masked_ce(self, x, y, mask):
        """Same weighted-mask semantics as the PAM path: sum(mask*CE)/sum(mask)."""
        B, T = x.shape
        net = self.net
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        h = net.token_embed(x) + net.pos_embed(pos)
        for block in net.blocks:
            h = block(h)
        h = net.ln_f(h)
        if float(mask[:, :-1].abs().max()) == 0.0:
            # Answer-only: score just the final position (no B x T x V tensor).
            logits = net.lm_head(h[:, -1]).float()
            return torch.nn.functional.cross_entropy(logits, y[:, -1])
        logits = net.lm_head(h)
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1),
            reduction='none')
        m = mask.reshape(-1)
        return (ce * m).sum() / m.sum().clamp_min(1.0)


def _build_schedule(steps, batch, length_buckets, bucket_weights, seed):
    """Pre-draw the per-step (length, [example-specs]) plan so both arms train
    on byte-identical data. Each spec is (ctx, associations, position, seed)."""
    rng = random.Random(seed)
    positions = (0.0, 0.5, 1.0)
    assoc_choices = (1, 2, 4, 6, 8)
    plan = []
    ex_seed = 10_000_000
    for _ in range(steps):
        L = rng.choices(length_buckets, weights=bucket_weights, k=1)[0]
        specs = []
        for _ in range(batch):
            a = rng.choice(assoc_choices)
            p = rng.choice(positions)
            specs.append((L, a, p, ex_seed))
            ex_seed += 1
        plan.append((L, specs))
    return plan


def _batch_from_specs(tokenizer, specs, device, aux_weight: float = 0.0):
    """Build one equal-length batch: input = records+filler+query, labels shifted,
    with the answer value appended so the final label IS the retrieval target.

    The loss mask is 1 at the final position (the one predicting the queried
    value). Full-sequence CE is ~all filler+record-copy (>=99.9 % of tokens are
    trivially predictable), so it drives train ppl to 1.0 while giving the
    retrieval token almost no gradient -- the first micro-bench run showed
    exactly that (train ppl 1.0, probe at chance for both arms). Answer-only
    supervision (aux_weight=0) concentrates every gradient on the read path.

    BUT answer-only removes the intermediate LM signal that key-matching
    circuits (previous-token -> match -> copy) normally form from; the
    transformer positive control plateaued at chance on a4/a8 too. With
    ``aux_weight=w>0`` the other positions get total weight ``w`` (each
    ``w/(T-1)``), so the fused CE (which divides by mask.sum()) computes
    exactly ``(CE_answer + w * mean CE_others) / (1 + w)`` per sequence."""
    seqs = []
    for (ctx, assoc, pos, seed) in specs:
        ex = build_example(
            tokenizer, context_tokens=ctx, target_position=pos,
            associations=assoc, seed=seed, candidate_count=8,
        )
        seqs.append(ex.prompt_ids + [ex.target_token_id])
    t = torch.tensor(seqs, dtype=torch.long, device=device)
    x, y = t[:, :-1].contiguous(), t[:, 1:].contiguous()
    T = y.shape[1]
    mask = torch.full_like(y, aux_weight / max(T - 1, 1), dtype=torch.float32)
    mask[:, -1] = 1.0
    return x, y, mask


@torch.no_grad()
def _eval_probe(model, tokenizer, device, ctxs, assocs, trials):
    model.eval()
    suite = build_suite(
        tokenizer, context_lengths=ctxs, positions=(0.0, 0.5, 1.0),
        association_counts=assocs, seeds=tuple(range(trials)), candidate_count=8,
    )
    # accuracy[(ctx, assoc)] over positions x seeds
    hit = {(c, a): [0, 0] for c in ctxs for a in assocs}
    for ex in suite:
        x = torch.tensor([ex.prompt_ids], dtype=torch.long, device=device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _, _ = model(x)
        last = logits[0, -1].float()
        cand = [float(last[i]) for i in ex.candidate_token_ids]
        res = score_candidate_logits(ex, cand)
        rec = hit[(ex.context_tokens, ex.associations)]
        rec[0] += int(res['correct'])
        rec[1] += 1
    model.train()
    return {k: v[0] / max(v[1], 1) for k, v in hit.items()}


def _fmt_table(name, acc, ctxs, assocs):
    lines = [f"== {name}  (accuracy; chance ~0.12)"]
    lines.append("ctx    " + "".join(f"{c:>7}" for c in ctxs))
    for a in assocs:
        row = f"a{a:<5}"
        for c in ctxs:
            row += f"{acc[(c, a)]:7.2f}"
        lines.append(row)
    return "\n".join(lines)


def _train_arm(name, cfg, tokenizer, plan, device, lr, log_every,
               ctxs, assocs, trials, aux_weight: float = 0.0):
    """``cfg`` is a PAMConfig (PAM arm) or an already-built ``_XfArm``. Both
    arms see the identical ``plan`` (same documents, same order, same seeds)."""
    torch.manual_seed(0)
    model = (cfg if isinstance(cfg, torch.nn.Module) else LM(cfg)).to(device)
    is_xf = isinstance(model, _XfArm)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=len(plan), pct_start=0.05,
        anneal_strategy='cos', div_factor=10, final_div_factor=10)
    model.train()
    t0 = time.time()
    toks = 0
    for step, (L, specs) in enumerate(plan, 1):
        x, y, mask = _batch_from_specs(tokenizer, specs, device, aux_weight)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            if is_xf:
                loss = model.masked_ce(x, y, mask)
            else:
                lm, _ = model._hidden_to_lm(x)
        if not is_xf:
            loss = model.ce_from_lm(lm, y, loss_mask=mask, chunk=4096,
                                    gemm_dtype=torch.float32)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        toks += x.numel()
        if step % log_every == 0 or step == 1:
            tps = toks / (time.time() - t0)
            print(f"  [{name}] step {step:4d}/{len(plan)} L={L:5d} "
                  f"loss={float(loss):.4f} ppl={math.exp(min(float(loss),20)):7.1f} "
                  f"| {tps:,.0f} tok/s", flush=True)
    acc = _eval_probe(model, tokenizer, device, ctxs, assocs, trials)
    print(f"  [{name}] trained {n_params/1e6:.1f}M params in "
          f"{time.time()-t0:.0f}s")
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--trials', type=int, default=24)
    p.add_argument('--log_every', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--aux_weight', type=float, default=0.0,
                   help="total loss weight of the non-answer positions "
                        "(0 = answer-only). loss=(CE_ans + w*mean CE_other)/(1+w)")
    p.add_argument('--arms', default='base,delta',
                   help="comma list of arms. Each: base|delta|xf with optional "
                        "@<head_dim> (default 64, PAM only) and :L<n_layers> "
                        "(default 4). 'xf' is the transformer positive control. "
                        "e.g. 'xf,delta,delta:L8,delta@128:L12'")
    args = p.parse_args()

    set_kernel_enabled(True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab = len(tokenizer)

    # Train lengths bucketed (no padding); short buckets weighted up for speed.
    # 8192 stays EVAL-ONLY -> tests extrapolation past the training window.
    length_buckets = [256, 512, 1024, 2048, 4096]
    bucket_weights = [0.30, 0.28, 0.22, 0.12, 0.08]
    max_seq_len = max(length_buckets + [8192])
    eval_ctxs = (128, 256, 512, 1024, 2048, 4096, 8192)
    eval_assocs = (1, 4, 8)

    plan = _build_schedule(args.steps, args.batch, length_buckets,
                           bucket_weights, args.seed)
    print(f"micro-bench: {args.steps} steps x B{args.batch}, buckets "
          f"{length_buckets} w={bucket_weights}, eval ctx {eval_ctxs}, "
          f"{args.trials} trials, vocab {vocab}, aux_weight={args.aux_weight}",
          flush=True)

    # Parse arms: base | delta | xf, optional '@<head_dim>' and ':L<n_layers>'.
    arm_specs = []
    for tok in args.arms.split(','):
        tok = tok.strip()
        hd, nl = 64, 4
        if ':L' in tok:
            tok, nl = tok.split(':L'); nl = int(nl)
        if '@' in tok:
            tok, hd = tok.split('@'); hd = int(hd)
        if tok not in ('base', 'delta', 'xf'):
            raise SystemExit(f"unknown arm '{tok}'")
        if tok == 'xf':
            label = f"transformer(ctrl)@L{nl}"
        else:
            label = f"chrono+gate{'+delta' if tok == 'delta' else ''}@hd{hd}L{nl}"
        arm_specs.append((label, tok, hd, nl))

    results = {}
    for label, kind, hd, nl in arm_specs:
        print(f"\n--- arm: {label} (kind={kind}, head_dim={hd}, n_layers={nl}) ---",
              flush=True)
        if kind == 'xf':
            cfg = _XfArm(vocab, max_seq_len, n_layers=nl)
        else:
            cfg = _make_cfg(vocab, kind == 'delta', max_seq_len, head_dim=hd,
                            n_layers=nl)
        results[label] = _train_arm(
            label, cfg, tokenizer, plan, args.device, args.lr,
            args.log_every, eval_ctxs, eval_assocs, args.trials,
            aux_weight=args.aux_weight)

    print("\n" + "=" * 60)
    for name in results:
        print(_fmt_table(name, results[name], eval_ctxs, eval_assocs))
        print()
    # Headline rows: a1 (clean single-binding readback) and a8 (interference).
    for a in (1, 8):
        print(f"a{a} by arm x ctx:")
        print("  arm".ljust(32) + "".join(f"{c:>7}" for c in eval_ctxs))
        for label in results:
            row = "  " + label.ljust(30)
            for c in eval_ctxs:
                row += f"{results[label][(c, a)]:7.2f}"
            print(row)


if __name__ == '__main__':
    main()
