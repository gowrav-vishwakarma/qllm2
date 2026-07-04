"""Full training-step benchmark for V11 (fwd + loss + bwd + optimizer).

Measures tok/s and peak VRAM for a real optimizer step, so speed/memory work
(fused E3, fused CE, selective recompute, ...) is compared apples-to-apples on
the actual training path — not just an isolated PAM kernel.

Run (needs a free GPU; compile warmup is ~minutes per config):
    .venv/bin/python -m v11.bench_step --preset v11_e3_k3_chat --compile \
        --variants loop,fused --B 18 --T 2048

Each `--variants` token toggles a config flag combo:
    loop   -> fused_e3=False
    fused  -> fused_e3=True
    fused+ckpt -> fused_e3=True, gradient_checkpointing=True
"""

import argparse
import time

import torch
import torch.nn.functional as F

from v11.model import V11LM, get_config


def _parse_variant(variant: str):
    """Return (cfg_mutator, use_fused_ce). Tokens joined by '+'."""
    use_fused_ce = False
    flags = dict(fused_e3=True, gradient_checkpointing=False, recompute_pam_chunks=False)
    for tok in variant.split('+'):
        tok = tok.strip()
        if tok in ('loop', 'nofuse'):
            flags['fused_e3'] = False
        elif tok == 'fused':
            flags['fused_e3'] = True
        elif tok == 'ckpt':
            flags['gradient_checkpointing'] = True
        elif tok in ('recompute', 'rc'):
            flags['recompute_pam_chunks'] = True
        elif tok == 'ce':
            use_fused_ce = True
        else:
            raise ValueError(f"unknown variant token '{tok}'")
    return flags, use_fused_ce


def run(variant, preset, B, T, iters, warmup, compile_, dtype):
    torch.manual_seed(0)
    flags, use_fused_ce = _parse_variant(variant)
    cfg = get_config(preset)
    for k, v in flags.items():
        setattr(cfg, k, v)
    raw = V11LM(cfg).cuda().train()
    opt = torch.optim.AdamW(raw.parameters(), lr=1e-4, betas=(0.9, 0.95), fused=True)
    # For fused CE we compile the stack (_hidden_to_lm) and keep chunked CE eager,
    # so the [N,vocab] logits tensor is never built. For the standard path we
    # compile the full model.
    if compile_:
        model_fn = torch.compile(raw)
        hidden_fn = torch.compile(raw._hidden_to_lm)
    else:
        model_fn, hidden_fn = raw, raw._hidden_to_lm
    ids = torch.randint(0, cfg.vocab_size, (B, T), device='cuda')
    lbl = torch.randint(0, cfg.vocab_size, (B, T), device='cuda')

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=dtype):
            if use_fused_ce:
                lm = hidden_fn(ids)
                loss = raw.ce_from_lm(lm, lbl)
            else:
                logits, _, _ = model_fn(ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), lbl.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
        opt.step()
        return loss

    print(f"[{variant}] warmup/compile ({warmup} steps)...", flush=True)
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    tok = B * T / dt
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[RESULT {variant:14s}] {dt*1000:7.1f} ms/step  {tok:8.0f} tok/s  "
          f"peak {peak:5.1f} GB  (B={B} T={T} preset={preset})", flush=True)
    del raw, opt, model_fn, hidden_fn
    torch.cuda.empty_cache()
    return tok, peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preset', default='v11_e3_k3_chat')
    p.add_argument('--variants', default='loop,fused')
    p.add_argument('--B', type=int, default=18)
    p.add_argument('--T', type=int, default=2048)
    p.add_argument('--iters', type=int, default=8)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--dtype', default='bf16', choices=['bf16', 'fp16'])
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float16

    results = {}
    for variant in args.variants.split(','):
        variant = variant.strip()
        results[variant] = run(variant, args.preset, args.B, args.T,
                                args.iters, args.warmup, args.compile, dtype)

    if 'loop' in results and 'fused' in results:
        base_tok = results['loop'][0]
        fused_tok = results['fused'][0]
        print(f"\nspeedup fused/loop: {fused_tok / base_tok:.2f}x", flush=True)


if __name__ == '__main__':
    main()
