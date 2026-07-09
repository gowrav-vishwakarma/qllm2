"""Correctness + speed harness for Flash-PAM (Track 1, chunk-parallel).

Compares `flash_pam.flash_pam_chunked_head` against the baseline
`V11PAMLayer._forward_chunked_head` for:
  * forward equality (max abs diff) in fp64 and bf16,
  * gradient equality (q,k,v,gamma),
  * wall-clock + peak memory, in eager AND under torch.compile.

Designed to run in *spare* VRAM next to a live training job: keep B small.

Run:
    .venv/bin/python -m v11.bench_flash_pam --B 2 --T 2048 --bench
"""

import argparse
import time
import torch

from v11.model import V11Config, V11PAMLayer
from v11.flash_pam import flash_pam_chunked_head
from v11.triton_kernels import set_triton_enabled


def _make_inputs(B, H, T, d, device, dtype, seed=0, requires_grad=False):
    g = torch.Generator(device=device).manual_seed(seed)
    def r():
        x = torch.randn(B, H, T, d, 2, device=device, dtype=dtype, generator=g)
        return x.requires_grad_(requires_grad)
    q, k, v = r(), r(), r()
    gamma = (torch.rand(B, H, T, device=device, dtype=dtype, generator=g) * 0.4 + 0.55)
    gamma = gamma.requires_grad_(requires_grad)
    return q, k, v, gamma


def _baseline_layer(H, d, C, device):
    cfg = V11Config(n_heads=H, head_dim=d, chunk_size=C, max_seq_len=4096)
    layer = V11PAMLayer(cfg).to(device)
    return layer


def correctness(args):
    dev = 'cuda'
    H, d, C = args.H, args.head_dim, args.chunk
    layer = _baseline_layer(H, d, C, dev)
    print(f"== Correctness  B={args.B} H={H} d={d} T={args.T} C={C} ==")
    # fp64 + triton off  -> exact math equivalence; fp32 + triton on -> production precision.
    for dtype, triton in ((torch.float64, False), (torch.float32, True)):
        set_triton_enabled(triton)
        layer_dt = layer.double() if dtype == torch.float64 else layer.float()
        q, k, v, gamma = _make_inputs(args.B, H, args.T, d, dev, dtype, requires_grad=True)
        # baseline
        yb, Sb = layer_dt._forward_chunked_head(q, k, v, gamma, d)
        lb = (yb.float() ** 2).sum()
        gb = torch.autograd.grad(lb, [q, k, v, gamma], retain_graph=False)
        # flash
        q2 = q.detach().clone().requires_grad_(True)
        k2 = k.detach().clone().requires_grad_(True)
        v2 = v.detach().clone().requires_grad_(True)
        gm2 = gamma.detach().clone().requires_grad_(True)
        yf, Sf = flash_pam_chunked_head(q2, k2, v2, gm2, d, C)
        lf = (yf.float() ** 2).sum()
        gf = torch.autograd.grad(lf, [q2, k2, v2, gm2], retain_graph=False)

        ydiff = (yb - yf).abs().max().item()
        sdiff = (Sb - Sf).abs().max().item()
        gnames = ['dq', 'dk', 'dv', 'dgamma']
        gdiffs = {n: (a - b).abs().max().item() for n, a, b in zip(gnames, gb, gf)}
        tag = str(dtype).replace('torch.', '') + ('/triton' if triton else '/notriton')
        print(f"  [{tag:18s}] y_max={ydiff:.2e} S_max={sdiff:.2e} "
              + " ".join(f"{n}={v:.2e}" for n, v in gdiffs.items()))


def _time_fn(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters * 1000
    peak = torch.cuda.max_memory_allocated() / 1e9
    return dt, peak


def bench(args):
    dev = 'cuda'
    H, d, C = args.H, args.head_dim, args.chunk
    layer = _baseline_layer(H, d, C, dev).float()
    q, k, v, gamma = _make_inputs(args.B, H, args.T, d, dev, torch.bfloat16)
    q, k, v = q.float(), k.float(), v.float()
    gamma = gamma.float()

    def base_fwd():
        return layer._forward_chunked_head(q, k, v, gamma, d)

    def flash_fwd():
        return flash_pam_chunked_head(q, k, v, gamma, d, C)

    def make_bwd(fwd):
        def run():
            qq = q.detach().requires_grad_(True)
            y, _ = fwd_with(qq)
            (y.float() ** 2).sum().backward()
        return run

    def fwd_with_base(qq):
        return layer._forward_chunked_head(qq, k, v, gamma, d)

    def fwd_with_flash(qq):
        return flash_pam_chunked_head(qq, k, v, gamma, d, C)

    print(f"\n== Bench  B={args.B} H={H} d={d} T={args.T} C={C}  "
          f"iters={args.iters} ==")
    print(f"  free VRAM headroom: keep B small next to training")

    variants = [("baseline", base_fwd, fwd_with_base),
                ("flash   ", flash_fwd, fwd_with_flash)]

    for name, fwd, fwd_w in variants:
        dt_f, pk_f = _time_fn(lambda: fwd(), args.iters, args.warmup)
        def bwd():
            qq = q.detach().requires_grad_(True)
            y, _ = fwd_w(qq)
            (y.float() ** 2).sum().backward()
        dt_b, pk_b = _time_fn(bwd, args.iters, args.warmup)
        print(f"  {name} eager      fwd={dt_f:6.2f}ms (pk {pk_f:4.1f}GB) "
              f"fwd+bwd={dt_b:6.2f}ms (pk {pk_b:4.1f}GB)")

    if args.compile:
        print("  -- torch.compile --")
        cbase = torch.compile(base_fwd)
        cflash = torch.compile(flash_fwd)
        for name, cfn in [("baseline", cbase), ("flash   ", cflash)]:
            try:
                dt, pk = _time_fn(lambda: cfn(), args.iters, args.warmup)
                print(f"  {name} compiled   fwd={dt:6.2f}ms (pk {pk:4.1f}GB)")
            except Exception as e:
                print(f"  {name} compiled   FAILED: {type(e).__name__}: {str(e)[:80]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--B', type=int, default=2)
    p.add_argument('--H', type=int, default=16)
    p.add_argument('--head_dim', type=int, default=64)
    p.add_argument('--T', type=int, default=2048)
    p.add_argument('--chunk', type=int, default=256)
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--bench', action='store_true')
    p.add_argument('--compile', action='store_true')
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    correctness(args)
    if args.bench:
        bench(args)


if __name__ == '__main__':
    main()
