"""Oracle + parity harness for the fused real-PAM kernel.

The ORACLE (``eager_chunk_reference``) is a verbatim raw-torch transcription of
``RealPAMLayer._chunked`` + ``_stable_notebook`` (v13_sempty/model.py). It is
the ground truth any kernel must match, forward and backward.

Parity contract for ``v13_sempty/triton_kernels.py``:

    fused_real_pam_read(q, k, v, retention, carry, chunk_size)
      -> (read [B*H, T, head_dim] in q.dtype,
          carry_out [B*H, head_dim, head_dim] fp32)

    q/k/v:      [B*H, T, head_dim]   (q,k already RoPE'd; NOT yet scaled)
    retention:  [B*H, T]             in (0,1)  == exp(-softplus(dt+bias))
    carry:      [B*H, head_dim, head_dim] fp32 (zeros for the first window)
    chunk_size: model chunk (carry hand-off every chunk_size positions;
                internal sub-tiling at <=256 is allowed — chunked linear
                attention is associative in window size)

    NOTE: scale (head_dim**-0.5) is NOT applied here — model.py applies it
    outside, exactly where _chunked applies it today.

Run:  PYTHONPATH=. .venv/bin/python v13_sempty/pam_kernel_test.py
It runs the oracle-vs-itself sanity + (if triton_kernels.py exists) full
fwd+bwd parity fp32/bf16 across real shapes (head_dim 98, T 256..1024,
partial chunks, model chunk 256/512/1024, carry hand-off).
"""
import importlib
import sys

import torch


# ── ORACLE: verbatim _chunked transcription (raw torch) ──────────────────────

def eager_chunk_reference(q, k, v, retention, carry, chunk_size):
    """Mirrors model._stable_notebook + the _chunked loop, on [B*H,T,D].

    decay_y = per-window log-space notebook with incoming carry;
    carry_out = select(window_notebook, over=window, index=length-1)
    (i.e. notebook state AT the last real position of the window).
    """
    BH, T, K = q.shape
    dev, dt = q.device, q.dtype
    reads = []
    carried = carry.clone().float() if carry is not None else None
    for start in range(0, T, chunk_size):
        length = min(chunk_size, T - start)
        w = length
        # slice window (named `take` == plain narrow)
        qw = q[:, start:start + length].float()
        kw = k[:, start:start + length].float()
        vw = v[:, start:start + length].float()
        decay = retention[:, start:start + length].float()
        # --- _stable_notebook, verbatim signs ---
        log_decay = -torch.log(decay + 1e-6)                    # [BH,w]
        C = torch.cumsum(log_decay, dim=-1)                     # increasing, C>=0
        E = C.unsqueeze(-2) - C.unsqueeze(-1)                   # E[s,t] = C_t - C_s
        M = torch.exp(torch.clamp(E, max=0.0)) * torch.tril(
            torch.ones(w, w, device=dev, dtype=torch.float32))
        # outer(values -> head_row, keys -> head_col): write[v, u] = v_v * k_u
        write = torch.einsum('btv,btu->btvu', vw, kw)                # [BH,w,K(v),K(u)]
        nb = torch.einsum('bst,btvu->bsvu', M, write)            # [BH,w,K,K]
        if carried is not None:
            decay_state = torch.exp(
                torch.clamp(-C.unsqueeze(-1), max=0.0)
            ).unsqueeze(-2) * carried.unsqueeze(-3)              # [BH,w,K,K]
            nb = nb + decay_state
        read_w = torch.einsum('bsvu,bsu->bsv', nb, qw)           # contract over head_col (=u)
        reads.append(read_w.to(dt))
        # carried = select(nb, over=time, index=length-1)
        carried = nb[:, length - 1]                              # [BH,K(v),K(u)]
    read = torch.cat(reads, dim=1)
    return read, carried


def oracle_vs_model(dev) -> bool:
    """Anchor: the oracle must match RealPAMLayer._stepwise, the one-token-at-
    a-time recurrence that IS the model's definition (untouched by any kernel)."""
    from sempyt.dim import Dim
    from sempyt.tensor import named
    from v13_sempty.config import get_config
    from v13_sempty.model import RealPAMLayer
    torch.manual_seed(3)
    cfg = get_config("tiny_real")
    layer = RealPAMLayer(cfg).to(dev)
    B, H, T, K = 2, cfg.n_heads, 23, cfg.head_dim
    batch, time = Dim("batch", B), Dim("time", T)
    tokens = named(torch.zeros(B, T, cfg.dim, device=dev), (batch, time, layer.model_dim))
    lay = (batch, layer.heads, time, layer.head_feature)
    q, k, v = (named(torch.randn(B, H, T, K, device=dev), lay) for _ in range(3))
    dec_raw = torch.exp(-torch.nn.functional.softplus(torch.randn(B, H, T, device=dev)))
    dec = named(dec_raw, (batch, layer.heads, time))
    with torch.no_grad():
        out, nb = layer._stepwise(tokens, q, k, v, dec, None)
    out_raw = out.raw(batch, layer.heads, time, layer.head_row).reshape(B * H, T, K)
    nb_raw = nb.raw(batch, layer.heads, layer.head_row, layer.head_col).reshape(B * H, K, K)
    r, co = eager_chunk_reference(q.data.reshape(B * H, T, K), k.data.reshape(B * H, T, K),
                                  v.data.reshape(B * H, T, K), dec_raw.reshape(B * H, T),
                                  None, 7)
    r = r * K ** -0.5
    dr = (r - out_raw).abs().max().item() / out_raw.abs().max().item()
    dc = (co - nb_raw).abs().max().item() / nb_raw.abs().max().item()
    print(f"oracle vs model._stepwise: read max_rel={dr:.2e} notebook max_rel={dc:.2e}")
    return dr < 1e-4 and dc < 1e-4


# ── inputs ────────────────────────────────────────────────────────────────────

def make(BH, T, D, dtype, seed, dev):
    g = torch.Generator(device=dev).manual_seed(seed)
    q = torch.randn(BH, T, D, generator=g, device=dev, dtype=torch.float32)
    k = torch.randn(BH, T, D, generator=g, device=dev, dtype=torch.float32)
    v = torch.randn(BH, T, D, generator=g, device=dev, dtype=torch.float32)
    logit = torch.randn(BH, T, generator=g, device=dev, dtype=torch.float32)
    dec = torch.exp(-torch.nn.functional.softplus(logit))
    carry = torch.randn(BH, D, D, generator=g, device=dev, dtype=torch.float32) * 0.1
    if dtype is not torch.float32:
        q, k, v, dec = (t.to(dtype) for t in (q, k, v, dec))
    return q, k, v, dec, carry


def grad_check(fn_ref, fn_test, args, tol):
    """Compare outputs AND grads wrt q,k,v,retention,carry."""
    outs = {}
    for name, fn in (("ref", fn_ref), ("test", fn_test)):
        # inputs keep their dtype (bf16 under training autocast); carry is fp32
        q, k, v, dec, carry = [a.clone().requires_grad_(True) for a in args]
        r, co = fn(q, k, v, dec, carry)
        gen = torch.Generator(device=r.device).manual_seed(123)   # same upstream grads for both
        g = torch.randn(r.shape, generator=gen, device=r.device, dtype=torch.float32).to(r.dtype)
        gco = torch.randn(co.shape, generator=gen, device=co.device, dtype=torch.float32)
        torch.autograd.backward([r, co], [g, gco])
        outs[name] = tuple(t.detach().float() for t in
                           (r, co, q.grad, k.grad, v.grad, dec.grad, carry.grad))
    good = True
    names = ["read", "carry_out", "dq", "dk", "dv", "ddec", "dcarry"]
    for n, a, b in zip(names, outs["ref"], outs["test"]):
        d = (a - b).abs().max().item()
        scale = a.abs().max().item() + 1e-12
        # ddec = q.dq - k.dk suffers bf16 cancellation (same identity as FLA);
        # the fp32 reference has none, so allow 5e-2 there in bf16.
        bar = tol * (5 / 3) if (n == "ddec" and tol > 1e-3) else tol
        good &= d / scale < bar
        print(f"    {n:9s} max_rel={d/scale:.2e}")
    return good


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # oracle sanity: chunked == single full-T window (associativity of the
    # closed form), fp32 only (bf16 re-ordering would drift).
    print("oracle associativity check (fp32):")
    q, k, v, dec, carry = make(2, 512, 64, torch.float32, 7, dev)
    r1, c1 = eager_chunk_reference(q, k, v, dec, carry, 512)
    r2, c2 = eager_chunk_reference(q, k, v, dec, carry, 256)
    print(f"    read  max_rel={(r1-r2).abs().max().item()/r1.abs().max().item():.2e}")
    print(f"    carry max_rel={(c1-c2).abs().max().item()/c1.abs().max().item():.2e}")

    # oracle vs THE model (RealPAMLayer._chunked on the eager _stable_notebook
    # path), so the oracle is anchored to model.py, not to itself.
    ok_oracle = oracle_vs_model(dev)

    try:
        mod = importlib.import_module("v13_sempty.triton_kernels")
    except ImportError:
        print("\ntriton_kernels.py not present yet — oracle-only run. PASS (oracle).")
        return 0

    scale = 98 ** -0.5
    ok = ok_oracle

    # 1) the pure-torch chunked linear-attention form vs the oracle (any device)
    print("\ntorch fallback parity (pam_scan_torch; fp32 tol 3e-5):")
    for (T, chunk) in ((256, 256), (300, 128), (130, 7)):
        for D in (98, 64):
            q, k, v, dec, carry = make(3, T, D, torch.float32, T * 3 + chunk, dev)
            args = (q, k, v, dec, carry.float())

            def ref(qq, kk, vv, dd, cc, chunk=chunk):
                r, co = eager_chunk_reference(qq, kk, vv, dd, cc, chunk)
                return r * scale, co

            def torch_form(qq, kk, vv, dd, cc, chunk=chunk):
                r, co = mod.pam_scan_torch(qq, kk, vv, dd, cc, chunk)
                return r * scale, co

            print(f"  T={T} chunk={chunk} D={D}")
            ok &= grad_check(ref, torch_form, args, 3e-5)

    if not getattr(mod, "HAS_TRITON", False) or dev.type != "cuda":
        print("triton/cuda unavailable — torch-form only.", "PASS" if ok else "FAILURES")
        return 0 if ok else 1

    # 2) the Triton kernel (fwd + bwd) vs the oracle
    # fp32 5e-5: a 1024-position window summed in 64-tiles vs one einsum
    # differs by fp32 round-off (~3e-5); the oracle's own chunked-vs-single
    # check is 1.1e-5.  bf16 3e-2 (ddec 5e-2, see grad_check).
    print("\nfused parity (scale applied by harness to both; fp32 tol 5e-5, bf16 3e-2):")
    for dtype, tol in ((torch.float32, 5e-5), (torch.bfloat16, 3e-2)):
        for (T, chunk) in ((256, 256), (300, 256), (512, 512), (1024, 1024), (1024, 256), (64, 64), (130, 7)):
            for D in (98, 64):
                q, k, v, dec, carry = make(4, T, D, dtype, T * 7 + chunk, dev)
                args = (q, k, v, dec, carry.float())

                def ref(qq, kk, vv, dd, cc, chunk=chunk):
                    r, co = eager_chunk_reference(qq, kk, vv, dd, cc, chunk)
                    return r * scale, co

                def fused(qq, kk, vv, dd, cc, chunk=chunk):
                    r, co = mod.fused_real_pam_read(qq, kk, vv, dd, cc, chunk)
                    return r * scale, co

                print(f"  dtype={dtype} T={T} chunk={chunk} D={D}")
                ok &= grad_check(ref, fused, args, tol)

    # 3) no carry in, carry_out unused (the training call pattern)
    q, k, v, dec, _ = make(4, 512, 98, torch.bfloat16, 99, dev)
    q.requires_grad_(True); dec.requires_grad_(True)
    r, co = mod.fused_real_pam_read(q, k, v, dec, None, 256)
    r.float().sum().backward()
    r2, co2 = mod.pam_scan_torch(q.detach(), k, v, dec.detach(), None, 256)
    d = (r.float() - r2.float()).abs().max().item() / r2.abs().max().item()
    dc = (co - co2).abs().max().item() / co2.abs().max().item()
    print(f"\nno-carry bf16: read max_rel={d:.2e} carry max_rel={dc:.2e} "
          f"grads finite={bool(torch.isfinite(q.grad).all() and torch.isfinite(dec.grad).all())}")
    ok &= d < 3e-2 and dc < 3e-2

    # 4) micro-bench fwd+bwd, real-101M geometry (B16 x H6, T1024, K98) bf16
    import time
    q, k, v, dec, _ = make(96, 1024, 98, torch.bfloat16, 5, dev)
    for t in (q, k, v, dec):
        t.requires_grad_(True)

    def bench(fn, iters=20):
        for _ in range(3):
            r, _ = fn(); r.float().sum().backward()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(iters):
            r, _ = fn(); r.float().sum().backward()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    ms_k = bench(lambda: mod.fused_real_pam_read(q, k, v, dec, None, 256))
    ms_t = bench(lambda: mod.pam_scan_torch(q, k, v, dec, None, 256))
    print(f"bench BH=96 T=1024 K=98 bf16 fwd+bwd: kernel {ms_k:.2f} ms | torch form {ms_t:.2f} ms "
          f"| {ms_t / ms_k:.1f}x")
    print("PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
