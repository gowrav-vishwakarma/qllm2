"""Why v11 compile wins but v13 delta does not: A/B on same trainer path.

    PYTHONPATH=. uv run python v13/tmp/bench_compile_v11_vs_v13.py
"""
import gc
import time

import torch
import torch.nn.functional as F

from v13.model import V13LM, get_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

SEQ = 2048
WARMUP = 3
ITERS = 6


def gate_surprisal_loss(gate_probs, nll, labels, m_cfg):
    _, T = labels.shape
    valid = labels != -100
    median_ce = nll[valid].median()
    tau = max(getattr(m_cfg, "gate_surprisal_tau", 1.0), 1e-3)
    sign = getattr(m_cfg, "gate_surprisal_sign", 1.0)
    target_p = torch.sigmoid(sign * (median_ce - nll) / tau).detach()
    gp = gate_probs.float().clamp(1e-4, 1 - 1e-4)
    tgt = target_p.float()
    val = valid.float()
    num = gp.new_zeros(())
    den = gp.new_zeros(())
    for c0 in range(0, T, 256):
        c1 = min(c0 + 256, T)
        g = gp[:, :, c0:c1]
        t = tgt[:, c0:c1].unsqueeze(0).expand_as(g)
        vm = val[:, c0:c1].unsqueeze(0).expand_as(g)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            bce = F.binary_cross_entropy(g, t, reduction="none")
        num = num + (bce * vm).sum()
        den = den + vm.sum()
    return num / den.clamp_min(1.0)


def bench(preset, batch, delta_chunk, grad_ckpt, compile_hidden):
    torch.cuda.empty_cache()
    gc.collect()
    cfg = get_config(preset)
    if hasattr(cfg, "delta_chunk") and delta_chunk is not None:
        cfg.delta_chunk = delta_chunk
    cfg.gradient_checkpointing = grad_ckpt
    torch.manual_seed(0)
    m = V13LM(cfg).cuda().train()
    hidden_fn = m._hidden_to_lm
    if compile_hidden:
        hidden_fn = torch.compile(hidden_fn, mode="default")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
    ids = torch.randint(0, cfg.vocab_size, (batch, SEQ), device="cuda")
    labels = torch.randint(0, cfg.vocab_size, (batch, SEQ), device="cuda")
    gsl = getattr(cfg, "gate_surprisal_lambda", 0.0)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lm, aux, gate_probs = hidden_fn(ids)
            loss, nll = m.ce_from_lm(lm, labels, chunk=4096, return_nll=True)
            loss = loss + aux
            if gsl > 0 and gate_probs is not None:
                loss = loss + gsl * gate_surprisal_loss(gate_probs, nll, labels, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / ITERS
    tok = batch * SEQ / dt
    peak = torch.cuda.max_memory_allocated() / 1e9
    del m, opt
    torch.cuda.empty_cache()
    return tok, peak


def row(label, preset, batch, dc, ckpt, comp):
    try:
        tok, gb = bench(preset, batch, dc, ckpt, comp)
    except torch.cuda.OutOfMemoryError:
        print("%-28s %s B=%2d  %8s       OOM" % (label, "compile" if comp else "eager  ", batch, "-"))
        torch.cuda.empty_cache()
        gc.collect()
        return
    tag = "compile" if comp else "eager  "
    print("%-28s %s B=%2d  %8.0f tok/s  %5.1f GB" % (label, tag, batch, tok, gb))


def main():
    print("Same trainer path, T=2048, fused_ce, bf16\n")
    # v11 additive: production used --no_grad_ckpt --compile B=18/32 on 6000 Pro
    for comp in (False, True):
        row("v11 additive (no ckpt)", "v11_e3_k3_chat", 18, None, False, comp)
    for comp in (False, True):
        row("v11 additive (no ckpt)", "v11_e3_k3_chat", 32, None, False, comp)
    print()
    # v13 delta: needs grad ckpt; compile tested here
    for comp in (False, True):
        row("v13 delta (ckpt ON)", "v13_e3_k3_selective", 16, 256, True, comp)


if __name__ == "__main__":
    main()
