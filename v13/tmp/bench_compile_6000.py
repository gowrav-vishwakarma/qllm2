"""Compare eager vs compile vs compile_blocks on trainer-faithful step.

    PYTHONPATH=. uv run python v13/tmp/bench_compile_6000.py
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
BATCH = 16
DELTA_CHUNK = 256
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


def bench_mode(mode, grad_ckpt=True):
    torch.cuda.empty_cache()
    gc.collect()
    cfg = get_config("v13_e3_k3_selective")
    cfg.delta_chunk = DELTA_CHUNK
    cfg.gradient_checkpointing = grad_ckpt
    torch.manual_seed(0)
    m = V13LM(cfg).cuda().train()
    if mode == "compile_blocks":
        m.compile_blocks(mode="default")
        hidden_fn = m._hidden_to_lm
    elif mode == "compile_hidden":
        hidden_fn = torch.compile(m._hidden_to_lm, mode="default")
    else:
        hidden_fn = m._hidden_to_lm
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
    ids = torch.randint(0, cfg.vocab_size, (BATCH, SEQ), device="cuda")
    labels = torch.randint(0, cfg.vocab_size, (BATCH, SEQ), device="cuda")

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lm, aux, gate_probs = hidden_fn(ids)
            loss, nll = m.ce_from_lm(lm, labels, chunk=4096, return_nll=True)
            loss = loss + aux
            if gate_probs is not None and cfg.gate_surprisal_lambda > 0:
                loss = loss + cfg.gate_surprisal_lambda * gate_surprisal_loss(
                    gate_probs, nll, labels, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        return loss

    try:
        for i in range(WARMUP):
            step()
            torch.cuda.synchronize()
            print("  warmup %d ok" % (i + 1), flush=True)
    except Exception as e:
        del m, opt
        torch.cuda.empty_cache()
        return None, None, str(e)[:120]

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / ITERS
    tok = BATCH * SEQ / dt
    peak = torch.cuda.max_memory_allocated() / 1e9
    del m, opt
    torch.cuda.empty_cache()
    return tok, peak, None


def main():
    print("B=%d C=%d grad_ckpt=ON trainer-faithful step" % (BATCH, DELTA_CHUNK))
    for mode in ("eager", "compile_hidden", "compile_blocks"):
        print("\n--- %s ---" % mode, flush=True)
        tok, peak, err = bench_mode(mode, grad_ckpt=True)
        if err:
            print("FAIL: %s" % err)
        else:
            print("RESULT: %8.0f tok/s  peak %.1f GB" % (tok, peak))


if __name__ == "__main__":
    main()
