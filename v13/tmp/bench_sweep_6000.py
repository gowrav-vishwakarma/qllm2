"""Trainer-faithful v13 throughput sweep: batch size x delta_chunk on big GPU.

    uv run python v13/tmp/bench_sweep_6000.py
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
WARMUP = 2
ITERS = 6


def gate_surprisal_loss(gate_probs, nll, labels, m_cfg):
    _, T = labels.shape
    surprisal = nll
    valid = labels != -100
    median_ce = surprisal[valid].median()
    tau = max(getattr(m_cfg, "gate_surprisal_tau", 1.0), 1e-3)
    sign = getattr(m_cfg, "gate_surprisal_sign", 1.0)
    target_p = torch.sigmoid(sign * (median_ce - surprisal) / tau).detach()
    gp = gate_probs.float().clamp(1e-4, 1 - 1e-4)
    tgt = target_p.float()
    val = valid.float()
    num = gp.new_zeros(())
    den = gp.new_zeros(())
    chunk = 256
    for c0 in range(0, T, chunk):
        c1 = min(c0 + chunk, T)
        g = gp[:, :, c0:c1]
        t = tgt[:, c0:c1].unsqueeze(0).expand_as(g)
        vm = val[:, c0:c1].unsqueeze(0).expand_as(g)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            bce = F.binary_cross_entropy(g, t, reduction="none")
        num = num + (bce * vm).sum()
        den = den + vm.sum()
    return num / den.clamp_min(1.0)


def bench(batch, delta_chunk, grad_ckpt):
    torch.cuda.empty_cache()
    gc.collect()
    try:
        cfg = get_config("v13_e3_k3_selective")
        cfg.delta_chunk = delta_chunk
        cfg.gradient_checkpointing = grad_ckpt
        torch.manual_seed(0)
        m = V13LM(cfg).cuda().train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
        ids = torch.randint(0, cfg.vocab_size, (batch, SEQ), device="cuda")
        labels = torch.randint(0, cfg.vocab_size, (batch, SEQ), device="cuda")

        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lm, aux, gate_probs = m._hidden_to_lm(ids)
                loss, nll = m.ce_from_lm(lm, labels, chunk=4096, return_nll=True)
                loss = loss + aux
                if gate_probs is not None and cfg.gate_surprisal_lambda > 0:
                    loss = loss + cfg.gate_surprisal_lambda * gate_surprisal_loss(
                        gate_probs, nll, labels, cfg)
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
        tok_s = batch * SEQ / dt
        peak = torch.cuda.max_memory_allocated() / 1e9
        del m, opt, ids, labels
        torch.cuda.empty_cache()
        return tok_s, peak, None
    except RuntimeError as e:
        torch.cuda.empty_cache()
        gc.collect()
        if "out of memory" in str(e).lower():
            return None, None, "OOM"
        raise


def main():
    print("=== v13_e3_k3_selective trainer-path sweep (T=2048, bf16, fused_ce, gate ON) ===")
    print("%7s %3s %4s %8s %6s" % ("ckpt", "B", "C", "tok/s", "GB"))
    print("-" * 36)

    results = []
    for ckpt in (True, False):
        label = "ckpt" if ckpt else "no-ckpt"
        for b in [8, 12, 16, 18, 20, 24, 28, 32, 36, 40, 44, 48, 56, 64]:
            tok, gb, err = bench(b, 128, ckpt)
            if err:
                print("%7s %3d %4d %8s %6s" % (label, b, 128, "OOM", "-"))
                break
            print("%7s %3d %4d %8.0f %6.1f" % (label, b, 128, tok, gb))
            results.append((tok, label, b, 128, gb))

    best_by_mode = {}
    for tok, label, b, c, gb in results:
        if label not in best_by_mode or tok > best_by_mode[label][0]:
            best_by_mode[label] = (tok, b, c, gb)

    for label, (best_tok, best_b, _, _) in best_by_mode.items():
        ckpt = label == "ckpt"
        print("\n=== C sweep @ B=%d (%s, peak %.0f tok/s) ===" % (best_b, label, best_tok))
        for c in [32, 64, 128, 256]:
            if c == 128:
                continue
            tok, gb, err = bench(best_b, c, ckpt)
            if err:
                print("  C=%3d OOM" % c)
            else:
                print("  C=%3d  %8.0f tok/s  %6.1f GB" % (c, tok, gb))
                results.append((tok, label, best_b, c, gb))

    print("\n=== TOP 5 overall ===")
    for tok, label, b, c, gb in sorted(results, reverse=True)[:5]:
        print("  %8.0f tok/s  B=%d C=%d %-7s  %5.1f GB" % (tok, b, c, label, gb))


if __name__ == "__main__":
    main()
