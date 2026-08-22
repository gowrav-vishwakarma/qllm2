"""Re-runnable: per-component timing of the trainer step, gate ON vs OFF.
Isolates which part costs the ~7K tok/s gap (2026-08-22): forward stash,
CE forward, gate-target/BCE compute, backward.

    cd qllm2 && .venv/bin/python v13/tmp/probe_gate_overhead.py --batch 16
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
import torch.nn.functional as F
from v13.model import V13LM, get_config


def gate_loss_fn(gate_probs, nll, labels, cfg):
    B, T = labels.shape
    s = nll
    valid = labels != -100
    med = s[valid].median()
    tau = max(getattr(cfg, 'gate_surprisal_tau', 1.0), 1e-3)
    sign = getattr(cfg, 'gate_surprisal_sign', 1.0)
    tgt = torch.sigmoid(sign * (med - s) / tau).detach()
    gp = gate_probs.float().clamp(1e-4, 1 - 1e-4)
    vm = valid.unsqueeze(0).expand_as(gp).float()
    with torch.amp.autocast(device_type='cuda', enabled=False):
        bce = F.binary_cross_entropy(gp, tgt.unsqueeze(0).expand_as(gp), reduction='none')
    return (bce * vm).sum() / vm.sum().clamp_min(1.0)


def main(batch, iters=8, warmup=2, T=2048):
    cfg = get_config('v13_e3_k3_selective')
    torch.manual_seed(0)
    m = V13LM(cfg).cuda()
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    ids = torch.randint(0, 50261, (batch, T), device='cuda')
    labels = torch.randint(0, 50261, (batch, T), device='cuda')

    def timed(fn, n=3):
        ts = []
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return sum(ts) / n

    def step(use_gate):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lm, aux, gp = m._hidden_to_lm(ids)
            loss, nll = m.ce_from_lm(lm, labels, chunk=1024, return_nll=True)
            loss = loss + aux
            if use_gate:
                loss = loss + cfg.gate_surprisal_lambda * gate_loss_fn(gp, nll, labels, cfg)
        loss.backward()
        opt.step()

    for _ in range(warmup):
        step(True)
        step(False)

    # full steps
    t_on = timed(lambda: step(True), iters)
    t_off = timed(lambda: step(False), iters)
    print(f"full step  gate-ON {t_on*1000:8.1f} ms  ({batch*T/t_on:7.0f} tok/s)")
    print(f"full step  gate-OFF {t_off*1000:8.1f} ms  ({batch*T/t_off:7.0f} tok/s)")

    # components (no backward)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        f = timed(lambda: m._hidden_to_lm(ids), 3)
    print(f"fwd _hidden_to_lm:        {f*1000:8.1f} ms")
    lm, aux, gp = m._hidden_to_lm(ids)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        c = timed(lambda: m.ce_from_lm(lm, labels, chunk=1024, return_nll=True), 3)
    print(f"fwd ce_from_lm(nll):      {c*1000:8.1f} ms")
    loss, nll = m.ce_from_lm(lm, labels, chunk=1024, return_nll=True)
    g = timed(lambda: gate_loss_fn(gp, nll, labels, cfg), 5)
    print(f"fwd gate loss compute:    {g*1000:8.1f} ms")
    # backward only
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss_g, _ = m.ce_from_lm(lm, labels, chunk=1024, return_nll=True)
        loss_g = loss_g + aux + cfg.gate_surprisal_lambda * gate_loss_fn(gp, nll, labels, cfg)
        loss_c, _ = m.ce_from_lm(lm, labels, chunk=1024, return_nll=True)
        loss_c = loss_c + aux
    b_on = timed(lambda: loss_g.backward(), 2)
    print(f"bwd full (gate ON):       {b_on*1000:8.1f} ms")
    m.zero_grad(set_to_none=True)
    b_off = timed(lambda: loss_c.backward(), 2)
    print(f"bwd full (gate OFF):      {b_off*1000:8.1f} ms")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--iters', type=int, default=8)
    a = ap.parse_args()
    main(a.batch, a.iters)
