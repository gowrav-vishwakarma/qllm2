"""Re-runnable: isolate the gate-surprisal backward cost (2026-08-22).

The NLL byproduct made the gate-target forward ~free (0.2ms), yet gate-ON
full steps still cost ~730ms more than gate-OFF at B16/T2048. This probe
times, in isolation (fresh graph each call):
  * main CE + PAM trunk forward+backward (the baseline everyone shares)
  * + gate stash forward (inside _hidden_to_lm, config-gated, in both)
  * + gate BCE loss forward
  * + gate BCE backward ALONE (zeroing all other grads first)
to pin down where the 730ms lives.

    cd qllm2 && .venv/bin/python v13/tmp/probe_gate_bwd2.py --batch 16
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


def timed(fn, n=3):
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return sum(ts) / n


def main(batch, iters=6, T=2048):
    cfg = get_config('v13_e3_k3_selective')
    torch.manual_seed(0)
    m = V13LM(cfg).cuda()
    m.train()
    ids = torch.randint(0, 50261, (batch, T), device='cuda')
    labels = torch.randint(0, 50261, (batch, T), device='cuda')

    def fwd():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lm, aux, gp = m._hidden_to_lm(ids)
            loss, nll = m.ce_from_lm(lm, labels, chunk=1024, return_nll=True)
        return lm, aux, gp, loss, nll

    # warm
    for _ in range(2):
        m.zero_grad(set_to_none=True)
        _, _, _, loss, _ = fwd()
        loss.backward()

    # 1) trunk fwd+bwd only (no gate loss term)
    def trunk_bwd():
        m.zero_grad(set_to_none=True)
        _, aux, _, loss, _ = fwd()
        (loss + aux).backward()
    t_trunk = timed(trunk_bwd, iters)

    # 2) full fwd+bwd WITH gate loss term
    def full_bwd():
        m.zero_grad(set_to_none=True)
        _, aux, gp, loss, nll = fwd()
        (loss + aux + cfg.gate_surprisal_lambda * gate_loss_fn(gp, nll, labels, cfg)).backward()
    t_full = timed(full_bwd, iters)

    # 3) gate BCE backward ALONE: build gate loss, zero trunk grads, bwd only it
    m.zero_grad(set_to_none=True)
    _, aux, gp, loss, nll = fwd()
    gl = gate_loss_fn(gp, nll, labels, cfg)
    # zero all param grads so only gate path contributes; gl graph is separate
    m.zero_grad(set_to_none=True)
    t_gate_bwd = timed(lambda: (cfg.gate_surprisal_lambda * gl).backward(retain_graph=True) or m.zero_grad(set_to_none=True), 3)

    print(f"trunk fwd+bwd (no gate term): {t_trunk*1000:8.1f} ms  ({batch*T/t_trunk:7.0f} tok/s)")
    print(f"full  fwd+bwd (gate term):    {t_full*1000:8.1f} ms  ({batch*T/t_full:7.0f} tok/s)")
    print(f"  -> gate term overhead:      {(t_full-t_trunk)*1000:8.1f} ms")
    print(f"gate BCE bwd alone (isolated):{t_gate_bwd*1000:8.1f} ms")
    # report protect_gate grad magnitudes to confirm it's real
    pg = [p.grad.abs().max().item() for n, p in m.named_parameters()
          if p.grad is not None and 'protect_gate' in n]
    print(f"protect_gate |grad|max after full: {max(pg):.3e} (n={len(pg)})")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--iters', type=int, default=6)
    a = ap.parse_args()
    main(a.batch, a.iters)
