"""Re-runnable: quantify the gate-surprisal backward cost at B8/T2048 (2026-08-22).

The gate aux stash is computed INSIDE each gradient-checkpointed block. When the
gate loss backprops, the engine re-runs the checkpointed block to recover the
stash node's saved tensors — a suspected second PAM forward. This measures:
  A) trunk fwd+bwd (no gate term)
  B) trunk fwd+bwd WITH the gate loss term (stash in ckpt)
  C) trunk fwd+bwd with gate term but gate_probs DETACHED (no stash node ->
     no recompute) = the "stash is free" lower bound.
B - C isolates the stash-in-checkpoint recompute cost.

    cd qllm2 && .venv/bin/python v13/tmp/probe_gate_ckpt_cost.py --batch 8
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


def timed(fn, n=4):
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return sum(ts) / n


def main(batch, iters=4, T=2048):
    cfg = get_config('v13_e3_k3_selective')
    torch.manual_seed(0)
    m = V13LM(cfg).cuda()
    m.train()
    ids = torch.randint(0, 50261, (batch, T), device='cuda')
    labels = torch.randint(0, 50261, (batch, T), device='cuda')

    def make(mode):
        m.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lm, aux, gp = m._hidden_to_lm(ids)
            loss, nll = m.ce_from_lm(lm, labels, chunk=1024, return_nll=True)
            loss = loss + aux
            if mode in ('gate', 'detach'):
                g = gp.detach() if mode == 'detach' else gp
                loss = loss + cfg.gate_surprisal_lambda * gate_loss_fn(g, nll, labels, cfg)
        return loss

    for mode in ('trunk', 'gate', 'detach'):
        for _ in range(2):
            make(mode).backward()

    t_trunk = timed(lambda: make('trunk').backward(), iters)
    t_gate = timed(lambda: make('gate').backward(), iters)
    t_det = timed(lambda: make('detach').backward(), iters)
    print(f"trunk  fwd+bwd (no gate):  {t_trunk*1000:8.1f} ms  ({batch*T/t_trunk:7.0f} tok/s)")
    print(f"gate   fwd+bwd (in ckpt):  {t_gate*1000:8.1f} ms  ({batch*T/t_gate:7.0f} tok/s)")
    print(f"detach fwd+bwd (no node):  {t_det*1000:8.1f} ms  ({batch*T/t_det:7.0f} tok/s)")
    print(f"  stash-in-ckpt recompute (gate - detach): {(t_gate-t_det)*1000:8.1f} ms")
    print(f"  total gate overhead       (gate - trunk): {(t_gate-t_trunk)*1000:8.1f} ms")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--iters', type=int, default=4)
    a = ap.parse_args()
    main(a.batch, a.iters)
