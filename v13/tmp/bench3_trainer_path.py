"""Re-runnable: TRAINER-FAITHFUL 100M train-step bench (bench2.py missed the
gate-surprisal aux loss, which cost ~6.8GB at B16 and caused the 500M-restart
OOM on 2026-08-22). Replicates v7/train.py:train_epoch: _hidden_to_lm under
autocast, ce_from_lm(chunk=...), aux + gate_surprisal_lambda * gate BCE loss,
backward, AdamW step. Wall-clock tok/s + peak mem.

    .venv/bin/python v13/tmp/bench3_trainer_path.py --batch 14
    .venv/bin/python v13/tmp/bench3_trainer_path.py --batch 14 --no_gate
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
import torch.nn.functional as F
from v13.model import V13LM, get_config


def gate_surprisal_loss(gate_probs, nll, labels, m_cfg):
    """Copy of V7Trainer._gate_surprisal_loss (v7/train.py) taking the EXACT
    per-token NLL byproduct from the main fused CE (O(1) in vocab — no second
    O(V) head GEMM; 2026-08-22). Target/BCE math is identical to the legacy
    linear_ce_per_token version (verified bit-exact,
    test_nll_byproduct_equivalence.py)."""
    B, T = labels.shape
    surprisal = nll  # [B,T] fp32, detached
    valid = labels != -100
    median_ce = surprisal[valid].median()
    tau = max(getattr(m_cfg, 'gate_surprisal_tau', 1.0), 1e-3)
    sign = getattr(m_cfg, 'gate_surprisal_sign', 1.0)
    target_p = torch.sigmoid(sign * (median_ce - surprisal) / tau).detach()
    gp = gate_probs.float().clamp(1e-4, 1 - 1e-4)
    tgt = target_p.float()
    val = valid.float()
    chunk = 256
    num = gp.new_zeros(())
    den = gp.new_zeros(())
    for c0 in range(0, T, chunk):
        c1 = min(c0 + chunk, T)
        g = gp[:, :, c0:c1]
        t = tgt[:, c0:c1].unsqueeze(0).expand_as(g)
        vm = val[:, c0:c1].unsqueeze(0).expand_as(g)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            bce = F.binary_cross_entropy(g, t, reduction='none')
        num = num + (bce * vm).sum()
        den = den + vm.sum()
    return num / den.clamp_min(1.0)


def main(batch, delta_chunk=128, iters=8, warmup=2, seq_len=2048, use_gate=True):
    cfg = get_config('v13_e3_k3_selective')
    assert cfg.fused_e3, "preset must have fused_e3=True"
    torch.manual_seed(0)
    m = V13LM(cfg).cuda()
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    ids = torch.randint(0, 50261, (batch, seq_len), device='cuda')
    labels = torch.randint(0, 50261, (batch, seq_len), device='cuda')

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lm, aux, gate_probs = m._hidden_to_lm(ids)
            loss, nll = m.ce_from_lm(lm, labels, chunk=1024, return_nll=True)
            loss = loss + aux
            if use_gate and gate_probs is not None and cfg.gate_surprisal_lambda > 0:
                loss = loss + cfg.gate_surprisal_lambda * gate_surprisal_loss(
                    gate_probs, nll, labels, cfg)
        loss.backward()
        opt.step()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() / 1e9
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    tok_s = batch * seq_len / dt
    tag = "gate " if use_gate else "no-gate"
    print(f"TRAINER-PATH delta chunk={delta_chunk} B{batch:<3d} {tag:8s}  "
          f"{tok_s:8.0f} tok/s   {mem:5.1f} GB   (limit 24)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=14)
    ap.add_argument('--delta_chunk', type=int, default=128)
    ap.add_argument('--iters', type=int, default=8)
    ap.add_argument('--no_gate', action='store_true')
    a = ap.parse_args()
    main(a.batch, a.delta_chunk, a.iters, use_gate=not a.no_gate)
