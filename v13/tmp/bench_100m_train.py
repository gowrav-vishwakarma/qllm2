"""Re-runnable: time the full 100M (v13_e3_k3_selective) training step on the 4090.

Measures tokens/s and peak GPU memory for a real backward+step, mirroring the
trainer (bf16 autocast around _hidden_to_lm, fused CE, grad checkpointing,
AdamW step).

    .venv/bin/python v13/tmp/bench_100m_train.py --write_mode delta --delta_chunk 256 --batch 8
    .venv/bin/python v13/tmp/bench_100m_train.py --write_mode delta --delta_chunk 256 --batch 8 --fused False
    .venv/bin/python v13/tmp/bench_100m_train.py --write_mode additive --batch 16
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
from v13.model import V13LM, V13Config


def build(over: dict):
    cfg = V13Config(
        vocab_size=50261, dim=384, n_heads=6, head_dim=64, n_layers=16,
        expand=3, dropout=0.1, max_seq_len=2048, chunk_size=256,
        n_states=3, state_dt_spread=2.0, write_mode='delta', delta_chunk=64,
        delta_erase_gate=True, gate_content_aware=True, vault_state=True,
        vault_state_idx=0, write_phase_address=True, gate_surprisal_lambda=0.1,
        fused_e3=True, gradient_checkpointing=True,
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def bench(write_mode, delta_chunk, batch, fused, seq_len=2048, iters=5, warmup=2):
    torch.manual_seed(0)
    m = V13LM(build(dict(write_mode=write_mode, delta_chunk=delta_chunk, fused_e3=fused))).cuda()
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    ids = torch.randint(0, 50261, (batch, seq_len), device='cuda')

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lm, aux, _ = m._hidden_to_lm(ids)
        loss = m.ce_from_lm(lm, ids) + aux
        loss.backward()
        opt.step()
        return loss

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = torch.cuda.Event(True)
    t1 = torch.cuda.Event(True)
    t0.record()
    for _ in range(iters):
        step()
    t1.record()
    torch.cuda.synchronize()
    dt = t0.elapsed_time(t1) / 1000.0 / iters
    tok_s = batch * seq_len / dt
    mem = torch.cuda.max_memory_allocated() / 1e9
    label = f"{write_mode:9s} chunk={delta_chunk:3d} B{batch:<3d} fused={fused!s:5s}"
    print(f"{label}  {tok_s:8.0f} tok/s   {mem:5.1f} GB")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--write_mode', default='delta')
    ap.add_argument('--delta_chunk', type=int, default=256)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--fused', default='True')
    ap.add_argument('--iters', type=int, default=5)
    a = ap.parse_args()
    bench(a.write_mode, a.delta_chunk, a.batch, a.fused == 'True', a.iters)
