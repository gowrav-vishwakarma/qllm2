"""Clean wall-clock 100M train-step benchmark (no event-timing, no reset_peak)."""
import os
import sys
import time

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


def bench(write_mode, delta_chunk, batch, fused, seq_len=2048, iters=8, warmup=2):
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
    print(f"{write_mode:9s} chunk={delta_chunk:3d} B{batch:<3d} fused={fused!s:5s}  "
          f"{tok_s:8.0f} tok/s   {mem:5.1f} GB")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--write_mode', default='delta')
    ap.add_argument('--delta_chunk', type=int, default=256)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--fused', default='True')
    a = ap.parse_args()
    bench(a.write_mode, a.delta_chunk, a.batch, a.fused == 'True')
