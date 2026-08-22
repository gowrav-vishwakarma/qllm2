"""One-off diagnostic: reconcile event timing vs wall time on the 100M model."""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
import torch
from bench_100m_train import build

m = V13LM = None
from v13.model import V13LM

cfg = build(dict(write_mode='delta', delta_chunk=256, fused_e3=True))
print("n_layers:", cfg.n_layers, "dim:", cfg.dim)
torch.manual_seed(0)
m = V13LM(cfg).cuda()
m.train()
n_params = sum(p.numel() for p in m.parameters())
print(f"params: {n_params/1e6:.1f}M  device: {next(m.parameters()).device}")
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
ids = torch.randint(0, 50261, (8, 2048), device='cuda')

with torch.autocast('cuda', dtype=torch.bfloat16):
    lm, aux, _ = m._hidden_to_lm(ids)
loss = m.ce_from_lm(lm, ids) + aux
print("loss:", loss.item())

torch.cuda.synchronize()
t = time.time()
for i in range(3):
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        lm, aux, _ = m._hidden_to_lm(ids)
    loss = m.ce_from_lm(lm, ids) + aux
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    print(f"step {i}: {time.time()-t:.2f}s")
print("peak mem GB:", torch.cuda.max_memory_allocated()/1e9)
