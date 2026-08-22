"""One-off: which loss term's BACKWARD eats the ~20GB? (forward peak is only 3.2GB)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
from v13.model import V13LM, get_config
from bench3_trainer_path import gate_surprisal_loss

B, T = 14, 2048
cfg = get_config('v13_e3_k3_selective')
torch.manual_seed(0)
m = V13LM(cfg).cuda()
m.train()
ids = torch.randint(0, 50261, (B, T), device='cuda')
labels = torch.randint(0, 50261, (B, T), device='cuda')


def peak_of(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9


def make_losses():
    with torch.autocast('cuda', dtype=torch.bfloat16):
        lm, aux, gate_probs = m._hidden_to_lm(ids)
        ce = m.ce_from_lm(lm, labels, chunk=1024)
        gl = gate_surprisal_loss(gate_probs, lm, labels, m, cfg)
    return ce, aux, gl


cases = [
    ("ce only            ", lambda: make_losses()[0].backward()),
    ("gate only          ", lambda: make_losses()[2].backward()),
    ("ce+gate (no aux)   ", lambda: (make_losses()[0] + cfg.gate_surprisal_lambda * make_losses()[2]).backward()),
]
for name, fn in cases:
    try:
        p = peak_of(fn)
        print(f"{name} backward peak = {p:6.2f} GB", flush=True)
    except Exception as e:
        print(f"{name} FAILED: {str(e)[:100]}", flush=True)
    torch.cuda.empty_cache()
