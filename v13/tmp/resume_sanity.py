"""Re-runnable: confirm the killed 500M checkpoint loads into the new fused-delta
model with zero missing/unexpected keys (the fused path adds no parameters), and
that a bf16 forward+CE runs. Run before restarting the detached run.

    .venv/bin/python v13/tmp/resume_sanity.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
from v13.model import V13LM, get_config

CKPT = 'checkpoints_v13/100m_realdat_500m/latest.pt'

cfg = get_config('v13_e3_k3_selective')
print(f"preset: fused_e3={cfg.fused_e3} write_mode={cfg.write_mode} "
      f"delta_chunk={cfg.delta_chunk} n_states={cfg.n_states}")

m = V13LM(cfg).cuda()
m.train()
ck = torch.load(CKPT, map_location='cuda', weights_only=False)
missing, unexpected = m.load_state_dict(ck['model_state_dict'], strict=True)
print(f"global_step={ck['global_step']} global_tokens={ck['global_tokens']:,}")
print(f"optimizer_state_dict present: {'optimizer_state_dict' in ck}")
print(f"state_dict: strict load OK (no missing/unexpected keys)")

# bf16 forward + fused CE smoke, B16/T2048 (the resume batch).
ids = torch.randint(0, 50261, (16, 2048), device='cuda')
with torch.autocast('cuda', dtype=torch.bfloat16):
    lm, aux, _ = m._hidden_to_lm(ids)
loss = m.ce_from_lm(lm, ids) + aux
print(f"smoke loss={loss.item():.4f}  peak_mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB")
print("PASS — checkpoint is a valid resume point for the fused-delta model.")
