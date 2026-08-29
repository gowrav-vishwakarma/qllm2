#!/usr/bin/env python
"""Is the 7.7e-7 ON-vs-OFF diff real, or the eager-CUDA non-determinism floor?

Run OFF twice (same seed, same aligned weights, same input): the OFF-vs-OFF
delta is the machine's determinism floor at this shape on this GPU.
"""
import sys
import torch

sys.path.insert(0, '/home/gowrav/Development/qllm2')
from v13.model import V13LM, get_config

dev = 'cuda'
cfg_off = get_config('v13_e3_k3_selective')
cfg_on = get_config('v13_e3_k3_selective')
cfg_on.ngram_read = True
cfg_on.ngram_size = 3
cfg_on.ngram_fusion = True

torch.manual_seed(123)
off = V13LM(cfg_off).eval()
torch.manual_seed(123)
on = V13LM(cfg_on).eval()
on.load_state_dict(off.state_dict(), strict=False)
off.to(dev)
on.to(dev)

B, T = 2, 512
ids = torch.randint(0, cfg_off.vocab_size, (B, T), device=dev)

with torch.no_grad():
    lg1, _, _ = off(ids)
    lg2, _, _ = off(ids)
    d_off_off = (lg1 - lg2).abs().max().item()
    on._ngram_ctx = None
    on._ngram_row_ctx = None
    lg_on, _, _ = on(ids)
    d_on_off = (lg_on - lg1).abs().max().item()
print(f'OFF-vs-OFF (same model, same input, back-to-back): max|d| = {d_off_off:.3e}')
print(f'ON-vs-OFF  (aligned weights, zero ngram):          max|d| = {d_on_off:.3e}')
print(f'diff ratio ON/OFF floor = {d_on_off / max(d_off_off, 1e-12):.2f}')

# And the exact-zero ngram property, at the smoke shape:
with torch.no_grad():
    ng = on._ngram_repr(ids, None)
    print(f'ngram exact_zero at smoke shape = {bool((ng == 0).all())}')
