"""Selftest (2026-11-05): the plain v13 complex LM trains one step. W1's `block.window` loop in LM._run_blocks had broken the
complex Block (AttributeError: 'Block' object has no attribute 'window').    python v13_sempty/tmp/complex_block_selftest.py"""
import sys, torch
sys.path.insert(0, '.')
from v13_sempty.config import get_config
from v13_sempty.model import LM
c = get_config('baseline'); c.dim, c.n_heads, c.head_dim, c.n_layers, c.dropout = 32, 2, 16, 2, 0.0
c.gradient_checkpointing = False
m = LM(c).train(); ids = torch.randint(0, 1000, (2, 64))
logits, _, _ = m(ids)
logits.float().pow(2).mean().backward()
g = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
print('plain complex LM: one fwd + bwd OK, grad sum %.3g' % g); assert g > 0
lm, _ = m._hidden_to_lm(ids); print('training path (_hidden_to_lm) OK'); print('PASS')
