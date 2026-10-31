# Batch consistency of the reset: model([s0, s1]) rows must equal model([s0]) and model([s1]),
# with separators at DIFFERENT positions in s0 and s1 (a batch-1 parity cannot see a layout bug).
import os, sys, torch
sys.path.insert(0, '/home/gowrav/Development/qllm2'); sys.path.insert(0, '/home/gowrav/Development/qllm2/scripts')
from run_memory_behavioral import _load_v13_sempty
torch.manual_seed(0)
m, _, _ = _load_v13_sempty(sys.argv[1], 'baseline_real_pm', torch.device('cpu')); m.eval(); m.config.reset_at_eos = True
T = 320
s0 = torch.randint(100, 50000, (T,)); s1 = torch.randint(100, 50000, (T,))
s0[90] = 50256; s1[230] = 50256; s1[40] = 50256
with torch.no_grad():
    both = m(torch.stack([s0, s1]))[0]
    a = m(s0[None])[0][0]; b = m(s1[None])[0][0]
print(f'row 0 vs alone {float((both[0] - a).abs().max()):.2e}   row 1 vs alone {float((both[1] - b).abs().max()):.2e}   (logit scale {float(a.abs().max()):.1f})')
