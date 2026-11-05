"""cx_m2 model-level parity (CxLM, complex M2: sparse notebooks + shared reads + delta + chrono + out_gate), CUDA:
  (e) full prefill logits == prefill of the first part + token-by-token decode (carried notebooks + clocks; readers read S_t);
  (f) the shared readers at init (zero-init complex out-proj, gate = 1) leave the model unchanged: logits with readers == without;
  (g) build_lm routes a complex config with ported parts to CxLM; a plain complex config still builds the v13 LM.
    python v16/complex/tests/cx_m2_parity.py
"""
import sys, copy, torch
sys.path.insert(0, '.')
from v13_sempty.config import get_config
from v14.hook import build_lm
from v16.complex.model import CxLM
torch.manual_seed(0); dev = 'cuda'

def cfg_m2(**kw):
    c = get_config('baseline'); c.dim, c.n_heads, c.head_dim, c.n_layers, c.dropout = 64, 2, 32, 6, 0.0
    c.delta = c.chrono = c.out_gate = c.share_reads = True; c.pam_layers = (1, 4); c.mlp_hidden = 160
    for k, v in kw.items(): setattr(c, k, v)
    return c

def worst(a, b): return (a - b).abs().max().item() / (b.abs().max().item() + 1e-9)
ok = True
m = CxLM(cfg_m2()).to(dev).eval()
with torch.no_grad():                                   # make readers / gates non-trivial for (e)
    for b in m.blocks:
        if b.reader is not None:
            torch.nn.init.normal_(b.reader.o_proj.weight_real, std=0.05); torch.nn.init.normal_(b.reader.o_proj.weight_imag, std=0.05)
        if b.pam is not None:
            for p in (b.pam.warp_proj, b.pam.gate_proj, b.pam.erase_proj, b.pam.write_proj):
                torch.nn.init.normal_(p.weight, std=0.2)
ids = torch.randint(0, 50000, (2, 90), device=dev)
with torch.no_grad():
    full, _, _ = m(ids)
    pre, st, _ = m(ids[:, :60])
    outs = [pre]
    for t in range(60, 90):
        lg, st, _ = m(ids[:, t:t + 1], states=st, step_offset=t); outs.append(lg)
e = worst(torch.cat(outs, 1).float(), full.float()); print(f'(e) prefill + decode vs full prefill (CxLM M2): rel max err {e:.2e}'); ok &= e < 3e-3

a = CxLM(cfg_m2()).to(dev).eval(); b = CxLM(cfg_m2(share_reads=False)).to(dev).eval()
b.load_state_dict({k: v for k, v in a.state_dict().items() if k in b.state_dict()}, strict=True)
with torch.no_grad():
    la, _, _ = a(ids); lb, _, _ = b(ids)
e = worst(la.float(), lb.float()); print(f'(f) readers at init vs no readers: rel max err {e:.2e}'); ok &= e < 1e-4
n_a = sum(p.numel() for p in a.parameters()); n_b = sum(p.numel() for p in b.parameters())
print(f'    params with / without readers: {n_a:,} / {n_b:,}')

r = type(build_lm(cfg_m2())).__name__; p = type(build_lm(get_config('baseline'))).__name__
print(f'(g) build_lm: ported complex -> {r}; plain complex -> {p}'); ok &= (r == 'CxLM' and p == 'LM')
print('PASS' if ok else 'FAIL')
