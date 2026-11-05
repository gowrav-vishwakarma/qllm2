"""cx_m2 port parity (complex PAMLayer: A3 delta, N1 chrono, N4 out_gate), run on CUDA:
  (a) realified fused prefill (real pam_delta_batched, two sub-steps per token) == true-complex stepwise reference;
  (b) prefill of the first part + token-by-token decode (carried notebook + clock) == the full prefill;
  (c) chrono + out_gate at init (zero-init) == the plain complex layer with the same weights (exact identity parts);
  (d) RESET_AT_EOS: a keep mask that zeroes retention gives the same read as two separate documents.
    python v13_sempty/tmp/cx_delta_parity.py
"""
import sys, copy, torch, torch.nn as nn
sys.path.insert(0, '.')
from sempyt.dim import Dim
from sempyt.tensor import named
from v13_sempty.config import get_config
from v13_sempty.model import PAMLayer
torch.manual_seed(0); dev = 'cuda'

def mk_cfg(**kw):
    c = get_config('baseline'); c.dim, c.n_heads, c.head_dim, c.dropout = 64, 2, 32, 0.0
    for k, v in kw.items(): setattr(c, k, v)
    return c

def run(L, x, state=None, off=0):
    B, T = x.shape[0], x.shape[1]
    nt = named(x, (Dim('batch', B), Dim('time', T), L.model_dim, L.complex_pair), L.policy)
    out, st = L(nt, state=state, step_offset=off)
    return out.data.float(), st

def worst(a, b): return (a - b).abs().max().item() / (b.abs().max().item() + 1e-9)

ok = True
cfg = mk_cfg(delta=True, chrono=True, out_gate=True)
L = PAMLayer(cfg).to(dev).eval()
with torch.no_grad():
    for m in (L.erase_proj, L.write_proj, L.warp_proj, L.gate_proj, L.dt_proj):
        nn.init.normal_(m.weight, std=0.3); nn.init.normal_(m.bias, std=0.3)
B, T = 2, 150
x = torch.randn(B, T, cfg.dim, 2, device=dev)
with torch.no_grad():
    y_full, st_full = run(L, x)                                                # chunked (fused, realified)
    y_step, st = [], None                                                     # (a) every token stepwise
    for t in range(T):
        yt, st = run(L, x[:, t:t + 1], st, t); y_step.append(yt)
    y_step = torch.cat(y_step, dim=1)
    e = worst(y_full, y_step); print(f'(a) chunked realified vs complex stepwise: rel max err {e:.2e}'); ok &= e < 2e-3
    y_pre, st = run(L, x[:, :100])                                            # (b) prefill 100 + decode 50
    y_dec = [y_pre]
    for t in range(100, T):
        yt, st = run(L, x[:, t:t + 1], st, t); y_dec.append(yt)
    e = worst(torch.cat(y_dec, 1), y_full); print(f'(b) prefill + decode vs full prefill:        rel max err {e:.2e}'); ok &= e < 2e-3
    e = worst(st[0].float(), st_full[0].float()); print(f'    carried notebook after T:               rel max err {e:.2e}'); ok &= e < 2e-3

# (c) identity at init: chrono + out_gate (zero-init) vs plain, same shared weights, delta OFF on both
P = PAMLayer(mk_cfg()).to(dev).eval()
C = PAMLayer(mk_cfg(chrono=True, out_gate=True)).to(dev).eval()
with torch.no_grad():
    nn.init.zeros_(C.warp_proj.weight); nn.init.zeros_(C.warp_proj.bias)
    nn.init.zeros_(C.gate_proj.weight); nn.init.constant_(C.gate_proj.bias, 1.2785)
    C.load_state_dict(P.state_dict(), strict=False)
    yp, _ = run(P, x); yc, _ = run(C, x)
e = worst(yc, yp); print(f'(c) chrono + out_gate at init vs plain:    rel max err {e:.2e}'); ok &= e < 1e-3

# (d) RESET_AT_EOS: keep=0 at the boundary == the second document run from an empty notebook
with torch.no_grad():
    keep = torch.ones(B, T, device=dev); keep[:, 80] = 0.0
    L._keep = keep; y_keep, _ = run(L, x); L._keep = None
    nt_dummy = None
    # the second document alone, with Chrono's clock continued (the reset touches retention only, as in the real arm)
    _, (_, clk) = run(L, x[:, :80])
    y_2, _ = run(L, x[:, 80:], state=(None, clk))
e = worst(y_keep[:, 80:], y_2); print(f'(d) keep mask vs a fresh notebook (clock kept): rel max err {e:.2e}'); ok &= e < 2e-3
print('PASS' if ok else 'FAIL')
