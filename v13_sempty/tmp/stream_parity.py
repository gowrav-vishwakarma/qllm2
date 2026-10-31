import sys, torch
import os; R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, R); sys.path.insert(0, R + '/scripts')
from run_memory_behavioral import _load_v13_sempty
torch.manual_seed(0)
ok = True
for ck in sys.argv[1:]:
    m, _, _ = _load_v13_sempty(ck, 'baseline_real_pm', torch.device('cpu')); m.eval()
    name = ck.split('/')[-2]
    x = torch.randint(100, 50000, (2, 300))
    with torch.no_grad():
        full = m(x)[0]
        P = 170
        la, st, _ = m(x[:, :P])
        lb, st2, _ = m(x[:, P:], states=st, step_offset=P)
        e_pref = float((la - full[:, :P]).abs().max()); e_suf = float((lb - full[:, P:]).abs().max())
        # decode: prefill 120, then 12 single steps
        Q = 120
        _, s, _ = m(x[:, :Q])
        dec = []
        for i in range(12):
            l1, s, _ = m(x[:, Q + i:Q + i + 1], states=s, step_offset=Q + i)
            dec.append(float((l1[:, 0] - full[:, Q + i]).abs().max()))
    scale = float(full.abs().max())
    print(f'{name}: prefix {e_pref:.2e}  suffix-after-carry {e_suf:.2e}  decode max {max(dec):.2e}  (logit scale {scale:.1f})')
    ok &= e_suf < 1e-3 * scale and max(dec) < 1e-3 * scale
print('STREAM PARITY', 'PASS' if ok else 'FAIL')
