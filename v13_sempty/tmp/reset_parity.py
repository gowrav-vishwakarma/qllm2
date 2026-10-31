# Reset-at-separator parity (lg8r): doc B after [A, EOS] with reset ON must equal a fresh run of
# [EOS, B]. Criteria: every position AFTER the separator within 1e-3 x logit scale (exact up to fp32),
# the separator position itself (it predicts B's first token) within 1 % of the scale; and the
# no-reset run must differ (the test can fail).
import os, sys, torch
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) if 'v13_sempty/tmp' in __file__ else '/home/gowrav/Development/qllm_reset2'
sys.path.insert(0, R); sys.path.insert(0, R + '/scripts')
from run_memory_behavioral import _load_v13_sempty
torch.manual_seed(0)
ok = True
for ck in sys.argv[1:]:
    m, _, _ = _load_v13_sempty(ck, 'baseline_real_pm', torch.device(os.environ.get('DEV', 'cpu'))); m.eval()
    A1, A2, B = (torch.randint(100, 50000, (n,), device=os.environ.get('DEV', 'cpu')) for n in (180, 140, 120))
    eos = torch.tensor([50256], device=A1.device)
    m.config.reset_at_eos = True
    with torch.no_grad():
        f1 = m(torch.cat([A1, eos, B])[None])[0][0, len(A1):]
        f2 = m(torch.cat([A2, eos, B])[None])[0][0, len(A2):]
        fr = m(torch.cat([eos, B])[None])[0][0]
        m.config.reset_at_eos = False
        off = m(torch.cat([A1, eos, B])[None])[0][0, len(A1):]
    sc = float(fr.abs().max())
    after = max(float((f1[1:] - fr[1:]).abs().max()), float((f2[1:] - fr[1:]).abs().max()))
    at = max(float((f1[0] - fr[0]).abs().max()), float((f2[0] - fr[0]).abs().max()))
    eo = float((off[1:] - fr[1:]).abs().max())
    print(f"{ck.split('/')[-2]}: after the separator {after:.2e} | at the separator {at:.2e} | no reset {eo:.2e} (scale {sc:.1f})")
    ok &= after < 1e-3 * sc and at < 1e-2 * sc and eo > 1e-2 * sc
print('RESET PARITY', 'PASS' if ok else 'FAIL')
