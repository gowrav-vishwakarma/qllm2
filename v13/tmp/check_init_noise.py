import sys, torch
sys.path.insert(0, '/home/gowrav/Development/qllm2')
from v13.model import V13LM, get_config
dev = 'cuda'
co = get_config('v13_e3_k3_selective')
cf = get_config('v13_e3_k3_selective'); cf.ngram_read=True; cf.ngram_size=3; cf.ngram_fusion=True
torch.manual_seed(42); off = V13LM(co).eval()
torch.manual_seed(42); on  = V13LM(cf).eval()   # NO alignment: RNG shifted by conv1d init
off.to(dev); on.to(dev)
B, T = 8, 2048
ids = torch.randint(0, 50261, (B, T), device=dev); lbl = torch.randint(0, 50261, (B, T), device=dev)
def ce(m):
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
        out = m._hidden_to_lm(ids)
        lm = out[0] if isinstance(out, tuple) else out
        r = m.ce_from_lm(lm, lbl, chunk=256)
        return float(r[0] if isinstance(r, tuple) else r)
lo, lf = ce(off), ce(on)
print(f'unaligned seed-42 pair: OFF={lo:.4f} ON={lf:.4f} |d|={abs(lf-lo):.4f}')
print(f'live log: D-ref=10.9055  F-step0=10.9066  |d|=0.0011')
