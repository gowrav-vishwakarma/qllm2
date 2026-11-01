# clk1 cost cell: prose NLL after an irrelevant pack with retention ** (1/s) over the prose (cd3's windows).
import json, os, sys, numpy as np, torch
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, R); sys.path.insert(0, R + '/scripts')
from run_memory_behavioral import _load_v13_sempty
from memory_probes.spread import split_units, filler_tokens, _SEP
from v14.rules_ood import build_example
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
ck, out = sys.argv[1], sys.argv[2]
WHERE = sys.argv[3] if len(sys.argv) > 3 else 'all'
S = (1.0, 2.0, 4.0)
m, tok, _ = _load_v13_sempty(ck, 'baseline_real_pm', torch.device('cpu')); m.eval()
filler = filler_tokens(R + '/.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt')
sep = tok.encode(_SEP, add_special_tokens=False); enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
res = {str(s): [] for s in S}
with torch.no_grad():
    for w in range(60):
        st = int(np.random.default_rng([31, w]).integers(0, len(filler) - 1100)); prose = filler[st:st + 1025].tolist()
        ex = build_example(tok, ('affine_map_new', 'shift_wide_new')[w % 2], 8, 9000 + w)
        H, U = split_units(ex); pack = enc(H) + sum((enc(u) for u in U[:-1]), [])
        P = len(pack) + len(sep); tgt = torch.tensor(prose[1:])
        for sc in S:
            m._dilate = (P, P + 1025, sc, WHERE)
            lp = m(torch.tensor([pack + sep + prose]))[0][0].float()[P:P + 1024]
            res[str(sc)].append(torch.nn.functional.cross_entropy(lp, tgt, reduction='none').tolist())
        m._dilate = None
        if w % 20 == 19: print(f'  {w + 1}/60', flush=True)
json.dump({'nll': res, 'ckpt': ck, 'where': WHERE}, open(out, 'w'))
mean = lambda k, s=slice(None): float(np.mean([np.mean(x[s]) for x in res[k]]))  # noqa: E731
print('prose NLL after an irrelevant pack, dilation over the prose: ' + '  '.join(f's={k}: {mean(k):.4f}' for k in res))
print('                                    tokens 512-1024:        ' + '  '.join(f's={k}: {mean(k, slice(512, 1024)):.4f}' for k in res))
