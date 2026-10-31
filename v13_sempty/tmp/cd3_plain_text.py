# cd3 (EXPERIMENTS_SEMPY "cd3"): state-contrast decoding on ordinary prose after an IRRELEVANT skill pack.
# Per prose token NLL for a in A: l = l_pack + a (l_pack - l_nopack); plus no pack at all (= l_nopack).
import json, os, sys, numpy as np, torch
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, R); sys.path.insert(0, R + '/scripts')
from run_memory_behavioral import _load_v13_sempty
from memory_probes.spread import split_units, filler_tokens, _SEP
from v14.rules_ood import build_example
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
ck, out = sys.argv[1], sys.argv[2]
A = (0.0, 4.0, 8.0)
m, tok, _ = _load_v13_sempty(ck, 'baseline_real_pm', torch.device('cpu')); m.eval()
filler = filler_tokens(R + '/.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt')
sep = tok.encode(_SEP, add_special_tokens=False)
enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
res = {str(a): [] for a in A}; res['nopack'] = []
with torch.no_grad():
    for w in range(60):
        st = int(np.random.default_rng([31, w]).integers(0, len(filler) - 1100))
        prose = filler[st:st + 1025].tolist()
        ex = build_example(tok, ('affine_map_new', 'shift_wide_new')[w % 2], 8, 9000 + w)
        H, U = split_units(ex); pack = enc(H) + sum((enc(u) for u in U[:-1]), [])
        lp = m(torch.tensor([pack + sep + prose]))[0][0].float(); ln = m(torch.tensor([sep + prose]))[0][0].float()
        P, Q = len(pack) + len(sep), len(sep)
        tgt = torch.tensor(prose[1:])
        lpp, lnn = lp[P:P + 1024], ln[Q:Q + 1024]
        res['nopack'].append(torch.nn.functional.cross_entropy(lnn, tgt, reduction='none').tolist())
        for a in A:
            res[str(a)].append(torch.nn.functional.cross_entropy(lpp + a * (lpp - lnn), tgt, reduction='none').tolist())
        if w % 10 == 9: print(f'  {w + 1}/60', flush=True)
    # one shown continuation: greedy 40 tokens after pack + the first 64 prose tokens of window 0
    st = int(np.random.default_rng([31, 0]).integers(0, len(filler) - 1100)); prose = filler[st:st + 64].tolist()
    ex = build_example(tok, 'affine_map_new', 8, 9000); H, U = split_units(ex); pack = enc(H) + sum((enc(u) for u in U[:-1]), [])
    shown = {}
    for a in (0.0, 4.0):
        gen = []
        for _ in range(40):
            lp = m(torch.tensor([pack + sep + prose + gen]))[0][0, -1].float(); ln = m(torch.tensor([sep + prose + gen]))[0][0, -1].float()
            gen.append(int((lp + a * (lp - ln)).argmax()))
        shown[str(a)] = tok.decode(gen)
json.dump({'nll': res, 'shown_prompt_tail': tok.decode(prose[-24:]), 'shown': shown, 'ckpt': ck}, open(out, 'w'))
mean = lambda k, s=slice(None): float(np.mean([np.mean(x[s]) for x in res[k]]))  # noqa: E731
print('prose NLL (60 windows x 1024 tokens): ' + '  '.join(f'{k} {mean(k):.4f}' for k in ['nopack'] + [str(a) for a in A]))
print('first 64 tokens:                      ' + '  '.join(f'{k} {mean(k, slice(0, 64)):.4f}' for k in ['nopack'] + [str(a) for a in A]))
print('shown, prompt tail:', repr(tok.decode(prose[-24:])))
for a, t in shown.items(): print(f'  a={a}:', repr(t))
