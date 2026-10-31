# Throwaway: are PAM's soft write gates SELECTIVE in a trained model?
# Spread label prompts (label_category, 8 shots, examples pulled apart by
# held-out web text to L tokens). Per token class (item / label / format /
# filler) we read: the A3 delta write gate b_w and erase gate b_e of each
# notebook layer, the notebook decay, and the echo memory's write gate and
# retention. Selectivity = AUROC(write gate at item+label tokens > at filler
# tokens), per head (0.5 = not selective, 1.0 = writes items, skips filler).
import sys, os, numpy as np, torch
torch.set_num_threads(8)
R = '/home/gowrav/Development/qllm2'
sys.path.insert(0, R); sys.path.insert(0, R + '/scripts')
from memory_probes.rules import build_rule_example
from memory_probes.spread import split_units, filler_tokens, _SEP
from run_memory_behavioral import _load_v13_sempty

ck, L, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
TASK = sys.argv[4] if len(sys.argv) > 4 else 'label_category'
model, tok, _ = _load_v13_sempty(ck, 'baseline_real_pm', torch.device('cpu')); model.eval()
filler = filler_tokens(R + '/.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt')

def build(ex):
    """Same ids as memory_probes.spread.spread_example, plus a class per token."""
    header, units = split_units(ex)
    hid = tok.encode(header, add_special_tokens=False)
    uids = [tok.encode(u, add_special_tokens=False) for u in units]
    base = len(hid) + sum(map(len, uids)); gaps = len(uids) - 1
    sep = tok.encode(_SEP, add_special_tokens=False)
    per_gap = (L - base) // gaps - 2 * len(sep)
    rng = np.random.default_rng([ex.seed, L, sum(map(ord, ex.task))])
    starts = rng.integers(0, len(filler) - per_gap, size=gaps)
    ids, cls = list(hid), ['fmt'] * len(hid)
    def unit(u, last):
        out, seen_is = [], False
        for t in u:
            s = tok.decode([t]).strip()
            if s in ('is', ',', '', '->'):
                out.append('fmt'); seen_is |= s in ('is', '->')
            else:
                out.append('query' if last else ('label' if seen_is else 'item'))
        return out
    ids += uids[0]; cls += unit(uids[0], False)
    for g, u in enumerate(uids[1:]):
        chunk = sep + filler[starts[g]:starts[g] + per_gap].tolist() + sep
        ids += chunk; cls += ['sep'] * len(sep) + ['filler'] * per_gap + ['sep'] * len(sep)
        ids += u; cls += unit(u, g == len(uids) - 2)
    return ids, np.array(cls)

cap = {}
def raw(x):
    x = x.data if hasattr(x, 'data') and not isinstance(x, torch.Tensor) else x
    return x.detach().float()
pam = [(name, m) for name, m in model.named_modules() if hasattr(m, '_delta_gates') and getattr(m, 'delta', False)]
for name, m in pam:
    og, od = m._delta_gates, m._decay
    def g(tokens, _og=og, _n=name):
        bw, be = _og(tokens); cap[_n + '.bw'] = raw(bw); cap[_n + '.be'] = raw(be); return bw, be
    def d(tokens, _od=od, _n=name):
        out = _od(tokens); cap[_n + '.decay'] = raw(out); return out
    m._delta_gates, m._decay = g, d
if getattr(model, 'echo', False):
    H = model.config.n_heads
    def eh(mod, inp, out):
        dt, wr, er = out.detach().float().split(H, dim=-1)
        cap['echo.bw'] = torch.sigmoid(wr)[0].T; cap['echo.ret'] = torch.exp(-torch.nn.functional.softplus(dt))[0].T
        cap['echo.be'] = torch.sigmoid(er - 3).clamp(max=0.95)[0].T
    model.echo_gates.register_forward_hook(eh)

acc = {}
for seed in range(5000, 5000 + n):
    if TASK == 'label_category':
        ex = build_rule_example(tok, TASK, 8, seed)
    else:
        from v14.rules_ood import build_example as b2_example
        ex = b2_example(tok, TASK, 8, seed)
    ids, cls = build(ex)
    with torch.no_grad(): model.forward(torch.tensor([ids]))
    for k, v in cap.items():
        v = v.reshape(-1, v.shape[-1]) if v.shape[-1] == len(ids) else v.reshape(v.shape[0], -1)
        v = v[-v.shape[0]:] if v.shape[-1] == len(ids) else v
        if v.shape[-1] != len(ids): print('shape?', k, tuple(cap[k].shape)); continue
        for c in ('item', 'label', 'fmt', 'filler', 'query'):
            acc.setdefault((k, c), []).append(v[:, cls == c].numpy())

def auroc(a, b):   # P(a > b), a/b 1-D
    x = np.concatenate([a, b]); r = x.argsort().argsort().astype(float) + 1
    return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
keys = sorted({k for k, _ in acc})
print(f'{os.path.basename(os.path.dirname(ck))}  task={TASK}  L={L}  prompts={n}')
print(f'{"gate":34s} {"item":>6s} {"label":>6s} {"fmt":>6s} {"filler":>6s} | filler<0.1  AUROC(item+label > filler) mean / max head')
for k in keys:
    m = {c: np.concatenate(acc[(k, c)], axis=1) for c in ('item', 'label', 'fmt', 'filler')}
    content = np.concatenate([m['item'], m['label']], axis=1)
    au = [auroc(content[h], m['filler'][h][::7]) for h in range(content.shape[0])]
    print(f'{k:34s} {m["item"].mean():6.3f} {m["label"].mean():6.3f} {m["fmt"].mean():6.3f} {m["filler"].mean():6.3f} | '
          f'{(m["filler"] < 0.1).mean():9.3f}   {np.mean(au):.3f} / {np.max(au):.3f}')

print('\nper head: half-life of retention on filler (tokens, geometric mean); write gate content vs filler; write AUROC')
layers = sorted({k.rsplit('.', 1)[0] for k in keys})
for lay in layers:
    rk = lay + ('.ret' if lay == 'echo' else '.decay'); wk = lay + '.bw'
    if (rk, 'filler') not in acc: continue
    rf = np.concatenate(acc[(rk, 'filler')], axis=1); rc = np.concatenate(acc[(rk, 'item')] + acc[(rk, 'label')], axis=1)
    wf = np.concatenate(acc[(wk, 'filler')], axis=1); wc = np.concatenate(acc[(wk, 'item')] + acc[(wk, 'label')], axis=1)
    cells = []
    for h in range(rf.shape[0]):
        hl_f = np.log(0.5) / np.mean(np.log(np.clip(rf[h], 1e-6, 1 - 1e-9)))
        hl_c = np.log(0.5) / np.mean(np.log(np.clip(rc[h], 1e-6, 1 - 1e-9)))
        cells.append(f'h{h}: hl {hl_f:7.0f}/{hl_c:5.0f} w {wc[h].mean():.2f}/{wf[h].mean():.2f} au {auroc(wc[h], wf[h][::7]):.2f}')
    print(lay); [print('   ', c) for c in cells]
