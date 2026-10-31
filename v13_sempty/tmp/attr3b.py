# Throwaway: where does PAM's 3B LM gap come from? Per-token loss of the 3B
# models on 2048-token windows (PG-19 books + web holdout docs), split by
# position and by token type: 'copy' = the bigram (x_t, y_t) already occurred
# earlier in the window (an induction head can copy it), 'seen' = the target
# token occurred earlier but not after this token, 'new' = first occurrence.
# Gap share of class c = frac_c * (loss_PAM_c - loss_base_c) / total gap.
# argv: out.json name=ckpt ...   (all model-type v13_sempty)
import json, sys, numpy as np, torch
R = '/home/gowrav/Development/qllm2'
sys.path.insert(0, R); sys.path.insert(0, R + '/scripts')
from eval_pos_loss import _docs, _token_losses
from run_memory_behavioral import _load_v13_sempty

out_path, specs = sys.argv[1], [a.split('=', 1) for a in sys.argv[2:]]
dev = torch.device('cuda')
CTX, NPG, NWEB = 2048, 60, 150

def web_docs(min_len=512):
    blob = torch.load(R + '/.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt', weights_only=False)
    t = blob['tokens'] if isinstance(blob, dict) else blob
    t = np.asarray(t.cpu().numpy() if hasattr(t, 'cpu') else t, dtype=np.int64).reshape(-1)
    cuts = np.flatnonzero(t >= 50256)
    docs, s = [], 0
    for c in list(cuts) + [len(t)]:
        d = t[s:c]; s = c + 1
        if len(d) >= min_len:
            docs.append(torch.tensor(d[:CTX + 1]))
        if len(docs) >= NWEB:
            break
    return docs

def classes(ids):
    x, y = ids[:-1].tolist(), ids[1:].tolist()
    seen_bi, seen_tok, cls = set(), set(x[:1]), []
    for t in range(len(y)):
        if (x[t], y[t]) in seen_bi: cls.append(0)        # copy
        elif y[t] in seen_tok: cls.append(1)            # seen
        else: cls.append(2)                             # new
        seen_bi.add((x[t], y[t])); seen_tok.add(y[t]); seen_tok.add(x[t])
    return np.array(cls)

_, tok, _ = _load_v13_sempty(specs[0][1], 'baseline_real_pm', torch.device('cpu'))
sets = {'pg19': _docs(tok, NPG, CTX, 1024, 'pg19'), 'web': web_docs()}
cls = {k: [classes(d) for d in v] for k, v in sets.items()}
loss = {}
for name, ck in specs:
    model, _, _ = _load_v13_sempty(ck, 'baseline_real_pm', dev); model.eval()
    loss[name] = {k: [_token_losses(model, 'v13_sempty', d, dev).numpy() for d in v] for k, v in sets.items()}
    print(name, {k: round(float(np.mean(np.concatenate(v))), 4) for k, v in loss[name].items()}, flush=True)
    del model; torch.cuda.empty_cache()

POS = [(0, 8), (8, 64), (64, 512), (512, 2048)]
res = {}
for k in sets:
    C = np.concatenate(cls[k]); P = np.concatenate([np.arange(len(c)) for c in cls[k]])
    res[k] = {'frac': {n: float((C == i).mean()) for i, n in enumerate(('copy', 'seen', 'new'))}, 'models': {}}
    for name in loss:
        L = np.concatenate(loss[name][k])
        res[k]['models'][name] = {
            'all': float(L.mean()),
            'by_class': {n: float(L[C == i].mean()) for i, n in enumerate(('copy', 'seen', 'new'))},
            'by_pos': {f'{a}-{b}': float(L[(P >= a) & (P < b)].mean()) for a, b in POS},
            'by_pos_class': {f'{a}-{b}': {n: float(L[(P >= a) & (P < b) & (C == i)].mean()) if ((P >= a) & (P < b) & (C == i)).any() else None
                                          for i, n in enumerate(('copy', 'seen', 'new'))} for a, b in POS}}
json.dump(res, open(out_path, 'w'), indent=1)
pam = specs[0][0]
for k in res:
    f = res[k]['frac']; m = res[k]['models']
    print(f'\n[{k}] token share copy {f["copy"]:.3f} seen {f["seen"]:.3f} new {f["new"]:.3f}')
    for name in m:
        if name == pam: continue
        tot = m[pam]['all'] - m[name]['all']
        sh = {c: f[c] * (m[pam]['by_class'][c] - m[name]['by_class'][c]) for c in f}
        print(f'  {pam} - {name}: total {tot:+.4f} | by class (gap, share of total): ' +
              ' '.join(f'{c} {m[pam]["by_class"][c] - m[name]["by_class"][c]:+.3f} ({sh[c] / tot:+.0%})' for c in f) +
              ' | by pos: ' + ' '.join(f'{p} {m[pam]["by_pos"][p] - m[name]["by_pos"][p]:+.3f}' for p in m[pam]['by_pos']))
