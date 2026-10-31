# Throwaway: where does PAM's 3B LM gap come from? Per-token loss of the 3B
# models on 2048-token windows (PG-19 books + web holdout docs), split by
# position and by token type: 'copy' = the bigram (x_t, y_t) already occurred
# earlier in the window (an induction head can copy it), 'seen' = the target
# token occurred earlier but not after this token, 'new' = first occurrence.
# Gap share of class c = frac_c * (loss_PAM_c - loss_base_c) / total gap.
# Three steps so models can be scored on different boxes on identical text:
#   dump   <data.npz>                         (needs the tokenizer + HF pg19)
#   score  <data.npz> <out.npz> name=ckpt ... (GPU; mamba2 needs mamba_ssm)
#   report <data.npz> <ref name> <loss.npz> ...
import json, os, sys, numpy as np, torch
R = os.environ.get('ATTR_ROOT') or __file__.rsplit('/v13_sempty/', 1)[0]
sys.path.insert(0, R); sys.path.insert(0, R + '/scripts')
CTX, NPG, NWEB = 2048, 60, 150
POS = [(0, 8), (8, 64), (64, 512), (512, 2048)]


def dump(path):
    from eval_pos_loss import _docs
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')
    pg = [d.numpy() for d in _docs(tok, NPG, CTX, 1024, 'pg19')]
    blob = torch.load(R + '/.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt', weights_only=False)
    t = blob['tokens'] if isinstance(blob, dict) else blob
    t = np.asarray(t.cpu().numpy() if hasattr(t, 'cpu') else t, dtype=np.int64).reshape(-1)
    web, s = [], 0
    for c in list(np.flatnonzero(t >= 50256)) + [len(t)]:
        d = t[s:c]; s = c + 1
        if len(d) >= 512:
            web.append(d[:CTX + 1])
        if len(web) >= NWEB:
            break
    np.savez(path, **{f'pg19_{i}': d for i, d in enumerate(pg)}, **{f'web_{i}': d for i, d in enumerate(web)})
    print('dumped', len(pg), 'pg19 windows,', len(web), 'web docs')


def load_sets(path):
    z = np.load(path); sets = {'pg19': [], 'web': []}
    for k in sorted(z.files, key=lambda k: (k.split('_')[0], int(k.split('_')[1]))):
        sets[k.split('_')[0]].append(z[k])
    return sets


def score(data, out, specs):
    from eval_pos_loss import _token_losses
    from run_memory_behavioral import _load_v13_sempty
    sets, dev, res = load_sets(data), torch.device('cuda'), {}
    if os.environ.get('ATTR_MEM_FRAC'):   # hard cap when sharing a busy card
        torch.cuda.set_per_process_memory_fraction(float(os.environ['ATTR_MEM_FRAC']))
    for name, ck in specs:
        model, _, _ = _load_v13_sempty(ck, 'baseline_real_pm', dev); model.eval()
        for k, docs in sets.items():
            for i, d in enumerate(docs):
                res[f'{name}|{k}_{i}'] = _token_losses(model, 'v13_sempty', torch.tensor(d), dev).numpy()
        print(name, {k: round(float(np.mean(np.concatenate([res[f'{name}|{k}_{i}'] for i in range(len(v))]))), 4)
                     for k, v in sets.items()}, flush=True)
        del model; torch.cuda.empty_cache()
    np.savez(out, **res)


def classes(ids):
    x, y = ids[:-1].tolist(), ids[1:].tolist()
    seen_bi, seen_tok, cls = set(), set(x[:1]), []
    for t in range(len(y)):
        cls.append(0 if (x[t], y[t]) in seen_bi else 1 if y[t] in seen_tok else 2)
        seen_bi.add((x[t], y[t])); seen_tok.add(y[t]); seen_tok.add(x[t])
    return np.array(cls)


def report(data, ref, files):
    sets = load_sets(data); loss = {}
    for f in files:
        z = np.load(f)
        for key in z.files:
            name, doc = key.split('|'); loss.setdefault(name, {})[doc] = z[key]
    names = [ref] + [n for n in loss if n != ref]
    out = {}
    for k, docs in sets.items():
        C = np.concatenate([classes(d) for d in docs]); P = np.concatenate([np.arange(len(d) - 1) for d in docs])
        frac = {n: float((C == i).mean()) for i, n in enumerate(('copy', 'seen', 'new'))}
        M = {n: np.concatenate([loss[n][f'{k}_{i}'] for i in range(len(docs))]) for n in names}
        out[k] = {'frac': frac, 'models': {n: {'all': float(L.mean()),
                  'by_class': {c: float(L[C == i].mean()) for i, c in enumerate(frac)},
                  'by_pos': {f'{a}-{b}': float(L[(P >= a) & (P < b)].mean()) for a, b in POS},
                  'new_by_pos': {f'{a}-{b}': float(L[(P >= a) & (P < b) & (C == 2)].mean()) for a, b in POS}}
                  for n, L in M.items()}}
        print(f'\n[{k}] tokens {len(C)}; share copy {frac["copy"]:.3f} seen {frac["seen"]:.3f} new {frac["new"]:.3f}')
        print('  loss  ' + '  '.join(f'{n} {out[k]["models"][n]["all"]:.4f}' for n in names))
        r = out[k]['models'][ref]
        for n in names[1:]:
            m = out[k]['models'][n]; tot = r['all'] - m['all']
            print(f'  {ref} - {n}: total {tot:+.4f}')
            print('     by class (gap / share of total): ' + '  '.join(
                f'{c} {r["by_class"][c] - m["by_class"][c]:+.3f} / {frac[c] * (r["by_class"][c] - m["by_class"][c]) / tot:+.0%}' for c in frac))
            print('     by position:                     ' + '  '.join(f'{p} {r["by_pos"][p] - m["by_pos"][p]:+.3f}' for p in r['by_pos']))
            print('     new tokens by position:          ' + '  '.join(f'{p} {r["new_by_pos"][p] - m["new_by_pos"][p]:+.3f}' for p in r['new_by_pos']))
    return out


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'dump':
        dump(sys.argv[2])
    elif mode == 'score':
        score(sys.argv[2], sys.argv[3], [a.split('=', 1) for a in sys.argv[4:]])
    else:
        res = report(sys.argv[2], sys.argv[3], sys.argv[4:])
        json.dump(res, open(sys.argv[4].rsplit('.', 1)[0] + '_report.json', 'w'), indent=1)
