# Throwaway: SL0, the shared library as knowledge (pre-registered in
# EXPERIMENTS_SEMPY "SL0"). kNN-LM over the frozen mind's own state:
#   build : stream training text (dclm + fineweb, training split), run the frozen
#           PAM, keep (PCA-128 of the pre-logit vector at t -> token t+1).
#   query : for the frozen attr3b text, the k nearest library keys per position
#           (exact, chunked on the GPU), saved as distances + next tokens.
#   report: lambda / T tuned on the first third of the docs, scored on the rest,
#           vs the mind alone and Mamba-2 (per-token losses from attr3b).
# argv: build <ckpt> <out_dir> <n_tokens> | query <ckpt> <out_dir> <data.npz> |
#       report <out_dir> <data.npz> <loss_local.npz> <loss_mamba2.npz>
import os, sys, time, numpy as np, torch
R = os.environ.get('SL0_ROOT') or __file__.rsplit('/v13_sempty/', 1)[0]
sys.path.insert(0, R); sys.path.insert(0, R + '/scripts'); sys.path.insert(0, R + '/v13_sempty/tmp')
K, DIM, KEYCHUNK, QCHUNK = 64, 128, 500_000, 1024
dev = torch.device('cuda')
if os.environ.get('SL0_MEM_FRAC'):
    torch.cuda.set_per_process_memory_fraction(float(os.environ['SL0_MEM_FRAC']))


def load(ck):
    from run_memory_behavioral import _load_v13_sempty
    m, tok, _ = _load_v13_sempty(ck, 'baseline_real_pm', dev); m.eval(); return m, tok


@torch.no_grad()
def vectors(m, ids):
    x = torch.tensor([ids], device=dev)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        z, b, t = m._stem(x, 0)
        z, _ = m._run_blocks(z, b, t, None, 0, x)
        lm, _ = m._lm_vector(z, x, b, t)
    return lm.raw(b, t, m.model_dim)[0].float()


def build(ck, out, n_tokens):
    from v7.data import SOURCE_REGISTRY, _open_source_iter
    m, tok = load(ck)
    its = [_open_source_iter(s, SOURCE_REGISTRY[s], edu_score_min=3, fineweb_name='sample-10BT',
                             exclude_holdout=True, holdout_pct=5, skip=0, doc_counters=None, mix_seed=42)
           for s in ('dclm', 'fineweb')]
    t0, keys, vals, buf, P, mu, n, d = time.time(), [], [], [], None, None, 0, 0
    W, B = 2049, 8                                        # packed windows like training, 8 per forward
    eos = tok.eos_token_id
    while n < n_tokens:
        while len(buf) < W * B:
            buf += tok(next(its[d % 2]), add_special_tokens=False)['input_ids'] + [eos]; d += 1
        win = torch.tensor(buf[:W * B]).view(B, W); buf = buf[W * B:]
        x = win[:, :-1].to(dev)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            z, b, t = m._stem(x, 0)
            z, _ = m._run_blocks(z, b, t, None, 0, x)
            lm, _ = m._lm_vector(z, x, b, t)
        v = lm.raw(b, t, m.model_dim).float(); v = v.reshape(-1, v.shape[-1])
        y = win[:, 1:].reshape(-1).to(torch.int32)
        if P is None:                                     # PCA on the first 8 x 2048 x ~3 windows
            keys.append(v.cpu()); vals.append(y); n += len(y)
            if n >= 50_000:
                S = torch.cat(keys).to(dev); mu = S.mean(0)
                _, _, V = torch.linalg.svd(S - mu, full_matrices=False); P = V[:DIM].T.contiguous()
                keys = [((S - mu) @ P).half().cpu()]; del S
            continue
        keys.append(((v - mu) @ P).half().cpu()); vals.append(y); n += len(y)
        if (n // (W * B)) % 50 == 0:
            print(f'  {n:,} rows  {n / (time.time() - t0):,.0f} tok/s', flush=True)
    keys, vals = torch.cat(keys)[:n_tokens], torch.cat(vals)[:n_tokens]
    sec = time.time() - t0
    torch.save({'keys': keys, 'vals': vals, 'P': P.cpu(), 'mu': mu.cpu(), 'docs': d, 'build_sec': sec, 'ckpt': ck}, f'{out}/library.pt')
    print(f'library: {len(keys):,} rows from {d} docs, {keys.numel() * 2 / 1e6:.0f} MB keys + {vals.numel() * 4 / 1e6:.0f} MB values, build {sec:.0f} s')


@torch.no_grad()
def query(ck, out, data):
    from attr3b import load_sets
    m, _ = load(ck); lib = torch.load(f'{out}/library.pt')
    keys = lib['keys'].to(dev); vals = lib['vals'].to(dev).long(); P, mu = lib['P'].to(dev), lib['mu'].to(dev)
    kk = torch.cat([(keys[b:b + KEYCHUNK].float() ** 2).sum(1) for b in range(0, len(keys), KEYCHUNK)])   # chunked: no fp32 copy of all keys
    res, t0 = {}, time.time()
    for name, docs in load_sets(data).items():
        for i, d in enumerate(docs):
            q = ((vectors(m, d[:-1].tolist()) - mu) @ P)  # [T, 128]
            D, I = [], []
            for a in range(0, len(q), QCHUNK):
                qc = q[a:a + QCHUNK]; qq = (qc ** 2).sum(1, keepdim=True)
                bd, bi = None, None
                for b in range(0, len(keys), KEYCHUNK):
                    dist = qq + kk[b:b + KEYCHUNK] - 2 * (qc.half() @ keys[b:b + KEYCHUNK].T).float()
                    td, ti = dist.topk(K, dim=1, largest=False); ti = ti + b
                    if bd is not None:
                        td, sel = torch.cat([bd, td], 1).topk(K, dim=1, largest=False)
                        ti = torch.cat([bi, ti], 1).gather(1, sel)
                    bd, bi = td, ti
                D.append(bd.cpu()); I.append(vals[bi].cpu())
            res[f'{name}_{i}_d'] = torch.cat(D).clamp(min=0).numpy().astype(np.float32)
            res[f'{name}_{i}_v'] = torch.cat(I).numpy().astype(np.int32)
    np.savez(f'{out}/neighbours.npz', **res)
    print(f'query done in {time.time() - t0:.0f} s')


def report(out, data, local, mamba):
    from attr3b import load_sets, classes, POS
    sets = load_sets(data); nb = np.load(f'{out}/neighbours.npz'); lib = torch.load(f'{out}/library.pt')
    L = {}
    for f in (local, mamba):
        z = np.load(f)
        for key in z.files:
            nm, doc = key.split('|'); L.setdefault(nm, {})[doc] = z[key]
    split = {'pg19': 20, 'web': 50}                       # first docs tune, the rest are scored

    def mixed(k, idx, lam, T):
        out = []
        for i in idx:
            d, v = nb[f'{k}_{i}_d'], nb[f'{k}_{i}_v']; y = sets[k][i][1:]
            w = np.exp(-(d - d.min(1, keepdims=True)) / (T * med[k])); w /= w.sum(1, keepdims=True)
            pk = (w * (v == y[:, None])).sum(1)
            plm = np.exp(-L['pam3b'][f'{k}_{i}'])
            out.append(-np.log(lam * pk + (1 - lam) * plm + 1e-12))
        return out

    best, med = {}, {}
    for k in sets:                                        # T is in units of the median nearest distance (tuning docs)
        med[k] = float(np.median(np.concatenate([nb[f'{k}_{i}_d'][:, 0] for i in range(split[k])])))
    for k in sets:
        tune = range(split[k])
        grid = [(float(np.mean(np.concatenate(mixed(k, tune, lam, T)))), lam, T)
                for lam in (0.02, 0.05, 0.1, 0.2, 0.3) for T in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)]
        best[k] = min(grid)[1:]
    print(f'library {len(lib["keys"]):,} rows, {(lib["keys"].numel() * 2 + lib["vals"].numel() * 4) / 1e6:.0f} MB, '
          f'install {lib["build_sec"]:.0f} s (one mind pass over the library text); tuned (lambda, T): {best}')
    for k in sets:
        idx = range(split[k], len(sets[k]))
        C = np.concatenate([classes(sets[k][i]) for i in idx]); Pp = np.concatenate([np.arange(len(sets[k][i]) - 1) for i in idx])
        arms = {'pam3b': np.concatenate([L['pam3b'][f'{k}_{i}'] for i in idx]),
                'pam3b+lib': np.concatenate(mixed(k, idx, *best[k])),
                'mamba2': np.concatenate([L['mamba2'][f'{k}_{i}'] for i in idx])}
        print(f'\n[{k}] scored docs {len(idx)}, tokens {len(C)}')
        for a, x in arms.items():
            cls = '  '.join(f'{c} {x[C == j].mean():.4f}' for j, c in enumerate(('copy', 'seen', 'new')))
            pos = '  '.join(f'{p0}-{p1} {x[(Pp >= p0) & (Pp < p1)].mean():.4f}' for p0, p1 in POS)
            newpos = x[(C == 2) & (Pp < 64)].mean()
            print(f'  {a:10s} all {x.mean():.4f} | {cls} | {pos} | new@0-64 {newpos:.4f}')
        g = arms['pam3b+lib'].mean()
        print(f'  rule 1 (lib helps, <= pam - 0.05): {g - arms["pam3b"].mean():+.4f}   '
              f'rule 2 (<= mamba2): {g - arms["mamba2"].mean():+.4f}')


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'build':
        build(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif mode == 'query':
        query(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        report(*sys.argv[2:6])
