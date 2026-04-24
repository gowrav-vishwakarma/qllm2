"""
WordNet synonym separability d_sep vs. parameter count, for QPAM (conjugate
inner product Re<z1*|z2>) and RPAM (cosine similarity), across the canonical
scaling sweep (5M, 10M, 50M).

Aggregated over ~10K synonym pairs per (arch, scale); bootstrap error bars
on d_sep.

Usage:
    cd /Users/caug/npcww/qnlp/qllm-private
    uv run python v6/paper/scripts/plot_synonym_scaling.py
"""

import os
import numpy as np
import mlx.core as mx
from transformers import AutoTokenizer
from scipy.stats import mannwhitneyu

import matplotlib
matplotlib.use('Agg')
matplotlib.rc_file(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'matplotlibrc'))
import matplotlib.pyplot as plt

from qpam_mlx import QPAMModel, RPAMModel


OUTDIR = '/Users/caug/npcww/qnlp/qllm-private/v6/paper/figures'
ROOT = '/Users/caug/npcww/qnlp/qllm-private'

QPAM_CFG = {
    '5M':  dict(dim= 44, num_layers=12, num_heads=2, head_dim=16),
    '10M': dict(dim= 80, num_layers=12, num_heads=4, head_dim=16),
    '50M': dict(dim=292, num_layers=12, num_heads=4, head_dim=16),
}
RPAM_CFG = {
    '5M':  dict(dim= 84, num_layers=11, num_heads=2, head_dim= 8),
    '10M': dict(dim=140, num_layers=11, num_heads=8, head_dim=16),
    '50M': dict(dim=496, num_layers=11, num_heads=2, head_dim= 8),
}
QPAM_CKPT = {s: os.path.join(ROOT, f'checkpoints_qpam_{s.lower()}_mlx', 'best_model.npz')
             for s in QPAM_CFG}
RPAM_CKPT = {s: os.path.join(ROOT, f'checkpoints_rpam_{s.lower()}_mlx', 'best_model.npz')
             for s in RPAM_CFG}


def count_params(model):
    return sum(p.size for _, p in model.parameters().items()
               if isinstance(p, mx.array)) if False else \
        sum(v.size for v in _flatten(model.parameters()))


def _flatten(d):
    out = []
    if isinstance(d, dict):
        for v in d.values():
            out.extend(_flatten(v))
    elif isinstance(d, (list, tuple)):
        for v in d:
            out.extend(_flatten(v))
    elif isinstance(d, mx.array):
        out.append(d)
    return out


def build_pairs(tokenizer, max_words=3000):
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    from nltk.corpus import wordnet as wn

    single_token_words = {}
    for word in wn.all_lemma_names():
        if '_' in word or len(word) < 3:
            continue
        ids = tokenizer.encode(' ' + word)
        if len(ids) == 1:
            single_token_words[word] = ids[0]

    syn_pairs = set()
    for word in list(single_token_words.keys())[:max_words]:
        for syn in wn.synsets(word):
            for lem in syn.lemmas():
                l = lem.name()
                if l != word and l in single_token_words:
                    syn_pairs.add(tuple(sorted([word, l])))
    syn_pairs = list(syn_pairs)

    np.random.seed(42)
    word_list = list(single_token_words.keys())
    rand_pairs = []
    seen = set(syn_pairs)
    while len(rand_pairs) < len(syn_pairs):
        i, j = np.random.randint(0, len(word_list), 2)
        if i != j:
            p = tuple(sorted([word_list[i], word_list[j]]))
            if p not in seen:
                rand_pairs.append(p)
                seen.add(p)
    return single_token_words, syn_pairs, rand_pairs


def qpam_sims(model, words, pairs):
    er = np.array(model.embed_r)
    ei = np.array(model.embed_i)
    out = np.zeros(len(pairs))
    for k, (w1, w2) in enumerate(pairs):
        i1, i2 = words[w1], words[w2]
        z1 = er[i1] + 1j * ei[i1]
        z2 = er[i2] + 1j * ei[i2]
        out[k] = np.real(np.sum(np.conj(z1) * z2)) / (np.linalg.norm(z1) * np.linalg.norm(z2) + 1e-10)
    return out


def rpam_sims(model, words, pairs):
    e = np.array(model.embed)
    out = np.zeros(len(pairs))
    for k, (w1, w2) in enumerate(pairs):
        i1, i2 = words[w1], words[w2]
        out[k] = np.dot(e[i1], e[i2]) / (np.linalg.norm(e[i1]) * np.linalg.norm(e[i2]) + 1e-10)
    return out


def cohens_d(a, b):
    return (np.mean(a) - np.mean(b)) / np.sqrt((np.var(a) + np.var(b)) / 2 + 1e-12)


def bootstrap_dsep(syn, rand, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n_s, n_r = len(syn), len(rand)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        s = syn[rng.integers(0, n_s, n_s)]
        r = rand[rng.integers(0, n_r, n_r)]
        vals[b] = cohens_d(s, r)
    return float(np.mean(vals)), float(np.std(vals))


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print("Building WordNet pairs...")
    words, syn_pairs, rand_pairs = build_pairs(tokenizer)
    print(f"  {len(syn_pairs)} synonym pairs / {len(rand_pairs)} random pairs")

    scales = ['5M', '10M', '50M']
    results = {'QPAM': {}, 'RPAM': {}}

    for scale in scales:
        print(f"\nQPAM {scale}...")
        cfg = QPAM_CFG[scale]
        m = QPAMModel(vocab_size=50257, expand=3, **cfg)
        m.load_weights(QPAM_CKPT[scale])
        mx.eval(m.parameters())
        n_params = sum(v.size for v in _flatten(m.parameters()))
        s_syn = qpam_sims(m, words, syn_pairs)
        s_rand = qpam_sims(m, words, rand_pairs)
        d, d_err = bootstrap_dsep(s_syn, s_rand)
        _, p = mannwhitneyu(s_syn, s_rand, alternative='greater')
        results['QPAM'][scale] = dict(N=n_params, d=d, d_err=d_err, p=p,
                                      mean_syn=float(np.mean(s_syn)),
                                      mean_rand=float(np.mean(s_rand)))
        print(f"  N={n_params:,}  d_sep={d:.4f}±{d_err:.4f}  p={p:.2e}")
        del m

    for scale in scales:
        print(f"\nRPAM {scale}...")
        cfg = RPAM_CFG[scale]
        m = RPAMModel(vocab_size=50257, expand=3, **cfg)
        m.load_weights(RPAM_CKPT[scale])
        mx.eval(m.parameters())
        n_params = sum(v.size for v in _flatten(m.parameters()))
        s_syn = rpam_sims(m, words, syn_pairs)
        s_rand = rpam_sims(m, words, rand_pairs)
        d, d_err = bootstrap_dsep(s_syn, s_rand)
        _, p = mannwhitneyu(s_syn, s_rand, alternative='greater')
        results['RPAM'][scale] = dict(N=n_params, d=d, d_err=d_err, p=p,
                                      mean_syn=float(np.mean(s_syn)),
                                      mean_rand=float(np.mean(s_rand)))
        print(f"  N={n_params:,}  d_sep={d:.4f}±{d_err:.4f}  p={p:.2e}")
        del m

    # Plot: d_sep vs N
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for arch, marker, label in (('QPAM', '*', 'QPAM (Re$\\langle z_1^*|z_2\\rangle$)'),
                                ('RPAM', '^', 'RPAM (cosine)')):
        Ns = np.array([results[arch][s]['N'] for s in scales])
        ds = np.array([results[arch][s]['d'] for s in scales])
        es = np.array([results[arch][s]['d_err'] for s in scales])
        ax.errorbar(Ns, ds, yerr=es, fmt=marker, color='black',
                    markersize=10 if marker == '*' else 7,
                    markerfacecolor='white' if arch == 'RPAM' else 'black',
                    markeredgecolor='black', capsize=3, label=label,
                    linestyle='-' if arch == 'QPAM' else '--', linewidth=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('parameters $N$')
    ax.set_ylabel(r'synonym--random separability  $d_{\rm sep}$')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.legend(loc='best', frameon=False, fontsize=9)

    out = os.path.join(OUTDIR, 'synonym_scaling.pdf')
    fig.subplots_adjust(left=0.18, right=0.96, top=0.96, bottom=0.14)
    fig.savefig(out, dpi=300)
    fig.savefig(out.replace('.pdf', '.png'), dpi=300)
    print(f"\nSaved {out}")

    print("\nSummary:")
    for arch in ('QPAM', 'RPAM'):
        for s in scales:
            r = results[arch][s]
            print(f"  {arch} {s}: N={r['N']:>10,}  d={r['d']:+.4f}±{r['d_err']:.4f}  "
                  f"p={r['p']:.1e}  syn={r['mean_syn']:+.4f}  rand={r['mean_rand']:+.4f}")


if __name__ == '__main__':
    main()
