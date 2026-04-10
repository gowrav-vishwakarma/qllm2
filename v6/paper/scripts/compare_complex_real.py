"""
Compare QPAM (complex) vs RPAM (real) on matched analyses.

Usage:
    cd /Users/caug/npcww/qnlp/qllm-private
    uv run python v6/paper/scripts/compare_complex_real.py
"""

import os, sys, math, numpy as np
import mlx.core as mx
import mlx.nn as nn

os.environ['MATPLOTLIBRC'] = '/Users/caug/npcww/qnlp/ket-nlp/matplotlibrc'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from chroptiks.plotting_utils import makefig, set_aesthetics, scatter as cscat, plot2dhist as chist2d
from qpam_mlx import QPAMModel, RPAMModel

OUTDIR = '/Users/caug/npcww/qnlp/qllm-private/v6/paper/figures'
LOGDIR = '/Users/caug/npcww/qnlp/ket-nlp'


def load_qpam():
    model = QPAMModel(vocab_size=50257, dim=384, num_layers=16, expand=3, num_heads=6, head_dim=64)
    model.load_weights(os.path.join(LOGDIR, 'checkpoints_qpam_mlx', 'best_model.npz'))
    mx.eval(model.parameters())
    return model


def load_rpam():
    model = RPAMModel(vocab_size=50257, dim=576, num_layers=16, expand=3, num_heads=9, head_dim=64)
    model.load_weights('/Users/caug/npcww/qnlp/qllm-private/checkpoints_rpam_mlx/best_model.npz')
    mx.eval(model.parameters())
    return model


# ══════════════════════════════════════════════════════════════
# 1. Val PPL comparison curve
# ══════════════════════════════════════════════════════════════

def plot_val_ppl_comparison():
    """Plot epoch-end val PPL for both models."""
    print("\n[1] Val PPL comparison...")

    # Parse QPAM log
    qpam_epochs, qpam_vals = [], []
    with open(os.path.join(LOGDIR, 'qpam_mlx_train_v2.log')) as f:
        for line in f:
            if 'Epoch' in line and 'complete' in line and 'val PPL=' in line:
                ep = int(line.split('Epoch ')[1].split(' ')[0])
                val = float(line.split('val PPL=')[1].split(',')[0])
                qpam_epochs.append(ep)
                qpam_vals.append(val)

    # Parse RPAM log
    rpam_epochs, rpam_vals = [], []
    with open(os.path.join(LOGDIR, 'rpam_mlx_train.log')) as f:
        for line in f:
            if 'Epoch' in line and 'complete' in line and 'val PPL=' in line:
                ep = int(line.split('Epoch ')[1].split(' ')[0])
                val = float(line.split('val PPL=')[1].split(',')[0])
                rpam_epochs.append(ep)
                rpam_vals.append(val)

    qpam_epochs = np.array(qpam_epochs, dtype=np.float64)
    qpam_vals = np.array(qpam_vals)
    rpam_epochs = np.array(rpam_epochs, dtype=np.float64)
    rpam_vals = np.array(rpam_vals)

    fig, ax = makefig()
    cscat(qpam_epochs, qpam_vals, color='#2166ac', s=50, edgecolor='k', marker='o',
          fig=fig, ax=ax, makefigax=False, label='QPAM (complex)', aspect='auto')
    ax.plot(qpam_epochs, qpam_vals, '-', color='#2166ac', linewidth=1.0, alpha=0.5)

    cscat(rpam_epochs, rpam_vals, color='#b2182b', s=50, edgecolor='k', marker='s',
          fig=fig, ax=ax, makefigax=False, label='RPAM (real)', aspect='auto')
    ax.plot(rpam_epochs, rpam_vals, '-', color='#b2182b', linewidth=1.0, alpha=0.5)

    ax.legend(fontsize=9, frameon=False)
    set_aesthetics(fig=fig, ax=ax, makefigax=False,
                   xlabel='Epoch', ylabel='Val PPL')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'cmp_val_ppl.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved cmp_val_ppl.pdf")
    print(f"  QPAM best: {min(qpam_vals):.2f} at epoch {qpam_epochs[np.argmin(qpam_vals)]:.0f}")
    print(f"  RPAM best: {min(rpam_vals):.2f} at epoch {rpam_epochs[np.argmin(rpam_vals)]:.0f}")


# ══════════════════════════════════════════════════════════════
# 2. K·Q vs K*·Q retrieval patterns on same sentence
# ══════════════════════════════════════════════════════════════

def plot_retrieval_comparison(qpam, rpam, tokenizer):
    """Same sentence through both models, compare retrieval patterns."""
    print("\n[2] Retrieval pattern comparison...")

    sentences = [
        "The bat flew out of the cave at dusk",
        "The bank approved his loan application last week",
    ]

    for sent_idx, sentence in enumerate(sentences):
        ids = tokenizer.encode(sentence)
        tokens_text = [tokenizer.decode([t]) for t in ids]
        T = len(ids)
        tokens_mx = mx.array([ids], dtype=mx.int32)

        # QPAM: get K*·Q at layer 8
        target_layer = 8
        z = mx.stack([qpam.embed_r[tokens_mx], qpam.embed_i[tokens_mx]], axis=-1)
        z = qpam.input_norm(z)
        for li in range(target_layer + 1):
            z = z + qpam.blocks[li].alpha_cgu * qpam.blocks[li].cgu(qpam.blocks[li].norm1(z))
            if li == target_layer:
                x_normed = qpam.blocks[li].norm2(z)
            z = z + qpam.blocks[li].alpha_pam * qpam.blocks[li].pam(
                qpam.blocks[li].norm2(z) if li < target_layer else x_normed)
            mx.eval(z)

        qkv = qpam.blocks[target_layer].pam.qkv(x_normed)
        mx.eval(qkv)
        qkv_np = np.array(qkv).reshape(1, T, 3, 6, 64, 2)
        Q_c, K_c = qkv_np[0, :, 0], qkv_np[0, :, 1]

        W_qpam = np.zeros((T, T))
        for h in range(6):
            Qr, Qi = Q_c[:, h, :, 0], Q_c[:, h, :, 1]
            Kr, Ki = K_c[:, h, :, 0], K_c[:, h, :, 1]
            W_qpam += (Kr @ Qr.T + Ki @ Qi.T)
        W_qpam /= 6

        # RPAM: get K·Q at layer 8
        x_r = rpam.embed[tokens_mx]
        x_r = rpam.input_norm(x_r)
        for li in range(target_layer + 1):
            x_r = x_r + rpam.blocks[li].alpha_cgu * rpam.blocks[li].ffn(rpam.blocks[li].norm1(x_r))
            if li == target_layer:
                x_normed_r = rpam.blocks[li].norm2(x_r)
            x_r = x_r + rpam.blocks[li].alpha_pam * rpam.blocks[li].pam(
                rpam.blocks[li].norm2(x_r) if li < target_layer else x_normed_r)
            mx.eval(x_r)

        qkv_r = rpam.blocks[target_layer].pam.qkv(x_normed_r)
        mx.eval(qkv_r)
        qkv_r_np = np.array(qkv_r).reshape(1, T, 3, 9, 64)
        Q_r, K_r = qkv_r_np[0, :, 0], qkv_r_np[0, :, 1]

        W_rpam = np.zeros((T, T))
        for h in range(9):
            W_rpam += Q_r[:, h] @ K_r[:, h].T
        W_rpam /= 9

        # Apply causal mask
        causal = np.tril(np.ones((T, T)))
        W_qpam_d = W_qpam * causal
        W_qpam_d[causal == 0] = np.nan
        W_rpam_d = W_rpam * causal
        W_rpam_d[causal == 0] = np.nan

        vmax = max(np.nanpercentile(np.abs(W_qpam_d[~np.isnan(W_qpam_d)]), 95),
                   np.nanpercentile(np.abs(W_rpam_d[~np.isnan(W_rpam_d)]), 95))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

        im1 = ax1.imshow(W_qpam_d, cmap='RdBu_r', aspect='equal', origin='upper',
                          vmin=-vmax, vmax=vmax)
        ax1.set_xticks(range(T))
        ax1.set_xticklabels(tokens_text, rotation=45, ha='right', fontsize=7)
        ax1.set_yticks(range(T))
        ax1.set_yticklabels(tokens_text, fontsize=7)
        ax1.set_title('QPAM (complex)', fontsize=10)

        im2 = ax2.imshow(W_rpam_d, cmap='RdBu_r', aspect='equal', origin='upper',
                          vmin=-vmax, vmax=vmax)
        ax2.set_xticks(range(T))
        ax2.set_xticklabels(tokens_text, rotation=45, ha='right', fontsize=7)
        ax2.set_yticks(range(T))
        ax2.set_yticklabels(tokens_text, fontsize=7)
        ax2.set_title('RPAM (real)', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f'cmp_retrieval_{sent_idx}.pdf'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved cmp_retrieval_{sent_idx}.pdf")

        # Stats
        qpam_neg = np.mean(W_qpam[causal.astype(bool)] < 0)
        rpam_neg = np.mean(W_rpam[causal.astype(bool)] < 0)
        print(f"  '{sentence}': QPAM {qpam_neg:.0%} destructive, RPAM {rpam_neg:.0%} destructive")


# ══════════════════════════════════════════════════════════════
# 3. Embedding similarity comparison
# ══════════════════════════════════════════════════════════════

def plot_embedding_comparison(qpam, rpam, tokenizer):
    """Compare how both models organize embeddings for synonym pairs."""
    print("\n[3] Embedding similarity comparison...")

    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    from nltk.corpus import wordnet as wn
    from scipy.stats import mannwhitneyu

    single_token_words = {}
    for word in wn.all_lemma_names():
        if '_' in word or len(word) < 3:
            continue
        ids = tokenizer.encode(' ' + word)
        if len(ids) == 1:
            single_token_words[word] = ids[0]

    syn_pairs = set()
    for word in list(single_token_words.keys())[:3000]:
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

    # QPAM: conjugate inner product
    qpam_er = np.array(qpam.embed_r)
    qpam_ei = np.array(qpam.embed_i)

    # RPAM: cosine similarity
    rpam_e = np.array(rpam.embed)

    def qpam_sim(w1, w2):
        id1, id2 = single_token_words[w1], single_token_words[w2]
        z1 = qpam_er[id1] + 1j * qpam_ei[id1]
        z2 = qpam_er[id2] + 1j * qpam_ei[id2]
        dot = np.sum(np.conj(z1) * z2)
        return dot.real / (np.linalg.norm(z1) * np.linalg.norm(z2) + 1e-10)

    def rpam_sim(w1, w2):
        id1, id2 = single_token_words[w1], single_token_words[w2]
        return np.dot(rpam_e[id1], rpam_e[id2]) / (np.linalg.norm(rpam_e[id1]) * np.linalg.norm(rpam_e[id2]) + 1e-10)

    qpam_syn = np.array([qpam_sim(*p) for p in syn_pairs])
    qpam_rand = np.array([qpam_sim(*p) for p in rand_pairs])
    rpam_syn = np.array([rpam_sim(*p) for p in syn_pairs])
    rpam_rand = np.array([rpam_sim(*p) for p in rand_pairs])

    def sep(a, b):
        return (np.mean(a) - np.mean(b)) / np.sqrt((np.var(a) + np.var(b)) / 2)

    u_q, p_q = mannwhitneyu(qpam_syn, qpam_rand, alternative='greater')
    u_r, p_r = mannwhitneyu(rpam_syn, rpam_rand, alternative='greater')

    print(f"  N={len(syn_pairs)} pairs")
    print(f"  QPAM Re(<z1*|z2>): d_sep={sep(qpam_syn, qpam_rand):.4f}, p={p_q:.2e}")
    print(f"  RPAM cosine:       d_sep={sep(rpam_syn, rpam_rand):.4f}, p={p_r:.2e}")

    # 2D density: QPAM sim vs RPAM sim, synonyms and random
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True, sharey=True)

    chist2d(rpam_syn.astype(np.float64), qpam_syn.astype(np.float64),
            nx=80, ny=80, fig=fig, ax=ax1, makefigax=False,
            xlabel='RPAM cosine', ylabel=r'QPAM Re$\langle z_1^*|z_2\rangle$',
            aspect='auto', dens_scale=0.3)
    ax1.set_title(f'Synonyms (N={len(syn_pairs)})', fontsize=10)

    chist2d(rpam_rand.astype(np.float64), qpam_rand.astype(np.float64),
            nx=80, ny=80, fig=fig, ax=ax2, makefigax=False,
            xlabel='RPAM cosine',
            aspect='auto', dens_scale=0.3)
    ax2.set_title(f'Random (N={len(rand_pairs)})', fontsize=10)

    for ax in [ax1, ax2]:
        ax.plot([-0.3, 0.3], [-0.3, 0.3], '--', color='gray', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'cmp_embedding_similarity.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved cmp_embedding_similarity.pdf")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading QPAM...")
    qpam = load_qpam()
    print("Loading RPAM...")
    rpam = load_rpam()

    plot_val_ppl_comparison()
    plot_retrieval_comparison(qpam, rpam, tokenizer)
    plot_embedding_comparison(qpam, rpam, tokenizer)

    print("\n" + "=" * 60)
    print(f"  All comparison figures saved to {OUTDIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
