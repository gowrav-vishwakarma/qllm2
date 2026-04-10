"""
Extended hypothesis tests for PAM paper.

1. Phase coherence vs cosine similarity — does complex add over real?
2. State matrix rank evolution over a sequence
3. Compositional binding/retrieval capacity test
4. Passkey retrieval at various distances

Usage:
    cd /Users/caug/npcww/qnlp/qllm-private
    uv run python v6/paper/scripts/analyze_extended.py
"""

import os, sys, math, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from scipy.stats import pearsonr, mannwhitneyu

os.environ['MATPLOTLIBRC'] = '/Users/caug/npcww/qnlp/ket-nlp/matplotlibrc'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from chroptiks.plotting_utils import makefig, set_aesthetics, scatter as cscat, plot2dhist as chist2d
from qpam_mlx import QPAMModel

OUTDIR = '/Users/caug/npcww/qnlp/qllm-private/v6/paper/figures'
LOGDIR = '/Users/caug/npcww/qnlp/ket-nlp'


def load_model(checkpoint_path):
    model = QPAMModel(
        vocab_size=50257, dim=384, num_layers=16,
        expand=3, num_heads=6, head_dim=64,
    )
    model.load_weights(checkpoint_path)
    mx.eval(model.parameters())
    return model


def get_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


# ══════════════════════════════════════════════════════════════
# 1. Phase coherence vs cosine similarity
# ══════════════════════════════════════════════════════════════

def test_phase_vs_cosine(model, tokenizer):
    """Compare complex conjugate inner product to real-only cosine similarity.
    If complex adds something, phase coherence should separate semantic
    categories better than cosine on the real parts alone."""
    print("\n[1] Phase coherence vs cosine similarity...")

    embed_r = np.array(model.embed_r)
    embed_i = np.array(model.embed_i)

    synonyms = [
        ("big","large"),("happy","glad"),("fast","quick"),("smart","clever"),
        ("begin","start"),("end","finish"),("angry","furious"),("cold","freezing"),
        ("hot","burning"),("small","tiny"),("rich","wealthy"),("poor","destitute"),
        ("king","monarch"),("house","home"),("car","automobile"),
    ]
    antonyms = [
        ("big","small"),("happy","sad"),("fast","slow"),("hot","cold"),
        ("rich","poor"),("light","dark"),("old","young"),("good","bad"),
        ("up","down"),("love","hate"),("war","peace"),("life","death"),
        ("open","closed"),("strong","weak"),("high","low"),
    ]
    randoms = [
        ("table","freedom"),("cloud","purple"),("river","syntax"),
        ("guitar","theorem"),("window","gravity"),("paper","dolphin"),
        ("mountain","algebra"),("forest","voltage"),("ocean","pencil"),
        ("chair","electron"),("lamp","justice"),("book","magnet"),
        ("tree","frequency"),("door","enzyme"),("rain","circuit"),
    ]

    def get_both_sims(w1, w2):
        id1 = tokenizer.encode(" " + w1)
        id2 = tokenizer.encode(" " + w2)
        if len(id1) != 1 or len(id2) != 1:
            return None

        # Complex conjugate inner product (phase coherence)
        z1 = embed_r[id1[0]] + 1j * embed_i[id1[0]]
        z2 = embed_r[id2[0]] + 1j * embed_i[id2[0]]
        dot = np.sum(np.conj(z1) * z2)
        phase_coh = np.abs(dot) / (np.linalg.norm(z1) * np.linalg.norm(z2) + 1e-10)

        # Real-only cosine similarity
        r1 = embed_r[id1[0]]
        r2 = embed_r[id2[0]]
        cos_sim = np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2) + 1e-10)

        return phase_coh, cos_sim

    all_phase = []
    all_cosine = []
    all_labels = []  # 0=synonym, 1=antonym, 2=random

    for pairs, label in [(synonyms, 0), (antonyms, 1), (randoms, 2)]:
        for w1, w2 in pairs:
            result = get_both_sims(w1, w2)
            if result:
                all_phase.append(result[0])
                all_cosine.append(result[1])
                all_labels.append(label)

    all_phase = np.array(all_phase)
    all_cosine = np.array(all_cosine)
    all_labels = np.array(all_labels)

    # Scatter: phase coherence vs cosine sim, colored by relation type
    fig, ax = makefig()
    colors = {0: '#2166ac', 1: '#b2182b', 2: '#636363'}
    markers = {0: 'o', 1: 's', 2: '^'}
    names = {0: 'Synonyms', 1: 'Antonyms', 2: 'Random'}

    for label in [0, 1, 2]:
        mask = all_labels == label
        cscat(all_cosine[mask], all_phase[mask],
              color=colors[label], marker=markers[label],
              s=50, edgecolor='k',
              fig=fig, ax=ax, makefigax=False,
              label=names[label], aspect='auto')

    # Diagonal: where phase == cosine
    lims = [min(all_cosine.min(), all_phase.min()) - 0.02,
            max(all_cosine.max(), all_phase.max()) + 0.02]
    ax.plot(lims, lims, '--', color='gray', linewidth=0.8, alpha=0.5)

    ax.legend(fontsize=8, frameon=False)
    set_aesthetics(fig=fig, ax=ax, makefigax=False,
                   xlabel=r'Cosine similarity (real part only)',
                   ylabel=r'Phase coherence (complex)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'ext_phase_vs_cosine.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved ext_phase_vs_cosine.pdf")

    # Separation power: Mann-Whitney for each metric
    for metric_name, vals in [("Phase coherence", all_phase), ("Cosine sim", all_cosine)]:
        syn_vals = vals[all_labels == 0]
        rand_vals = vals[all_labels == 2]
        u, p = mannwhitneyu(syn_vals, rand_vals, alternative='greater')
        print(f"  {metric_name}: syn vs rand U={u}, p={p:.6f}")


# ══════════════════════════════════════════════════════════════
# 2. State matrix rank evolution over a sequence
# ══════════════════════════════════════════════════════════════

def test_state_rank_evolution(model, tokenizer):
    """Track the effective rank of S_t as tokens accumulate.
    Run in recurrent mode, capture state after each token."""
    print("\n[2] State matrix rank evolution...")

    text = ("It was the best of times, it was the worst of times, it was the age of "
            "wisdom, it was the age of foolishness, it was the epoch of belief, it was "
            "the epoch of incredulity, it was the season of Light, it was the season of "
            "Darkness, it was the spring of hope, it was the winter of despair.")
    ids = tokenizer.encode(text)
    T = len(ids)

    # We need to simulate recurrent mode manually for layer 0
    # S_t = gamma * S_{t-1} + V'_t outer K_t*
    block = model.blocks[0]
    H, d = block.pam.num_heads, block.pam.head_dim

    tokens = mx.array([ids], dtype=mx.int32)
    B = 1

    # Get embeddings
    er = model.embed_r[tokens]
    ei = model.embed_i[tokens]
    z = mx.stack([er, ei], axis=-1)
    z = model.input_norm(z)
    # Apply CGU
    z = z + block.alpha_cgu * block.cgu(block.norm1(z))
    x = block.norm2(z)

    # Get Q, K, V
    qkv = block.pam.qkv(x)
    mx.eval(qkv)
    qkv_np = np.array(qkv).reshape(B, T, 3, H, d, 2)
    K = qkv_np[0, :, 1]  # [T, H, d, 2]
    V = qkv_np[0, :, 2]  # [T, H, d, 2]

    # Get decay
    x_np = np.array(x)
    x_flat = np.concatenate([x_np[0, :, :, 0], x_np[0, :, :, 1]], axis=-1)  # [T, 2D]
    dt_proj = np.array(block.pam.dt_proj)  # [H, 2D]
    dt_bias = np.array(block.pam.dt_bias)  # [H]
    dt_input = x_flat @ dt_proj.T + dt_bias  # [T, H]
    dt = np.log(1 + np.exp(dt_input))
    gamma = np.exp(-dt)  # [T, H]

    # Simulate recurrent state evolution for head 0
    h = 0
    S = np.zeros((d, d), dtype=np.complex128)
    eff_ranks = []
    token_positions = []

    for t in range(T):
        g = gamma[t, h]
        k = K[t, h, :, 0] + 1j * K[t, h, :, 1]  # [d]
        v = V[t, h, :, 0] + 1j * V[t, h, :, 1]  # [d]

        S = g * S + np.outer(v, np.conj(k))

        # Effective rank via SVD
        sv = np.linalg.svd(S, compute_uv=False)
        sv_norm = sv / (sv.sum() + 1e-10)
        sv_norm = sv_norm[sv_norm > 1e-10]
        eff_rank = np.exp(-np.sum(sv_norm * np.log(sv_norm)))

        eff_ranks.append(eff_rank)
        token_positions.append(float(t))

    eff_ranks = np.array(eff_ranks)
    token_positions = np.array(token_positions)

    fig, ax = makefig()
    cscat(token_positions, eff_ranks, ccode=token_positions, cmap='plasma',
          s=20, edgecolor='k', fig=fig, ax=ax, makefigax=False,
          xlabel='Token position', ylabel='Effective rank of $S_t$',
          aspect='auto')
    ax.axhline(d, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(T - 2, d + 0.5, f'd = {d}', fontsize=8, color='gray', ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'ext_state_rank_evolution.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved ext_state_rank_evolution.pdf")
    print(f"  Final effective rank: {eff_ranks[-1]:.1f} / {d}")
    print(f"  Max effective rank: {eff_ranks.max():.1f}")


# ══════════════════════════════════════════════════════════════
# 3. Compositional binding / retrieval capacity
# ══════════════════════════════════════════════════════════════

def test_binding_capacity(model, tokenizer):
    """Write N key-value pairs into a PAM state, query for each one,
    measure retrieval accuracy as N grows. Compare to 1/sqrt(n) theory
    for vector-state holographic binding."""
    print("\n[3] Compositional binding capacity...")

    block = model.blocks[0]
    H, d = block.pam.num_heads, block.pam.head_dim

    # Generate random complex key-value pairs
    np.random.seed(42)
    max_n = 200
    n_trials = 20

    ns = list(range(1, 65, 4)) + list(range(65, max_n + 1, 8))
    retrieval_accs = []

    for n in ns:
        trial_accs = []
        for trial in range(n_trials):
            # Random complex keys and values
            keys = np.random.randn(n, d) + 1j * np.random.randn(n, d)
            keys = keys / np.linalg.norm(keys, axis=1, keepdims=True)
            values = np.random.randn(n, d) + 1j * np.random.randn(n, d)
            values = values / np.linalg.norm(values, axis=1, keepdims=True)

            # Build state: S = sum_i v_i outer k_i*
            S = np.zeros((d, d), dtype=np.complex128)
            for i in range(n):
                S += np.outer(values[i], np.conj(keys[i]))

            # Query each key, check if retrieved value is closest to correct
            correct = 0
            for i in range(n):
                retrieved = S @ keys[i]  # conjugate already in the state
                # Compare to all values
                sims = np.array([np.abs(np.sum(np.conj(retrieved) * values[j]))
                                 for j in range(n)])
                if np.argmax(sims) == i:
                    correct += 1
            trial_accs.append(correct / n)

        retrieval_accs.append(np.mean(trial_accs))

    ns = np.array(ns, dtype=np.float64)
    retrieval_accs = np.array(retrieval_accs)

    # Theoretical 1/sqrt(n) curve for vector holographic binding
    # (normalized so it starts at 1.0 for n=1)
    theory_vector = 1.0 / np.sqrt(ns)

    fig, ax = makefig()
    cscat(ns, retrieval_accs, ccode=ns, cmap='plasma',
          s=30, edgecolor='k', fig=fig, ax=ax, makefigax=False,
          xlabel='Number of stored associations $N$',
          ylabel='Retrieval accuracy',
          aspect='auto')
    ax.plot(ns, theory_vector, '--', color='#b2182b', linewidth=1.5,
            label=r'$O(1/\sqrt{N})$ (vector state)')
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'ext_binding_capacity.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved ext_binding_capacity.pdf")
    print(f"  Accuracy at N=1: {retrieval_accs[0]:.3f}")
    print(f"  Accuracy at N={int(ns[-1])}: {retrieval_accs[-1]:.3f}")


# ══════════════════════════════════════════════════════════════
# 4. Passkey retrieval
# ══════════════════════════════════════════════════════════════

def test_passkey_retrieval(model, tokenizer):
    """Embed a random passkey in distractor text at various distances,
    ask the model to predict the next token after a retrieval cue.
    Measures whether conjugate retrieval can recover specific associations
    from a long context."""
    print("\n[4] Passkey retrieval...")

    # Generate distractor text
    distractor_sentences = [
        "The weather was pleasant and the birds were singing in the trees.",
        "Mathematics provides a framework for understanding the natural world.",
        "The committee discussed several proposals for the new building.",
        "Fresh water is essential for the survival of all living organisms.",
        "The history of science is marked by paradigm shifts and discoveries.",
        "Music has the power to evoke strong emotional responses in listeners.",
        "The economic implications of the policy were debated extensively.",
        "Advances in technology have transformed communication globally.",
        "The philosophical implications of consciousness remain unresolved.",
        "Agricultural practices have evolved significantly over millennia.",
    ]

    # Passkey: a specific number embedded in context
    passkey_values = [42, 73, 156, 891, 2047]
    distances = [5, 10, 20, 50, 100, 200]  # tokens between passkey and query

    all_distances = []
    all_losses = []

    for passkey in passkey_values:
        passkey_text = f" The secret number is {passkey}."
        passkey_ids = tokenizer.encode(passkey_text)

        for target_dist in distances:
            # Build: [distractor...] [passkey] [distractor...target_dist tokens...] [cue]
            cue_text = " The secret number is"
            cue_ids = tokenizer.encode(cue_text)

            # Fill distractor to reach target distance
            dist_ids = []
            while len(dist_ids) < target_dist:
                sent = distractor_sentences[np.random.randint(len(distractor_sentences))]
                dist_ids.extend(tokenizer.encode(" " + sent))
            dist_ids = dist_ids[:target_dist]

            # Some leading context
            lead_ids = []
            for _ in range(3):
                sent = distractor_sentences[np.random.randint(len(distractor_sentences))]
                lead_ids.extend(tokenizer.encode(" " + sent))

            full_ids = lead_ids + passkey_ids + dist_ids + cue_ids
            if len(full_ids) > 512:
                full_ids = full_ids[-512:]

            # Run through model, get logits at the cue position
            tokens = mx.array([full_ids], dtype=mx.int32)
            logits = model(tokens)
            mx.eval(logits)
            logits_np = np.array(logits[0])

            # The target token is the first token of the passkey number
            target_token_ids = tokenizer.encode(" " + str(passkey))
            if len(target_token_ids) == 0:
                continue
            target_id = target_token_ids[0]

            # Get the logit at the last position (predicting next token after cue)
            last_logits = logits_np[-1]
            # Log softmax
            last_logits_shifted = last_logits - np.max(last_logits)
            log_probs = last_logits_shifted - np.log(np.sum(np.exp(last_logits_shifted)))

            loss = -log_probs[target_id]
            rank = np.sum(log_probs > log_probs[target_id]) + 1

            all_distances.append(float(target_dist))
            all_losses.append(loss)

    all_distances = np.array(all_distances)
    all_losses = np.array(all_losses)

    # Scatter: distance vs loss, colored by distance
    fig, ax = makefig()
    cscat(all_distances, all_losses, ccode=all_distances, cmap='plasma',
          s=30, edgecolor='k', fig=fig, ax=ax, makefigax=False,
          xlabel='Distance (tokens)',
          ylabel='Cross-entropy loss for passkey',
          aspect='auto')

    # Running mean
    for dist in sorted(set(all_distances)):
        mask = all_distances == dist
        mean_loss = np.mean(all_losses[mask])
        ax.plot(dist, mean_loss, 'D', color='#b2182b', markersize=8,
                markeredgecolor='k', markeredgewidth=0.5, zorder=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'ext_passkey_retrieval.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved ext_passkey_retrieval.pdf")

    # Summary by distance
    for dist in sorted(set(all_distances)):
        mask = all_distances == dist
        print(f"  dist={int(dist):3d}: mean loss={np.mean(all_losses[mask]):.2f}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    checkpoint = os.path.join(LOGDIR, 'checkpoints_qpam_mlx', 'best_model.npz')
    print(f"Loading model from {checkpoint}")
    model = load_model(checkpoint)
    print("  Model loaded")

    tokenizer = get_tokenizer()

    test_phase_vs_cosine(model, tokenizer)
    test_state_rank_evolution(model, tokenizer)
    test_binding_capacity(model, tokenizer)
    test_passkey_retrieval(model, tokenizer)

    print("\n" + "="*60)
    print(f"  All extended figures saved to {OUTDIR}")
    print("="*60)


if __name__ == '__main__':
    main()
