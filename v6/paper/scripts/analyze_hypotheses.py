"""
Hypothesis-driven analyses of trained QPAM model.

Tests specific predictions from the quantum semantics literature:
1. Phase coherence encodes semantic similarity (Agostino et al. 2025)
2. Von Neumann entropy of state tracks semantic complexity
3. Magnitude-phase dissociation: phase ablation vs magnitude ablation
4. Eigenvalue phase structure of complex embeddings
5. Constructive/destructive interference in conjugate retrieval

Usage:
    cd /Users/caug/npcww/qnlp/qllm-private
    uv run python v6/paper/scripts/analyze_hypotheses.py
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
os.makedirs(OUTDIR, exist_ok=True)


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
# 1. Phase coherence encodes semantic similarity
# ══════════════════════════════════════════════════════════════

def test_phase_semantic_similarity(model, tokenizer):
    """Test whether phase alignment between word embeddings correlates
    with semantic relatedness. Agostino et al. (2025) found z > 4000
    for related word pairs via phase interference in GPT-2."""
    print("\n[1] Phase coherence vs semantic similarity...")

    # Word pairs: (word1, word2, relation)
    synonym_pairs = [
        ("big", "large"), ("happy", "glad"), ("fast", "quick"),
        ("smart", "clever"), ("begin", "start"), ("end", "finish"),
        ("angry", "furious"), ("cold", "freezing"), ("hot", "burning"),
        ("small", "tiny"), ("rich", "wealthy"), ("poor", "destitute"),
        ("king", "monarch"), ("house", "home"), ("car", "automobile"),
    ]
    antonym_pairs = [
        ("big", "small"), ("happy", "sad"), ("fast", "slow"),
        ("hot", "cold"), ("rich", "poor"), ("light", "dark"),
        ("old", "young"), ("good", "bad"), ("up", "down"),
        ("love", "hate"), ("war", "peace"), ("life", "death"),
        ("open", "closed"), ("strong", "weak"), ("high", "low"),
    ]
    random_pairs = [
        ("table", "freedom"), ("cloud", "purple"), ("river", "syntax"),
        ("guitar", "theorem"), ("window", "gravity"), ("paper", "dolphin"),
        ("mountain", "algebra"), ("forest", "voltage"), ("ocean", "pencil"),
        ("chair", "electron"), ("lamp", "justice"), ("book", "magnet"),
        ("tree", "frequency"), ("door", "enzyme"), ("rain", "circuit"),
    ]

    # Get complex embeddings
    embed_r = np.array(model.embed_r)  # [V, D]
    embed_i = np.array(model.embed_i)

    def phase_coherence(w1, w2):
        """Compute phase-interference score between two words."""
        id1 = tokenizer.encode(" " + w1)
        id2 = tokenizer.encode(" " + w2)
        if len(id1) != 1 or len(id2) != 1:
            return None
        id1, id2 = id1[0], id2[0]

        # Complex embeddings
        z1 = embed_r[id1] + 1j * embed_i[id1]
        z2 = embed_r[id2] + 1j * embed_i[id2]

        # Conjugate inner product: <z1* | z2>
        conj_dot = np.sum(np.conj(z1) * z2)

        # Phase coherence: |<z1*|z2>| / (|z1| |z2|)
        norm1 = np.sqrt(np.sum(np.abs(z1)**2))
        norm2 = np.sqrt(np.sum(np.abs(z2)**2))
        coherence = np.abs(conj_dot) / (norm1 * norm2 + 1e-10)

        # Phase difference: angle of the conjugate dot product
        phase_diff = np.angle(conj_dot)

        return coherence, phase_diff

    syn_coh, syn_phase = [], []
    ant_coh, ant_phase = [], []
    rand_coh, rand_phase = [], []

    for w1, w2 in synonym_pairs:
        result = phase_coherence(w1, w2)
        if result:
            syn_coh.append(result[0])
            syn_phase.append(result[1])

    for w1, w2 in antonym_pairs:
        result = phase_coherence(w1, w2)
        if result:
            ant_coh.append(result[0])
            ant_phase.append(result[1])

    for w1, w2 in random_pairs:
        result = phase_coherence(w1, w2)
        if result:
            rand_coh.append(result[0])
            rand_phase.append(result[1])

    syn_coh = np.array(syn_coh)
    ant_coh = np.array(ant_coh)
    rand_coh = np.array(rand_coh)
    syn_phase = np.array(syn_phase)
    ant_phase = np.array(ant_phase)
    rand_phase = np.array(rand_phase)

    print(f"  Synonyms:  coherence={np.mean(syn_coh):.4f} +/- {np.std(syn_coh):.4f}")
    print(f"  Antonyms:  coherence={np.mean(ant_coh):.4f} +/- {np.std(ant_coh):.4f}")
    print(f"  Random:    coherence={np.mean(rand_coh):.4f} +/- {np.std(rand_coh):.4f}")

    # Mann-Whitney U test: synonyms vs random
    u_syn, p_syn = mannwhitneyu(syn_coh, rand_coh, alternative='greater')
    u_ant, p_ant = mannwhitneyu(ant_coh, rand_coh, alternative='greater')
    print(f"  Synonyms > Random: U={u_syn}, p={p_syn:.4f}")
    print(f"  Antonyms > Random: U={u_ant}, p={p_ant:.4f}")

    # Scatter: coherence by relation type, color-coded
    fig, ax = makefig()
    n_syn = len(syn_coh)
    n_ant = len(ant_coh)
    n_rand = len(rand_coh)

    # Jitter x positions
    x_syn = np.random.normal(0, 0.08, n_syn)
    x_ant = np.random.normal(1, 0.08, n_ant)
    x_rand = np.random.normal(2, 0.08, n_rand)

    cscat(x_syn, syn_coh, color='#2166ac', s=40, edgecolor='k',
          fig=fig, ax=ax, makefigax=False, label='Synonyms', aspect='auto')
    cscat(x_ant, ant_coh, color='#b2182b', s=40, edgecolor='k',
          fig=fig, ax=ax, makefigax=False, label='Antonyms', aspect='auto')
    cscat(x_rand, rand_coh, color='#636363', s=40, edgecolor='k',
          fig=fig, ax=ax, makefigax=False, label='Random', aspect='auto')

    # Means
    for x_pos, vals, c in [(0, syn_coh, '#2166ac'), (1, ant_coh, '#b2182b'), (2, rand_coh, '#636363')]:
        ax.plot([x_pos - 0.2, x_pos + 0.2], [np.mean(vals), np.mean(vals)],
                '-', color=c, linewidth=2)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Synonyms', 'Antonyms', 'Random'], fontsize=10)
    set_aesthetics(fig=fig, ax=ax, makefigax=False,
                   ylabel=r'Phase coherence $|\langle z_1^* | z_2 \rangle| / (|z_1||z_2|)$')
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'hyp_phase_semantic_coherence.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved hyp_phase_semantic_coherence.pdf")

    # Phase difference distribution
    fig, ax = makefig()
    bins = np.linspace(-np.pi, np.pi, 40)
    ax.hist(syn_phase, bins=bins, density=True, alpha=0.5, color='#2166ac', label='Synonyms')
    ax.hist(ant_phase, bins=bins, density=True, alpha=0.5, color='#b2182b', label='Antonyms')
    ax.hist(rand_phase, bins=bins, density=True, alpha=0.5, color='#636363', label='Random')
    set_aesthetics(fig=fig, ax=ax, makefigax=False,
                   xlabel=r'Phase difference $\angle\langle z_1^* | z_2 \rangle$',
                   ylabel='Density')
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'hyp_phase_diff_distribution.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved hyp_phase_diff_distribution.pdf")


# ══════════════════════════════════════════════════════════════
# 2. Magnitude-phase dissociation ablation
# ══════════════════════════════════════════════════════════════

def test_magnitude_phase_ablation(model, tokenizer):
    """Test whether phase and magnitude carry independent information.
    Three ablations on the embedding:
    (a) Zero imaginary part (destroy phase, keep magnitude ~ real part)
    (b) Normalize to unit magnitude (keep phase, destroy magnitude)
    (c) Randomize phase (keep magnitude, destroy phase structure)
    Measure validation perplexity under each."""
    print("\n[2] Magnitude-phase dissociation ablation...")

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    full_text = "\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer.encode(full_text)[:50000]  # 50k tokens for speed

    seq_len = 512
    n_chunks = len(ids) // (seq_len + 1)
    ids = ids[:n_chunks * (seq_len + 1)]
    chunks = np.array(ids, dtype=np.int32).reshape(n_chunks, seq_len + 1)

    def eval_ppl(model_fn, label, n_eval=50):
        total_loss, total_tokens = 0.0, 0
        for i in range(min(n_eval, n_chunks)):
            batch = mx.array(chunks[i:i+1])
            x, y = batch[:, :-1], batch[:, 1:]
            logits = model_fn(x)
            logits_flat = mx.reshape(logits, (-1, logits.shape[-1]))
            targets_flat = mx.reshape(y, (-1,))
            loss = mx.mean(nn.losses.cross_entropy(logits_flat, targets_flat))
            mx.eval(loss)
            total_loss += loss.item() * x.shape[1]
            total_tokens += x.shape[1]
        ppl = math.exp(total_loss / total_tokens)
        print(f"  {label}: PPL = {ppl:.2f}")
        return ppl

    # Baseline
    baseline_ppl = eval_ppl(model, "Baseline (full complex)")

    # Ablation (a): zero imaginary part
    orig_embed_i = np.array(model.embed_i).copy()
    model.embed_i = mx.zeros_like(model.embed_i)
    mx.eval(model.parameters())
    zero_imag_ppl = eval_ppl(model, "Zero imaginary (phase destroyed)")
    model.embed_i = mx.array(orig_embed_i)
    mx.eval(model.parameters())

    # Ablation (b): normalize to unit magnitude
    orig_embed_r = np.array(model.embed_r).copy()
    orig_embed_i_np = np.array(model.embed_i).copy()
    mag = np.sqrt(orig_embed_r**2 + orig_embed_i_np**2 + 1e-8)
    model.embed_r = mx.array(orig_embed_r / mag)
    model.embed_i = mx.array(orig_embed_i_np / mag)
    mx.eval(model.parameters())
    unit_mag_ppl = eval_ppl(model, "Unit magnitude (phase only)")
    model.embed_r = mx.array(orig_embed_r)
    model.embed_i = mx.array(orig_embed_i_np)
    mx.eval(model.parameters())

    # Ablation (c): randomize phase, keep magnitude
    mag = np.sqrt(orig_embed_r**2 + orig_embed_i_np**2)
    rand_phase = np.random.uniform(-np.pi, np.pi, mag.shape)
    model.embed_r = mx.array(mag * np.cos(rand_phase))
    model.embed_i = mx.array(mag * np.sin(rand_phase))
    mx.eval(model.parameters())
    rand_phase_ppl = eval_ppl(model, "Random phase (magnitude only)")
    model.embed_r = mx.array(orig_embed_r)
    model.embed_i = mx.array(orig_embed_i_np)
    mx.eval(model.parameters())

    # Plot
    labels = ['Full\ncomplex', 'Zero\nimaginary', 'Unit\nmagnitude', 'Random\nphase']
    ppls = [baseline_ppl, zero_imag_ppl, unit_mag_ppl, rand_phase_ppl]
    colors = ['#2166ac', '#636363', '#1b7837', '#b2182b']

    fig, ax = makefig()
    x_pos = np.arange(len(labels), dtype=np.float32)
    for i, (xp, ppl, c) in enumerate(zip(x_pos, ppls, colors)):
        cscat(np.array([xp]), np.array([ppl]), color=c, s=120, edgecolor='k',
              fig=fig, ax=ax, makefigax=False, aspect='auto')
        ax.annotate(f'{ppl:.1f}', (xp, ppl), fontsize=9, ha='center',
                    xytext=(0, 8), textcoords='offset points')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    set_aesthetics(fig=fig, ax=ax, makefigax=False,
                   ylabel='Validation perplexity')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'hyp_mag_phase_ablation.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved hyp_mag_phase_ablation.pdf")


# ══════════════════════════════════════════════════════════════
# 3. Eigenvalue phase structure of complex embeddings
# ══════════════════════════════════════════════════════════════

def test_eigenvalue_phase(model, tokenizer):
    """Compute eigenvalues of the complex embedding covariance matrix.
    If phase encodes semantic categories, eigenvalue phases should cluster."""
    print("\n[3] Eigenvalue phase structure...")

    embed_r = np.array(model.embed_r)  # [V, D]
    embed_i = np.array(model.embed_i)
    Z = embed_r + 1j * embed_i  # [V, D]

    # Complex covariance: Z^H Z / V
    cov = Z.conj().T @ Z / Z.shape[0]  # [D, D]
    eigenvalues = np.linalg.eigvals(cov)

    # Sort by magnitude
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]

    ev_mag = np.abs(eigenvalues)
    ev_phase = np.angle(eigenvalues)
    ev_idx = np.arange(len(eigenvalues), dtype=np.float32)

    # Scatter: eigenvalue in complex plane
    fig, ax = makefig()
    cscat(eigenvalues.real.astype(np.float64), eigenvalues.imag.astype(np.float64),
          ccode=ev_idx, cmap='plasma', s=8, alpha=0.6,
          fig=fig, ax=ax, makefigax=False,
          xlabel=r'Re($\lambda$)', ylabel=r'Im($\lambda$)',
          aspect='equal')
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Eigenvalue index')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'hyp_eigenvalue_complex_plane.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved hyp_eigenvalue_complex_plane.pdf")

    # 2D hist: eigenvalue magnitude vs phase
    fig, ax = makefig()
    chist2d(ev_mag.astype(np.float64), ev_phase.astype(np.float64),
            nx=80, ny=80, fig=fig, ax=ax, makefigax=False,
            xlabel=r'$|\lambda|$', ylabel=r'$\angle\lambda$',
            aspect='auto', dens_scale=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'hyp_eigenvalue_mag_phase.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved hyp_eigenvalue_mag_phase.pdf")

    print(f"  Top 10 eigenvalue magnitudes: {np.abs(eigenvalues[:10]).real}")
    print(f"  Phase range: [{ev_phase.min():.3f}, {ev_phase.max():.3f}]")


# ══════════════════════════════════════════════════════════════
# 4. Interference in retrieval: K* · Q decomposition
# ══════════════════════════════════════════════════════════════

def test_retrieval_interference(model, tokenizer):
    """Run text through one PAM layer, extract Q and K, compute K*·Q,
    and decompose into constructive (positive real) vs destructive
    (negative real) interference. Show that the model actively uses
    both, not just positive similarity."""
    print("\n[4] Constructive vs destructive interference in retrieval...")

    text = ("The bat flew out of the cave at dusk and the bank of the river "
            "was lined with old willows whose roots reached down into the dark water.")
    ids = tokenizer.encode(text)
    tokens = mx.array([ids], dtype=mx.int32)

    B, T = tokens.shape
    H, d = model.blocks[0].pam.num_heads, model.blocks[0].pam.head_dim

    # Get input to first PAM layer
    er = model.embed_r[tokens]
    ei = model.embed_i[tokens]
    z = mx.stack([er, ei], axis=-1)
    z = model.input_norm(z)
    z = z + model.blocks[0].alpha_cgu * model.blocks[0].cgu(model.blocks[0].norm1(z))
    x = model.blocks[0].norm2(z)

    # Extract Q, K from PAM layer 0
    qkv = model.blocks[0].pam.qkv(x)  # [B, T, 3*H*d, 2]
    mx.eval(qkv)
    qkv_np = np.array(qkv)
    qkv_r = qkv_np.reshape(B, T, 3, H, d, 2)
    Q = qkv_r[:, :, 0]  # [B, T, H, d, 2]
    K = qkv_r[:, :, 1]

    # Compute conjugate inner product K* · Q for all pairs
    # For head 0
    h = 0
    Qr, Qi = Q[0, :, h, :, 0], Q[0, :, h, :, 1]  # [T, d]
    Kr, Ki = K[0, :, h, :, 0], K[0, :, h, :, 1]

    # K* · Q = (Kr - iKi)(Qr + iQi) = (KrQr + KiQi) + i(KrQi - KiQr)
    # Real part: constructive/destructive, Imag part: phase rotation
    W_real = Kr @ Qr.T + Ki @ Qi.T  # [T, T]
    W_imag = Kr @ Qi.T - Ki @ Qr.T  # [T, T]

    # Only look at causal (lower triangle)
    mask = np.tril(np.ones((T, T), dtype=bool), k=-1)
    w_real = W_real[mask].ravel()
    w_imag = W_imag[mask].ravel()

    # 2D hist: Re(K*Q) vs Im(K*Q)
    fig, ax = makefig()
    chist2d(w_real.astype(np.float64), w_imag.astype(np.float64),
            nx=200, ny=200, fig=fig, ax=ax, makefigax=False,
            xlabel=r'Re$(K^* \cdot Q)$ (constructive/destructive)',
            ylabel=r'Im$(K^* \cdot Q)$ (phase rotation)',
            aspect='equal', dens_scale=0.3)
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'hyp_retrieval_interference.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved hyp_retrieval_interference.pdf")

    # Fraction negative (destructive interference)
    frac_neg = np.mean(w_real < 0)
    print(f"  Fraction of K*Q with negative real part (destructive): {frac_neg:.3f}")
    print(f"  Mean Re(K*Q): {np.mean(w_real):.4f}")
    print(f"  Std  Re(K*Q): {np.std(w_real):.4f}")

    # Per-query: what fraction of keys produce destructive interference?
    fig, ax = makefig()
    frac_destructive_per_query = []
    for t in range(1, T):
        real_vals = W_real[:t, t]
        frac = np.mean(real_vals < 0)
        frac_destructive_per_query.append(frac)

    positions = np.arange(1, T, dtype=np.float32)
    frac_arr = np.array(frac_destructive_per_query)
    cscat(positions, frac_arr, ccode=positions, cmap='plasma',
          s=15, fig=fig, ax=ax, makefigax=False,
          xlabel='Query position', ylabel='Fraction destructive interference',
          aspect='auto')
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(T - 2, 0.51, '0.5 (symmetric)', fontsize=8, color='gray', ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'hyp_destructive_fraction.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved hyp_destructive_fraction.pdf")


# ══════════════════════════════════════════════════════════════
# 5. Contextual disambiguation via phase rotation
# ══════════════════════════════════════════════════════════════

def test_contextual_disambiguation(model, tokenizer):
    """Polysemous words get different phase representations depending on
    context. Run the same word through different sentences, extract its
    final-layer complex representation, and measure phase distance between
    same-sense vs different-sense usages."""
    print("\n[5] Contextual disambiguation via phase...")

    # Ambiguous words with two contexts each
    test_cases = {
        'bank': {
            'financial': [
                "She deposited the check at the bank downtown",
                "The bank approved his mortgage application last week",
                "Interest rates at the bank have been rising steadily",
            ],
            'river': [
                "The fisherman sat on the bank of the river",
                "Willows grew along the muddy bank of the stream",
                "The boat was tied to a post on the bank",
            ],
        },
        'bat': {
            'animal': [
                "The bat flew out of the cave at dusk",
                "A colony of bats roosted in the old barn",
                "The bat hung upside down from the ceiling",
            ],
            'sports': [
                "He swung the bat and hit a home run",
                "The cricket bat was made of English willow",
                "She picked up the bat and walked to the plate",
            ],
        },
        'crane': {
            'bird': [
                "The crane waded through the shallow marsh",
                "A flock of cranes migrated south for winter",
                "The white crane stood motionless in the water",
            ],
            'machine': [
                "The crane lifted the steel beam into place",
                "A massive crane towered over the construction site",
                "The operator positioned the crane above the foundation",
            ],
        },
        'spring': {
            'season': [
                "The flowers bloomed in the warm spring air",
                "Spring arrived late that year with heavy rains",
                "Birds returned to the garden every spring",
            ],
            'water': [
                "Fresh water bubbled up from the spring",
                "The mountain spring fed a clear cold stream",
                "They found a natural spring near the campsite",
            ],
        },
    }

    # Literary passages for richer context
    literary_passages = [
        # Dickens
        "It was the best of times, it was the worst of times, it was the age of "
        "wisdom, it was the age of foolishness, it was the epoch of belief, it was "
        "the epoch of incredulity.",
        # Shakespeare
        "To be, or not to be, that is the question: whether it is nobler in the "
        "mind to suffer the slings and arrows of outrageous fortune, or to take "
        "arms against a sea of troubles.",
        # Marquez
        "Many years later, as he faced the firing squad, Colonel Aureliano Buendia "
        "was to remember that distant afternoon when his father took him to discover ice.",
        # Bolano
        "The sun set and the whole desert seemed to be on fire, and the shadows "
        "of the rocks stretched out like fingers reaching for something that was "
        "always just beyond their grasp.",
    ]

    def get_word_activation(sentence, target_word):
        """Get the complex activation of target_word at the final layer."""
        ids = tokenizer.encode(sentence)
        # Find the target token
        target_ids = tokenizer.encode(" " + target_word)
        if len(target_ids) != 1:
            return None
        target_id = target_ids[0]

        # Find position of target in sentence
        positions = [i for i, tid in enumerate(ids) if tid == target_id]
        if not positions:
            return None
        pos = positions[0]

        tokens = mx.array([ids], dtype=mx.int32)

        # Forward pass through all layers
        er = model.embed_r[tokens]
        ei = model.embed_i[tokens]
        z = mx.stack([er, ei], axis=-1)
        z = model.input_norm(z)
        for block in model.blocks:
            z = z + block.alpha_cgu * block.cgu(block.norm1(z))
            z = z + block.alpha_pam * block.pam(block.norm2(z))
        z = model.final_proj(z)
        z = model.final_norm(z)
        mx.eval(z)

        z_np = np.array(z[0, pos])  # [D, 2]
        return z_np[..., 0] + 1j * z_np[..., 1]  # [D] complex

    # For each ambiguous word, collect activations by sense
    all_within_dists = []
    all_between_dists = []
    word_results = {}

    for word, senses in test_cases.items():
        sense_names = list(senses.keys())
        activations = {s: [] for s in sense_names}

        for sense_name, sentences in senses.items():
            for sent in sentences:
                act = get_word_activation(sent, word)
                if act is not None:
                    activations[sense_name].append(act)

        if all(len(v) >= 2 for v in activations.values()):
            s0, s1 = sense_names
            acts0 = activations[s0]
            acts1 = activations[s1]

            # Within-sense phase distances
            for i in range(len(acts0)):
                for j in range(i+1, len(acts0)):
                    dot = np.sum(np.conj(acts0[i]) * acts0[j])
                    coh = np.abs(dot) / (np.linalg.norm(acts0[i]) * np.linalg.norm(acts0[j]) + 1e-10)
                    all_within_dists.append(coh)
            for i in range(len(acts1)):
                for j in range(i+1, len(acts1)):
                    dot = np.sum(np.conj(acts1[i]) * acts1[j])
                    coh = np.abs(dot) / (np.linalg.norm(acts1[i]) * np.linalg.norm(acts1[j]) + 1e-10)
                    all_within_dists.append(coh)

            # Between-sense phase distances
            for a0 in acts0:
                for a1 in acts1:
                    dot = np.sum(np.conj(a0) * a1)
                    coh = np.abs(dot) / (np.linalg.norm(a0) * np.linalg.norm(a1) + 1e-10)
                    all_between_dists.append(coh)

            word_results[word] = {
                'within_mean': np.mean([all_within_dists[-len(acts0)*(len(acts0)-1)//2:]]),
                'between_mean': np.mean([all_between_dists[-len(acts0)*len(acts1):]]),
            }
            print(f"  {word}: within-sense coherence={word_results[word]['within_mean']:.4f}, "
                  f"between-sense={word_results[word]['between_mean']:.4f}")

    within = np.array(all_within_dists)
    between = np.array(all_between_dists)

    if len(within) > 0 and len(between) > 0:
        u, p = mannwhitneyu(within, between, alternative='greater')
        print(f"  Within > Between: U={u}, p={p:.6f}")

        # Scatter
        fig, ax = makefig()
        x_within = np.random.normal(0, 0.08, len(within))
        x_between = np.random.normal(1, 0.08, len(between))
        cscat(x_within, within, color='#2166ac', s=50, edgecolor='k',
              fig=fig, ax=ax, makefigax=False, label='Same sense', aspect='auto')
        cscat(x_between, between, color='#b2182b', s=50, edgecolor='k',
              fig=fig, ax=ax, makefigax=False, label='Different sense', aspect='auto')
        ax.plot([-0.2, 0.2], [np.mean(within), np.mean(within)], '-', color='#2166ac', linewidth=2)
        ax.plot([0.8, 1.2], [np.mean(between), np.mean(between)], '-', color='#b2182b', linewidth=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Same sense', 'Different sense'], fontsize=10)
        ax.text(0.05, 0.92, f'$U={u}$, $p={p:.4f}$',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='gray', alpha=0.9))
        set_aesthetics(fig=fig, ax=ax, makefigax=False,
                       ylabel='Phase coherence (same word, different context)')
        ax.legend(fontsize=8, frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, 'hyp_contextual_disambiguation.pdf'), dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved hyp_contextual_disambiguation.pdf")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    checkpoint = os.path.join(LOGDIR, 'checkpoints_qpam_mlx', 'best_model.npz')
    print(f"Loading model from {checkpoint}")
    model = load_model(checkpoint)
    print("  Model loaded")

    tokenizer = get_tokenizer()

    test_phase_semantic_similarity(model, tokenizer)
    test_magnitude_phase_ablation(model, tokenizer)
    test_eigenvalue_phase(model, tokenizer)
    test_retrieval_interference(model, tokenizer)
    test_contextual_disambiguation(model, tokenizer)

    print("\n" + "="*60)
    print(f"  All hypothesis figures saved to {OUTDIR}")
    print("="*60)


if __name__ == '__main__':
    main()
