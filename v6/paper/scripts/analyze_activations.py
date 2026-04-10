"""
Activation-level analysis of trained QPAM model.

Runs real text through the model, captures intermediate representations,
and analyzes what complex-valued computation actually does.

Usage:
    cd /Users/caug/npcww/qnlp/qllm-private
    uv run python v6/paper/scripts/analyze_activations.py
"""

import os, sys, math, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn

os.environ['MATPLOTLIBRC'] = '/Users/caug/npcww/qnlp/ket-nlp/matplotlibrc'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.stats as st

from chroptiks.plotting_utils import makefig, set_aesthetics, scatter as cscat, plot2dhist as chist2d
from qpam_mlx import QPAMModel
from qpam_mlx.model import cmag, cphase, cmul, cconj

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


def get_text_batch(tokenizer, texts, max_len=512):
    """Tokenize a list of texts, return [B, T] array."""
    all_ids = []
    for t in texts:
        ids = tokenizer.encode(t)[:max_len]
        all_ids.append(ids)
    # Pad to same length
    maxl = max(len(x) for x in all_ids)
    padded = [x + [tokenizer.pad_token_id] * (maxl - len(x)) for x in all_ids]
    return mx.array(padded, dtype=mx.int32)


# ══════════════════════════════════════════════════════════════
# Hook-based activation capture
# ══════════════════════════════════════════════════════════════

def forward_with_hooks(model, tokens):
    """Run forward pass, capture activations at every layer boundary."""
    B, T = tokens.shape
    activations = {}

    # Embedding
    er = model.embed_r[tokens]
    ei = model.embed_i[tokens]
    z = mx.stack([er, ei], axis=-1)
    z = model.input_norm(z)
    mx.eval(z)
    activations['embed'] = np.array(z)

    for i, block in enumerate(model.blocks):
        # Before CGU
        z_pre = z
        z = z + block.alpha_cgu * block.cgu(block.norm1(z))
        mx.eval(z)
        activations[f'post_cgu_{i}'] = np.array(z)

        # Before PAM
        z = z + block.alpha_pam * block.pam(block.norm2(z))
        mx.eval(z)
        activations[f'post_pam_{i}'] = np.array(z)

    z = model.final_proj(z)
    z = model.final_norm(z)
    mx.eval(z)
    activations['final'] = np.array(z)

    return activations


# ══════════════════════════════════════════════════════════════
# 1. Phase evolution through layers
# ══════════════════════════════════════════════════════════════

def plot_phase_evolution(activations, n_layers=16):
    """How does the distribution of activation phases change layer by layer?
    2D hist: layer depth vs phase angle, density-coded."""
    print("\n[1] Phase evolution through layers...")

    all_phases = []
    all_depths = []

    # Embedding
    z = activations['embed']
    phase = np.arctan2(z[..., 1], z[..., 0]).ravel()
    all_phases.append(phase)
    all_depths.append(np.full(len(phase), -0.5))

    for i in range(n_layers):
        z = activations[f'post_pam_{i}']
        phase = np.arctan2(z[..., 1], z[..., 0]).ravel()
        # Subsample for tractability
        idx = np.random.choice(len(phase), size=min(100000, len(phase)), replace=False)
        all_phases.append(phase[idx])
        all_depths.append(np.full(len(idx), float(i)))

    all_phases = np.concatenate(all_phases)
    all_depths = np.concatenate(all_depths)

    fig, ax = makefig()
    chist2d(all_depths, all_phases, nx=17, ny=120, fig=fig, ax=ax, makefigax=False,
            xlabel='Layer', ylabel=r'Activation phase $\theta$',
            aspect='auto', dens_scale=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'act_phase_evolution.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved act_phase_evolution.pdf")


# ══════════════════════════════════════════════════════════════
# 2. Phase coherence: do nearby tokens develop correlated phases?
# ══════════════════════════════════════════════════════════════

def plot_phase_coherence(activations, n_layers=16):
    """For each layer, compute mean |phase_t - phase_{t+1}| across dimension.
    If phase alignment matters for retrieval, consecutive tokens should show
    structured phase relationships, not random ones."""
    print("\n[2] Phase coherence between consecutive tokens...")

    layer_indices = []
    coherences = []

    for i in range(n_layers):
        z = activations[f'post_pam_{i}']  # [B, T, D, 2]
        phase = np.arctan2(z[..., 1], z[..., 0])  # [B, T, D]
        # Phase difference between consecutive tokens
        dphase = phase[:, 1:, :] - phase[:, :-1, :]
        # Wrap to [-pi, pi]
        dphase = (dphase + np.pi) % (2 * np.pi) - np.pi
        # Mean absolute phase difference per (batch, time) pair
        mean_dphi = np.mean(np.abs(dphase), axis=-1).ravel()  # [B*(T-1)]
        # Subsample
        idx = np.random.choice(len(mean_dphi), size=min(5000, len(mean_dphi)), replace=False)
        layer_indices.append(np.full(len(idx), float(i)))
        coherences.append(mean_dphi[idx])

    layer_indices = np.concatenate(layer_indices)
    coherences = np.concatenate(coherences)

    # Random baseline: uniform phases would give mean |dphase| = pi/2
    fig, ax = makefig()
    chist2d(layer_indices, coherences, nx=16, ny=80, fig=fig, ax=ax, makefigax=False,
            xlabel='Layer', ylabel=r'Mean $|\Delta\theta_{t,t+1}|$',
            aspect='auto', dens_scale=0.4,
            bin_y=True, size_y_bin=1.0, counting_thresh=10,
            ybincolsty='r-', linewid=2)
    ax.axhline(y=np.pi/2, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.text(14, np.pi/2 + 0.03, r'$\pi/2$ (random)', fontsize=8, color='gray', ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'act_phase_coherence.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved act_phase_coherence.pdf")


# ══════════════════════════════════════════════════════════════
# 3. Magnitude vs phase: which carries more information?
# ══════════════════════════════════════════════════════════════

def plot_mag_phase_info(activations, n_layers=16):
    """Scatter: per-layer variance of activation magnitudes vs phases.
    If complex computation is doing something, phase variance should be
    structured (not just noise), and its relationship to magnitude
    should change across layers."""
    print("\n[3] Activation magnitude vs phase variance per layer...")

    mag_vars = []
    phase_vars = []
    mag_means = []
    phase_concentrations = []

    for i in range(n_layers):
        z = activations[f'post_pam_{i}']
        mag = np.sqrt(z[..., 0]**2 + z[..., 1]**2 + 1e-8)
        phase = np.arctan2(z[..., 1], z[..., 0])

        mag_vars.append(np.var(mag))
        phase_vars.append(np.var(phase))
        mag_means.append(np.mean(mag))
        # Circular concentration: R = |mean(e^{i*phase})|
        R = np.abs(np.mean(np.exp(1j * phase.ravel())))
        phase_concentrations.append(R)

    layers = np.arange(n_layers, dtype=np.float32)

    # Scatter: mag variance vs phase concentration, colored by layer
    fig, ax = makefig()
    cscat(np.array(mag_vars), np.array(phase_concentrations),
          ccode=layers, cmap='plasma', s=80, edgecolor='k',
          fig=fig, ax=ax, makefigax=False,
          xlabel='Activation magnitude variance',
          ylabel=r'Phase concentration $|\langle e^{i\theta} \rangle|$',
          aspect='auto')
    for i in range(n_layers):
        ax.annotate(f'{i}', (mag_vars[i], phase_concentrations[i]),
                    fontsize=7, ha='center', va='bottom',
                    xytext=(0, 4), textcoords='offset points')
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Layer')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'act_mag_vs_phase_conc.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved act_mag_vs_phase_conc.pdf")

    # Also: magnitude evolution through layers (line + scatter)
    fig, ax = makefig()
    cscat(layers, np.array(mag_means), ccode=layers, cmap='plasma',
          s=80, edgecolor='k', fig=fig, ax=ax, makefigax=False,
          xlabel='Layer', ylabel='Mean activation magnitude', aspect='auto')
    ax.plot(layers, mag_means, '-', color='gray', linewidth=0.8, alpha=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'act_mag_evolution.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved act_mag_evolution.pdf")


# ══════════════════════════════════════════════════════════════
# 4. Interference patterns: constructive vs destructive
# ══════════════════════════════════════════════════════════════

def plot_interference(activations, n_layers=16):
    """The conjugate inner product K* · Q produces interference.
    Look at the real part (constructive/destructive) vs imaginary part
    at each layer. If interference is structured, this should not be
    a symmetric blob."""
    print("\n[4] Interference structure in activations...")

    # Use post-PAM activations at different layers
    # Compare: correlation between real and imaginary parts of z
    for layer_set, label in [([0, 4, 8, 15], 'selected')]:
        fig, axes = plt.subplots(1, len(layer_set), figsize=(3.5 * len(layer_set), 3.5))

        for j, i in enumerate(layer_set):
            z = activations[f'post_pam_{i}']  # [B, T, D, 2]
            zr = z[..., 0].ravel()
            zi = z[..., 1].ravel()
            # Subsample
            idx = np.random.choice(len(zr), size=min(50000, len(zr)), replace=False)

            ax = axes[j] if len(layer_set) > 1 else axes
            chist2d(zr[idx], zi[idx], nx=150, ny=150,
                    fig=fig, ax=ax, makefigax=False,
                    xlabel=r'Re$(z)$' if j == 0 else '',
                    ylabel=r'Im$(z)$' if j == 0 else '',
                    aspect='equal', dens_scale=0.3)
            ax.set_title(f'Layer {i}', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f'act_real_vs_imag.pdf'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved act_real_vs_imag.pdf")


# ══════════════════════════════════════════════════════════════
# 5. Phase rotation per layer: how much does each layer rotate?
# ══════════════════════════════════════════════════════════════

def plot_phase_rotation(activations, n_layers=16):
    """Measure the phase change induced by each block.
    2D hist: dimension index vs phase rotation, for each layer."""
    print("\n[5] Per-layer phase rotation...")

    all_rotations = []
    all_layers = []

    for i in range(n_layers):
        if i == 0:
            z_pre = activations['embed']
        else:
            z_pre = activations[f'post_pam_{i-1}']
        z_post = activations[f'post_pam_{i}']

        phase_pre = np.arctan2(z_pre[..., 1], z_pre[..., 0])
        phase_post = np.arctan2(z_post[..., 1], z_post[..., 0])
        dphase = phase_post - phase_pre
        dphase = (dphase + np.pi) % (2 * np.pi) - np.pi

        # Mean rotation per dimension (averaged over batch and time)
        mean_rot = np.mean(dphase, axis=(0, 1))  # [D]
        all_rotations.append(mean_rot)
        all_layers.append(np.full(len(mean_rot), float(i)))

    all_rotations = np.concatenate(all_rotations)
    all_layers = np.concatenate(all_layers)

    fig, ax = makefig()
    chist2d(all_layers, all_rotations, nx=16, ny=100,
            fig=fig, ax=ax, makefigax=False,
            xlabel='Layer', ylabel=r'Mean phase rotation $\langle\Delta\theta\rangle_d$',
            aspect='auto', dens_scale=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'act_phase_rotation.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved act_phase_rotation.pdf")


# ══════════════════════════════════════════════════════════════
# 6. Token-level: phase divergence predicts next-token difficulty
# ══════════════════════════════════════════════════════════════

def plot_phase_vs_loss(model, tokenizer):
    """For each token in a passage, measure phase spread at the final layer
    and the per-token cross-entropy loss. Scatter: phase spread vs loss.
    If phase encodes uncertainty, high phase spread → high loss."""
    print("\n[6] Phase spread vs per-token loss...")

    text = ("The discovery of the Higgs boson at the Large Hadron Collider "
            "confirmed a central prediction of the Standard Model of particle "
            "physics. The particle, with a mass of approximately 125 GeV, was "
            "observed through its decay channels into pairs of photons, Z bosons, "
            "and W bosons. The measured properties are consistent with the "
            "predictions of the electroweak theory developed by Glashow, Weinberg, "
            "and Salam in the 1960s and 1970s.")

    ids = tokenizer.encode(text)
    tokens = mx.array([ids], dtype=mx.int32)

    # Get final-layer activations
    acts = forward_with_hooks(model, tokens)
    z_final = acts['final']  # [1, T, D, 2]
    phase_final = np.arctan2(z_final[0, :, :, 1], z_final[0, :, :, 0])  # [T, D]

    # Phase spread per token: circular variance
    # R = |mean(e^{i*phase})|, circular variance = 1 - R
    phase_spread = []
    for t in range(phase_final.shape[0]):
        R = np.abs(np.mean(np.exp(1j * phase_final[t])))
        phase_spread.append(1.0 - R)
    phase_spread = np.array(phase_spread)

    # Per-token loss
    logits = model(tokens)
    mx.eval(logits)
    logits_np = np.array(logits[0])  # [T, V]
    log_probs = logits_np - np.log(np.sum(np.exp(logits_np - np.max(logits_np, axis=-1, keepdims=True)), axis=-1, keepdims=True)) - np.max(logits_np, axis=-1, keepdims=True)
    # Properly: log_softmax
    logits_shifted = logits_np - np.max(logits_np, axis=-1, keepdims=True)
    log_probs = logits_shifted - np.log(np.sum(np.exp(logits_shifted), axis=-1, keepdims=True))

    per_token_loss = []
    for t in range(len(ids) - 1):
        target = ids[t + 1]
        per_token_loss.append(-log_probs[t, target])
    per_token_loss = np.array(per_token_loss)

    # Align: phase_spread[:-1] predicts loss for next token
    ps = phase_spread[:-1]

    # Correlation
    r, p = st.pearsonr(ps, per_token_loss)
    print(f"  Phase spread vs loss: r={r:.3f}, p={p:.4f}")

    fig, ax = makefig()
    cscat(ps, per_token_loss, ccode=np.arange(len(ps), dtype=np.float32),
          cmap='plasma', s=30, edgecolor='k',
          fig=fig, ax=ax, makefigax=False,
          xlabel='Phase spread (circular variance)',
          ylabel='Next-token cross-entropy loss',
          aspect='auto')
    # Annotation
    ax.text(0.05, 0.92, f'$r = {r:.3f}$\n$p = {p:.3f}$',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='gray', alpha=0.9))
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Token position')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'act_phase_spread_vs_loss.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved act_phase_spread_vs_loss.pdf")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    checkpoint = os.path.join(LOGDIR, 'checkpoints_qpam_mlx', 'best_model.npz')
    print(f"Loading model from {checkpoint}")
    model = load_model(checkpoint)
    print("  Model loaded")

    tokenizer = get_tokenizer()

    # Get a batch of real text
    texts = [
        "The discovery of the Higgs boson at the Large Hadron Collider confirmed a central prediction of the Standard Model.",
        "In quantum mechanics, the state of a system is described by a wave function that evolves according to the Schrodinger equation.",
        "Language models trained on large corpora learn statistical regularities that enable them to generate coherent text.",
        "The evolution of galaxies is driven by the interplay of gravitational collapse, stellar feedback, and dark matter dynamics.",
    ]

    print("Running forward pass with activation capture...")
    tokens = get_text_batch(tokenizer, texts)
    activations = forward_with_hooks(model, tokens)
    print(f"  Captured activations at {len(activations)} points")

    plot_phase_evolution(activations)
    plot_phase_coherence(activations)
    plot_mag_phase_info(activations)
    plot_interference(activations)
    plot_phase_rotation(activations)
    plot_phase_vs_loss(model, tokenizer)

    print("\n" + "="*60)
    print(f"  All activation figures saved to {OUTDIR}")
    print("="*60)


if __name__ == '__main__':
    main()
