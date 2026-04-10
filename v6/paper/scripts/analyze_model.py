"""
Comprehensive analysis of trained QPAM model.

Usage:
    cd /Users/caug/npcww/qnlp/qllm-private
    uv run python v6/paper/scripts/analyze_model.py
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

from chroptiks.plotting_utils import makefig, set_aesthetics, scatter as cscat, plot2dhist as chist2d
from qpam_mlx import QPAMModel

OUTDIR = '/Users/caug/npcww/qnlp/qllm-private/v6/paper/figures'
LOGDIR = '/Users/caug/npcww/qnlp/ket-nlp'
os.makedirs(OUTDIR, exist_ok=True)

C_COMPLEX = '#2166ac'
C_REAL = '#762a83'
C_TRANS = '#b2182b'
C_GRAY = '#636363'
C_GREEN = '#1b7837'


def load_model(checkpoint_path):
    model = QPAMModel(
        vocab_size=50257, dim=384, num_layers=16,
        expand=3, num_heads=6, head_dim=64,
    )
    model.load_weights(checkpoint_path)
    mx.eval(model.parameters())
    return model


# ══════════════════════════════════════════════════════════════
# 1. Phase vs magnitude — 2D density plot per layer
# ══════════════════════════════════════════════════════════════

def analyze_phase_distribution(model):
    """2D histogram: magnitude vs phase for all complex weights, color-coded by layer."""
    print("\n[1/6] Phase vs magnitude 2D density...")

    n_layers = len(model.blocks)

    # Collect all weights with layer index
    all_mags = []
    all_phases = []
    all_layers = []

    for i, block in enumerate(model.blocks):
        for module in [block.pam.qkv, block.pam.out_proj,
                       block.cgu.up, block.cgu.gate, block.cgu.down]:
            Wr = np.array(module.Wr).ravel()
            Wi = np.array(module.Wi).ravel()
            mag = np.sqrt(Wr**2 + Wi**2)
            phase = np.arctan2(Wi, Wr)
            all_mags.append(mag)
            all_phases.append(phase)
            all_layers.append(np.full(len(mag), i, dtype=np.float32))

    all_mags = np.concatenate(all_mags)
    all_phases = np.concatenate(all_phases)
    all_layers = np.concatenate(all_layers)

    # 2D hist: phase vs magnitude (gray_r density)
    fig, ax = makefig()
    chist2d(all_phases, all_mags, nx=200, ny=200, fig=fig, ax=ax, makefigax=False,
            xlabel=r'Phase angle $\theta$ (rad)',
            ylabel=r'Magnitude $|w|$',
            aspect='auto', dens_scale=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'phase_vs_mag_density.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved phase_vs_mag_density.pdf")

    # Scatter: phase vs magnitude, color-coded by layer
    fig, ax = makefig()
    # Subsample for readability
    idx = np.random.choice(len(all_phases), size=min(50000, len(all_phases)), replace=False)
    cscat(all_phases[idx], all_mags[idx], ccode=all_layers[idx],
          cmap='plasma', s=2, alpha=0.3, fig=fig, ax=ax, makefigax=False,
          xlabel=r'Phase angle $\theta$ (rad)',
          ylabel=r'Magnitude $|w|$',
          aspect='auto')
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Layer index')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'phase_vs_mag_by_layer.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved phase_vs_mag_by_layer.pdf")

    # 2D hist: layer vs phase, color = density
    fig, ax = makefig()
    chist2d(all_layers, all_phases, nx=16, ny=100, fig=fig, ax=ax, makefigax=False,
            xlabel='Layer', ylabel=r'Phase angle $\theta$ (rad)',
            aspect='auto', dens_scale=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'layer_vs_phase_density.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved layer_vs_phase_density.pdf")


# ══════════════════════════════════════════════════════════════
# 2. Magnitude vs phase variance per layer — scatter
# ══════════════════════════════════════════════════════════════

def analyze_mag_vs_phase(model):
    """Scatter: magnitude variance vs phase variance per layer, color-coded by layer."""
    print("\n[2/6] Magnitude vs phase variance...")

    layers = []
    mag_vars = []
    phase_vars = []

    for i, block in enumerate(model.blocks):
        all_Wr, all_Wi = [], []
        for module in [block.pam.qkv, block.pam.out_proj,
                       block.cgu.up, block.cgu.gate, block.cgu.down]:
            all_Wr.append(np.array(module.Wr).ravel())
            all_Wi.append(np.array(module.Wi).ravel())

        Wr = np.concatenate(all_Wr)
        Wi = np.concatenate(all_Wi)
        mag = np.sqrt(Wr**2 + Wi**2)
        phase = np.arctan2(Wi, Wr)

        layers.append(float(i))
        mag_vars.append(np.var(mag))
        phase_vars.append(np.var(phase))

    layers = np.array(layers)
    mag_vars = np.array(mag_vars)
    phase_vars = np.array(phase_vars)

    fig, ax = makefig()
    cscat(mag_vars, phase_vars, ccode=layers, cmap='plasma',
          s=80, edgecolor='k', fig=fig, ax=ax, makefigax=False,
          xlabel='Magnitude variance', ylabel='Phase variance',
          aspect='auto')
    # Annotate each point with layer number
    for i in range(len(layers)):
        ax.annotate(f'{int(layers[i])}', (mag_vars[i], phase_vars[i]),
                    fontsize=7, ha='center', va='bottom',
                    xytext=(0, 4), textcoords='offset points')
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Layer')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'mag_vs_phase_variance.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved mag_vs_phase_variance.pdf")


# ══════════════════════════════════════════════════════════════
# 3. SVD spectrum — scatter, color-coded by layer
# ══════════════════════════════════════════════════════════════

def analyze_svd_spectrum(model):
    """SVD of complex QKV matrices. Scatter: SV index vs normalized SV, colored by layer."""
    print("\n[3/6] Singular value spectrum...")

    n_layers = len(model.blocks)
    all_sv_idx = []
    all_sv_val = []
    all_sv_layer = []
    effective_ranks = []

    for i, block in enumerate(model.blocks):
        Wr = np.array(block.pam.qkv.Wr)
        Wi = np.array(block.pam.qkv.Wi)
        W = Wr + 1j * Wi
        sv = np.linalg.svd(W, compute_uv=False)
        sv = sv / sv[0]

        all_sv_idx.append(np.arange(len(sv), dtype=np.float32))
        all_sv_val.append(sv)
        all_sv_layer.append(np.full(len(sv), float(i)))

        sv_norm = sv / sv.sum()
        eff_rank = np.exp(-np.sum(sv_norm[sv_norm > 1e-10] * np.log(sv_norm[sv_norm > 1e-10])))
        effective_ranks.append(eff_rank)

    all_sv_idx = np.concatenate(all_sv_idx)
    all_sv_val = np.concatenate(all_sv_val)
    all_sv_layer = np.concatenate(all_sv_layer)

    fig, ax = makefig()
    cscat(all_sv_idx, all_sv_val, ccode=all_sv_layer, cmap='plasma',
          s=2, alpha=0.5, fig=fig, ax=ax, makefigax=False,
          xlabel='Singular value index',
          ylabel='Normalized singular value',
          aspect='auto')
    ax.set_yscale('log')
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Layer')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'svd_spectrum.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved svd_spectrum.pdf")

    # Effective rank vs layer — scatter
    fig, ax = makefig()
    cscat(np.arange(n_layers, dtype=np.float32), np.array(effective_ranks),
          ccode=np.arange(n_layers, dtype=np.float32), cmap='plasma',
          s=80, edgecolor='k', fig=fig, ax=ax, makefigax=False,
          xlabel='Layer', ylabel='Effective rank', aspect='auto')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'effective_rank.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved effective_rank.pdf")
    print(f"  Effective ranks: {[f'{r:.1f}' for r in effective_ranks]}")


# ══════════════════════════════════════════════════════════════
# 4. Dense validation PPL curve
# ══════════════════════════════════════════════════════════════

def plot_dense_val_ppl():
    """Plot all mid-epoch val PPL checkpoints from training log."""
    print("\n[4/6] Dense validation PPL curve...")

    log_path = os.path.join(LOGDIR, 'qpam_mlx_train_v2.log')
    if not os.path.exists(log_path):
        print("  No training log found, skipping.")
        return

    val_ppls = []
    step = 0

    with open(log_path) as f:
        for line in f:
            if '** Val loss=' in line and 'ppl=' in line:
                ppl = float(line.split('ppl=')[1].strip())
                val_ppls.append(ppl)
                step += 1

    val_ppls = np.array(val_ppls)
    val_steps = np.arange(len(val_ppls), dtype=np.float32)

    # 2D hist style: step vs PPL as density
    fig, ax = makefig()
    ax.plot(val_steps, val_ppls, '-', color=C_COMPLEX, linewidth=0.6, alpha=0.8)
    set_aesthetics(fig=fig, ax=ax, makefigax=False,
                   xlabel='Validation checkpoint index',
                   ylabel='Validation perplexity',
                   ylim=[30, 120])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'val_ppl_dense.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved val_ppl_dense.pdf")
    print(f"  Total val checkpoints: {len(val_ppls)}, best: {val_ppls.min():.2f}")


# ══════════════════════════════════════════════════════════════
# 5. Generation quality — scatter: temperature vs metrics
# ══════════════════════════════════════════════════════════════

def analyze_generation(model):
    """Generate text at multiple temperatures, scatter plot quality metrics."""
    print("\n[5/6] Generation quality analysis...")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    prompt = "In 1923, the University of"
    prompt_ids = tokenizer.encode(prompt)
    temperatures = [0.3, 0.5, 0.7, 1.0, 1.3]
    max_new = 100

    results = {}
    for temp in temperatures:
        tokens = list(prompt_ids)
        for _ in range(max_new):
            x = mx.array([tokens[-512:]], dtype=mx.int32)
            logits = model(x)
            mx.eval(logits)
            next_logits = np.array(logits[0, -1, :])
            next_logits = next_logits / temp
            probs = np.exp(next_logits - np.max(next_logits))
            probs = probs / probs.sum()
            next_token = np.random.choice(len(probs), p=probs)
            tokens.append(int(next_token))

        gen_tokens = tokens[len(prompt_ids):]

        def ngram_rep(toks, n):
            ngrams = [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
            if len(ngrams) == 0:
                return 0
            return 1 - len(set(ngrams)) / len(ngrams)

        results[temp] = {
            'rep_2': ngram_rep(gen_tokens, 2),
            'rep_3': ngram_rep(gen_tokens, 3),
            'unique_ratio': len(set(gen_tokens)) / len(gen_tokens),
        }
        print(f"  T={temp}: rep2={results[temp]['rep_2']:.3f} "
              f"rep3={results[temp]['rep_3']:.3f} "
              f"unique={results[temp]['unique_ratio']:.3f}")

    temps = np.array(sorted(results.keys()))
    rep2s = np.array([results[t]['rep_2'] for t in temps])
    rep3s = np.array([results[t]['rep_3'] for t in temps])
    uniques = np.array([results[t]['unique_ratio'] for t in temps])

    fig, ax = makefig()
    cscat(temps, rep2s, color=C_COMPLEX, s=60, edgecolor='k', marker='o',
          fig=fig, ax=ax, makefigax=False, label='2-gram rep.', aspect='auto')
    cscat(temps, rep3s, color=C_GREEN, s=60, edgecolor='k', marker='s',
          fig=fig, ax=ax, makefigax=False, label='3-gram rep.', aspect='auto')
    cscat(temps, uniques, color=C_TRANS, s=60, edgecolor='k', marker='^',
          fig=fig, ax=ax, makefigax=False, label='Unique ratio', aspect='auto')
    # Connect with lines
    ax.plot(temps, rep2s, '-', color=C_COMPLEX, linewidth=1.0, alpha=0.5)
    ax.plot(temps, rep3s, '-', color=C_GREEN, linewidth=1.0, alpha=0.5)
    ax.plot(temps, uniques, '-', color=C_TRANS, linewidth=1.0, alpha=0.5)
    ax.legend(fontsize=8, frameon=False, loc='upper center')
    set_aesthetics(fig=fig, ax=ax, makefigax=False,
                   xlabel='Temperature', ylabel='Rate',
                   ylim=[0, 1.0])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'generation_quality.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved generation_quality.pdf")


# ══════════════════════════════════════════════════════════════
# 6. Inference profiling — scatter
# ══════════════════════════════════════════════════════════════

def profile_inference(model):
    """Measure tokens/sec at different sequence lengths."""
    print("\n[6/6] Inference profiling...")

    seq_lengths = [64, 128, 256, 512]
    n_warmup = 3
    n_trials = 10

    sls, tps = [], []
    for sl in seq_lengths:
        tokens = mx.random.randint(0, 50257, (1, sl))
        for _ in range(n_warmup):
            out = model(tokens)
            mx.eval(out)
        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            out = model(tokens)
            mx.eval(out)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        mean_t = np.mean(times)
        tok_per_sec = sl / mean_t
        sls.append(float(sl))
        tps.append(tok_per_sec)
        print(f"  seq_len={sl:4d}: {mean_t*1000:.1f}ms ({tok_per_sec:.0f} tok/s)")

    fig, ax = makefig()
    cscat(np.array(sls), np.array(tps), color=C_COMPLEX, s=80, edgecolor='k',
          fig=fig, ax=ax, makefigax=False,
          xlabel='Sequence length', ylabel='Tokens / second', aspect='auto')
    ax.plot(sls, tps, '-', color=C_COMPLEX, linewidth=1.2, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'inference_throughput.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved inference_throughput.pdf")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    checkpoint = os.path.join(LOGDIR, 'checkpoints_qpam_mlx', 'best_model.npz')
    print(f"Loading model from {checkpoint}")
    model = load_model(checkpoint)
    print(f"  Model loaded (119.5M params)")

    analyze_phase_distribution(model)
    analyze_mag_vs_phase(model)
    analyze_svd_spectrum(model)
    plot_dense_val_ppl()
    analyze_generation(model)
    profile_inference(model)

    print("\n" + "="*60)
    print(f"  All figures saved to {OUTDIR}")
    print("="*60)


if __name__ == '__main__':
    main()
