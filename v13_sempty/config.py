"""PAMConfig + presets for the simple PAM.

Only the knobs the model actually has. Everything v13 learned around the
recurrence (E1/E2/E3, GSP, vault, gate supervision) is gone by design —
see EXPERIMENTS_SEMPY.md for the decision log.
"""

import copy
from dataclasses import dataclass


@dataclass
class PAMConfig:
    vocab_size: int = 50257
    dim: int = 384
    n_heads: int = 6
    head_dim: int = 64
    n_layers: int = 16
    expand: int = 3
    dropout: float = 0.1
    max_seq_len: int = 2048
    use_rope: bool = True
    tie_weights: bool = True
    gradient_checkpointing: bool = True
    activation: str = 'swish'        # 'swish' (v7/v11 default) | 'modrelu' | 'phase_mod'
    chunk_size: int = 256            # notebook carry window (train / prefill)
    base_dt_bias: float = -4.0       # decay bias: init decay ~= e^-4 ~= 0.018
    # (R1 `dt_bias_spread` -- per-head ladder of initial decay biases -- was
    # removed 2026-09-05 after the L1 run: no horizon gain, 128-ctx recall
    # worse, biases frozen at init. See EXPERIMENTS_SEMPY.md "L1 / R1".)
    # R3 (2026-09-08): a SPLIT prior instead of a ladder -- `long_heads` heads
    # start at `long_dt_bias` (-9 => softplus 1.2e-4 => ~0.37 of a binding
    # survives 8192 tokens of passive decay), the rest keep `base_dt_bias`.
    # Motivation: L-2 (8K ctx, 8 % recall_long docs with <=6k-token gaps, 2B
    # tok) left every dt_bias at -4.00 +/- 0.1 -- data pressure does NOT move
    # the decay prior, so the horizon has to be given structurally.
    long_heads: int = 0
    long_dt_bias: float = -9.0
    is_complex: bool = True          # SplitComplex (phase) | fully-real PAM

    # ── architecture ladder (real arm; EXPERIMENTS_SEMPY "Architecture ladder")
    # Each is off by default so 'baseline_real_pm' stays the control. A rung
    # that fails its decision gate is removed from the code (AGENTS rule).
    # A1 short conv: REMOVED (fair rung 23.49 vs 23.14 chrono ref, -17% tok/s;
    # EXPERIMENTS_SEMPY "A1 short conv").
    n_states: int = 1                # A2: independent PAM states per head (E3)
    state_dt_spread: float = 2.0     # A2: +/- spread of per-state decay-logit offsets
    vault: bool = False              # A2b: state 0 pinned (retention=1) + protect gate
    delta: bool = False              # A3: delta erase/write (unit keys, beta_w/beta_e)
    # (R2 `no_decay` -- retention pinned to 1.0, delta erase-only memory -- was
    # REMOVED 2026-09-07: at 100M/3B tok holdout PPL 30.16 vs 26.38 and recall
    # WORSE at every length (state overflow: the fresh-binding read decays with
    # context). Passive decay is load-bearing. See EXPERIMENTS_SEMPY "R2".)
    # (A2r content-routed delta -- route writes by key / reads by query to S
    # states -- was REMOVED 2026-09-05: a8 stayed at chance at 4-8x cost even
    # with sharpened routing; multi-way is not a memory-structure problem.
    # See EXPERIMENTS_SEMPY "A2r content-routed delta".)
    cond_mem: bool = False           # A4: Engram-style hashed n-gram conditional memory
    cond_mem_slots: int = 1 << 18    # table rows per (order, head)
    cond_mem_dim: int = 64           # table row width
    cond_mem_layers: tuple = (0, 1)  # blocks after which to add the memory read
    # N1 Chrono-PAM: content-modulated rotary retention (novel). A per-head
    # learned time-warp g_t=exp(clamp(W x)) scales the per-step RoPE angle;
    # cumulative phase = cumsum_t(inv_freq * g_t). Zero-init W => g=1 => this is
    # EXACTLY standard RoPE (bit-parity), so it is a safe drop-in on the 23.81
    # baseline. Equivalent to a complex rotating retention folded into q,k, so
    # the fused kernel is untouched (speed preserved). Decode carries
    # (notebook, clock). Real arm only. RUNG DONE: 23.14 vs 23.81 (KEEP).
    chrono: bool = False
    # N4 content-dependent read-out gate (real arm). The block's static
    # `pam_scale` is the only thing deciding how much of the memory read
    # reaches the residual, and it stays small (0.11-0.31 after 10 ep). N4
    # makes that per token, per head: read_h <- read_h * silu(W_g x + b_g),
    # W_g zero-init and b_g = 1.2785 (silu(b_g) = 1) => identity at start,
    # bit-parity with the chrono reference. dim -> n_heads params (~3.5k per
    # layer), one broadcast multiply: elementwise, kernel untouched.
    out_gate: bool = False


def _base_flat(**kw) -> PAMConfig:
    cfg = PAMConfig(
        vocab_size=50257, dim=384, n_heads=6, head_dim=64,
        n_layers=16, expand=3, dropout=0.1, max_seq_len=2048,
        activation='swish', chunk_size=256,
    )
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


PRESETS = {
    # The production shape: v11 hero's bare minimum (7d baseline geometry).
    'baseline': _base_flat(),
    # The fully-real twin of baseline at matched real width: 768 real channels
    # (== 384 complex), 128-dim real heads. Same recurrence, real arithmetic.
    'baseline_real': _base_flat(dim=768, head_dim=128, is_complex=False),
    # Param-matched real control: real dim 588 (== 98 x 6 heads, even
    # head_dim required by real RoPE) -> 101.89M params, within +1.5% of
    # 'baseline' (100.36M). Unlike 'baseline_real' (dim 768, matched real
    # *width*, 161.97M params), this isolates the complex-vs-real arithmetic
    # question from the model-size confound, since a complex-linear map is
    # ~2x more parameter-efficient than a real one at the same real channel
    # count.
    'baseline_real_pm': _base_flat(dim=588, head_dim=98, is_complex=False),
    # Small enough to run on CPU in a minute.
    'micro': PAMConfig(
        vocab_size=50261, dim=96, n_heads=3, head_dim=32, n_layers=6,
        expand=3, dropout=0.0, max_seq_len=2048, chunk_size=64,
        gradient_checkpointing=False,
    ),
    # The dev preset: tiny, fast, for selftest / smoke runs.
    'tiny': PAMConfig(
        vocab_size=50257, dim=64, n_heads=2, head_dim=32, n_layers=2,
        expand=2, dropout=0.0, max_seq_len=512, chunk_size=64,
        gradient_checkpointing=False,
    ),
    # The dev twin of tiny: fully-real, CPU-friendly selftest preset.
    'tiny_real': PAMConfig(
        vocab_size=50257, dim=128, n_heads=2, head_dim=64, n_layers=2,
        expand=2, dropout=0.0, max_seq_len=512, chunk_size=64,
        gradient_checkpointing=False, is_complex=False,
    ),
}


def get_config(preset: str = 'baseline') -> PAMConfig:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}")
    return copy.deepcopy(PRESETS[preset])
