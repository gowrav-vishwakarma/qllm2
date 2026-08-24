"""V13Config + presets (copied from v13/model.py; no framework dependence)."""

import copy
from dataclasses import dataclass


@dataclass
class V13Config:
    vocab_size: int = 50257
    dim: int = 384
    n_heads: int = 6
    head_dim: int = 64
    n_layers: int = 16
    expand: int = 3
    dropout: float = 0.1
    max_seq_len: int = 2048
    use_learned_pos: bool = False
    use_rope: bool = True
    use_gsp: bool = True
    fused_qkv: bool = True
    qk_norm: bool = False
    tie_weights: bool = True
    gradient_checkpointing: bool = True
    activation: str = 'swish'           # 'swish' (7d default) | 'modrelu' | 'phase_mod'
    chunk_size: int = 256

    # ── Memory dynamics (V13 defaults: selective E2+E3+Stage-6) ─────────────
    decay_mode: str = 'head'            # E1: 'head' | 'per_channel'
    write_mode: str = 'delta'           # E2: 'additive' | 'delta' (error-correcting write)
    # E2b: delta_erase_gate splits the delta erase into its own LEARNED gate.
    #   Off (legacy): u = beta * (v_prot - pred)  -- erase fires on EVERY token.
    #   On:          u = beta_w * v_prot - beta_e * pred,  beta_e = sigmoid(erase_beta_proj(|x|))
    #   erase_beta_proj init bias -3.0 (sigmoid -> 0.047): training starts in the
    #   additive regime and the erase (competition) is only learned where it pays.
    delta_erase_gate: bool = False
    n_states: int = 3                   # E3: K superposed states
    delta_chunk: int = 64               # E2 chunk size for the UT transform
    state_dt_spread: float = 2.0        # E3 spread of per-state decay biases
    base_dt_bias: float = -4.0          # uniform decay bias (flat stack)
    gate_content_aware: bool = True     # GSP gate reads real+imag (2*dim) vs magnitude-only
    protect_gate_bias: float = -3.0     # init bias for the GSP write-protect gate
    routing_content_aware: bool = False # E3 phase/score router reads real+imag vs magnitude-only
    state_compete: bool = False         # E3 magnitude competition: c_k = K*softmax(score)*e^{i phi_k}
    phase_init: str = 'zero'            # 'zero' | 'spread' (biases 0,±2π/3) | 'ortho'
    route_balance_lambda: float = 0.0   # MoE-style load balance on batch-mean routing (needs state_compete)
    aux_loss_weight: float = 1.0        # trainer weight for route_balance aux
    fused_e3: bool = True               # E3: fused multistate path (exact-equiv, K-independent matmuls)
    delta_key_norm: bool = True
    delta_erase_beta_cap: float = 0.95
    recompute_pam_chunks: bool = False
    delta_decay_factored: bool = False
    delta_decay_factor_min_a: float = 1e-6

    # ── Recall program (V12): longer memory horizon + gate supervision ───────
    gamma_floor: float = 0.0
    gate_surprisal_lambda: float = 0.1
    gate_surprisal_tau: float = 1.0
    gate_surprisal_sign: float = 1.0
    vault_state: bool = True
    vault_state_idx: int = 0
    write_phase_address: bool = True


def _base_flat(**kw) -> V13Config:
    cfg = V13Config(
        vocab_size=50257, dim=384, n_heads=6, head_dim=64,
        n_layers=16, expand=3, dropout=0.1, max_seq_len=2048,
        activation='swish', chunk_size=256,
    )
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


PRESETS = {
    'v11_baseline': _base_flat(),
    'v11_e1_perchannel': _base_flat(decay_mode='per_channel'),
    'v11_e2_delta': _base_flat(write_mode='delta', delta_chunk=64),
    'v11_e3_multistate': _base_flat(n_states=2, state_dt_spread=2.0),
    'v11_e3_k3': _base_flat(n_states=3, state_dt_spread=2.0, gate_content_aware=True),
    'v11_e3_k3_chat': _base_flat(
        n_states=3, state_dt_spread=2.0, vocab_size=50261, gate_content_aware=True,
    ),
    'v11_e3_k3_chat_gate': _base_flat(
        n_states=3, state_dt_spread=2.0, vocab_size=50261, gate_content_aware=True,
    ),
    'v11_e3_k3_chat_recall': _base_flat(
        n_states=3, state_dt_spread=2.0, vocab_size=50261, gate_content_aware=True,
        gamma_floor=0.98, gate_surprisal_lambda=0.1, gate_surprisal_tau=1.0,
        gate_surprisal_sign=1.0,
    ),
    'v11_e3_k3_chat_compete': _base_flat(
        n_states=3, state_dt_spread=2.0, vocab_size=50261, gate_content_aware=True,
        routing_content_aware=True, state_compete=True, phase_init='spread',
        route_balance_lambda=0.01,
    ),
    'v11_e1e3_combo': _base_flat(decay_mode='per_channel', n_states=2, state_dt_spread=2.0),
    'tiny': V13Config(
        vocab_size=50257, dim=64, n_heads=2, head_dim=32, n_layers=2,
        expand=2, dropout=0.0, max_seq_len=512, chunk_size=64,
        gradient_checkpointing=False,
    ),
    'tiny_e1': V13Config(
        vocab_size=50257, dim=64, n_heads=2, head_dim=32, n_layers=2,
        expand=2, dropout=0.0, max_seq_len=512, chunk_size=64,
        gradient_checkpointing=False, decay_mode='per_channel',
    ),
    'tiny_e2': V13Config(
        vocab_size=50257, dim=64, n_heads=2, head_dim=32, n_layers=2,
        expand=2, dropout=0.0, max_seq_len=512, chunk_size=64,
        gradient_checkpointing=False, write_mode='delta', delta_chunk=32,
    ),
    'tiny_e3': V13Config(
        vocab_size=50257, dim=64, n_heads=2, head_dim=32, n_layers=2,
        expand=2, dropout=0.0, max_seq_len=512, chunk_size=64,
        gradient_checkpointing=False, n_states=2,
    ),
    'tiny_e1e3': V13Config(
        vocab_size=50257, dim=64, n_heads=2, head_dim=32, n_layers=2,
        expand=2, dropout=0.0, max_seq_len=512, chunk_size=64,
        gradient_checkpointing=False, decay_mode='per_channel', n_states=2,
    ),
    'v11_micro_10m': V13Config(
        vocab_size=50261, dim=96, n_heads=3, head_dim=32, n_layers=6,
        expand=3, dropout=0.0, max_seq_len=2048, chunk_size=64,
        gradient_checkpointing=False, n_states=3, state_dt_spread=2.0,
        gate_content_aware=True,
    ),
    'v11_micro_10m_delta': V13Config(
        vocab_size=50261, dim=96, n_heads=3, head_dim=32, n_layers=6,
        expand=3, dropout=0.0, max_seq_len=2048, chunk_size=64,
        gradient_checkpointing=False, n_states=1, write_mode='delta',
        delta_chunk=32, gate_content_aware=True,
    ),
    'v11_micro_10m_vault': V13Config(
        vocab_size=50261, dim=96, n_heads=3, head_dim=32, n_layers=6,
        expand=3, dropout=0.0, max_seq_len=2048, chunk_size=64,
        gradient_checkpointing=False, n_states=3, state_dt_spread=2.0,
        gate_content_aware=True, vault_state=True, vault_state_idx=0,
    ),
    'v11_micro_10m_phase': V13Config(
        vocab_size=50261, dim=96, n_heads=3, head_dim=32, n_layers=6,
        expand=3, dropout=0.0, max_seq_len=2048, chunk_size=64,
        gradient_checkpointing=False, n_states=3, state_dt_spread=2.0,
        gate_content_aware=True, write_phase_address=True,
    ),
    'v13_e3_k3_selective': _base_flat(
        n_states=3, state_dt_spread=2.0, vocab_size=50261,
        write_mode='delta', delta_chunk=128,
        delta_erase_gate=True,
        gate_content_aware=True, vault_state=True, vault_state_idx=0,
        write_phase_address=True, gate_surprisal_lambda=0.1,
        fused_e3=True,
    ),
    'v13_micro_10m_recall': V13Config(
        vocab_size=50261, dim=96, n_heads=3, head_dim=32, n_layers=6,
        expand=3, dropout=0.0, max_seq_len=2048, chunk_size=64,
        gradient_checkpointing=False, n_states=3, state_dt_spread=2.0,
        write_mode='delta', delta_chunk=32,
        delta_erase_gate=True,
        gate_content_aware=True, vault_state=True, vault_state_idx=0,
        write_phase_address=True, gate_surprisal_lambda=0.1,
        fused_e3=False,
    ),
}


def get_config(preset: str = 'v13_e3_k3_selective') -> V13Config:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}")
    return copy.deepcopy(PRESETS[preset])
