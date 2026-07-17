"""
V12 model: leaner PAM core (V11 proven winners) + learnable phase-band heads.

Reuses the stable V7/V11 complex primitives (ComplexLinear, ComplexNorm, CGU,
ModSwish/ModReLU, ComplexEmbed, RoPE) via vendored `v12.complex_ops`. Carried
forward from V11: E3 K=3 multistate phase-interference retrieval, phase-aware
GSP write gate, fused E3 + chunked CE, RoPE, tied complex LM head.

`V12PAMLayer` keeps the proven dynamics:
    write_mode : 'additive'    -> S += V (x) K*           (V11 winner)
                 'delta'        -> error-correcting write  (kept as M3 starting point)
    n_states   : 1             -> single matrix state      (control)
                 K>1           -> superposed states, phase-routed retrieval (E3)

Novel V12 work: learnable phase-band heads (M1), a progressive frozen-head-growth
curriculum (M2), low-interference fact writes (M3), and a spec-driven depth-growth
framework (M4) — non-uniform stacks built from ``V12Config.layer_specs`` with
grow/freeze/manifest APIs and attach-mode-aware composition (sequential now, MoE
router reserved). See ``V12LM.grow_layers`` / ``layer_manifest`` and v12/{losses,
compact,pack}.py.

All paths expose a parallel training form and an O(1) recurrent inference form,
numerically verified to agree (see v12/selftest.py).

Complex representation: split-real `[..., dim, 2]`. Never torch.complex64/128.
"""

import math
import copy
import hashlib
from dataclasses import dataclass, field, fields
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

# Vendored V7 complex primitives — V12 no longer imports v7.model on the forward path.
from v12.complex_ops import (
    real_part, imag_part, stack_complex, scale_complex, as_complex_dropout_mask,
    cmul, cconj, cabs, cnormalize, to_real_concat,
    ComplexLinear, ComplexNorm, ComplexEmbed, ComplexPosEmbed,
    ComplexGatedUnit, build_rope_cache, _build_activation,
)
from v12.triton_kernels import fused_decay_matrix


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class V12Config:
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

    # ── New memory dynamics (defaults == V7 7d) ──────────────────────────────
    decay_mode: str = 'head'            # E1: 'head' | 'per_channel'
    write_mode: str = 'additive'        # E2: 'additive' | 'delta'
    n_states: int = 1                   # E3: K superposed states (1 == baseline)
    delta_chunk: int = 64               # E2 chunk size for the UT transform
    state_dt_spread: float = 2.0        # E3 spread of per-state decay biases
    base_dt_bias: float = -4.0          # uniform decay bias (flat stack)
    gate_content_aware: bool = False    # GSP gate reads real+imag (2*dim) vs magnitude-only
    protect_gate_bias: float = -3.0     # init bias for the GSP write-protect gate
    routing_content_aware: bool = False # E3 phase/score router reads real+imag vs magnitude-only
    state_compete: bool = False         # E3 magnitude competition: c_k = K*softmax(score)*e^{i phi_k}
    phase_init: str = 'zero'            # 'zero' | 'spread' (biases 0,±2π/3) | 'ortho'
    route_balance_lambda: float = 0.0   # MoE-style load balance on batch-mean routing (needs state_compete)
    aux_loss_weight: float = 1.0        # trainer weight for route_balance aux (v7.train hook)
    fused_e3: bool = True               # E3: fused multistate path (exact-equiv, K-independent matmuls)
    recompute_pam_chunks: bool = False  # recompute per-chunk W/D/A in backward (exact; less VRAM, more FLOPs)

    # ── Recall program (V12): longer memory horizon + gate supervision ───────
    # gamma_floor: minimum per-step decay applied to the *base* (pre-GSP) decay.
    #   Reparam: base_decay = gamma_floor + (1-gamma_floor)*exp(-softplus_dt).
    #   0.0 disables (identical to old behaviour). ~0.98 keeps state ~50x longer
    #   before the GSP protect blend, attacking the ~1-2K token recall cliff.
    gamma_floor: float = 0.0
    # gate_surprisal_lambda: weight of the self-supervised gate-selectivity loss
    #   (0 disables). Ties the GSP write-protect prob to per-token surprisal so the
    #   gate learns *when* to write vs freeze instead of a flat ~0.4 on every token.
    gate_surprisal_lambda: float = 0.0
    gate_surprisal_tau: float = 1.0     # temperature (nats) mapping surprisal->target protect prob
    # gate_surprisal_sign: +1 => LOW-surprisal (filler) tokens get HIGH protect target
    #   (freeze state through filler, write on content). This is the recall-oriented
    #   direction and drives (p_content - p_filler) NEGATIVE. -1 flips it to the
    #   probe's "protect content more" convention. Default +1 optimizes for recall.
    gate_surprisal_sign: float = 1.0
    # Stage-6 architecture levers (defaults OFF = bit-identical to prior behaviour).
    # vault_state: pin one of the K states to γ≈1 (no decay); writes still GSP-gated.
    vault_state: bool = False
    vault_state_idx: int = 0
    # write_phase_address: key-conditioned write phase + matching query phase on read.
    write_phase_address: bool = False

    # ── M1: learnable phase-band heads ───────────────────────────────────────
    # head_gate: treat `n_heads` as a MAX head budget (H_max). Each head slot gets
    #   a hard-concrete L0 gate applied at the output merge, so the *effective*
    #   number of heads is learned (unused slots prune to exactly 0). Default OFF
    #   is bit-identical to V11. Kernel-safe: shapes stay [B, n_heads, T, d, 2];
    #   a gated-off head simply contributes 0 to the residual stream.
    head_gate: bool = False
    # head_gate_l0_lambda: weight of the expected-L0 sparsity penalty (0 => no
    #   pressure, all slots stay open). Routed through the trainer aux-loss hook.
    head_gate_l0_lambda: float = 0.0
    # head_gate_init_logalpha: initial gate logit. ~3.0 opens every slot at init
    #   (so training starts using all H_max heads, then L0 prunes down).
    head_gate_init_logalpha: float = 3.0

    # ── M4: depth-growth framework (spec-driven per-layer construction) ───────
    # layer_specs: when set, V12LM builds ONE block per entry, so the stack can be
    #   non-uniform (the progressive-growth curriculum grows specialist layer groups
    #   on top of a frozen grammar base). Each entry is a plain dict of:
    #     - structural overrides (any per-layer V12Config field, e.g. n_heads,
    #       head_dim, n_states, write_mode, vault_state, write_phase_address, ...);
    #       omitted fields inherit the top-level cfg.
    #     - provenance / curriculum keys (NOT config fields): 'skill', 'group_id',
    #       'stage', 'frozen', 'substrate_hash', 'attach_mode'.
    #   attach_mode is 'sequential' (default; always-on depth) or 'moe' (reserved;
    #   pack-time routing hook, currently runs sequentially — see _apply_moe_group).
    #   When None the stack is uniform (n_layers x top-level cfg), bit-identical to
    #   the pre-M4 build. V12LM materializes an explicit manifest here after init.
    layer_specs: Optional[List[dict]] = None


# ── M4: per-layer spec handling for the depth-growth framework ──────────────

# Provenance / curriculum keys that live in a layer_spec but are NOT V12Config
# fields (so _layer_cfg ignores them when cloning the per-layer config).
_SPEC_PROVENANCE_KEYS = frozenset(
    {'skill', 'group_id', 'stage', 'frozen', 'substrate_hash', 'attach_mode', 'layer_idx'}
)
# Config fields that are GLOBAL to the model (shared residual stream / vocab /
# stack length) and therefore must not be overridden per layer.
_SPEC_GLOBAL_FIELDS = frozenset(
    {'vocab_size', 'dim', 'n_layers', 'max_seq_len', 'use_learned_pos',
     'tie_weights', 'gradient_checkpointing', 'layer_specs'}
)


def _layer_cfg(base_cfg: 'V12Config', spec: dict) -> 'V12Config':
    """Clone ``base_cfg`` and apply the structural overrides from ``spec``.

    Provenance keys are ignored; global fields cannot be overridden; the per-layer
    cfg carries ``layer_specs=None`` so V12Block builds a single uniform block.
    """
    lc = copy.deepcopy(base_cfg)
    lc.layer_specs = None
    valid = {f.name for f in fields(V12Config)}
    for key, value in spec.items():
        if key in _SPEC_PROVENANCE_KEYS:
            continue
        if key not in valid:
            raise ValueError(
                f"layer_spec key '{key}' is neither a V12Config field nor a "
                f"provenance key {sorted(_SPEC_PROVENANCE_KEYS)}"
            )
        if key in _SPEC_GLOBAL_FIELDS:
            raise ValueError(
                f"layer_spec cannot override global field '{key}' "
                f"(shared across the stack): {sorted(_SPEC_GLOBAL_FIELDS)}"
            )
        setattr(lc, key, value)
    return lc


def _normalize_spec(spec: dict) -> dict:
    """Return a copy of ``spec`` with the standard provenance defaults filled in."""
    s = dict(spec)
    s.setdefault('attach_mode', 'sequential')
    s.setdefault('frozen', False)
    s.setdefault('group_id', s.get('skill'))
    s.setdefault('stage', 0)
    s.pop('layer_idx', None)  # positional, never stored in the spec itself
    return s


# ── M1: learnable head count via hard-concrete L0 gates ─────────────────────

class HardConcreteGate(nn.Module):
    """Per-head hard-concrete gate for a learnable/prunable number of heads.

    Based on Louizos, Welling & Kingma, "Learning Sparse Neural Networks through
    L0 Regularization" (2018). Each of ``num_gates`` head slots owns a logit
    ``log_alpha``; the gate stretches a sigmoid through (gamma, zeta) so it can
    hit *exactly* 0 (prune the head) or 1 (keep it). The expected-L0 penalty
    ``num_active`` is added to the training loss so unused slots are driven shut.

    We use the *deterministic* (noise-free) hard-concrete so the gate value is
    stable under gradient checkpointing (production uses it) while still being
    differentiable and able to reach 0/1. Effective head count therefore emerges
    from data + the L0 pressure rather than being hand-set.
    """

    beta: float = 2.0 / 3.0
    gamma: float = -0.1
    zeta: float = 1.1

    def __init__(self, num_gates: int, init_logalpha: float = 3.0):
        super().__init__()
        self.num_gates = num_gates
        self.log_alpha = nn.Parameter(torch.full((num_gates,), float(init_logalpha)))

    def _z(self) -> torch.Tensor:
        """Deterministic stretched-sigmoid gate in [0, 1], shape [num_gates]."""
        s = torch.sigmoid(self.log_alpha)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        return s_bar.clamp(0.0, 1.0)

    def gate(self) -> torch.Tensor:
        return self._z()

    def num_active(self) -> torch.Tensor:
        """Differentiable expected number of open gates (the L0 surrogate)."""
        # P(gate > 0) under the hard-concrete stretch.
        shift = self.beta * math.log(-self.gamma / self.zeta)
        return torch.sigmoid(self.log_alpha - shift).sum()

    @torch.no_grad()
    def active_mask(self, threshold: float = 1e-3) -> torch.Tensor:
        return self._z() > threshold


# ── Phase-Associative Memory (V12) ──────────────────────────────────────────

class V12PAMLayer(nn.Module):
    r"""Matrix-state memory with complex-conjugate retrieval and pluggable dynamics.

    Baseline:  S_t = gamma_t * S_{t-1} + V_t (x) K_t^* ;  Y_t = S_t * Q_t
    E1:        gamma_t becomes per-key-channel (vector decay).
    E2:        write becomes delta-rule (erase stale assoc for K_t before write).
    E3:        K states with distinct decay; retrieval = sum_k e^{i phi_k} S_k Q.
    """

    def __init__(self, cfg: V12Config, layer_idx: int = 0):
        super().__init__()
        self.num_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        inner = cfg.n_heads * cfg.head_dim
        self.inner_dim = inner
        self.dim = cfg.dim
        self.fused_qkv = cfg.fused_qkv
        self.use_rope = cfg.use_rope
        self.use_gsp = cfg.use_gsp
        self.qk_norm = cfg.qk_norm
        self.decay_mode = cfg.decay_mode
        self.write_mode = cfg.write_mode
        self.n_states = cfg.n_states
        self.delta_chunk = cfg.delta_chunk
        self.fused_e3 = getattr(cfg, 'fused_e3', True)
        self.recompute_pam_chunks = getattr(cfg, 'recompute_pam_chunks', False)

        if cfg.fused_qkv:
            self.qkv_proj = ComplexLinear(cfg.dim, 3 * inner, bias=False)
        else:
            self.q_proj = ComplexLinear(cfg.dim, inner, bias=False)
            self.k_proj = ComplexLinear(cfg.dim, inner, bias=False)
            self.v_proj = ComplexLinear(cfg.dim, inner, bias=False)
        self.o_proj = ComplexLinear(inner, cfg.dim, bias=False)

        # Decay projection: per-head scalar, or per-(head, key-channel) for E1.
        decay_out = cfg.n_heads * (cfg.head_dim if cfg.decay_mode == 'per_channel' else 1)
        self.dt_proj = nn.Linear(cfg.dim * 2, decay_out)
        if cfg.decay_mode == 'per_channel':
            self.dt_bias = nn.Parameter(torch.zeros(cfg.n_heads, cfg.head_dim) + cfg.base_dt_bias)
        else:
            self.dt_bias = nn.Parameter(torch.zeros(cfg.n_heads) + cfg.base_dt_bias)

        self.gate_content_aware = getattr(cfg, 'gate_content_aware', False)
        self.protect_gate_bias = getattr(cfg, 'protect_gate_bias', -3.0)
        self.routing_content_aware = getattr(cfg, 'routing_content_aware', False)
        self.state_compete = getattr(cfg, 'state_compete', False)
        self.phase_init = getattr(cfg, 'phase_init', 'zero')
        self.route_balance_lambda = getattr(cfg, 'route_balance_lambda', 0.0)
        self.gamma_floor = getattr(cfg, 'gamma_floor', 0.0)
        self.gate_surprisal_lambda = getattr(cfg, 'gate_surprisal_lambda', 0.0)
        self.vault_state = getattr(cfg, 'vault_state', False)
        self.vault_state_idx = int(getattr(cfg, 'vault_state_idx', 0))
        # M1: learnable head count. n_heads is the MAX budget; the gate prunes it.
        self.head_gate_enabled = getattr(cfg, 'head_gate', False)
        self.head_gate_l0_lambda = getattr(cfg, 'head_gate_l0_lambda', 0.0)
        if self.head_gate_enabled:
            self.head_gate = HardConcreteGate(
                cfg.n_heads, init_logalpha=getattr(cfg, 'head_gate_init_logalpha', 3.0)
            )
        self.write_phase_address = getattr(cfg, 'write_phase_address', False)
        if self.write_phase_address:
            # M3 phase bands: each head slot learns its OWN key/query->phase map, so
            # heads occupy distinct, content-dependent phase bands (non-vacuous —
            # a constant per-head phase would be absorbed by o_proj). Params are
            # per-head [H, d] + [H]; zero-init => identity rotation at start.
            self.write_phase_w = nn.Parameter(torch.zeros(cfg.n_heads, cfg.head_dim))
            self.write_phase_b = nn.Parameter(torch.zeros(cfg.n_heads))
        if cfg.use_gsp:
            gate_in = cfg.dim * 2 if self.gate_content_aware else cfg.dim
            self.protect_gate = nn.Linear(gate_in, cfg.n_heads)
            nn.init.constant_(self.protect_gate.bias, self.protect_gate_bias)

        # E2: delta-rule write strength beta_t in (0, 1) per head.
        if cfg.write_mode == 'delta':
            self.beta_proj = nn.Linear(cfg.dim, cfg.n_heads)
            nn.init.constant_(self.beta_proj.bias, 0.0)

        # E3: per-state decay bias offsets + per-(head,state) retrieval phase.
        if cfg.n_states > 1:
            offs = torch.linspace(-cfg.state_dt_spread, cfg.state_dt_spread, cfg.n_states)
            self.state_dt_offset = nn.Parameter(offs.clone())          # [K]
            route_in = cfg.dim * 2 if self.routing_content_aware else cfg.dim
            self.phase_proj = nn.Linear(route_in, cfg.n_heads * cfg.n_states)
            if self.state_compete:
                self.score_proj = nn.Linear(route_in, cfg.n_heads * cfg.n_states)
                nn.init.zeros_(self.score_proj.weight)
                nn.init.zeros_(self.score_proj.bias)
            self._init_phase_proj()

        if cfg.use_rope:
            self.register_buffer(
                'rope_cache',
                build_rope_cache(cfg.max_seq_len, cfg.head_dim),
                persistent=False,
            )

        self.dropout = nn.Dropout(cfg.dropout)
        self.chunk_size = cfg.chunk_size
        _causal_size = cfg.chunk_size if cfg.chunk_size > 0 else cfg.max_seq_len
        self.register_buffer(
            '_causal',
            torch.tril(torch.ones(_causal_size, _causal_size)),
            persistent=False,
        )
        self._route_aux = None
        self._gate_prob_bt = None   # [B,T] mean protect prob per token (gate-surprisal aux)
        # M2: progressive frozen-head growth. Frozen head slots keep zero grad on
        # their slices of the fused projections (see set_trainable_heads).
        self._frozen_head_mask = None
        self._freeze_handles = []
        # Hard open/closed mask: reserved (not-yet-grown) head slots contribute 0
        # to the residual stream until their curriculum stage opens them. Default
        # all-open => identity (bit-identical to a non-staged model).
        self.register_buffer('_head_open_mask', torch.ones(cfg.n_heads), persistent=False)

    def _apply_gamma_floor(self, base_decay: torch.Tensor) -> torch.Tensor:
        """Lift the base (pre-GSP) decay onto [gamma_floor, 1) to lengthen memory.

        base_decay = exp(-softplus_dt) in (0,1); reparam keeps the learned shape
        but caps the minimum retention so unprotected state survives far longer.
        """
        if self.gamma_floor and self.gamma_floor > 0.0:
            return self.gamma_floor + (1.0 - self.gamma_floor) * base_decay
        return base_decay

    def _init_phase_proj(self):
        """Custom init for phase_proj (re-applied after V12LM._init_weights)."""
        if self.n_states <= 1:
            return
        num_memory_states = self.n_states
        num_heads = self.num_heads
        if self.phase_init == 'spread':
            nn.init.zeros_(self.phase_proj.weight)
            biases = torch.zeros(num_heads * num_memory_states)
            for head_idx in range(num_heads):
                for state_idx in range(num_memory_states):
                    if num_memory_states == 3:
                        biases[head_idx * num_memory_states + state_idx] = [0.0, 2 * math.pi / 3, -2 * math.pi / 3][state_idx]
                    else:
                        biases[head_idx * num_memory_states + state_idx] = state_idx * 2 * math.pi / num_memory_states
            with torch.no_grad():
                self.phase_proj.bias.copy_(biases)
        elif self.phase_init == 'ortho':
            nn.init.orthogonal_(self.phase_proj.weight)
            nn.init.zeros_(self.phase_proj.bias)
        else:
            nn.init.zeros_(self.phase_proj.weight)
            nn.init.zeros_(self.phase_proj.bias)

    def _routing_input(self, x: torch.Tensor) -> torch.Tensor:
        return to_real_concat(x) if self.routing_content_aware else cabs(x)

    def _phase_and_alpha(self, x: torch.Tensor):
        """Phase and K-scaled routing weights for E3 superposition.

        Winner (state_compete off): routing_weights_k == 1; only phases matter.
        phase_proj sees magnitudes by default (cabs) — angle of x is ignored for
        routing so "how loud" a token is, not its phase, picks retrieval phases.

        Returns retrieval_phase [B,T,H,K], routing_weights [B,T,H,K] where
        c_k = routing_weights_k * e^{i retrieval_phase_k} rotates each state's read.
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        num_heads, num_memory_states = self.num_heads, self.n_states
        routing_input = self._routing_input(x)
        retrieval_phase = self.phase_proj(routing_input).view(
            batch_size, seq_len, num_heads, num_memory_states
        )
        if self.state_compete:
            routing_scores = self.score_proj(routing_input).view(
                batch_size, seq_len, num_heads, num_memory_states
            )
            # scale by K so uniform softmax init keeps routing_weights == 1
            routing_weights = F.softmax(routing_scores, dim=-1) * num_memory_states
        else:
            routing_weights = torch.ones(
                batch_size, seq_len, num_heads, num_memory_states,
                device=x.device, dtype=x.dtype,
            )
        return retrieval_phase, routing_weights

    def _route_balance_loss(self, routing_weights: torch.Tensor):
        """MoE-style load balance: maximize entropy of batch-mean routing per head."""
        balance_lambda = self.route_balance_lambda
        if balance_lambda <= 0 or not self.state_compete or not self.training:
            return None
        routing_prob = routing_weights / self.n_states
        mean_routing_prob = routing_prob.mean(dim=(0, 1))
        entropy = -(mean_routing_prob * (mean_routing_prob + 1e-8).log()).sum(dim=-1)
        return -balance_lambda * entropy.mean()

    # ── Projections + position + decay/gate prep (shared) ─────────────────────

    def _project(self, x: torch.Tensor, step_offset: int):
        """Build Q, K, V in PAM layout [B, H, T, d, 2].

        Heads move before time so later matmuls are batched over (B,H). After
        transpose the tensor is non-contiguous; .contiguous() makes views/matmuls dense.
        """
        batch_size, seq_len, _, _ = x.shape
        num_heads, head_dim = self.num_heads, self.head_dim
        if self.fused_qkv:
            qkv = self.qkv_proj(x).view(batch_size, seq_len, 3, num_heads, head_dim, 2)
            # fused QKV -> [B,T,3,H,d,2]; move heads before time for PAM matmuls
            queries = qkv[:, :, 0].transpose(1, 2).contiguous()
            keys = qkv[:, :, 1].transpose(1, 2).contiguous()
            values = qkv[:, :, 2].transpose(1, 2).contiguous()
        else:
            queries = self.q_proj(x).view(batch_size, seq_len, num_heads, head_dim, 2).transpose(1, 2).contiguous()
            keys = self.k_proj(x).view(batch_size, seq_len, num_heads, head_dim, 2).transpose(1, 2).contiguous()
            values = self.v_proj(x).view(batch_size, seq_len, num_heads, head_dim, 2).transpose(1, 2).contiguous()

        if self.use_rope:
            position_end = step_offset + seq_len
            if position_end > self.rope_cache.shape[0]:
                self.register_buffer(
                    'rope_cache',
                    build_rope_cache(position_end * 2, head_dim).to(x.device),
                    persistent=False,
                )
            # Complex multiply by e^{i·θ}: rotates Q/K by position without changing magnitude.
            rope_positions = self.rope_cache[step_offset:position_end].to(dtype=x.dtype)
            queries = cmul(queries, rope_positions)
            keys = cmul(keys, rope_positions)

        if self.qk_norm:
            queries = cnormalize(queries)
            keys = cnormalize(keys)

        # Stage-6 phase addressing: rotate V by ψ(K) on write and Q by ψ(Q) on read
        # so matching bindings reinforce via conjugation. Off by default.
        if self.write_phase_address:
            queries, values = self._apply_write_phase_address(queries, keys, values)
        return queries, keys, values

    def _apply_write_phase_address(self, queries, keys, values):
        """Rotate values by e^{iψ_h(k)} and queries by e^{iψ_h(q)}; ψ_h is a
        per-head map of magnitude -> phase angle (radians), so each head slot
        addresses writes/reads in its own learned phase band."""
        # cabs layout: [B,H,T,d]; per-head projection -> [B,H,T]
        bias = self.write_phase_b.view(1, -1, 1)
        key_phase = torch.einsum('bhtd,hd->bht', cabs(keys), self.write_phase_w) + bias
        query_phase = torch.einsum('bhtd,hd->bht', cabs(queries), self.write_phase_w) + bias
        values = self._rotate_complex(values, key_phase)
        queries = self._rotate_complex(queries, query_phase)
        return queries, values

    @staticmethod
    def _rotate_complex(z: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Multiply complex z [..., d, 2] by e^{i phase} with phase broadcastable to z[...,0]."""
        # phase [B,H,T] → [B,H,T,1] over head_dim
        while phase.dim() < z.dim() - 1:
            phase = phase.unsqueeze(-1)
        cos_p, sin_p = torch.cos(phase), torch.sin(phase)
        real, imag = z[..., 0], z[..., 1]
        return torch.stack([real * cos_p - imag * sin_p, real * sin_p + imag * cos_p], dim=-1)

    def _gamma_and_vprime(self, x: torch.Tensor, values: torch.Tensor, state_offset: float = 0.0,
                         state_idx: Optional[int] = None):
        """Return decay `decay_gamma` and protected value `protected_values`.

        decay_gamma shape: [B,H,T] (head decay) or [B,H,T,d] (per-channel decay).

        GSP protect gate (winner uses this): when protect_prob→1, decay_gamma→1
        (notebook barely forgets) and the write value is scaled down by (1-p)
        so we lock existing associations instead of overwriting them.
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        num_heads, head_dim = self.num_heads, self.head_dim
        x_flat = to_real_concat(x)
        if self.decay_mode == 'per_channel':
            decay_logits = self.dt_proj(x_flat).view(batch_size, seq_len, num_heads, head_dim)  # [B,T,H,d]
            softplus_dt = F.softplus(decay_logits + self.dt_bias + state_offset)                  # bias [H,d]
            # permute to [B,H,T,d] so decay aligns with PAM time axis before head matmuls
            softplus_dt = softplus_dt.permute(0, 2, 1, 3).contiguous()
        else:
            decay_logits = self.dt_proj(x_flat)                              # [B,T,H]
            softplus_dt = F.softplus(decay_logits + self.dt_bias + state_offset)
            # transpose to [B,H,T] so decay aligns with PAM time axis
            softplus_dt = softplus_dt.transpose(1, 2).contiguous()

        base_decay = self._apply_gamma_floor(torch.exp(-softplus_dt))
        # Vault state: pin base decay to 1 (no forgetting); GSP still gates writes.
        if (
            self.vault_state
            and state_idx is not None
            and state_idx == self.vault_state_idx
        ):
            base_decay = torch.ones_like(base_decay)
        if self.use_gsp:
            # Content-aware gate (winner): sees concat(real,imag); else magnitude only.
            gate_input = to_real_concat(x) if self.gate_content_aware else cabs(x)
            protect_prob = torch.sigmoid(self.protect_gate(gate_input)).transpose(1, 2)  # [B,H,T]
            if self.gate_surprisal_lambda > 0 and self.training:
                self._gate_prob_bt = protect_prob.mean(dim=1)  # [B,T]
            if self.decay_mode == 'per_channel':
                # base_decay is [B,H,T,d] here; protect broadcasts over the channel dim.
                protect_prob_expanded = protect_prob.unsqueeze(-1)
                decay_gamma = base_decay * (1 - protect_prob_expanded) + protect_prob_expanded
            else:
                # Blend: γ = base*(1-p) + p  →  p=1 freezes decay at 1.
                decay_gamma = base_decay * (1 - protect_prob) + protect_prob
            # Same protect scalar on real+imag of values [B,H,T,d,2].
            protected_values = scale_complex(values, 1 - protect_prob)
        else:
            decay_gamma = base_decay
            protected_values = values
        return decay_gamma, protected_values

    # ── Baseline dual-form block (head scalar decay, additive write) ──────────

    @staticmethod
    def _dual_form_block(scaled_queries, keys, protected_values, decay_gamma, causal_mask):
        batch_size, num_heads, seq_len = decay_gamma.shape
        decay_gamma_flat = decay_gamma.reshape(batch_size * num_heads, seq_len)
        decay_matrix = fused_decay_matrix(decay_gamma_flat, seq_len).reshape(
            batch_size, num_heads, seq_len, seq_len
        )
        query_real, query_imag = scaled_queries[..., 0], scaled_queries[..., 1]
        key_real, key_imag = keys[..., 0], keys[..., 1]
        score_real = query_real @ key_real.transpose(-1, -2) + query_imag @ key_imag.transpose(-1, -2)
        score_imag = query_imag @ key_real.transpose(-1, -2) - query_real @ key_imag.transpose(-1, -2)
        weighted_real, weighted_imag = score_real * decay_matrix, score_imag * decay_matrix
        value_real, value_imag = protected_values[..., 0], protected_values[..., 1]
        output_real = weighted_real @ value_real - weighted_imag @ value_imag
        output_imag = weighted_real @ value_imag + weighted_imag @ value_real
        output = torch.stack([output_real, output_imag], dim=-1)
        decay_last_row = decay_matrix[:, :, -1, :]
        write_value_real = value_real * decay_last_row.unsqueeze(-1)
        write_value_imag = value_imag * decay_last_row.unsqueeze(-1)
        state_real = write_value_real.transpose(-1, -2) @ key_real + write_value_imag.transpose(-1, -2) @ key_imag
        state_imag = write_value_imag.transpose(-1, -2) @ key_real - write_value_real.transpose(-1, -2) @ key_imag
        memory_state = torch.stack([state_real, state_imag], dim=-1)
        return output, memory_state

    def _forward_chunked_head(self, queries, keys, protected_values, decay_gamma, head_dim):
        batch_size, num_heads, seq_len = queries.shape[:3]
        chunk_size = self.chunk_size
        query_scale = head_dim ** -0.5
        scaled_queries = queries * query_scale
        memory_state = queries.new_zeros(batch_size, num_heads, head_dim, head_dim, 2)
        outputs = []
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk_len = chunk_end - chunk_start
            queries_chunk = scaled_queries[:, :, chunk_start:chunk_end]
            keys_chunk = keys[:, :, chunk_start:chunk_end]
            values_chunk = protected_values[:, :, chunk_start:chunk_end]
            decay_gamma_chunk = decay_gamma[:, :, chunk_start:chunk_end]
            causal = self._causal[:chunk_len, :chunk_len]
            output_chunk, state_chunk = self._dual_form_block(
                queries_chunk, keys_chunk, values_chunk, decay_gamma_chunk, causal
            )
            log_decay = torch.log(decay_gamma_chunk + 1e-6)
            cumulative_decay = torch.exp(torch.cumsum(log_decay, dim=-1))
            if chunk_start > 0:
                state_real, state_imag = memory_state[..., 0], memory_state[..., 1]
                query_real_chunk, query_imag_chunk = queries_chunk[..., 0], queries_chunk[..., 1]
                carried_real = (
                    state_real @ query_real_chunk.transpose(-1, -2)
                    - state_imag @ query_imag_chunk.transpose(-1, -2)
                ).transpose(-1, -2)
                carried_imag = (
                    state_real @ query_imag_chunk.transpose(-1, -2)
                    + state_imag @ query_real_chunk.transpose(-1, -2)
                ).transpose(-1, -2)
                cumulative_decay_expanded = cumulative_decay.unsqueeze(-1)
                output_chunk = output_chunk + torch.stack(
                    [carried_real * cumulative_decay_expanded, carried_imag * cumulative_decay_expanded],
                    dim=-1,
                )
            outputs.append(output_chunk)
            total_decay = cumulative_decay[:, :, -1]
            memory_state = memory_state * total_decay[..., None, None, None] + state_chunk
        return torch.cat(outputs, dim=2), memory_state

    # ── E1: per-channel decay (GLA-style fold), chunked ───────────────────────

    def _forward_chunked_perchannel(self, queries, keys, protected_values, decay_gamma, head_dim):
        """decay_gamma: [B,H,T,d] per key-channel. Stable chunk-local cumulative fold."""
        batch_size, num_heads, seq_len = queries.shape[:3]
        chunk_size = self.chunk_size
        query_scale = head_dim ** -0.5
        memory_state = queries.new_zeros(batch_size, num_heads, head_dim, head_dim, 2)  # value(i) x key(j)
        outputs = []
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk_len = chunk_end - chunk_start
            queries_chunk = queries[:, :, chunk_start:chunk_end]                  # [B,H,Tc,d,2]
            keys_chunk = keys[:, :, chunk_start:chunk_end]
            values_chunk = protected_values[:, :, chunk_start:chunk_end]
            decay_gamma_chunk = decay_gamma[:, :, chunk_start:chunk_end]        # [B,H,Tc,d]

            log_decay = torch.log(decay_gamma_chunk.clamp_min(1e-6)).float()
            cumulative_log_decay = torch.cumsum(log_decay, dim=2)             # inclusive cumsum, [B,H,Tc,d]
            cumulative_log_decay = cumulative_log_decay.clamp(min=-30.0)
            alpha = torch.exp(cumulative_log_decay)                             # prod_{0..t} g  (<=1)
            inv_alpha = torch.exp(-cumulative_log_decay)                        # 1/alpha        (>=1, bounded)
            cumulative_log_total = cumulative_log_decay[:, :, -1:, :]           # [B,H,1,d]
            decay_tail = torch.exp(cumulative_log_total - cumulative_log_decay)   # alpha_total/alpha_s  (<=1)
            alpha = alpha.to(queries.dtype)
            inv_alpha = inv_alpha.to(queries.dtype)
            decay_tail = decay_tail.to(queries.dtype)
            alpha_total = torch.exp(cumulative_log_total).to(queries.dtype)     # [B,H,1,d]

            # Fold decay into queries (q*alpha) and keys (k/alpha) -> plain conjugate score.
            folded_queries = queries_chunk * alpha.unsqueeze(-1) * query_scale
            folded_keys = keys_chunk * inv_alpha.unsqueeze(-1)
            query_real, query_imag = folded_queries[..., 0], folded_queries[..., 1]
            key_real, key_imag = folded_keys[..., 0], folded_keys[..., 1]
            score_real = query_real @ key_real.transpose(-1, -2) + query_imag @ key_imag.transpose(-1, -2)
            score_imag = query_imag @ key_real.transpose(-1, -2) - query_real @ key_imag.transpose(-1, -2)
            causal = self._causal[:chunk_len, :chunk_len]
            score_real, score_imag = score_real * causal, score_imag * causal
            value_real, value_imag = values_chunk[..., 0], values_chunk[..., 1]
            output_real = score_real @ value_real - score_imag @ value_imag
            output_imag = score_real @ value_imag + score_imag @ value_real
            output_chunk = torch.stack([output_real, output_imag], dim=-1)

            if chunk_start > 0:
                # carried state read: output += (memory_state @ (queries*alpha)) per channel.
                queries_with_decay = queries_chunk * alpha.unsqueeze(-1) * query_scale  # [B,H,Tc,d,2]
                state_real, state_imag = memory_state[..., 0], memory_state[..., 1]      # [B,H,d(i),d(j)]
                query_real_decay, query_imag_decay = queries_with_decay[..., 0], queries_with_decay[..., 1]
                carried_real = (
                    query_real_decay @ state_real.transpose(-1, -2)
                    - query_imag_decay @ state_imag.transpose(-1, -2)
                )
                carried_imag = (
                    query_real_decay @ state_imag.transpose(-1, -2)
                    + query_imag_decay @ state_real.transpose(-1, -2)
                )
                output_chunk = output_chunk + torch.stack([carried_real, carried_imag], dim=-1)

            outputs.append(output_chunk)

            # state update: S_new[i,j] = alpha_total[j]*S[i,j] + sum_s v_s[i] (k_s* decay_tail)[j]
            decayed_keys = keys_chunk * decay_tail.unsqueeze(-1)                    # [B,H,Tc,d,2]
            decayed_key_real, decayed_key_imag = decayed_keys[..., 0], decayed_keys[..., 1]
            state_real = value_real.transpose(-1, -2) @ decayed_key_real + value_imag.transpose(-1, -2) @ decayed_key_imag
            state_imag = value_imag.transpose(-1, -2) @ decayed_key_real - value_real.transpose(-1, -2) @ decayed_key_imag
            state_chunk = torch.stack([state_real, state_imag], dim=-1)
            alpha_total_squeezed = alpha_total.squeeze(2)                           # [B,H,d(j)]
            memory_state = memory_state * alpha_total_squeezed.unsqueeze(2).unsqueeze(-1) + state_chunk
        return torch.cat(outputs, dim=2), memory_state

    # ── E2: delta-rule write (UT transform), chunked, head scalar decay ───────

    def _forward_delta(self, queries, keys, protected_values, decay_gamma, write_beta, head_dim):
        """Gated delta rule via per-chunk UT transform. decay_gamma: [B,H,T] head scalar."""
        batch_size, num_heads, seq_len = queries.shape[:3]
        chunk_size = self.delta_chunk
        query_scale = head_dim ** -0.5
        memory_state = queries.new_zeros(batch_size, num_heads, head_dim, head_dim, 2)
        outputs = []
        identity = torch.eye(chunk_size, device=queries.device, dtype=torch.float32)
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk_len = chunk_end - chunk_start
            queries_chunk = queries[:, :, chunk_start:chunk_end]
            keys_chunk = keys[:, :, chunk_start:chunk_end]
            values_chunk = protected_values[:, :, chunk_start:chunk_end]
            decay_gamma_chunk = decay_gamma[:, :, chunk_start:chunk_end]   # [B,H,Tc]
            write_beta_chunk = write_beta[:, :, chunk_start:chunk_end]     # [B,H,Tc]

            decay_gamma_flat = decay_gamma_chunk.reshape(batch_size * num_heads, chunk_len)
            decay_matrix = fused_decay_matrix(decay_gamma_flat, chunk_len).reshape(
                batch_size, num_heads, chunk_len, chunk_len
            )  # decay_matrix[t,s]=prod_{s+1..t} g
            log_decay = torch.log(decay_gamma_chunk + 1e-6)
            cumulative_alpha = torch.exp(torch.cumsum(log_decay, dim=-1))  # alpha_t = prod_{0..t} g

            key_real, key_imag = keys_chunk[..., 0], keys_chunk[..., 1]
            query_real, query_imag = queries_chunk[..., 0], queries_chunk[..., 1]
            key_gram_real = key_real @ key_real.transpose(-1, -2) + key_imag @ key_imag.transpose(-1, -2)
            key_gram_imag = key_imag @ key_real.transpose(-1, -2) - key_real @ key_imag.transpose(-1, -2)
            strict_lower = torch.tril(torch.ones(chunk_len, chunk_len, device=queries.device), -1)
            decay_masked = decay_matrix * strict_lower
            mass_real = write_beta_chunk.unsqueeze(-1) * decay_masked * key_gram_real
            mass_imag = write_beta_chunk.unsqueeze(-1) * decay_masked * key_gram_imag

            value_real, value_imag = values_chunk[..., 0], values_chunk[..., 1]
            if chunk_start > 0:
                state_real, state_imag = memory_state[..., 0], memory_state[..., 1]
                state_key_real = (
                    key_real @ state_real.transpose(-1, -2) - key_imag @ state_imag.transpose(-1, -2)
                )
                state_key_imag = (
                    key_real @ state_imag.transpose(-1, -2) + key_imag @ state_real.transpose(-1, -2)
                )
                state_key_real = state_key_real * cumulative_alpha.unsqueeze(-1)
                state_key_imag = state_key_imag * cumulative_alpha.unsqueeze(-1)
                write_real = write_beta_chunk.unsqueeze(-1) * (value_real - state_key_real)
                write_imag = write_beta_chunk.unsqueeze(-1) * (value_imag - state_key_imag)
            else:
                write_real = write_beta_chunk.unsqueeze(-1) * value_real
                write_imag = write_beta_chunk.unsqueeze(-1) * value_imag

            update_real, update_imag = _complex_triangular_solve(
                mass_real, mass_imag, write_real, write_imag, identity[:chunk_len, :chunk_len]
            )

            query_key_real = query_real @ key_real.transpose(-1, -2) + query_imag @ key_imag.transpose(-1, -2)
            query_key_imag = query_imag @ key_real.transpose(-1, -2) - query_real @ key_imag.transpose(-1, -2)
            causal_inclusive = self._causal[:chunk_len, :chunk_len]
            projection_real = (decay_matrix * causal_inclusive) * query_key_real
            projection_imag = (decay_matrix * causal_inclusive) * query_key_imag
            output_real = (projection_real @ update_real - projection_imag @ update_imag) * query_scale
            output_imag = (projection_real @ update_imag + projection_imag @ update_real) * query_scale
            output_chunk = torch.stack([output_real, output_imag], dim=-1)
            if chunk_start > 0:
                state_real, state_imag = memory_state[..., 0], memory_state[..., 1]
                scaled_query_real = query_real * query_scale * cumulative_alpha.unsqueeze(-1)
                scaled_query_imag = query_imag * query_scale * cumulative_alpha.unsqueeze(-1)
                carried_real = (
                    scaled_query_real @ state_real.transpose(-1, -2)
                    - scaled_query_imag @ state_imag.transpose(-1, -2)
                )
                carried_imag = (
                    scaled_query_real @ state_imag.transpose(-1, -2)
                    + scaled_query_imag @ state_real.transpose(-1, -2)
                )
                output_chunk = output_chunk + torch.stack([carried_real, carried_imag], dim=-1)
            outputs.append(output_chunk)

            cumulative_total = cumulative_alpha[:, :, -1:]                       # [B,H,1]
            decay_tail = cumulative_total / (cumulative_alpha + 1e-12)           # alpha_T/alpha_s
            update_decayed_real = update_real * decay_tail.unsqueeze(-1)
            update_decayed_imag = update_imag * decay_tail.unsqueeze(-1)
            state_real = update_decayed_real.transpose(-1, -2) @ key_real + update_decayed_imag.transpose(-1, -2) @ key_imag
            state_imag = update_decayed_imag.transpose(-1, -2) @ key_real - update_decayed_real.transpose(-1, -2) @ key_imag
            state_chunk = torch.stack([state_real, state_imag], dim=-1)
            memory_state = memory_state * cumulative_total.unsqueeze(-1).unsqueeze(-1) + state_chunk
        return torch.cat(outputs, dim=2), memory_state

    # ── E3: multi-state superposition (loop over states, phase-combine) ───────

    def _forward_multistate(self, x, queries, keys, protected_values, head_dim):
        batch_size, seq_len = x.shape[0], x.shape[1]
        num_heads, num_memory_states = self.num_heads, self.n_states
        query_scale = head_dim ** -0.5
        retrieval_phase, routing_weights = self._phase_and_alpha(x)
        retrieval_phase = retrieval_phase.permute(0, 2, 3, 1)    # [B,H,K,T]
        routing_weights = routing_weights.permute(0, 2, 3, 1)
        self._route_aux = self._route_balance_loss(routing_weights.permute(0, 3, 1, 2))
        output_sum = None
        state_list = []
        for state_idx in range(num_memory_states):
            decay_gamma_state, protected_values_state = self._gamma_and_vprime(
                x, protected_values, state_offset=self.state_dt_offset[state_idx],
                state_idx=state_idx,
            )
            if self.decay_mode == 'per_channel':
                output_state, memory_state = self._forward_chunked_perchannel(
                    queries, keys, protected_values_state, decay_gamma_state, head_dim
                )
            elif self.chunk_size > 0 and seq_len > self.chunk_size:
                output_state, memory_state = self._forward_chunked_head(
                    queries, keys, protected_values_state, decay_gamma_state, head_dim
                )
            else:
                scaled_queries = queries * query_scale
                output_state, memory_state = self._dual_form_block(
                    scaled_queries, keys, protected_values_state, decay_gamma_state,
                    self._causal[:seq_len, :seq_len],
                )
            rotation_real = routing_weights[:, :, state_idx] * torch.cos(retrieval_phase[:, :, state_idx])
            rotation_imag = routing_weights[:, :, state_idx] * torch.sin(retrieval_phase[:, :, state_idx])
            rotation = torch.stack([rotation_real, rotation_imag], dim=-1)  # [B,H,T,2]
            output_state = cmul(output_state, rotation.unsqueeze(-2))         # rotate+scale complex output
            output_sum = output_state if output_sum is None else output_sum + output_state
            state_list.append(memory_state)
        return output_sum, torch.stack(state_list, dim=0)                     # [K,B,H,d,d,2]

    # ── E3 fused: state-independent work hoisted, K states collapsed ──────────
    #
    # Exact algebraic identity with `_forward_multistate` (head decay, additive
    # write). Two facts make it work:
    #   * The QK* score W and the protected value v' do NOT depend on the state
    #     index k, so they are computed ONCE per chunk (not K times).
    #   * Phase-routed retrieval is linear, so the per-state decay matrices D_k
    #     collapse into a single COMPLEX decay matrix
    #         Dtilde[t,s] = sum_k e^{i phi_k(t)} * D_k[t,s]
    #     and the intra-chunk output is one complex matmul  y = (W (.) Dtilde) @ v'
    #     instead of K separate (W (.) D_k) @ v' matmuls.
    # The carried-state read and the per-chunk state write stay per-state but are
    # the cheap O(C d^2) ops; they are folded into K-batched matmuls.

    def _gamma_all_and_vprime(self, x, values):
        """Head-scalar decay for all K states at once (single dt_proj/gate matmul).

        Returns decay_gamma_all [K,B,H,T] and shared protected_values [B,H,T,d,2].
        Same GSP idea as _gamma_and_vprime: protect freezes decay and shrinks writes.
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        num_heads, num_memory_states = self.num_heads, self.n_states
        x_flat = to_real_concat(x)
        decay_logits = self.dt_proj(x_flat)                              # [B,T,H]
        state_offsets = self.state_dt_offset.view(num_memory_states, 1, 1, 1)  # [K,1,1,1]
        softplus_dt = F.softplus(decay_logits + self.dt_bias + state_offsets)  # [K,B,T,H]
        # permute to [K,B,H,T] so decay aligns with PAM time axis
        softplus_dt = softplus_dt.permute(0, 1, 3, 2).contiguous()
        base_decay = self._apply_gamma_floor(torch.exp(-softplus_dt))  # [K,B,H,T]
        if self.vault_state and 0 <= self.vault_state_idx < num_memory_states:
            # Compile-safe: functional where over a static per-state mask (no in-place/clone).
            state_ids = torch.arange(num_memory_states, device=base_decay.device)
            is_vault = (state_ids == self.vault_state_idx).view(num_memory_states, 1, 1, 1)
            base_decay = torch.where(is_vault, torch.ones_like(base_decay), base_decay)
        if self.use_gsp:
            gate_input = to_real_concat(x) if self.gate_content_aware else cabs(x)
            protect_prob = torch.sigmoid(self.protect_gate(gate_input)).transpose(1, 2)  # [B,H,T]
            if self.gate_surprisal_lambda > 0 and self.training:
                self._gate_prob_bt = protect_prob.mean(dim=1)  # [B,T]
            decay_gamma_all = base_decay * (1 - protect_prob) + protect_prob  # [K,B,H,T]
            protected_values = scale_complex(values, 1 - protect_prob)
        else:
            decay_gamma_all = base_decay
            protected_values = values
        return decay_gamma_all, protected_values

    def _fused_chunk_step(
        self, queries_chunk, keys_chunk, protected_values_chunk,
        decay_gamma_chunk, retrieval_phase_chunk, routing_weights_chunk,
        memory_state, is_first_chunk,
    ):
        """One fused E3 chunk. Returns (output_chunk, memory_state_new).

        Dual form (training story) — same math as looping outer products, different layout:
          Inference writes each token as S = γ·S + V⊗K* then reads y = S@Q.
          Unrolling that over a chunk equals decay-weighted complex scores
          (Q·K*) ⊙ D̃ @ V, plus a carried read from the previous chunk's S.
          "Dual" = equivalent rewrite for GPU matmuls, not a second memory.

        Winner also collapses K decay matrices into one complex D̃ via phase routing
        so we do one complex matmul instead of K separate ones.
        """
        num_memory_states = decay_gamma_chunk.shape[0]
        chunk_len = decay_gamma_chunk.shape[-1]
        query_real, query_imag = real_part(queries_chunk), imag_part(queries_chunk)
        key_real, key_imag = real_part(keys_chunk), imag_part(keys_chunk)
        # Complex conjugate inner product Q·K* (score before decay weighting).
        score_real = query_real @ key_real.transpose(-1, -2) + query_imag @ key_imag.transpose(-1, -2)
        score_imag = query_imag @ key_real.transpose(-1, -2) - query_real @ key_imag.transpose(-1, -2)

        batch_heads_states = num_memory_states * decay_gamma_chunk.shape[1] * decay_gamma_chunk.shape[2]
        decay_matrix = fused_decay_matrix(
            decay_gamma_chunk.reshape(batch_heads_states, chunk_len), chunk_len
        ).reshape(decay_gamma_chunk.shape + (chunk_len,))
        cos_phase = torch.cos(retrieval_phase_chunk)
        sin_phase = torch.sin(retrieval_phase_chunk)
        # Collapse K real decay mats into one complex D̃ = Σ_k α_k e^{iφ_k} D_k.
        decay_real = ((routing_weights_chunk * cos_phase).unsqueeze(-1) * decay_matrix).sum(dim=0)
        decay_imag = ((routing_weights_chunk * sin_phase).unsqueeze(-1) * decay_matrix).sum(dim=0)

        # Intra-chunk dual read: y = (W ⊙ D̃) @ V.
        weighted_real = score_real * decay_real - score_imag * decay_imag
        weighted_imag = score_real * decay_imag + score_imag * decay_real
        value_real, value_imag = real_part(protected_values_chunk), imag_part(protected_values_chunk)
        output_real = weighted_real @ value_real - weighted_imag @ value_imag
        output_imag = weighted_real @ value_imag + weighted_imag @ value_real
        output_chunk = stack_complex(output_real, output_imag)

        log_decay = torch.log(decay_gamma_chunk + 1e-6)
        cumulative_decay = torch.exp(torch.cumsum(log_decay, dim=-1))   # [K,B,H,Tc]

        if not is_first_chunk:
            # Carried-state read: how much previous-chunk notebook still contributes
            # after decaying into this chunk, then phase-route and sum over K.
            state_real, state_imag = real_part(memory_state), imag_part(memory_state)
            query_real_states = query_real.unsqueeze(0)
            query_imag_states = query_imag.unsqueeze(0)
            carried_real = (
                state_real @ query_real_states.transpose(-1, -2)
                - state_imag @ query_imag_states.transpose(-1, -2)
            ).transpose(-1, -2)
            carried_imag = (
                state_real @ query_imag_states.transpose(-1, -2)
                + state_imag @ query_real_states.transpose(-1, -2)
            ).transpose(-1, -2)
            combined_real = routing_weights_chunk * cos_phase * cumulative_decay
            combined_imag = routing_weights_chunk * sin_phase * cumulative_decay
            routed_real = (carried_real * combined_real.unsqueeze(-1) - carried_imag * combined_imag.unsqueeze(-1)).sum(dim=0)
            routed_imag = (carried_real * combined_imag.unsqueeze(-1) + carried_imag * combined_real.unsqueeze(-1)).sum(dim=0)
            output_chunk = output_chunk + stack_complex(routed_real, routed_imag)

        # Chunk write into notebook: last row of decay says what survives to chunk end;
        # outer-product-equivalent state update via (decayed V) @ K* (batched over K).
        decay_last = decay_matrix[:, :, :, -1, :]
        value_real_states = value_real.unsqueeze(0)
        value_imag_states = value_imag.unsqueeze(0)
        write_value_real = value_real_states * decay_last.unsqueeze(-1)
        write_value_imag = value_imag_states * decay_last.unsqueeze(-1)
        key_real_states = key_real.unsqueeze(0)
        key_imag_states = key_imag.unsqueeze(0)
        state_real = write_value_real.transpose(-1, -2) @ key_real_states + write_value_imag.transpose(-1, -2) @ key_imag_states
        state_imag = write_value_imag.transpose(-1, -2) @ key_real_states - write_value_real.transpose(-1, -2) @ key_imag_states
        state_chunk = stack_complex(state_real, state_imag)
        total_decay = cumulative_decay[:, :, :, -1]
        # Broadcast per-(K,B,H) decay over the d×d matrix indices.
        memory_state_new = memory_state * total_decay[..., None, None, None] + state_chunk
        return output_chunk, memory_state_new

    def _forward_multistate_fused(self, x, queries, keys, values, head_dim):
        batch_size, seq_len = x.shape[0], x.shape[1]
        num_heads, num_memory_states = self.num_heads, self.n_states
        chunk_size = self.chunk_size if self.chunk_size > 0 else seq_len
        query_scale = head_dim ** -0.5
        retrieval_phase, routing_weights = self._phase_and_alpha(x)
        retrieval_phase = retrieval_phase.permute(3, 0, 2, 1)                        # [K,B,H,T]
        routing_weights = routing_weights.permute(3, 0, 2, 1)
        self._route_aux = self._route_balance_loss(
            routing_weights.permute(1, 3, 2, 0)
        )
        decay_gamma_all, protected_values = self._gamma_all_and_vprime(x, values)
        scaled_queries = queries * query_scale
        recompute = getattr(self, 'recompute_pam_chunks', False) and self.training

        memory_state = queries.new_zeros(num_memory_states, batch_size, num_heads, head_dim, head_dim, 2)
        outputs = []
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            queries_chunk = scaled_queries[:, :, chunk_start:chunk_end]
            keys_chunk = keys[:, :, chunk_start:chunk_end]
            protected_values_chunk = protected_values[:, :, chunk_start:chunk_end]
            decay_gamma_chunk = decay_gamma_all[:, :, :, chunk_start:chunk_end]
            retrieval_phase_chunk = retrieval_phase[:, :, :, chunk_start:chunk_end]
            routing_weights_chunk = routing_weights[:, :, :, chunk_start:chunk_end]
            is_first_chunk = chunk_start == 0
            if recompute:
                output_chunk, memory_state = grad_checkpoint(
                    self._fused_chunk_step,
                    queries_chunk, keys_chunk, protected_values_chunk,
                    decay_gamma_chunk, retrieval_phase_chunk, routing_weights_chunk,
                    memory_state, is_first_chunk,
                    use_reentrant=False,
                )
            else:
                output_chunk, memory_state = self._fused_chunk_step(
                    queries_chunk, keys_chunk, protected_values_chunk,
                    decay_gamma_chunk, retrieval_phase_chunk, routing_weights_chunk,
                    memory_state, is_first_chunk,
                )
            outputs.append(output_chunk)

        return torch.cat(outputs, dim=2), memory_state

    # ── Main forward ──────────────────────────────────────────────────────────

    def forward(self, x, state=None, step_offset: int = 0):
        batch_size, seq_len, _, _ = x.shape
        num_heads, head_dim = self.num_heads, self.head_dim
        queries, keys, values = self._project(x, step_offset)

        # Training / prefill (parallel): state is None and seq_len>1.
        # Winner (E3 K=3, fused_e3, head decay, additive): _forward_multistate_fused.
        # Other branches below are ablation paths (E1/E2 / non-fused); production skips them.
        if state is None and seq_len > 1:
            if self.n_states > 1:
                use_fused = (
                    getattr(self, 'fused_e3', True)
                    and self.decay_mode != 'per_channel'
                    and self.write_mode == 'additive'
                )
                if use_fused:
                    output, new_state = self._forward_multistate_fused(x, queries, keys, values, head_dim)
                else:
                    # Ablation: K-loop multistate without D̃ collapse.
                    output, new_state = self._forward_multistate(x, queries, keys, values, head_dim)
            elif self.write_mode == 'delta':
                # Ablation E2 — not used by winner.
                decay_gamma, protected_values = self._gamma_and_vprime(x, values)
                write_beta = torch.sigmoid(self.beta_proj(cabs(x))).transpose(1, 2)  # [B,H,T]
                output, new_state = self._forward_delta(
                    queries, keys, protected_values, decay_gamma, write_beta, head_dim
                )
            elif self.decay_mode == 'per_channel':
                # Ablation E1 — not used by winner.
                decay_gamma, protected_values = self._gamma_and_vprime(x, values)
                output, new_state = self._forward_chunked_perchannel(
                    queries, keys, protected_values, decay_gamma, head_dim
                )
            else:
                decay_gamma, protected_values = self._gamma_and_vprime(x, values)
                if self.chunk_size > 0 and seq_len > self.chunk_size:
                    output, new_state = self._forward_chunked_head(
                        queries, keys, protected_values, decay_gamma, head_dim
                    )
                else:
                    scaled_queries = queries * (head_dim ** -0.5)
                    output, new_state = self._dual_form_block(
                        scaled_queries, keys, protected_values, decay_gamma,
                        self._causal[:seq_len, :seq_len],
                    )
        else:
            # Decode / single-token: O(1) recurrent outer-product updates.
            output, new_state = self._recurrent(x, queries, keys, values, state, head_dim)

        # M1: learnable head count. Scale each head slot [B,H,T,d,2] by its gate
        # z_h in [0,1]; slots pruned to 0 drop out of the residual stream. Applied
        # here (post-PAM, pre-merge) so training and O(1) decode share one path,
        # and the fused E3 kernel stays untouched. L0 penalty rides the aux hook.
        if self.head_gate_enabled:
            z = self.head_gate.gate().to(output.dtype)             # [H]
            output = output * z.view(1, -1, 1, 1, 1)
            if self.training and self.head_gate_l0_lambda > 0:
                self._route_aux = self.head_gate_l0_lambda * self.head_gate.num_active()
            else:
                self._route_aux = None

        # M2: hard open/closed mask — reserved head slots contribute 0 until grown.
        if self._head_open_mask is not None and not bool((self._head_open_mask == 1).all()):
            output = output * self._head_open_mask.to(output.dtype).view(1, -1, 1, 1, 1)

        # merge heads back to [B,T,inner_dim,2] for output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.inner_dim, 2)
        out = self.o_proj(output)
        if self.training:
            # Manual complex dropout: one mask for both real and imag.
            dropout_mask = as_complex_dropout_mask(self.dropout, out)
            out = scale_complex(out, dropout_mask)
        return out, new_state

    # ── M2: progressive frozen-head growth ────────────────────────────────────

    def set_open_heads(self, open_indices):
        """Hard-open only ``open_indices`` head slots; close the rest (contribute 0).

        Used by the M2 curriculum to keep reserved slots silent until their stage.
        Pass the full head set to open everything (default state).
        """
        num_heads = self.num_heads
        open_set = {int(i) for i in open_indices}
        mask = torch.tensor([1.0 if h in open_set else 0.0 for h in range(num_heads)])
        self._head_open_mask = mask.to(self._head_open_mask.device)

    def set_trainable_heads(self, active_indices):
        """Freeze every head slot NOT in ``active_indices`` (list/set of ints).

        Heads live in *fused* projection tensors, so ``requires_grad`` cannot
        target one slot. Instead we register gradient hooks that zero the frozen
        heads' rows/columns of every per-head parameter, leaving active heads to
        train normally. NOTE: run growth stages with ``weight_decay=0`` so
        decoupled AdamW does not drift the frozen (zero-grad) slices.
        Call with the full head set to unfreeze everything.
        """
        num_heads = self.num_heads
        active = {int(i) for i in active_indices}
        frozen = torch.tensor([h not in active for h in range(num_heads)], dtype=torch.bool)
        self._frozen_head_mask = frozen
        self._apply_freeze_hooks()

    def _clear_freeze_hooks(self):
        for handle in getattr(self, '_freeze_handles', []):
            handle.remove()
        self._freeze_handles = []

    def _apply_freeze_hooks(self):
        self._clear_freeze_hooks()
        frozen = self._frozen_head_mask
        if frozen is None or not bool(frozen.any()):
            return
        head_dim, num_heads, num_states = self.head_dim, self.num_heads, self.n_states
        inner = self.inner_dim
        handles = self._freeze_handles

        def add_hook(param, keep_mask):
            # keep_mask: float tensor broadcastable to grad; 0 where frozen.
            keep = keep_mask.to(param.dtype)
            handles.append(param.register_hook(lambda g: g * keep.to(g.device)))

        def rows_keep(num_out, frozen_rows):
            m = torch.ones(num_out)
            m[frozen_rows] = 0.0
            return m

        # Per-head frozen row indices for width-`head_dim` blocks.
        def block_rows(width_per_head, per_head_stride=None, num_blocks=1):
            stride = per_head_stride if per_head_stride is not None else width_per_head
            idx = []
            for h in range(num_heads):
                if frozen[h]:
                    for b in range(num_blocks):
                        base = b * (num_heads * stride) + h * stride
                        idx.extend(range(base, base + width_per_head))
            return idx

        # qkv_proj rows: layout [3, H, head_dim] over out dim = 3*inner.
        if self.fused_qkv:
            qkv_rows = block_rows(head_dim, per_head_stride=head_dim, num_blocks=3)
            keep = rows_keep(3 * inner, qkv_rows).view(-1, 1)
            add_hook(self.qkv_proj.weight_real, keep)
            add_hook(self.qkv_proj.weight_imag, keep)
        else:
            rows = block_rows(head_dim)
            keep = rows_keep(inner, rows).view(-1, 1)
            for proj in (self.q_proj, self.k_proj, self.v_proj):
                add_hook(proj.weight_real, keep)
                add_hook(proj.weight_imag, keep)

        # o_proj columns: head h owns input cols [h*d:(h+1)*d].
        o_cols = block_rows(head_dim)
        keep_col = torch.ones(inner)
        keep_col[o_cols] = 0.0
        add_hook(self.o_proj.weight_real, keep_col.view(1, -1))
        add_hook(self.o_proj.weight_imag, keep_col.view(1, -1))

        # dt_proj / dt_bias / protect_gate: one row per head (head decay mode).
        head_keep = rows_keep(num_heads, [h for h in range(num_heads) if frozen[h]])
        if self.decay_mode != 'per_channel':
            add_hook(self.dt_proj.weight, head_keep.view(-1, 1))
            add_hook(self.dt_proj.bias, head_keep)
            add_hook(self.dt_bias, head_keep)
        if self.use_gsp:
            add_hook(self.protect_gate.weight, head_keep.view(-1, 1))
            add_hook(self.protect_gate.bias, head_keep)

        # phase_proj rows: layout [H, K] over out dim = H*K.
        if num_states > 1:
            pp_rows = []
            for h in range(num_heads):
                if frozen[h]:
                    pp_rows.extend(range(h * num_states, (h + 1) * num_states))
            keep_pp = rows_keep(num_heads * num_states, pp_rows)
            add_hook(self.phase_proj.weight, keep_pp.view(-1, 1))
            add_hook(self.phase_proj.bias, keep_pp)

        # head_gate: freeze frozen heads' gate logits (keep them open/as-is).
        if self.head_gate_enabled:
            add_hook(self.head_gate.log_alpha, head_keep)

        # delta write strength beta_proj: one row per head.
        if self.write_mode == 'delta':
            add_hook(self.beta_proj.weight, head_keep.view(-1, 1))
            add_hook(self.beta_proj.bias, head_keep)

        # M3 per-head phase-band params: one row per head.
        if self.write_phase_address:
            add_hook(self.write_phase_w, head_keep.view(-1, 1))
            add_hook(self.write_phase_b, head_keep)

        self._freeze_handles = handles

    # ── O(1) recurrent inference (covers all modes) ──────────────────────────

    def _recurrent(self, x, queries, keys, values, state, head_dim):
        """Token loop: fixed-size notebook S [K,B,H,d,d,2] — cost independent of past length."""
        batch_size, seq_len = x.shape[0], x.shape[1]
        num_heads, num_memory_states = self.num_heads, self.n_states
        query_scale = head_dim ** -0.5
        write_beta = None
        if self.write_mode == 'delta':
            write_beta = torch.sigmoid(self.beta_proj(cabs(x))).transpose(1, 2)  # [B,H,T]
        if self.n_states > 1:
            retrieval_phase, routing_weights = self._phase_and_alpha(x)
            retrieval_phase = retrieval_phase.permute(0, 2, 3, 1)                  # [B,H,K,T]
            routing_weights = routing_weights.permute(0, 2, 3, 1)
            self._route_aux = self._route_balance_loss(routing_weights.permute(0, 3, 1, 2))

        if state is None:
            if self.n_states > 1:
                memory_state = torch.zeros(
                    num_memory_states, batch_size, num_heads, head_dim, head_dim, 2,
                    device=x.device, dtype=x.dtype,
                )
            else:
                memory_state = torch.zeros(
                    batch_size, num_heads, head_dim, head_dim, 2,
                    device=x.device, dtype=x.dtype,
                )
        else:
            memory_state = state

        output_steps = []
        for time_idx in range(seq_len):
            token_input = x[:, time_idx:time_idx + 1]
            key_t = keys[:, :, time_idx]
            query_t = queries[:, :, time_idx] * query_scale
            value_t = values[:, :, time_idx]
            if self.n_states > 1:
                output_accum = None
                new_states = []
                for state_idx in range(num_memory_states):
                    decay_gamma_state, protected_values_state = self._gamma_and_vprime(
                        token_input, values[:, :, time_idx:time_idx + 1],
                        state_offset=self.state_dt_offset[state_idx],
                        state_idx=state_idx,
                    )
                    decay_gamma_t = decay_gamma_state[:, :, 0]  # [B,H]
                    # Outer-product write + S@Q read for this state.
                    output_state, state_new = self._recur_step_additive(
                        memory_state[state_idx], decay_gamma_t,
                        protected_values_state[:, :, 0], key_t, query_t,
                    )
                    # Phase-route: multiply read by α·e^{iφ} then sum over K states.
                    rotation_real = routing_weights[:, :, state_idx, time_idx] * torch.cos(
                        retrieval_phase[:, :, state_idx, time_idx]
                    )
                    rotation_imag = routing_weights[:, :, state_idx, time_idx] * torch.sin(
                        retrieval_phase[:, :, state_idx, time_idx]
                    )
                    rotation = stack_complex(rotation_real, rotation_imag)
                    output_state = cmul(output_state, rotation.unsqueeze(-2))
                    output_accum = output_state if output_accum is None else output_accum + output_state
                    new_states.append(state_new)
                output_steps.append(output_accum)
                memory_state = torch.stack(new_states, dim=0)
                continue

            decay_gamma, protected_values = self._gamma_and_vprime(
                token_input, values[:, :, time_idx:time_idx + 1]
            )
            decay_gamma_t = decay_gamma[:, :, 0]  # [B,H] or [B,H,d]
            protected_value_t = protected_values[:, :, 0]
            if self.write_mode == 'delta':
                output_step, memory_state = self._recur_step_delta(
                    memory_state, decay_gamma_t, protected_value_t, key_t, query_t,
                    write_beta[:, :, time_idx],
                )
            else:
                output_step, memory_state = self._recur_step_additive(
                    memory_state, decay_gamma_t, protected_value_t, key_t, query_t,
                )
            output_steps.append(output_step)

        output = torch.stack(output_steps, dim=2)
        return output, memory_state

    def _recur_step_additive(self, memory_state, decay_gamma, value_t, key_t, query_t):
        """One additive PAM step (inference / outer-product story).

        Natural notebook picture — same result as dual-form training, step-by-step:
          1. Forget: S ← γ · S
          2. Write:  S ← S + V ⊗ K*   where outer[i,j] = v[i] * conj(k)[j]
          3. Read:   y = S @ Q

        decay_gamma: [B,H] (winner head decay) or [B,H,d] (per-channel ablation).
        """
        # Conjugate of key: flip imag sign. unsqueeze inserts key-dim for outer product.
        key_conj = stack_complex(real_part(key_t), -imag_part(key_t)).unsqueeze(-3)
        outer_real = (
            real_part(value_t).unsqueeze(-1) * real_part(key_conj)
            - imag_part(value_t).unsqueeze(-1) * imag_part(key_conj)
        )
        outer_imag = (
            real_part(value_t).unsqueeze(-1) * imag_part(key_conj)
            + imag_part(value_t).unsqueeze(-1) * real_part(key_conj)
        )
        outer_product = stack_complex(outer_real, outer_imag)
        if decay_gamma.dim() == memory_state.dim() - 3:
            # Head-scalar γ → broadcast over d×d×2.
            decay_factor = decay_gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            # Per-channel γ → broadcast over key dim and complex axis.
            decay_factor = decay_gamma.unsqueeze(-2).unsqueeze(-1)
        memory_state = memory_state * decay_factor + outer_product
        # Complex matvec y = S @ q (sum over key dim).
        state_query_real = (
            real_part(memory_state) * real_part(query_t).unsqueeze(-2)
            - imag_part(memory_state) * imag_part(query_t).unsqueeze(-2)
        )
        state_query_imag = (
            real_part(memory_state) * imag_part(query_t).unsqueeze(-2)
            + imag_part(memory_state) * real_part(query_t).unsqueeze(-2)
        )
        output = stack_complex(state_query_real.sum(dim=-1), state_query_imag.sum(dim=-1))
        return output, memory_state

    def _recur_step_delta(self, memory_state, decay_gamma, value_t, key_t, query_t, write_beta_t):
        """One gated delta step. decay_gamma:[B,H], write_beta_t:[B,H]."""
        decay_factor = decay_gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        memory_state = memory_state * decay_factor
        predicted_real = (
            memory_state[..., 0] * key_t[..., 0].unsqueeze(-2)
            - memory_state[..., 1] * key_t[..., 1].unsqueeze(-2)
        ).sum(dim=-1)
        predicted_imag = (
            memory_state[..., 0] * key_t[..., 1].unsqueeze(-2)
            + memory_state[..., 1] * key_t[..., 0].unsqueeze(-2)
        ).sum(dim=-1)
        beta_expanded = write_beta_t.unsqueeze(-1)
        update_real = beta_expanded * (value_t[..., 0] - predicted_real)
        update_imag = beta_expanded * (value_t[..., 1] - predicted_imag)
        update = torch.stack([update_real, update_imag], dim=-1)
        key_conj = torch.stack([key_t[..., 0], -key_t[..., 1]], dim=-1)
        outer_real = (
            update[..., 0].unsqueeze(-1) * key_conj[..., 0].unsqueeze(-2)
            - update[..., 1].unsqueeze(-1) * key_conj[..., 1].unsqueeze(-2)
        )
        outer_imag = (
            update[..., 0].unsqueeze(-1) * key_conj[..., 1].unsqueeze(-2)
            + update[..., 1].unsqueeze(-1) * key_conj[..., 0].unsqueeze(-2)
        )
        memory_state = memory_state + torch.stack([outer_real, outer_imag], dim=-1)
        state_query_real = (
            memory_state[..., 0] * query_t[..., 0].unsqueeze(-2)
            - memory_state[..., 1] * query_t[..., 1].unsqueeze(-2)
        )
        state_query_imag = (
            memory_state[..., 0] * query_t[..., 1].unsqueeze(-2)
            + memory_state[..., 1] * query_t[..., 0].unsqueeze(-2)
        )
        output = torch.stack([state_query_real.sum(dim=-1), state_query_imag.sum(dim=-1)], dim=-1)
        return output, memory_state


@torch.compiler.disable
def _complex_triangular_solve(mass_real, mass_imag, write_real, write_imag, identity):
    """Solve (I + M) update = write for complex update, M strictly lower-tri.

    Eager-island: torch.linalg.solve hangs under torch.compile (Stage-6 / E2 revival).
    """
    chunk_len = mass_real.shape[-1]
    system_real = (identity + mass_real).float()
    system_imag = mass_imag.float()
    top = torch.cat([system_real, -system_imag], dim=-1)
    bot = torch.cat([system_imag, system_real], dim=-1)
    system_matrix = torch.cat([top, bot], dim=-2)
    rhs = torch.cat([write_real.float(), write_imag.float()], dim=-2)
    solution = torch.linalg.solve(system_matrix, rhs)
    update_real, update_imag = solution[..., :chunk_len, :], solution[..., chunk_len:, :]
    return update_real.to(write_real.dtype), update_imag.to(write_imag.dtype)


# ── V12 Block ────────────────────────────────────────────────────────────────

class V12Block(nn.Module):
    """Pre-norm residual: CGU (channel mix) + PAM (sequence mix)."""

    def __init__(self, cfg: V12Config, layer_idx: int = 0):
        super().__init__()
        # Pre-norm before CGU: stabilize magnitude; phase untouched.
        self.norm1 = ComplexNorm(cfg.dim)
        self.cgu = ComplexGatedUnit(cfg.dim, cfg.expand, activation=cfg.activation)
        self.cgu_scale = nn.Parameter(torch.tensor(1.0))
        self.cgu_dropout = nn.Dropout(cfg.dropout)
        # Pre-norm before PAM: same idea — PAM sees magnitude-stable inputs.
        self.norm2 = ComplexNorm(cfg.dim)
        self.pam = V12PAMLayer(cfg, layer_idx=layer_idx)
        # Start PAM residual weak (0.1) so early training is dominated by CGU;
        # deep stacks stay stable while memory pathways learn slowly.
        self.pam_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, pam_state=None, step_offset: int = 0):
        # Channel mix within each token (not across time).
        cgu_out = self.cgu(self.norm1(x))
        if self.training:
            # Same keep/drop on real+imag together (never drop only imag).
            drop = as_complex_dropout_mask(self.cgu_dropout, cgu_out)
            cgu_out = scale_complex(cgu_out, drop)
        x = x + cgu_out * self.cgu_scale
        # Sequence/memory mix via fixed-size PAM state.
        pam_out, new_state = self.pam(self.norm2(x), state=pam_state, step_offset=step_offset)
        x = x + pam_out * self.pam_scale
        return x, new_state


# ── V12 Language Model ──────────────────────────────────────────────────────

class V12LM(nn.Module):
    """ComplexEmbed -> [V12Block] x N -> tied complex LM head."""

    def __init__(self, cfg: V12Config):
        super().__init__()
        self.config = cfg
        self.embed = ComplexEmbed(cfg.vocab_size, cfg.dim)
        self.pos_embed = (
            ComplexPosEmbed(cfg.max_seq_len, cfg.dim) if cfg.use_learned_pos else None
        )
        self.embed_norm = ComplexNorm(cfg.dim)
        # Spec-driven (non-uniform) stack for the depth-growth curriculum, else the
        # uniform n_layers stack (bit-identical to the pre-M4 build).
        if cfg.layer_specs:
            self.blocks = nn.ModuleList([
                V12Block(_layer_cfg(cfg, spec), layer_idx=i)
                for i, spec in enumerate(cfg.layer_specs)
            ])
        else:
            self.blocks = nn.ModuleList([V12Block(cfg, layer_idx=i) for i in range(cfg.n_layers)])
        # Final magnitude stabilize after residual stack, before readout.
        self.output_norm = ComplexNorm(cfg.dim)
        # Small complex feature mix (not a second memory) + norm before vocab scores.
        self.lm_head_proj = ComplexLinear(cfg.dim, cfg.dim)
        self.lm_head_norm = ComplexNorm(cfg.dim)
        self._init_weights()
        # Materialize an explicit per-layer manifest (source of truth for growth /
        # composition). For a uniform base this records one 'base' spec per layer;
        # module init above is untouched, so the state_dict stays identical.
        self.config.layer_specs = self._materialize_specs()
        self.config.n_layers = len(self.blocks)
        self._refresh_layer_meta()

    @staticmethod
    def _init_module_tree(root: nn.Module, skip_embeddings=frozenset()):
        """Apply the standard V12 init to every submodule of ``root``.

        Shared by full-model init and by ``grow_layers`` so appended blocks get
        the same init recipe (normal(0.02) linears, zeroed biases, then the custom
        protect-gate bias and phase_proj re-init).
        """
        for module in root.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding) and module not in skip_embeddings:
                nn.init.normal_(module.weight, std=0.02)
        for _, module in root.named_modules():
            if hasattr(module, 'protect_gate') and isinstance(module.protect_gate, nn.Linear):
                nn.init.constant_(module.protect_gate.bias, getattr(module, 'protect_gate_bias', -3.0))
            if isinstance(module, V12PAMLayer) and module.n_states > 1:
                module._init_phase_proj()

    def _init_weights(self):
        embed_embeddings = {self.embed.embed_real, self.embed.embed_imag}
        self._init_module_tree(self, skip_embeddings=embed_embeddings)

    # ── M4: depth-growth manifest / composition metadata ──────────────────────

    def _materialize_specs(self) -> List[dict]:
        """Return one normalized spec per current block (source of truth)."""
        raw = self.config.layer_specs
        if raw:
            return [_normalize_spec(s) for s in raw]
        return [_normalize_spec({'group_id': 'base', 'stage': 0}) for _ in self.blocks]

    def _refresh_layer_meta(self):
        """Cache per-block attach_mode / group_id from the materialized manifest."""
        specs = self.config.layer_specs or []
        n = len(self.blocks)
        self._attach_modes = [
            (specs[i].get('attach_mode', 'sequential') if i < len(specs) else 'sequential')
            for i in range(n)
        ]
        self._group_ids = [
            (specs[i].get('group_id') if i < len(specs) else None) for i in range(n)
        ]

    @torch.no_grad()
    def _hash_blocks(self, indices) -> str:
        """SHA-256 over the (name-sorted) params of ``indices`` blocks.

        Used to stamp a grown group's ``substrate_hash`` = the frozen substrate it
        was built on, so a checkpoint records exactly what each group depends on.
        """
        h = hashlib.sha256()
        for i in sorted(indices):
            block = self.blocks[i]
            for name, p in sorted(block.named_parameters()):
                h.update(name.encode())
                h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def layer_manifest(self) -> List[dict]:
        """Structured per-layer manifest (specs + positional layer_idx)."""
        specs = self.config.layer_specs or self._materialize_specs()
        manifest = []
        for i in range(len(self.blocks)):
            entry = dict(specs[i]) if i < len(specs) else _normalize_spec({})
            entry = _normalize_spec(entry)
            entry['layer_idx'] = i
            manifest.append(entry)
        return manifest

    def grow_layers(self, specs: List[dict], *, stamp_substrate: bool = True) -> List[int]:
        """Append new blocks described by ``specs`` on top of the current stack.

        Existing block indices are preserved so a prior state_dict still loads by
        position. New blocks are freshly initialized and moved to the model device.
        Each new spec is stamped with ``substrate_hash`` = hash of the (frozen)
        pre-existing prefix it was grown on. Returns the new block indices.
        """
        start = len(self.blocks)
        substrate = self._hash_blocks(range(start)) if (stamp_substrate and start) else None
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
        new_indices, new_specs = [], []
        for offset, spec in enumerate(specs):
            idx = start + offset
            block = V12Block(_layer_cfg(self.config, spec), layer_idx=idx)
            self._init_module_tree(block)
            block.to(device)
            self.blocks.append(block)
            norm = _normalize_spec(spec)
            if substrate is not None and 'substrate_hash' not in spec:
                norm['substrate_hash'] = substrate
            new_specs.append(norm)
            new_indices.append(idx)
        self.config.layer_specs = list(self.config.layer_specs or self._materialize_specs()) + new_specs
        self.config.n_layers = len(self.blocks)
        self._refresh_layer_meta()
        return new_indices

    def freeze_layers(self, indices, *, frozen: bool = True):
        """Freeze (or unfreeze) whole blocks by index.

        Uses ``requires_grad_`` so frozen blocks are excluded from the optimizer
        param groups entirely (no weight-decay drift). Call BEFORE building the
        trainer/optimizer. Also records ``frozen`` in the manifest.
        """
        idx = {int(i) for i in indices}
        specs = self.config.layer_specs
        for i, block in enumerate(self.blocks):
            if i in idx:
                for p in block.parameters():
                    p.requires_grad_(not frozen)
                if specs and i < len(specs):
                    specs[i]['frozen'] = frozen

    @staticmethod
    def _collect_route_aux(blocks) -> torch.Tensor:
        total = None
        for block in blocks:
            aux = getattr(block.pam, '_route_aux', None)
            if aux is not None:
                total = aux if total is None else total + aux
        if total is None:
            return None
        return total

    @staticmethod
    def _collect_gate_probs(blocks):
        """Stack per-layer mean protect-prob [B,T] into [L,B,T] (or None).

        Grad flows to each layer's protect_gate; the gate-surprisal loss is
        computed in the trainer against a detached per-token surprisal target.
        """
        probs = []
        for block in blocks:
            gp = getattr(block.pam, '_gate_prob_bt', None)
            if gp is not None:
                probs.append(gp)
        if not probs:
            return None
        return torch.stack(probs, dim=0)

    def forward(self, input_ids, states=None, step_offset: int = 0, labels=None):
        # Token ids → complex vectors [B,T,dim,2].
        z = self.embed(input_ids)
        if self.pos_embed is not None:
            z = self.pos_embed(z, step_offset=step_offset)
        z = self.embed_norm(z)
        use_ckpt = self.config.gradient_checkpointing and self.training and states is None
        z, new_states = self._run_blocks(z, states, step_offset, use_ckpt)
        # Stabilize → mix features → stabilize again before tied embedding scores.
        z = self.output_norm(z)
        lm = self.lm_head_norm(self.lm_head_proj(z))
        # Tied head: reuse embed weights; real and imag contribute then add.
        logits = (
            real_part(lm) @ self.embed.embed_real.weight.T
            + imag_part(lm) @ self.embed.embed_imag.weight.T
        )
        route_aux = self._collect_route_aux(self.blocks)
        aux_loss = (
            route_aux
            if route_aux is not None
            else torch.tensor(0.0, device=input_ids.device)
        )
        return logits, new_states, aux_loss

    def _hidden_to_lm(self, input_ids, step_offset: int = 0):
        """Training path: stack + head norm, stop before full [B,T,V] logits.

        Fused CE scores vocab in chunks from `lm` so we never materialize the
        full logit tensor (memory win on long sequences / large vocab).
        """
        z = self.embed(input_ids)
        if self.pos_embed is not None:
            z = self.pos_embed(z, step_offset=step_offset)
        z = self.embed_norm(z)
        use_ckpt = self.config.gradient_checkpointing and self.training
        z, _ = self._run_blocks(z, None, step_offset, use_ckpt)
        z = self.output_norm(z)
        lm = self.lm_head_norm(self.lm_head_proj(z))
        route_aux = self._collect_route_aux(self.blocks)
        aux_loss = (
            route_aux
            if route_aux is not None
            else torch.tensor(0.0, device=input_ids.device)
        )
        gate_probs = self._collect_gate_probs(self.blocks)  # [L,B,T] or None
        return lm, aux_loss, gate_probs

    def ce_from_lm(self, lm, labels, loss_mask=None, ignore_index=-100, chunk: int = 4096):
        """Chunked cross-entropy from pre-logit complex hidden `lm` [B,T,dim,2].

        The tied head `lm_r @ E_r.T + lm_i @ E_i.T` folds into one real matmul
        H @ W.T with H=concat(lm_r,lm_i), W=concat(E_r,E_i); chunked-CE never holds
        the full [N,vocab] logits/softmax. Kept eager (compile the stack, not this).
        """
        from v12.fused_ce import fused_linear_cross_entropy
        batch_size, seq_len = labels.shape
        hidden_concat = torch.cat([real_part(lm), imag_part(lm)], dim=-1).reshape(batch_size * seq_len, -1)
        weight_concat = torch.cat([self.embed.embed_real.weight, self.embed.embed_imag.weight], dim=-1)
        mask = loss_mask.reshape(-1) if loss_mask is not None else None
        return fused_linear_cross_entropy(
            hidden_concat, weight_concat, labels.reshape(-1), mask=mask,
            chunk=chunk, ignore_index=ignore_index,
        )

    def fused_ce_loss(self, input_ids, labels, loss_mask=None, ignore_index=-100,
                      chunk: int = 4096):
        """Convenience eager path: hidden stack + chunked CE (exact == forward+CE)."""
        lm, aux_loss, _gate_probs = self._hidden_to_lm(input_ids)
        main = self.ce_from_lm(lm, labels, loss_mask=loss_mask,
                               ignore_index=ignore_index, chunk=chunk)
        return main, aux_loss

    @staticmethod
    def _ckpt_block(block, z, step_offset):
        def run(z_in):
            return block(z_in, pam_state=None, step_offset=step_offset)
        return grad_checkpoint(run, z, use_reentrant=False)

    # ── M4: attach-mode-aware block composition ───────────────────────────────

    def _apply_block(self, i, z, states, step_offset, use_ckpt):
        """Run one block; checkpoint only on the stateless (training) path."""
        block = self.blocks[i]
        s = states[i] if states is not None else None
        if use_ckpt and s is None:
            return self._ckpt_block(block, z, step_offset)
        return block(z, pam_state=s, step_offset=step_offset)

    def _apply_moe_group(self, indices, z, states, step_offset, use_ckpt):
        """Composition hook for a contiguous attach_mode='moe' group.

        TODO(moe-router): learn a gate that activates only the relevant expert
        block(s) per token so a large specialist library stays cheap. Until then
        MoE groups run always-on sequentially — identical to depth stacking — so
        the schema is forward-compatible and behavior is unchanged.
        """
        gstates = []
        for i in indices:
            z, s = self._apply_block(i, z, states, step_offset, use_ckpt)
            gstates.append(s)
        return z, gstates

    def _run_blocks(self, z, states, step_offset, use_ckpt):
        """Apply the block stack with attach-mode-aware composition.

        Default (all-sequential) is bit-identical to a plain per-block loop.
        Contiguous 'moe' groups (same group_id) route through _apply_moe_group.
        Returns (z, new_states) with one state entry per block.
        """
        modes = getattr(self, '_attach_modes', None) or ['sequential'] * len(self.blocks)
        groups = getattr(self, '_group_ids', None) or [None] * len(self.blocks)
        n = len(self.blocks)
        new_states = [None] * n
        i = 0
        while i < n:
            if modes[i] == 'moe':
                j, gid = i, groups[i]
                while j < n and modes[j] == 'moe' and groups[j] == gid:
                    j += 1
                z, gstates = self._apply_moe_group(range(i, j), z, states, step_offset, use_ckpt)
                for k, s in zip(range(i, j), gstates):
                    new_states[k] = s
                i = j
            else:
                z, new_states[i] = self._apply_block(i, z, states, step_offset, use_ckpt)
                i += 1
        return z, new_states

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0,
                 top_k=50, top_p=0.0, repetition_penalty=1.0, eos_token_id=None):
        """Autoregressive decode: one full forward builds PAM states, then O(1)/token.

        First call runs the parallel/chunked path over the prompt and returns
        per-layer memory states. Each new token is forwarded alone with those
        states + step_offset (RoPE / position) so cost stays fixed in context length.
        """
        self.eval()
        generated = input_ids.clone()
        # Prefill: build logits for the whole prompt and initialize PAM notebooks.
        logits, states, _ = self.forward(generated)
        step = generated.shape[1]
        finished = torch.zeros(generated.shape[0], dtype=torch.bool, device=generated.device)
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1] / temperature
            if repetition_penalty != 1.0:
                score = torch.gather(next_logits, 1, generated)
                score = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
                next_logits.scatter_(1, generated, score)
            if top_k > 0:
                v, _ = next_logits.topk(min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, -1:]] = float('-inf')
            if top_p > 0:
                sl, si = next_logits.sort(descending=True)
                cum = sl.softmax(dim=-1).cumsum(dim=-1)
                rm = cum - sl.softmax(dim=-1) >= top_p
                sl[rm] = float('-inf')
                next_logits = sl.scatter(1, si, sl)
            nxt = torch.multinomial(next_logits.softmax(dim=-1), 1)
            generated = torch.cat([generated, nxt], dim=1)
            if eos_token_id is not None:
                finished |= nxt.squeeze(1) == eos_token_id
                if bool(finished.all()):
                    break
            # One-token recurrent step: update each layer's fixed-size state.
            logits, states, _ = self.forward(nxt, states=states, step_offset=step)
            step += 1
        return generated

    def set_stage_active_heads(self, active_indices, *, open_indices=None,
                               freeze_embeddings=False, freeze_lm_head=False,
                               freeze_cgu=False):
        """M2 curriculum: make only ``active_indices`` head slots trainable across
        all PAM layers; optionally freeze embeddings / LM head / CGU channel mixers.

        ``open_indices`` (default = all past+current, i.e. 0..max(active)) hard-opens
        those slots and closes the rest so reserved future heads stay silent.

        Growth recipe: stage A trains+opens slots [0..a); stage B calls
        ``set_stage_active_heads(range(a, b), open_indices=range(0, b))`` so only the
        new slots [a..b) learn on top of the frozen, still-contributing [0..a).
        Use ``weight_decay=0`` on frozen stages (decoupled AdamW would otherwise
        shrink the frozen zero-grad slices).
        """
        active_list = list(active_indices)
        if open_indices is None:
            hi = (max(active_list) + 1) if active_list else 0
            open_indices = range(0, hi)
        for block in self.blocks:
            block.pam.set_open_heads(open_indices)
            block.pam.set_trainable_heads(active_list)
        if freeze_embeddings:
            for p in self.embed.parameters():
                p.requires_grad_(False)
        if freeze_lm_head:
            for module in (self.lm_head_proj, self.lm_head_norm):
                for p in module.parameters():
                    p.requires_grad_(False)
        if freeze_cgu:
            for block in self.blocks:
                for p in block.cgu.parameters():
                    p.requires_grad_(False)

    @torch.no_grad()
    def head_gate_report(self, threshold: float = 1e-3):
        """M1 observability: per-layer effective head count + gate values.

        Returns {'per_layer_active': [...], 'total_active': int, 'max_heads': int,
        'gates': [[z per head] per layer]} or None if head gating is disabled.
        """
        layers = [b.pam for b in self.blocks if getattr(b.pam, 'head_gate_enabled', False)]
        if not layers:
            return None
        gates = [pam.head_gate.gate().detach().cpu().tolist() for pam in layers]
        per_layer_active = [int(pam.head_gate.active_mask(threshold).sum()) for pam in layers]
        return {
            'per_layer_active': per_layer_active,
            'total_active': int(sum(per_layer_active)),
            'max_heads': self.config.n_heads * len(layers),
            'gates': gates,
        }

    def count_parameters(self) -> Dict[str, int]:
        embed_p = sum(p.numel() for p in self.embed.parameters())
        if self.pos_embed is not None:
            embed_p += sum(p.numel() for p in self.pos_embed.parameters())
        block_p = sum(p.numel() for b in self.blocks for p in b.parameters())
        head_p = (sum(p.numel() for p in self.lm_head_proj.parameters())
                  + sum(p.numel() for p in self.lm_head_norm.parameters()))
        norm_p = (sum(p.numel() for p in self.embed_norm.parameters())
                  + sum(p.numel() for p in self.output_norm.parameters()))
        total = embed_p + block_p + head_p + norm_p
        return {
            'embedding (tied)': embed_p, 'blocks': block_p,
            'norms': norm_p, 'lm_head': head_p, 'total': total,
        }


# ── Presets ───────────────────────────────────────────────────────────────────

def _base_flat(**kw) -> V12Config:
    cfg = V12Config(
        vocab_size=50257, dim=384, n_heads=6, head_dim=64,
        n_layers=16, expand=3, dropout=0.1, max_seq_len=2048,
        activation='swish', chunk_size=256,
    )
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


PRESETS = {
    # ── Lean core (carried forward from V11 proven winners) ──────────────────
    # Baseline == single-state control (K=1). Reproduces the V7 7d line (~26.88).
    'v12_baseline': _base_flat(),
    # Proven core: E3 K=3 multistate + phase-aware GSP gate (V11 best, PPL 25.77).
    'v12_e3_k3': _base_flat(n_states=3, state_dt_spread=2.0, gate_content_aware=True),
    # Core + ChatML/reasoning vocab (50261) — pretrain/SFT base preset.
    'v12_e3_k3_chat': _base_flat(
        n_states=3, state_dt_spread=2.0, vocab_size=50261, gate_content_aware=True,
    ),
    # Recall arm: core + gate-surprisal supervision at the best swept hypers
    # (GSL=0.3, GST=0.5). gamma_floor stays OFF (it cost PPL in V11).
    'v12_e3_k3_recall': _base_flat(
        n_states=3, state_dt_spread=2.0, vocab_size=50261, gate_content_aware=True,
        gate_surprisal_lambda=0.3, gate_surprisal_tau=0.5, gate_surprisal_sign=1.0,
    ),
    # ── M1: learnable phase-band heads ───────────────────────────────────────
    # Same geometry as v12_e3_k3 (6 slots) but with L0 head gates: tests whether
    # the model prunes any of the 6 heads at the proven configuration.
    'v12_e3_k3_headgate': _base_flat(
        n_states=3, state_dt_spread=2.0, gate_content_aware=True,
        head_gate=True, head_gate_l0_lambda=0.001,
    ),
    # Over-provisioned budget: 10 head slots, let L0 discover the effective count.
    'v12_headgate_hmax10': _base_flat(
        n_heads=10, head_dim=64, n_states=3, state_dt_spread=2.0,
        gate_content_aware=True, head_gate=True, head_gate_l0_lambda=0.001,
    ),
    # M4.4: high max-head budget (16 slots). L0 prunes down; v12.compact then drops
    # the closed slots into a slim inference checkpoint (per-layer n_heads).
    'v12_headgate_hmax16': _base_flat(
        n_heads=16, head_dim=64, n_states=3, state_dt_spread=2.0,
        vocab_size=50261, gate_content_aware=True,
        head_gate=True, head_gate_l0_lambda=0.001,
    ),
    # ── M2: progressive frozen-head growth ───────────────────────────────────
    # 10 head slots, L0 OFF (staging controls open/frozen slots explicitly via
    # --active_heads / --open_heads). Used by scripts/run_v12_stage.sh.
    'v12_grow10': _base_flat(
        n_heads=10, head_dim=64, n_states=3, state_dt_spread=2.0,
        vocab_size=50261, gate_content_aware=True,
    ),
    # ── M4: depth-growth curriculum ──────────────────────────────────────────
    # Small grammar base (few layers, little data) that later stages freeze and
    # grow specialist layer groups on top of (see scripts/run_v12_stage.sh).
    'v12_base_grammar': _base_flat(
        n_layers=4, n_states=3, state_dt_spread=2.0, vocab_size=50261,
        gate_content_aware=True,
    ),
    # Playable module system: dynamic-head grammar base. Few layers, high H_max
    # (16 slots) with L0 head gates + per-head phase-band addressing, so the head
    # count AND each head's phase map are LEARNED for grammar, then v12.compact
    # drops the closed slots. This is the role=base module the curriculum grows
    # fact_retrieval / reasoning groups on top of.
    'v12_grammar_dyn': _base_flat(
        n_layers=4, n_heads=16, head_dim=64, n_states=3, state_dt_spread=2.0,
        vocab_size=50261, gate_content_aware=True,
        head_gate=True, head_gate_l0_lambda=0.001, write_phase_address=True,
    ),

    # ── M3: low-interference fact writes ─────────────────────────────────────
    # Core + a vault (no-decay) memory state for long-horizon facts + per-head
    # phase-band addressing to reduce write cross-talk. This is the recall-oriented
    # architecture (attacks the additive-superposition interference ceiling).
    'v12_factband': _base_flat(
        n_states=3, state_dt_spread=2.0, vocab_size=50261, gate_content_aware=True,
        vault_state=True, vault_state_idx=0, write_phase_address=True,
        gate_surprisal_lambda=0.3, gate_surprisal_tau=0.5,
    ),

    # ── Local dev / smoke (RTX-4090 friendly; small enough for CPU selftest) ──
    'tiny': V12Config(
        vocab_size=50257, dim=64, n_heads=2, head_dim=32, n_layers=2,
        expand=2, dropout=0.0, max_seq_len=512, chunk_size=64,
        gradient_checkpointing=False,
    ),
    'tiny_e3': V12Config(
        vocab_size=50257, dim=64, n_heads=2, head_dim=32, n_layers=2,
        expand=2, dropout=0.0, max_seq_len=512, chunk_size=64,
        gradient_checkpointing=False, n_states=3,
    ),
    # Dev config for the M3 write-mechanism work (delta write kept as starting point).
    'tiny_delta': V12Config(
        vocab_size=50257, dim=64, n_heads=2, head_dim=32, n_layers=2,
        expand=2, dropout=0.0, max_seq_len=512, chunk_size=64,
        gradient_checkpointing=False, write_mode='delta', delta_chunk=32,
    ),
    # ~11M capacity micro (chat vocab) for fast local recall-substrate iteration.
    'v12_micro': V12Config(
        vocab_size=50261, dim=96, n_heads=3, head_dim=32, n_layers=6,
        expand=3, dropout=0.0, max_seq_len=2048, chunk_size=64,
        gradient_checkpointing=False, n_states=3, state_dt_spread=2.0,
        gate_content_aware=True,
    ),
}


def get_config(preset: str = 'v12_baseline') -> V12Config:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}")
    return copy.deepcopy(PRESETS[preset])
