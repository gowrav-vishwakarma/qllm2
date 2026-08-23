"""
V13 model: selective PAM — V11 core + delta/vault/phase defaults (E1/E2/E3).

Reuses the stable V7 complex primitives (ComplexLinear, ComplexNorm, CGU,
ModSwish/ModReLU, ComplexEmbed, RoPE) via vendored `v13.complex_ops`. The only
lives in `V13PAMLayer`, which dispatches on three flags:

    decay_mode : 'head'        -> per-head scalar decay (V7 baseline)
                 'per_channel'  -> per-key-channel decay (E1, GLA-style fold)
    write_mode : 'additive'    -> S += V (x) K*           (V7 baseline)
                 'delta'        -> error-correcting write  (E2, UT transform)
    n_states   : 1             -> single matrix state      (V7 baseline)
                 K>1           -> superposed states, phase-routed retrieval (E3)

All paths expose:
  * a parallel training form (dual / chunked / UT), and
  * an O(1) recurrent inference form,
and the two are numerically verified to agree (see v11/selftest.py).

Complex representation: split-real `[..., dim, 2]`. Never torch.complex64/128.
"""

import math
import os
import copy
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

# Vendored V7 complex primitives — V11 no longer imports v7.model on the forward path.
from v13.complex_ops import (
    real_part, imag_part, stack_complex, scale_complex, as_complex_dropout_mask,
    cmul, cconj, cabs, cnormalize, cnormalize_vec, to_real_concat,
    ComplexLinear, ComplexNorm, ComplexEmbed, ComplexPosEmbed,
    ComplexGatedUnit, build_rope_cache, _build_activation,
)
from v13.triton_kernels import fused_decay_matrix

# Shape-debug: V13_DEBUG=1 prints intermediate tensor shapes in the hot PAM paths
# (default off; zero cost when unset). Use it to check layouts instead of
# re-deriving them by hand.
_V13_DEBUG = os.environ.get('V13_DEBUG', '0') == '1'


# ── Config ────────────────────────────────────────────────────────────────────

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
    #   Why: with the shared beta, the erase term beta*pred leaks protected memory
    #   at ~beta/d per filler token (measured: facts dead by ~500 tokens), which is
    #   the V13 recall killer. Evidence: v13/experiments/mem_dynamics.py +
    #   v13/experiments/layer_equivalence.py (check C).
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
    aux_loss_weight: float = 1.0        # trainer weight for route_balance aux (v7.train hook)
    fused_e3: bool = True               # E3: fused multistate path (exact-equiv, K-independent matmuls)
    # delta_key_norm: per-vector (across head_dim) unit-norm keys for the delta
    # rule. The k-direction eigenvalue of S <- g*S + (b_w*v - b_e*k@S)*k^H is
    # g*(1 - b_e*||k||^2); with the vault state pinned to g=1, unnormalized
    # keys (||k||^2 ~ d) flip this past -1 -> positive feedback -> unbounded
    # state growth -> NaN (500M run died at ~57M). Unit keys make it a strict
    # contraction in [1-b_e, 1) for any g in [0,1]. Applied in _project (after
    # RoPE + phase addressing), so the write AND read paths see unit keys.
    # NOTE: qk_norm (per-element cnormalize) is NOT sufficient (||k||^2 -> d).
    delta_key_norm: bool = True
    # delta_erase_beta_cap: clamp the learned erase gain beta_e to this max
    # (0 = off). The k-direction eigenvalue of the vault delta state is
    #   eig = gamma - beta_e * ||k||^2  ==  1 - beta_e   (vault gamma=1, key-norm ||k||^2=1)
    # which is a stable contraction only while beta_e < 2; once the learned
    # erase_beta_proj trains beta_e past 2 the eigenvalue flips past -1 ->
    # oscillatory blowup -> unbounded state -> NaN (500M trainer died ~step 151,
    # a DIFFERENT weight init than the diagnostic, which stayed beta_e<2 and
    # passed 2000 steps). Capping beta_e <= 0.95 keeps eig in [0.05, 1) for ANY
    # init: the erase strength is still learned (sigmoid range 0.047..0.95), only
    # the pathological >2 overshoot is removed. Complements delta_key_norm.
    delta_erase_beta_cap: float = 0.95
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
    gate_surprisal_lambda: float = 0.1
    gate_surprisal_tau: float = 1.0     # temperature (nats) mapping surprisal->target protect prob
    # gate_surprisal_sign: +1 => LOW-surprisal (filler) tokens get HIGH protect target
    #   (freeze state through filler, write on content). This is the recall-oriented
    #   direction and drives (p_content - p_filler) NEGATIVE. -1 flips it to the
    #   probe's "protect content more" convention. Default +1 optimizes for recall.
    gate_surprisal_sign: float = 1.0
    # Stage-6 architecture levers (defaults OFF = bit-identical to prior behaviour).
    # vault_state: pin one of the K states to γ≈1 (no decay); writes still GSP-gated.
    vault_state: bool = True
    vault_state_idx: int = 0
    # write_phase_address: key-conditioned write phase + matching query phase on read.
    write_phase_address: bool = True


# ── Phase-Associative Memory (V11) ──────────────────────────────────────────

class V13PAMLayer(nn.Module):
    r"""Matrix-state memory with complex-conjugate retrieval and pluggable dynamics.

    Baseline:  S_t = gamma_t * S_{t-1} + V_t (x) K_t^* ;  Y_t = S_t * Q_t
    E1:        gamma_t becomes per-key-channel (vector decay).
    E2:        write becomes delta-rule (erase stale assoc for K_t before write).
    E3:        K states with distinct decay; retrieval = sum_k e^{i phi_k} S_k Q.
    """

    def __init__(self, cfg: V13Config, layer_idx: int = 0):
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
        self.delta_key_norm = getattr(cfg, 'delta_key_norm', False)
        self.delta_erase_beta_cap = getattr(cfg, 'delta_erase_beta_cap', 0.0)
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
        self.write_phase_address = getattr(cfg, 'write_phase_address', False)
        if self.write_phase_address:
            # Map per-channel key/query magnitude → scalar phase angle (radians).
            self.write_phase_proj = nn.Linear(cfg.head_dim, 1, bias=True)
            nn.init.zeros_(self.write_phase_proj.weight)
            nn.init.zeros_(self.write_phase_proj.bias)
        if cfg.use_gsp:
            gate_in = cfg.dim * 2 if self.gate_content_aware else cfg.dim
            self.protect_gate = nn.Linear(gate_in, cfg.n_heads)
            nn.init.constant_(self.protect_gate.bias, self.protect_gate_bias)

        # E2: delta-rule write strength beta_t in (0, 1) per head.
        if cfg.write_mode == 'delta':
            self.beta_proj = nn.Linear(cfg.dim, cfg.n_heads)
            nn.init.constant_(self.beta_proj.bias, 0.0)
            # E2b: separate learned erase gate. Low init -> starts additive-like;
            # the model learns to erase (overwrite) only for repeat/update tokens.
            if cfg.delta_erase_gate:
                self.erase_beta_proj = nn.Linear(cfg.dim, cfg.n_heads)
                nn.init.constant_(self.erase_beta_proj.bias, -3.0)
            else:
                self.erase_beta_proj = None

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

    def _gate_betas(self, x):
        """Write/erase gates, each [B,H,T]. Erase falls back to write (legacy).

        The erase gain is clamped to `delta_erase_beta_cap` (0 = off) so the
        vault delta eigenvalue (1 - beta_e, with key-norm ||k||^2=1) stays a
        stable contraction for any learned init (see V13Config comment).
        """
        write_beta = torch.sigmoid(self.beta_proj(cabs(x))).transpose(1, 2)
        if self.erase_beta_proj is None:
            return write_beta, write_beta
        erase_beta = torch.sigmoid(self.erase_beta_proj(cabs(x))).transpose(1, 2)
        if self.delta_erase_beta_cap and self.delta_erase_beta_cap < 1.0:
            erase_beta = erase_beta.clamp(max=self.delta_erase_beta_cap)
        return write_beta, erase_beta

    def _apply_gamma_floor(self, base_decay: torch.Tensor) -> torch.Tensor:
        """Lift the base (pre-GSP) decay onto [gamma_floor, 1) to lengthen memory.

        base_decay = exp(-softplus_dt) in (0,1); reparam keeps the learned shape
        but caps the minimum retention so unprotected state survives far longer.
        """
        if self.gamma_floor and self.gamma_floor > 0.0:
            return self.gamma_floor + (1.0 - self.gamma_floor) * base_decay
        return base_decay

    def _init_phase_proj(self):
        """Custom init for phase_proj (re-applied after V13LM._init_weights)."""
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

        # Delta-rule stability: unit-norm keys across head_dim (see
        # V13Config.delta_key_norm). Done after RoPE/phase addressing so the
        # phase gate sees the same key content it did pre-fix; only magnitude
        # is removed, which the delta mass term needs bounded.
        if self.write_mode == 'delta' and self.delta_key_norm:
            keys = cnormalize_vec(keys)
        return queries, keys, values

    def _apply_write_phase_address(self, queries, keys, values):
        """Rotate values by e^{iψ(k)} and queries by e^{iψ(q)}; ψ = Linear(|·|)."""
        # cabs layout: [B,H,T,d]
        key_phase = self.write_phase_proj(cabs(keys)).squeeze(-1)      # [B,H,T]
        query_phase = self.write_phase_proj(cabs(queries)).squeeze(-1)
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
            # (gate-surprisal aux stash moved to V13LM._collect_gate_probs,
            # built from the block's detached gate input OUTSIDE the
            # gradient-checkpoint region — see V13Block.forward.)
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

    def _forward_delta(self, queries, keys, protected_values, decay_gamma, write_beta, head_dim,
                       erase_beta=None):
        """Gated delta rule via per-chunk UT transform. decay_gamma: [B,H,T] head scalar.

        E2b: erase_beta [B,H,T] (default = write_beta, legacy) weights the UT mass
        matrix (the erase), write_beta weights the write RHS. Splitting the two kills
        the isotropic self-erosion leak while keeping exact closed form + dedup.
        """
        if erase_beta is None:
            erase_beta = write_beta
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
            erase_beta_chunk = erase_beta[:, :, chunk_start:chunk_end]     # [B,H,Tc]

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
            # E2b: the mass matrix encodes A_t = I - beta_e*kk^H -> erase gate.
            mass_real = erase_beta_chunk.unsqueeze(-1) * decay_masked * key_gram_real
            mass_imag = erase_beta_chunk.unsqueeze(-1) * decay_masked * key_gram_imag

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
                # E2b: the cross-chunk erase (state_key term) is gated by the ERASE
                # gate; only the fresh write v_t is gated by the write gate.
                write_real = (write_beta_chunk.unsqueeze(-1) * value_real
                              - erase_beta_chunk.unsqueeze(-1) * state_key_real)
                write_imag = (write_beta_chunk.unsqueeze(-1) * value_imag
                              - erase_beta_chunk.unsqueeze(-1) * state_key_imag)
            else:
                write_real = write_beta_chunk.unsqueeze(-1) * value_real
                write_imag = write_beta_chunk.unsqueeze(-1) * value_imag

            update_real, update_imag = _complex_triangular_solve(
                mass_real, mass_imag, write_real, write_imag, identity[:chunk_len, :chunk_len]
            )

            query_key_real = query_real @ key_real.transpose(-1, -2) + query_imag @ key_imag.transpose(-1, -2)
            query_key_imag = query_imag @ key_real.transpose(-1, -2) - query_real @ key_imag.transpose(-1, -2)
            causal_inclusive = torch.tril(
                torch.ones(chunk_len, chunk_len, device=queries.device, dtype=decay_matrix.dtype)
            )
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
        write_beta = None
        erase_beta = None
        if self.write_mode == 'delta':
            write_beta, erase_beta = self._gate_betas(x)  # [B,H,T] each (E2b)
        for state_idx in range(num_memory_states):
            decay_gamma_state, protected_values_state = self._gamma_and_vprime(
                x, protected_values, state_offset=self.state_dt_offset[state_idx],
                state_idx=state_idx,
            )
            if self.write_mode == 'delta':
                output_state, memory_state = self._forward_delta(
                    queries, keys, protected_values_state, decay_gamma_state,
                    write_beta, head_dim, erase_beta=erase_beta,
                )
            elif self.decay_mode == 'per_channel':
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
            # (gate-surprisal aux stash moved to V13LM._collect_gate_probs,
            # built from the block's detached gate input OUTSIDE the
            # gradient-checkpoint region — see V13Block.forward.)
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

    # ── E2+E3 fused-delta: K states collapsed into ONE batched chunk-solve ──
    #
    # Exact regrouping of `_forward_multistate`'s per-state `_forward_delta`
    # (head decay, delta write). The three facts that make it work:
    #   * The key-gram, query-key scores, decay-matrix layout, and the (I+M)
    #     unit-triangular solve are all INDEPENDENT per state, so the K per-chunk
    #     solves collapse into ONE batched solve over K·B·H.  K·C sequential UT
    #     steps -> C steps (96 -> 24 at K=3, C=32; 24 -> 8 at C=8).
    #   * The GSP protect gate and the write/erase betas are state-independent,
    #     so they are computed once (not K times) — same as `_gamma_all_and_vprime`.
    #   * The phase-rotate of each state's read is elementwise, so it batches over
    #     K without a Python loop.
    # Bit-identical to the K-loop (same ops, batched on the state dim).

    def _forward_multistate_delta_fused(self, x, queries, keys, values, head_dim):
        batch_size, seq_len = x.shape[0], x.shape[1]
        num_heads, K = self.num_heads, self.n_states
        chunk_size = self.delta_chunk if self.delta_chunk > 0 else seq_len
        query_scale = head_dim ** -0.5
        retrieval_phase, routing_weights = self._phase_and_alpha(x)
        retrieval_phase = retrieval_phase.permute(3, 0, 2, 1)    # [K,B,H,T]
        routing_weights = routing_weights.permute(3, 0, 2, 1)    # [K,B,H,T]
        self._route_aux = self._route_balance_loss(routing_weights.permute(1, 3, 2, 0))
        decay_gamma_all, protected_values = self._gamma_all_and_vprime(x, values)  # [K,B,H,T], [B,H,T,d,2]
        write_beta, erase_beta = self._gate_betas(x)            # [B,H,T] each
        if erase_beta is None:
            erase_beta = write_beta
        key_real = keys[..., 0]
        key_imag = keys[..., 1]
        query_real = queries[..., 0]
        query_imag = queries[..., 1]
        strict_lower = torch.tril(torch.ones(chunk_size, chunk_size, device=x.device), -1)
        identity = torch.eye(chunk_size, device=x.device, dtype=torch.float32)

        memory_state = queries.new_zeros(K, batch_size, num_heads, head_dim, head_dim, 2)
        outputs = []
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk_len = chunk_end - chunk_start
            kc = slice(chunk_start, chunk_end)
            keys_chunk = keys[:, :, kc]
            values_chunk = protected_values[:, :, kc]
            decay_gamma_chunk = decay_gamma_all[:, :, :, kc]          # [K,B,H,C]
            write_beta_chunk = write_beta[:, :, kc]                    # [B,H,C]
            erase_beta_chunk = erase_beta[:, :, kc]                    # [B,H,C]

            # K-independent: key-gram + query-key scores computed ONCE per chunk.
            key_gram_real = (key_real[:, :, kc] @ key_real[:, :, kc].transpose(-1, -2)
                             + key_imag[:, :, kc] @ key_imag[:, :, kc].transpose(-1, -2))
            key_gram_imag = (key_imag[:, :, kc] @ key_real[:, :, kc].transpose(-1, -2)
                             - key_real[:, :, kc] @ key_imag[:, :, kc].transpose(-1, -2))
            query_key_real = (query_real[:, :, kc] @ key_real[:, :, kc].transpose(-1, -2)
                              + query_imag[:, :, kc] @ key_imag[:, :, kc].transpose(-1, -2))
            query_key_imag = (query_imag[:, :, kc] @ key_real[:, :, kc].transpose(-1, -2)
                              - query_real[:, :, kc] @ key_imag[:, :, kc].transpose(-1, -2))

            # Decay matrix batched over K·B·H (state-independent layout).
            decay_matrix = fused_decay_matrix(
                decay_gamma_chunk.reshape(K * batch_size * num_heads, chunk_len), chunk_len
            ).reshape(K, batch_size, num_heads, chunk_len, chunk_len)  # [K,B,H,C,C]
            log_decay = torch.log(decay_gamma_chunk + 1e-6)
            cumulative_alpha = torch.exp(torch.cumsum(log_decay, dim=-1))  # [K,B,H,C]

            # Erase (mass) matrix: M[t,s] = beta_e[t] D[t,s] (K K^H)[t,s], strictly lower.
            mass_real = erase_beta_chunk.unsqueeze(0).unsqueeze(-1) * decay_matrix * strict_lower[:chunk_len, :chunk_len] * key_gram_real.unsqueeze(0)
            mass_imag = erase_beta_chunk.unsqueeze(0).unsqueeze(-1) * decay_matrix * strict_lower[:chunk_len, :chunk_len] * key_gram_imag.unsqueeze(0)

            value_real, value_imag = values_chunk[..., 0], values_chunk[..., 1]
            if chunk_start > 0:
                state_real = memory_state[..., 0]     # [K,B,H,d,d]
                state_imag = memory_state[..., 1]
                state_key_real = (
                    key_real[:, :, kc] @ state_real.transpose(-1, -2)
                    - key_imag[:, :, kc] @ state_imag.transpose(-1, -2)
                )
                state_key_imag = (
                    key_real[:, :, kc] @ state_imag.transpose(-1, -2)
                    + key_imag[:, :, kc] @ state_real.transpose(-1, -2)
                )
                state_key_real = state_key_real * cumulative_alpha.unsqueeze(-1)
                state_key_imag = state_key_imag * cumulative_alpha.unsqueeze(-1)
                write_real = (write_beta_chunk.unsqueeze(0).unsqueeze(-1) * value_real.unsqueeze(0)
                              - erase_beta_chunk.unsqueeze(0).unsqueeze(-1) * state_key_real)
                write_imag = (write_beta_chunk.unsqueeze(0).unsqueeze(-1) * value_imag.unsqueeze(0)
                              - erase_beta_chunk.unsqueeze(0).unsqueeze(-1) * state_key_imag)
            else:
                write_real = write_beta_chunk.unsqueeze(0).unsqueeze(-1) * value_real.unsqueeze(0)
                write_imag = write_beta_chunk.unsqueeze(0).unsqueeze(-1) * value_imag.unsqueeze(0)
            # Broadcast the write to all K states (it is state-independent; only the
            # erase/state_key term above differs per state and forces the K broadcast).
            write_real = write_real.expand(K, batch_size, num_heads, chunk_len, head_dim)
            write_imag = write_imag.expand(K, batch_size, num_heads, chunk_len, head_dim)
            if _V13_DEBUG:
                print(f"[delta_fused] chunk {chunk_start}: "
                      f"decay {tuple(decay_matrix.shape)} mass {tuple(mass_real.shape)} "
                      f"write {tuple(write_real.shape)} strict_lower {tuple(strict_lower.shape)}",
                      flush=True)

            # ONE batched complex triangular solve over K·B·H (per-state independent).
            solve_n = K * batch_size * num_heads
            sol = _complex_triangular_solve(
                mass_real.reshape(solve_n, chunk_len, chunk_len),
                mass_imag.reshape(solve_n, chunk_len, chunk_len),
                write_real.reshape(solve_n, chunk_len, head_dim),
                write_imag.reshape(solve_n, chunk_len, head_dim),
                identity[:chunk_len, :chunk_len],
            )
            update_real = sol[0].reshape(K, batch_size, num_heads, chunk_len, head_dim)
            update_imag = sol[1].reshape(K, batch_size, num_heads, chunk_len, head_dim)

            causal_inclusive = torch.tril(torch.ones(chunk_len, chunk_len, device=x.device, dtype=decay_matrix.dtype))
            projection_real = (decay_matrix * causal_inclusive) * query_key_real.unsqueeze(0)
            projection_imag = (decay_matrix * causal_inclusive) * query_key_imag.unsqueeze(0)
            output_real = (projection_real @ update_real - projection_imag @ update_imag) * query_scale
            output_imag = (projection_real @ update_imag + projection_imag @ update_real) * query_scale
            output_chunk = torch.stack([output_real, output_imag], dim=-1)   # [K,B,H,T,d,2]

            if chunk_start > 0:
                # Carry: q*α_k @ S_k per state. q is K-independent [B,H,C,d]; α is
                # [K,B,H,C]. Apply α AFTER unsqueeze(0) so K appears exactly once.
                scaled_query_real = query_real[:, :, kc] * query_scale   # [B,H,C,d]
                scaled_query_imag = query_imag[:, :, kc] * query_scale   # [B,H,C,d]
                state_real = memory_state[..., 0]
                state_imag = memory_state[..., 1]
                alpha_r = cumulative_alpha.unsqueeze(-1)                # [K,B,H,C,1]
                carried_real = (
                    (scaled_query_real.unsqueeze(0) * alpha_r) @ state_real.transpose(-1, -2)
                    - (scaled_query_imag.unsqueeze(0) * alpha_r) @ state_imag.transpose(-1, -2)
                )
                carried_imag = (
                    (scaled_query_real.unsqueeze(0) * alpha_r) @ state_imag.transpose(-1, -2)
                    + (scaled_query_imag.unsqueeze(0) * alpha_r) @ state_real.transpose(-1, -2)
                )
                output_chunk = output_chunk + torch.stack([carried_real, carried_imag], dim=-1)

            cumulative_total = cumulative_alpha[..., -1:]                      # [K,B,H,1]
            decay_tail = cumulative_total / (cumulative_alpha + 1e-12)
            update_decayed_real = update_real * decay_tail.unsqueeze(-1)
            update_decayed_imag = update_imag * decay_tail.unsqueeze(-1)
            state_real = (
                update_decayed_real.transpose(-1, -2) @ key_real[:, :, kc]
                + update_decayed_imag.transpose(-1, -2) @ key_imag[:, :, kc]
            )
            state_imag = (
                update_decayed_imag.transpose(-1, -2) @ key_real[:, :, kc]
                - update_decayed_real.transpose(-1, -2) @ key_imag[:, :, kc]
            )
            state_chunk = torch.stack([state_real, state_imag], dim=-1)        # [K,B,H,d,d,2]
            memory_state = memory_state * cumulative_total.unsqueeze(-1).unsqueeze(-1) + state_chunk
            # Phase-rotate each state's read and sum over K (elementwise, batched).
            # Slice phase/routing to THIS chunk (they are full-T [K,B,H,T]).
            rotation_real = routing_weights[:, :, :, kc] * torch.cos(retrieval_phase[:, :, :, kc])
            rotation_imag = routing_weights[:, :, :, kc] * torch.sin(retrieval_phase[:, :, :, kc])  # [K,B,H,C]
            if _V13_DEBUG:
                print(f"[delta_fused] rot {tuple(rotation_real.shape)} "
                      f"out {tuple(output_chunk.shape)} state {tuple(memory_state.shape)} "
                      f"causal {tuple(causal_inclusive.shape)}", flush=True)
            rotated_real = output_chunk[..., 0] * rotation_real.unsqueeze(-1) - output_chunk[..., 1] * rotation_imag.unsqueeze(-1)
            rotated_imag = output_chunk[..., 0] * rotation_imag.unsqueeze(-1) + output_chunk[..., 1] * rotation_real.unsqueeze(-1)
            outputs.append(torch.stack([rotated_real.sum(dim=0), rotated_imag.sum(dim=0)], dim=-1))

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
                use_fused_delta = (
                    self.write_mode == 'delta'
                    and self.decay_mode != 'per_channel'
                    and getattr(self, 'fused_e3', True)
                )
                use_fused_add = (
                    self.write_mode == 'additive'
                    and self.decay_mode != 'per_channel'
                    and getattr(self, 'fused_e3', True)
                )
                if use_fused_delta:
                    # E2+E3 fused: K states collapsed into one batched chunk-solve.
                    output, new_state = self._forward_multistate_delta_fused(x, queries, keys, values, head_dim)
                elif use_fused_add:
                    output, new_state = self._forward_multistate_fused(x, queries, keys, values, head_dim)
                else:
                    # Ablation: K-loop multistate without collapse.
                    output, new_state = self._forward_multistate(x, queries, keys, values, head_dim)
            elif self.write_mode == 'delta':
                # Ablation E2 — not used by winner.
                decay_gamma, protected_values = self._gamma_and_vprime(x, values)
                write_beta, erase_beta = self._gate_betas(x)  # [B,H,T] (E2b)
                output, new_state = self._forward_delta(
                    queries, keys, protected_values, decay_gamma, write_beta, head_dim,
                    erase_beta=erase_beta,
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

        # merge heads back to [B,T,inner_dim,2] for output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.inner_dim, 2)
        out = self.o_proj(output)
        if self.training:
            # Manual complex dropout: one mask for both real and imag.
            dropout_mask = as_complex_dropout_mask(self.dropout, out)
            out = scale_complex(out, dropout_mask)
        return out, new_state

    # ── O(1) recurrent inference (covers all modes) ──────────────────────────

    def _recurrent(self, x, queries, keys, values, state, head_dim):
        """Token loop: fixed-size notebook S [K,B,H,d,d,2] — cost independent of past length."""
        batch_size, seq_len = x.shape[0], x.shape[1]
        num_heads, num_memory_states = self.num_heads, self.n_states
        query_scale = head_dim ** -0.5
        write_beta = None
        erase_beta = None
        if self.write_mode == 'delta':
            write_beta, erase_beta = self._gate_betas(x)  # [B,H,T] (E2b)
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
                    protected_value_t = protected_values_state[:, :, 0]
                    if self.write_mode == 'delta':
                        output_state, state_new = self._recur_step_delta(
                            memory_state[state_idx], decay_gamma_t,
                            protected_value_t, key_t, query_t,
                            write_beta[:, :, time_idx],
                            erase_beta_t=erase_beta[:, :, time_idx],
                        )
                    else:
                        output_state, state_new = self._recur_step_additive(
                            memory_state[state_idx], decay_gamma_t,
                            protected_value_t, key_t, query_t,
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
                    erase_beta_t=erase_beta[:, :, time_idx],
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

    def _recur_step_delta(self, memory_state, decay_gamma, value_t, key_t, query_t, write_beta_t,
                          erase_beta_t=None):
        """One gated delta step. decay_gamma:[B,H], write_beta_t:[B,H].

        E2b: erase_beta_t defaults to write_beta_t (legacy shared-beta behaviour).
        """
        if erase_beta_t is None:
            erase_beta_t = write_beta_t
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
        # E2b: u = beta_w * v_prot - beta_e * pred  (legacy: beta_w == beta_e)
        write_expanded = write_beta_t.unsqueeze(-1)
        erase_expanded = erase_beta_t.unsqueeze(-1)
        update_real = write_expanded * value_t[..., 0] - erase_expanded * predicted_real
        update_imag = write_expanded * value_t[..., 1] - erase_expanded * predicted_imag
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

    (I + M) is unit lower-triangular, so a complex triangular solve is exact and
    ~26x faster than the old real 2N×2N general solve, which paid for LU
    factorization + row pivoting (sgetf2/slaswp/setup_pivinfo ≈ 40% of GPU time)
    that a triangular system never needs. Matches the general solve to ~4e-6 in
    fp32. Measured on the 4090, T=2048, chunk=256.
    """
    system = torch.complex((identity + mass_real).float(), mass_imag.float())
    rhs = torch.complex(write_real.float(), write_imag.float())
    solution = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
    return solution.real.to(write_real.dtype), solution.imag.to(write_imag.dtype)


# ── V11 Block ────────────────────────────────────────────────────────────────

class V13Block(nn.Module):
    """Pre-norm residual: CGU (channel mix) + PAM (sequence mix)."""

    def __init__(self, cfg: V13Config, layer_idx: int = 0):
        super().__init__()
        # Pre-norm before CGU: stabilize magnitude; phase untouched.
        self.norm1 = ComplexNorm(cfg.dim)
        self.cgu = ComplexGatedUnit(cfg.dim, cfg.expand, activation=cfg.activation)
        self.cgu_scale = nn.Parameter(torch.tensor(1.0))
        self.cgu_dropout = nn.Dropout(cfg.dropout)
        # Pre-norm before PAM: same idea — PAM sees magnitude-stable inputs.
        self.norm2 = ComplexNorm(cfg.dim)
        self.pam = V13PAMLayer(cfg, layer_idx=layer_idx)
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
        pam_in = self.norm2(x)
        # Detached gate input for the gate-surprisal aux (2026-08-22): stashed
        # here as a DETACHED LEAF and consumed by V13LM._collect_gate_probs
        # OUTSIDE the gradient-checkpoint region. Building the stash from this
        # leaf means its backward reaches only this layer's protect_gate — it
        # no longer touches the checkpointed block's graph, so no PAM recompute
        # (the old in-checkpoint stash forced a full block recompute on bwd:
        # 364ms of an 816ms trunk step at B8/T2048 on the 4090).
        pam = self.pam
        if pam.use_gsp and pam.gate_surprisal_lambda > 0 and self.training:
            self._gate_in_det = (
                to_real_concat(pam_in.detach()) if pam.gate_content_aware
                else cabs(pam_in.detach())
            )
        else:
            self._gate_in_det = None
        # Sequence/memory mix via fixed-size PAM state.
        pam_out, new_state = pam(pam_in, state=pam_state, step_offset=step_offset)
        x = x + pam_out * self.pam_scale
        return x, new_state


class V13LM(nn.Module):
    """ComplexEmbed -> [V13Block] x N -> tied complex LM head."""

    def __init__(self, cfg: V13Config):
        super().__init__()
        self.config = cfg
        self.embed = ComplexEmbed(cfg.vocab_size, cfg.dim)
        self.pos_embed = (
            ComplexPosEmbed(cfg.max_seq_len, cfg.dim) if cfg.use_learned_pos else None
        )
        self.embed_norm = ComplexNorm(cfg.dim)
        self.blocks = nn.ModuleList([V13Block(cfg, layer_idx=i) for i in range(cfg.n_layers)])
        # Final magnitude stabilize after residual stack, before readout.
        self.output_norm = ComplexNorm(cfg.dim)
        # Small complex feature mix (not a second memory) + norm before vocab scores.
        self.lm_head_proj = ComplexLinear(cfg.dim, cfg.dim)
        self.lm_head_norm = ComplexNorm(cfg.dim)
        self._init_weights()

    def _init_weights(self):
        embed_embeddings = {self.embed.embed_real, self.embed.embed_imag}
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding) and module not in embed_embeddings:
                nn.init.normal_(module.weight, std=0.02)
        # re-apply custom biases zeroed above
        for _, module in self.named_modules():
            if hasattr(module, 'protect_gate') and isinstance(module.protect_gate, nn.Linear):
                nn.init.constant_(module.protect_gate.bias, getattr(module, 'protect_gate_bias', -3.0))
            if isinstance(module, V13PAMLayer) and module.n_states > 1:
                module._init_phase_proj()

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
        """Build per-layer mean protect-prob [B,T] and stack to [L,B,T] (or None).

        Built here — OUTSIDE the gradient-checkpoint region — from each block's
        detached gate input (`_gate_in_det`, a detached leaf captured in
        V13Block.forward). The aux gradient therefore reaches only that layer's
        protect_gate weights, and its backward node is not inside the checkpoint
        so backpropping the gate loss does NOT force a PAM block recompute
        (the in-checkpoint stash cost ~364ms/step at B8/T2048 on the 4090,
        2026-08-22). The gate-surprisal BCE in the trainer is against a
        detached per-token surprisal target (the exact NLL byproduct of the
        main fused CE).
        """
        probs = []
        for block in blocks:
            gi = getattr(block, '_gate_in_det', None)
            if gi is not None:
                probs.append(torch.sigmoid(block.pam.protect_gate(gi)).mean(dim=-1))
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
        new_states = []
        for i, block in enumerate(self.blocks):
            # states[i] is that layer's PAM notebook (None = build from scratch / train).
            s = states[i] if states is not None else None
            if use_ckpt:
                z, new_s = self._ckpt_block(block, z, step_offset)
            else:
                z, new_s = block(z, pam_state=s, step_offset=step_offset)
            new_states.append(new_s)
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
        for block in self.blocks:
            if use_ckpt:
                z, _ = self._ckpt_block(block, z, step_offset)
            else:
                z, _ = block(z, pam_state=None, step_offset=step_offset)
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

    def ce_from_lm(self, lm, labels, loss_mask=None, ignore_index=-100, chunk: int = 4096, return_nll: bool = False):
        """Chunked cross-entropy from pre-logit complex hidden `lm` [B,T,dim,2].

        The tied head `lm_r @ E_r.T + lm_i @ E_i.T` folds into one real matmul
        H @ W.T with H=concat(lm_r,lm_i), W=concat(E_r,E_i); chunked-CE never holds
        the full [N,vocab] logits/softmax. Kept eager (compile the stack, not this).
        """
        from v13.fused_ce import fused_linear_cross_entropy
        batch_size, seq_len = labels.shape
        hidden_concat = torch.cat([real_part(lm), imag_part(lm)], dim=-1).reshape(batch_size * seq_len, -1)
        weight_concat = torch.cat([self.embed.embed_real.weight, self.embed.embed_imag.weight], dim=-1)
        mask = loss_mask.reshape(-1) if loss_mask is not None else None
        out = fused_linear_cross_entropy(
            hidden_concat, weight_concat, labels.reshape(-1), mask=mask,
            chunk=chunk, ignore_index=ignore_index, return_nll=return_nll,
        )
        if return_nll:
            return out, getattr(out, '_nll', None).reshape(batch_size, seq_len)
        return out

    def fused_ce_loss(self, input_ids, labels, loss_mask=None, ignore_index=-100,
                      chunk: int = 4096):
        """Convenience eager path: hidden stack + chunked CE (exact == forward+CE)."""
        lm, aux_loss, _gate_probs = self._hidden_to_lm(input_ids)
        main = self.ce_from_lm(lm, labels, loss_mask=loss_mask,
                               ignore_index=ignore_index, chunk=chunk)
        return main, aux_loss

    @staticmethod
    def _ckpt_block(block, z, step_offset):
        # Non-reentrant checkpoint: forward runs `run(z)` without saving the
        # block's internals; backward recomputes with grad and backprops through
        # that recomputed graph, which yields dL/dz and continues up the stack.
        # DO NOT detach the block input here: a non-reentrant checkpoint relies on
        # the input edge to propagate gradient to the preceding block. Detaching
        # it (a 2026-08-22 "determinism_check" workaround) silently froze every
        # block except the last on the main loss (commit d0abeed). The nested
        # per-chunk PAM checkpoint (use_reentrant=False, no detach) is unaffected.
        def run(z_in):
            return block(z_in, pam_state=None, step_offset=step_offset)
        return grad_checkpoint(run, z, use_reentrant=False)

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
    # Baseline == V7 7d (control). Should reproduce ~26.88.
    'v11_baseline': _base_flat(),
    # E1: per-channel decay.
    'v11_e1_perchannel': _base_flat(decay_mode='per_channel'),
    # E2: delta-rule write.
    'v11_e2_delta': _base_flat(write_mode='delta', delta_chunk=64),
    # E3: 2-state superposition.
    'v11_e3_multistate': _base_flat(n_states=2, state_dt_spread=2.0),
    # E3 K=3: phase-aware GSP gate default (WikiText A/B 2026-06-30: −0.97 val PPL).
    'v11_e3_k3': _base_flat(n_states=3, state_dt_spread=2.0, gate_content_aware=True),
    # E3 K=3 + ChatML+reasoning vocab (50261) — production pretrain/SFT base preset.
    'v11_e3_k3_chat': _base_flat(
        n_states=3, state_dt_spread=2.0, vocab_size=50261, gate_content_aware=True,
    ),
    # Alias (same as v11_e3_k3_chat); kept for old launch scripts/docs.
    'v11_e3_k3_chat_gate': _base_flat(
        n_states=3, state_dt_spread=2.0, vocab_size=50261, gate_content_aware=True,
    ),
    # Recall program (V12): production chat config + longer memory horizon
    # (gamma_floor) + gate-surprisal supervision. Same params/vocab as chat so it
    # loads round-8b-gate weights for warm A/B; recall levers off by default here
    # and switched on per-arm via CLI so the A/B isolates each change.
    'v11_e3_k3_chat_recall': _base_flat(
        n_states=3, state_dt_spread=2.0, vocab_size=50261, gate_content_aware=True,
        gamma_floor=0.98, gate_surprisal_lambda=0.1, gate_surprisal_tau=1.0,
        gate_surprisal_sign=1.0,
    ),
    # Competitive retrieval (Tier 1): enable after WikiText A/B smoke wins.
    'v11_e3_k3_chat_compete': _base_flat(
        n_states=3, state_dt_spread=2.0, vocab_size=50261, gate_content_aware=True,
        routing_content_aware=True, state_compete=True, phase_init='spread',
        route_balance_lambda=0.01,
    ),
    # E1+E3 combo: per-channel decay inside each of K=2 superposed states.
    'v11_e1e3_combo': _base_flat(decay_mode='per_channel', n_states=2, state_dt_spread=2.0),
    # tiny smoke
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
    # Stage-6 capacity micro (~11M chat-vocab): pure-recall substrate tests.
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
    # ── V13 production presets (selective dynamics ON by default) ───────────
    'v13_e3_k3_selective': _base_flat(
        n_states=3, state_dt_spread=2.0, vocab_size=50261,
        write_mode='delta', delta_chunk=128,
        delta_erase_gate=True,
        gate_content_aware=True, vault_state=True, vault_state_idx=0,
        write_phase_address=True, gate_surprisal_lambda=0.1,
        # K-batched fused delta path (2026-08-22): math-exact vs the K-loop
        # (1e-7 on 4090 fp32) and ~8.6x faster at B16/chunk128 on the 4090
        # (2.3K -> ~20K tok/s train step). See v13/tmp/bench2.py.
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
