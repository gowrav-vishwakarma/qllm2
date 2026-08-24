"""
V13 selective PAM on sempyt: same architecture/params as v13.V13LM.

Named axes (never abbreviated in code):
  batch            sequence items in the minibatch
  time             token positions
  model_dim        residual / embedding width (v13 ``dim``)
  heads            attention/PAM heads
  head_feature     per-head channel width (v13 ``head_dim``)
  complex_pair     last axis of size 2: real then imag (was ``px``)
  qkv_slot         fused Q / K / V slice (size 3)
  qkv_fused        packed QKV feature (3 * heads * head_feature)
  memory_states    E3 notebooks (v13 ``n_states`` / K)
  real_imag_feature  concat(real, imag) along model_dim (size 2 * model_dim)
  head_row/head_col  the two axes of the d×d notebook matrix
  chunk_time       token positions inside one delta chunk
  write_row/source_col  chunk_time under two identities: the position being
                   written / queried, and the past position being read

All layout work is named: ``.to()`` / ``.alias()`` / ``contract`` / ``take`` /
``select``. The only kernels that still speak raw torch are the ones sempyt
cannot name: the ``[time, time]`` decay lag table (``complex_ops``), the
triangular solve (``sempyt.solve_triangular``), and the fused-CE autograd
Function, which exists precisely to avoid materialising ``[N, vocab]``.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from sempyt.dim import Dim, ProductDim
from sempyt.nn import Linear as NamedLinear
from sempyt.ops import as_complex, contract, imag, mean, real, softmax, sum as nsum
from sempyt.policies import SplitComplex
from sempyt.structural import (
    arange,
    cat,
    clamp,
    cos,
    eq,
    exp,
    flatten,
    full_like,
    log,
    ones,
    select,
    sigmoid,
    sin,
    softplus,
    stack,
    take,
    unstack,
    where,
    zeros,
)
from sempyt.tensor import NamedTensor, named

from v13_sempty.config import V13Config, get_config, PRESETS
from v13_sempty.complex_ops import (
    as_complex_dropout_mask,
    to_real_concat,
    ComplexLinear, ComplexNorm, ComplexEmbed, ComplexPosEmbed,
    ComplexGatedUnit, build_rope_cache,
)
from v13_sempty.pam_ops import (
    conjugate_scores,
    cumulative_decay,
    delta_chunk,
    phase_rotation,
    recur_step_delta,
)

# ── Phase-Associative Memory (V11) ──────────────────────────────────────────

class V13PAMLayer(nn.Module):
    r"""Matrix-state memory with complex-conjugate retrieval and pluggable dynamics.

    Baseline:  S_t = gamma_t * S_{t-1} + V_t (x) K_t^* ;  Y_t = S_t * Q_t
    E1:        gamma_t becomes per-key-channel (vector decay).
    E2:        write becomes delta-rule (erase stale assoc for K_t before write).
    E3:        K states with distinct decay; retrieval = sum_k e^{i phi_k} S_k Q.
    """

    def __init__(self, cfg: V13Config, layer_idx: int = 0,
                 model_dim: Dim | None = None, complex_pair: Dim | None = None):
        super().__init__()
        self.num_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        inner = cfg.n_heads * cfg.head_dim
        self.inner_dim = inner
        self.dim = cfg.dim
        # Shared Dim objects (identity, not just size). Prefer the LM's Dims.
        self.model_dim = model_dim or Dim("model_dim", cfg.dim)
        self.heads = Dim("heads", cfg.n_heads)
        self.head_feature = Dim("head_feature", cfg.head_dim)
        self.complex_pair = complex_pair or Dim("complex_pair", 2)
        self.inner = ProductDim(self.heads, self.head_feature)
        self.qkv_slot = Dim("qkv_slot", 3)
        self.qkv_fused = Dim("qkv_fused", 3 * inner)
        self.head_row = Dim("head_row", cfg.head_dim)
        self.head_col = Dim("head_col", cfg.head_dim)
        self.real_imag_feature = Dim("real_imag_feature", cfg.dim * 2)
        self.memory_states = Dim("memory_states", cfg.n_states)
        self.phase_scalar = Dim("phase_scalar", 1)
        self.policy = SplitComplex(self.complex_pair)
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
        self.delta_decay_factored = getattr(cfg, 'delta_decay_factored', False)
        self.delta_decay_factor_min_a = getattr(cfg, 'delta_decay_factor_min_a', 1e-6)

        if cfg.fused_qkv:
            self.qkv_proj = ComplexLinear(self.model_dim, self.qkv_fused, bias=False, pair=self.complex_pair)
        else:
            self.q_proj = ComplexLinear(self.model_dim, self.inner, bias=False, pair=self.complex_pair)
            self.k_proj = ComplexLinear(self.model_dim, self.inner, bias=False, pair=self.complex_pair)
            self.v_proj = ComplexLinear(self.model_dim, self.inner, bias=False, pair=self.complex_pair)
        self.o_proj = ComplexLinear(self.inner, self.model_dim, bias=False, pair=self.complex_pair)

        # Decay projection: per-head scalar, or per-(head, key-channel) for E1.
        decay_out = cfg.n_heads * (cfg.head_dim if cfg.decay_mode == 'per_channel' else 1)
        self.decay_out = Dim("decay_out", decay_out)
        self.dt_proj = NamedLinear(self.real_imag_feature, self.decay_out)
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
            self.write_phase_proj = NamedLinear(self.head_feature, self.phase_scalar, bias=True)
            nn.init.zeros_(self.write_phase_proj.weight)
            nn.init.zeros_(self.write_phase_proj.bias)
        if cfg.use_gsp:
            gate_in = self.real_imag_feature if self.gate_content_aware else self.model_dim
            self.protect_gate = NamedLinear(gate_in, self.heads)
            nn.init.constant_(self.protect_gate.bias, self.protect_gate_bias)

        # E2: delta-rule write strength beta_t in (0, 1) per head.
        if cfg.write_mode == 'delta':
            self.beta_proj = NamedLinear(self.model_dim, self.heads)
            nn.init.constant_(self.beta_proj.bias, 0.0)
            # E2b: separate learned erase gate. Low init -> starts additive-like;
            # the model learns to erase (overwrite) only for repeat/update tokens.
            if cfg.delta_erase_gate:
                self.erase_beta_proj = NamedLinear(self.model_dim, self.heads)
                nn.init.constant_(self.erase_beta_proj.bias, -3.0)
            else:
                self.erase_beta_proj = None

        # E3: per-state decay bias offsets + per-(head,state) retrieval phase.
        if cfg.n_states > 1:
            offs = torch.linspace(-cfg.state_dt_spread, cfg.state_dt_spread, cfg.n_states)
            self.state_dt_offset = nn.Parameter(offs.clone())          # [memory_states]
            self.heads_times_states = Dim("heads_times_states", cfg.n_heads * cfg.n_states)
            route_in = self.real_imag_feature if self.routing_content_aware else self.model_dim
            self.phase_proj = NamedLinear(route_in, self.heads_times_states)
            if self.state_compete:
                self.score_proj = NamedLinear(route_in, self.heads_times_states)
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
        self._route_aux = None

    def _gate_betas(self, x):
        """Write/erase gates, each [batch, heads, time]. Erase falls back to write.

        The erase gain is clamped to `delta_erase_beta_cap` (0 = off) so the
        vault delta eigenvalue (1 - beta_e, with key-norm ||k||^2=1) stays a
        stable contraction for any learned init (see V13Config comment).
        """
        tokens = self._as_token(x)
        batch, time = tokens.layout[0], tokens.layout[1]
        magnitude = tokens.abs()
        write_beta = sigmoid(self.beta_proj(magnitude)).to(batch, self.heads, time)
        if self.erase_beta_proj is None:
            return write_beta, write_beta
        erase_beta = sigmoid(self.erase_beta_proj(magnitude)).to(batch, self.heads, time)
        if self.delta_erase_beta_cap and self.delta_erase_beta_cap < 1.0:
            erase_beta = clamp(erase_beta, max=self.delta_erase_beta_cap)
        return write_beta, erase_beta

    def _apply_gamma_floor(self, base_decay: NamedTensor) -> NamedTensor:
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

    def _routing_input(self, tokens: NamedTensor) -> NamedTensor:
        if self.routing_content_aware:
            return to_real_concat(tokens, into=self.real_imag_feature)
        return tokens.abs()

    def _phase_and_alpha(self, x):
        """Phase and memory-state-scaled routing weights for E3 superposition.

        Winner (state_compete off): routing_weights_k == 1; only phases matter.
        phase_proj sees magnitudes by default (abs) — angle of x is ignored for
        routing so "how loud" a token is, not its phase, picks retrieval phases.

        Returns NamedTensors on (batch, time, heads, memory_states).
        """
        tokens = self._as_token(x)
        batch, time = tokens.layout[0], tokens.layout[1]
        routing_input = self._routing_input(tokens)
        retrieval_phase = self.phase_proj(routing_input).to(
            batch, time, self.heads, self.memory_states,
        )
        if self.state_compete:
            routing_scores = self.score_proj(routing_input).to(
                batch, time, self.heads, self.memory_states,
            )
            routing_weights = softmax(routing_scores, over=self.memory_states) * self.n_states
        else:
            routing_weights = ones(
                batch, time, self.heads, self.memory_states,
                device=tokens.device, dtype=tokens.dtype,
            )
        return retrieval_phase, routing_weights

    def _phase_route(self, x):
        """Phase / routing weights on ``(memory_states, batch, heads, time)``."""
        tokens = self._as_token(x)
        batch, time = tokens.layout[0], tokens.layout[1]
        retrieval_phase, routing_weights = self._phase_and_alpha(tokens)
        self._route_aux = self._route_balance_loss(routing_weights, batch, time)
        layout = (self.memory_states, batch, self.heads, time)
        return retrieval_phase.to(*layout), routing_weights.to(*layout)

    def _route_balance_loss(self, routing_weights: NamedTensor, batch: Dim, time: Dim):
        """MoE-style load balance: maximize entropy of batch-mean routing per head."""
        balance_lambda = self.route_balance_lambda
        if balance_lambda <= 0 or not self.state_compete or not self.training:
            return None
        routing_prob = routing_weights / self.n_states
        mean_routing_prob = mean(routing_prob, over=(batch, time))
        entropy = -nsum(mean_routing_prob * log(mean_routing_prob + 1e-8), over=self.memory_states)
        return -balance_lambda * mean(entropy, over=self.heads).data  # named-exit: scalar aux loss for the trainer

    # ── Projections + position + decay/gate prep (shared) ─────────────────────

    def _as_token(self, x: NamedTensor) -> NamedTensor:
        """Rebind a token NamedTensor onto this layer's Dims."""
        batch, time = x.layout[0], x.layout[1]
        return named(x.data, (batch, time, self.model_dim, self.complex_pair), self.policy)  # named-exit: re-state axis identities, no data moves

    def _as_memory(self, state, batch: Dim) -> Optional[NamedTensor]:
        """Rebind a carried notebook onto this call's Dims.

        Decode threads the state through many forwards, and each call wraps its
        tokens with a fresh ``batch`` axis, so the state's axis identities have
        to be re-stated (no data moves) exactly as ``_as_token`` does for x.
        """
        if state is None:
            return None
        layout = (
            self.memory_states, batch, self.heads,
            self.head_row, self.head_col, self.complex_pair,
        )
        return named(state.data if isinstance(state, NamedTensor) else state,  # named-exit: re-state axis identities, no data moves
                     layout, self.policy)

    def _project(self, x, step_offset: int):
        """Build Q, K, V on ``(batch, heads, time, head_feature, complex_pair)``."""
        tokens = self._as_token(x)
        batch, time = tokens.layout[0], tokens.layout[1]
        heads, head_feature, complex_pair, policy = (
            self.heads, self.head_feature, self.complex_pair, self.policy,
        )
        if self.fused_qkv:
            qkv = self.qkv_proj(tokens)
            split = qkv.to(batch, time, self.qkv_slot, heads, head_feature, complex_pair)
            q, k, v = unstack(split.to(batch, self.qkv_slot, heads, time, head_feature, complex_pair), self.qkv_slot)
        else:
            def _to_heads(proj):
                return proj(tokens).to(batch, heads, time, head_feature, complex_pair)
            q, k, v = _to_heads(self.q_proj), _to_heads(self.k_proj), _to_heads(self.v_proj)

        if self.use_rope:
            seq_len = tokens.size(time)
            position_end = step_offset + seq_len
            if position_end > self.rope_cache.shape[0]:
                self.register_buffer(
                    'rope_cache',
                    build_rope_cache(position_end * 2, self.head_dim).to(tokens.device),
                    persistent=False,
                )
            rope = named(
                self.rope_cache[step_offset:position_end].to(dtype=tokens.dtype),
                (time, head_feature, complex_pair), policy,
            )
            q, k = q * rope, k * rope

        if self.qk_norm:
            q, k = q / q.abs(), k / k.abs()

        if self.write_phase_address:
            q, v = self._apply_write_phase_address(q, k, v)

        if self.write_mode == 'delta' and self.delta_key_norm:
            k = k.normalize(over=head_feature)
        layout = (batch, heads, time, head_feature, complex_pair)
        return q.to(*layout), k.to(*layout), v.to(*layout)

    def _apply_write_phase_address(self, queries, keys, values):
        """Rotate values by e^{iψ(k)} and queries by e^{iψ(q)}; ψ = Linear(|·|)."""
        key_phase = select(self.write_phase_proj(keys.abs()), over=self.phase_scalar, index=0)
        query_phase = select(self.write_phase_proj(queries.abs()), over=self.phase_scalar, index=0)
        values = self._rotate_complex(values, key_phase)
        queries = self._rotate_complex(queries, query_phase)
        return queries, values

    def _rotate_complex(self, z: NamedTensor, phase: NamedTensor) -> NamedTensor:
        """Multiply SplitComplex ``z`` by e^{i phase}. Phase broadcasts over head_feature."""
        rotation = as_complex(cos(phase), sin(phase), self.complex_pair)
        return z * rotation

    def _gamma_all_and_vprime(self, x, values):
        """Head-scalar decay for all memory states at once and GSP-protected write value.

        Returns ``decay_gamma_all [memory_states, batch, heads, time]`` and the shared
        ``protected_values [batch, heads, time, head_feature, complex_pair]``.
        GSP: protect freezes decay (→1) and shrinks the write (× (1-p)).
        """
        tokens = self._as_token(x)
        batch, time = tokens.layout[0], tokens.layout[1]
        real_imag = to_real_concat(tokens, into=self.real_imag_feature)
        decay_logits = self.dt_proj(real_imag).alias(self.decay_out, self.heads)
        softplus_dt = softplus(
            decay_logits
            + named(self.dt_bias, (self.heads,))
            + named(self.state_dt_offset, (self.memory_states,))
        ).to(self.memory_states, batch, self.heads, time)
        base_decay = self._apply_gamma_floor(exp(-softplus_dt))
        if self.vault_state and 0 <= self.vault_state_idx < self.n_states:
            # The vault notebook never forgets: force its decay to exactly 1.
            is_vault = eq(arange(self.memory_states, device=base_decay.device), self.vault_state_idx)
            base_decay = where(is_vault, full_like(base_decay, 1.0), base_decay)
        if self.use_gsp:
            gate_input = real_imag if self.gate_content_aware else tokens.abs()
            protect_prob = sigmoid(self.protect_gate(gate_input)).to(batch, self.heads, time)
            decay_gamma_all = base_decay * (1 - protect_prob) + protect_prob
            return decay_gamma_all, values * (1 - protect_prob)
        return base_decay, values

    def _forward_multistate_delta_fused(self, x, queries, keys, values):
        """Parallel (training / prefill) E3 delta path: K notebooks, one solve per chunk.

        Per chunk: score keys against keys and queries against keys, solve the
        intra-chunk delta system for the corrections to write, add the read of
        the notebooks carried in from earlier chunks, fold this chunk's writes
        into the notebooks, and phase-route the K reads into one output.
        """
        tokens = self._as_token(x)
        batch, time = tokens.layout[0], tokens.layout[1]
        seq_len = tokens.size(time)
        chunk_size = self.delta_chunk if self.delta_chunk > 0 else seq_len
        query_scale = self.head_dim ** -0.5

        retrieval_phase, routing_weights = self._phase_route(tokens)
        decay_gamma_all, protected_values = self._gamma_all_and_vprime(tokens, values)
        write_beta, erase_beta = self._gate_betas(tokens)

        memory = zeros(
            self.memory_states, batch, self.heads,
            self.head_row, self.head_col, self.complex_pair,
            policy=self.policy, device=tokens.device, dtype=tokens.dtype,
        )
        reads = []

        for chunk_start in range(0, seq_len, chunk_size):
            chunk_len = min(chunk_size, seq_len - chunk_start)
            chunk_time = Dim("chunk_time", chunk_len)
            write_row = Dim("write_row", chunk_len)
            source_col = Dim("source_col", chunk_len)

            def chunk_of(t):
                return take(t, over=time, start=chunk_start, length=chunk_len, new=chunk_time)

            queries_chunk = chunk_of(queries)
            keys_chunk = chunk_of(keys)
            values_chunk = chunk_of(protected_values)
            decay_gamma = chunk_of(decay_gamma_all)
            erase_chunk = chunk_of(erase_beta)
            cumulative = cumulative_decay(decay_gamma, over=chunk_time)
            keys_on_col = keys_chunk.alias(self.head_feature, self.head_col)

            # Scores are state-independent, so they are computed once per chunk.
            key_gram = conjugate_scores(
                keys_chunk.alias(chunk_time, write_row),
                keys_chunk.alias(chunk_time, source_col),
                over=self.head_feature,
            )
            query_key = conjugate_scores(
                queries_chunk.alias(chunk_time, write_row),
                keys_chunk.alias(chunk_time, source_col),
                over=self.head_feature,
            )

            # The delta rule erases whatever the notebooks already predict for
            # these keys before writing the new association.
            write = chunk_of(write_beta) * values_chunk.alias(self.head_feature, self.head_row)
            if chunk_start > 0:
                predicted = contract(memory, keys_on_col, over=self.head_col)
                write = write - erase_chunk * (predicted * cumulative)

            read, written = delta_chunk(
                key_gram=key_gram,
                query_key=query_key,
                write=write.alias(chunk_time, write_row),
                decay_gamma=decay_gamma,
                cumulative=cumulative,
                erase_beta=erase_chunk,
                chunk_time=chunk_time,
                write_row=write_row,
                source_col=source_col,
                head_row=self.head_row,
                query_scale=query_scale,
                factored=self._use_factored_decay(cumulative),
            )

            if chunk_start > 0:
                carried_query = (queries_chunk * query_scale * cumulative).alias(
                    self.head_feature, self.head_col
                )
                read = read + contract(
                    memory, carried_query, over=self.head_col
                ).alias(chunk_time, write_row)

            memory = memory * select(cumulative, over=chunk_time, index=-1) + contract(
                written,
                keys_on_col.alias(chunk_time, source_col).conj(),
                over=source_col,
            )

            rotation = phase_rotation(
                chunk_of(routing_weights), chunk_of(retrieval_phase),
                complex_pair=self.complex_pair,
            ).alias(chunk_time, write_row)
            routed = nsum(read * rotation, over=self.memory_states)
            reads.append(routed.alias(write_row, chunk_time))

        output = cat(reads, over="chunk_time", into=time)
        return (
            output.alias(self.head_row, self.head_feature).to(
                batch, self.heads, time, self.head_feature, self.complex_pair
            ),
            memory,
        )

    def _use_factored_decay(self, cumulative: NamedTensor) -> bool:
        """The factored solve divides by the decay product, so guard on min alpha."""
        if not self.delta_decay_factored:
            return False
        return float(cumulative.detach().data.min()) >= self.delta_decay_factor_min_a  # named-exit: a Python bool picks the solve variant

    # ── Main forward ──────────────────────────────────────────────────────────

    def forward(self, x, state=None, step_offset: int = 0):
        tokens = self._as_token(x)
        batch, time = tokens.layout[0], tokens.layout[1]
        seq_len = tokens.size(time)
        queries, keys, values = self._project(tokens, step_offset)

        if self.n_states <= 1 or self.write_mode != 'delta' or self.decay_mode == 'per_channel':
            raise NotImplementedError(
                "v13_sempty implements the production E3 delta path only "
                "(n_states > 1, write_mode='delta', head decay)"
            )
        if state is None and seq_len > 1:
            # Training / prefill: chunked parallel form.
            output, new_state = self._forward_multistate_delta_fused(
                tokens, queries, keys, values,
            )
        else:
            # Decode: O(1) per token, notebook carried in `state`.
            output, new_state = self._recurrent(tokens, queries, keys, values, state)

        out = self.o_proj(output.to(batch, time, self.inner, self.complex_pair))
        if self.training:
            out = out * as_complex_dropout_mask(self.dropout, out)
        return out, new_state

    # ── O(1) recurrent decode ────────────────────────────────────────────────

    def _recurrent(self, x, queries, keys, values, state):
        """Token loop over a fixed-size notebook — cost independent of past length."""
        tokens = self._as_token(x)
        batch, time = tokens.layout[0], tokens.layout[1]
        seq_len = tokens.size(time)
        query_scale = self.head_dim ** -0.5
        write_beta, erase_beta = self._gate_betas(tokens)
        retrieval_phase, routing_weights = self._phase_route(tokens)

        memory = self._as_memory(state, batch)
        if memory is None:
            memory = zeros(
                self.memory_states, batch, self.heads,
                self.head_row, self.head_col, self.complex_pair,
                policy=self.policy, device=tokens.device, dtype=tokens.dtype,
            )

        step_time = Dim("step_time", 1)
        reads = []
        for time_idx in range(seq_len):
            token = take(tokens, over=time, start=time_idx, length=1, new=step_time)
            value_step = take(values, over=time, start=time_idx, length=1, new=step_time)
            key = select(keys, over=time, index=time_idx).alias(self.head_feature, self.head_col)
            query = (
                select(queries, over=time, index=time_idx) * query_scale
            ).alias(self.head_feature, self.head_col)
            write_step = select(write_beta, over=time, index=time_idx)
            erase_step = select(erase_beta, over=time, index=time_idx)

            # All states' decay and the shared protected value, once per token.
            decay_gamma_all, protected_values = self._gamma_all_and_vprime(token, value_step)
            protected_value = select(protected_values, over=step_time, index=0).alias(
                self.head_feature, self.head_row
            )

            read = None
            notebooks = []
            for state_idx in range(self.n_states):
                decay_gamma = select(
                    select(decay_gamma_all, over=self.memory_states, index=state_idx),
                    over=step_time, index=0,
                )
                state_read, notebook = recur_step_delta(
                    select(memory, over=self.memory_states, index=state_idx),
                    decay_gamma, protected_value,
                    key, query, write_step, erase_step,
                    head_row=self.head_row, head_col=self.head_col,
                )
                rotation = phase_rotation(
                    select(routing_weights, over=self.memory_states, index=state_idx),
                    select(retrieval_phase, over=self.memory_states, index=state_idx),
                    complex_pair=self.complex_pair,
                )
                state_read = state_read * select(rotation, over=time, index=time_idx)
                read = state_read if read is None else read + state_read
                notebooks.append(notebook)
            reads.append(read.to(batch, self.heads, self.head_row, self.complex_pair))
            memory = stack(notebooks, into=self.memory_states, at=batch)

        output = stack(reads, into=time, at=self.head_row)
        return output.alias(self.head_row, self.head_feature), memory


# ── V11 Block ────────────────────────────────────────────────────────────────

class V13Block(nn.Module):
    """Pre-norm residual: CGU (channel mix) + PAM (sequence mix)."""

    def __init__(self, cfg: V13Config, layer_idx: int = 0,
                 model_dim: Dim | None = None, complex_pair: Dim | None = None):
        super().__init__()
        self.model_dim = model_dim or Dim("model_dim", cfg.dim)
        self.complex_pair = complex_pair or Dim("complex_pair", 2)
        self.policy = SplitComplex(self.complex_pair)
        self.norm1 = ComplexNorm(self.model_dim, pair=self.complex_pair)
        self.cgu = ComplexGatedUnit(
            self.model_dim, cfg.expand, activation=cfg.activation, pair=self.complex_pair,
        )
        self.cgu_scale = nn.Parameter(torch.tensor(1.0))
        self.cgu_dropout = nn.Dropout(cfg.dropout)
        self.norm2 = ComplexNorm(self.model_dim, pair=self.complex_pair)
        self.pam = V13PAMLayer(
            cfg, layer_idx=layer_idx,
            model_dim=self.model_dim, complex_pair=self.complex_pair,
        )
        self.pam_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: NamedTensor, pam_state=None, step_offset: int = 0):
        cgu_out = self.cgu(self.norm1(x))
        if self.training:
            cgu_out = cgu_out * as_complex_dropout_mask(self.cgu_dropout, cgu_out)
        x = x + cgu_out * self.cgu_scale
        pam_in = self.norm2(x)
        pam = self.pam
        if pam.use_gsp and pam.gate_surprisal_lambda > 0 and self.training:
            pin = pam_in.detach()
            self._gate_in_det = (
                to_real_concat(pin, into=pam.real_imag_feature)
                if pam.gate_content_aware
                else pin.abs()
            )
        else:
            self._gate_in_det = None
        pam_out, new_state = pam(pam_in, state=pam_state, step_offset=step_offset)
        return x + pam_out * self.pam_scale, new_state


class V13LM(nn.Module):
    """ComplexEmbed -> [V13Block] x N -> tied complex LM head."""

    def __init__(self, cfg: V13Config):
        super().__init__()
        self.config = cfg
        self.model_dim = Dim("model_dim", cfg.dim)
        self.complex_pair = Dim("complex_pair", 2)
        self.vocab = Dim("vocab", cfg.vocab_size)
        self.real_imag_feature = Dim("real_imag_feature", cfg.dim * 2)
        self.policy = SplitComplex(self.complex_pair)
        self.embed = ComplexEmbed(cfg.vocab_size, self.model_dim, self.complex_pair)
        self.pos_embed = (
            ComplexPosEmbed(cfg.max_seq_len, self.model_dim) if cfg.use_learned_pos else None
        )
        self.embed_norm = ComplexNorm(self.model_dim, pair=self.complex_pair)
        self.blocks = nn.ModuleList([
            V13Block(cfg, layer_idx=i, model_dim=self.model_dim, complex_pair=self.complex_pair)
            for i in range(cfg.n_layers)
        ])
        self.output_norm = ComplexNorm(self.model_dim, pair=self.complex_pair)
        # Distinct out-axis: contract cannot tell two copies of the same Dim apart.
        self.lm_head_out = Dim("lm_head_out", cfg.dim)
        self.lm_head_proj = ComplexLinear(self.model_dim, self.lm_head_out, pair=self.complex_pair)
        self.lm_head_norm = ComplexNorm(self.model_dim, pair=self.complex_pair)
        self._init_weights()

    def _init_weights(self):
        embed_embeddings = {self.embed.embed_real, self.embed.embed_imag}
        for module in self.modules():
            if isinstance(module, (nn.Linear, NamedLinear)):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding) and module not in embed_embeddings:
                nn.init.normal_(module.weight, std=0.02)
        # re-apply custom biases zeroed above
        for _, module in self.named_modules():
            if hasattr(module, 'protect_gate') and isinstance(module.protect_gate, (nn.Linear, NamedLinear)):
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
        """Per-layer mean protect-prob, stacked to ``[layers, batch, time]`` (or None).

        Built OUTSIDE the gradient-checkpoint region from each block's detached
        gate input, so the aux gradient reaches only that layer's protect_gate
        weights and its backward node is not inside the checkpoint.
        """
        probs = []
        for block in blocks:
            gi = getattr(block, '_gate_in_det', None)
            if gi is not None:
                protect = sigmoid(block.pam.protect_gate(gi))
                probs.append(mean(protect, over=block.pam.heads))
        if not probs:
            return None
        batch_dim = probs[0].layout[0]
        return stack(probs, into=Dim("layers", len(probs)), at=batch_dim).data  # named-exit: trainer expects raw torch

    def _stem(self, input_ids, step_offset: int = 0) -> tuple:
        batch, time = Dim("batch", input_ids.shape[0]), Dim("time", input_ids.shape[1])
        z = self.embed(input_ids, batch, time)
        if self.pos_embed is not None:
            z = self.pos_embed(z, step_offset=step_offset, time=time)
        return self.embed_norm(z), batch, time

    def _tied_logits(self, lm: NamedTensor, batch: Dim, time: Dim):
        embed_real = named(self.embed.embed_real.weight, (self.vocab, self.model_dim))
        embed_imag = named(self.embed.embed_imag.weight, (self.vocab, self.model_dim))
        logits = contract(real(lm), embed_real, over=self.model_dim) + contract(
            imag(lm), embed_imag, over=self.model_dim
        )
        return logits.raw(batch, time, self.vocab)  # named-exit: v13's public logits are raw torch

    def forward(self, input_ids, states=None, step_offset: int = 0, labels=None):
        z, batch, time = self._stem(input_ids, step_offset)
        use_ckpt = self.config.gradient_checkpointing and self.training and states is None
        new_states = []
        for i, block in enumerate(self.blocks):
            s = states[i] if states is not None else None
            if use_ckpt:
                z, new_s = self._ckpt_block(block, z, step_offset)
            else:
                z, new_s = block(z, pam_state=s, step_offset=step_offset)
            new_states.append(new_s)
        lm = self.lm_head_norm(self.lm_head_proj(self.output_norm(z)))
        logits = self._tied_logits(lm, batch, time)
        route_aux = self._collect_route_aux(self.blocks)
        aux_loss = (
            route_aux
            if route_aux is not None
            else torch.tensor(0.0, device=input_ids.device)
        )
        return logits, new_states, aux_loss

    def _hidden_to_lm(self, input_ids, step_offset: int = 0):
        """Training path: stack + head norm, stop before full [B,T,V] logits."""
        z, batch, time = self._stem(input_ids, step_offset)
        use_ckpt = self.config.gradient_checkpointing and self.training
        for block in self.blocks:
            if use_ckpt:
                z, _ = self._ckpt_block(block, z, step_offset)
            else:
                z, _ = block(z, pam_state=None, step_offset=step_offset)
        lm = self.lm_head_norm(self.lm_head_proj(self.output_norm(z)))
        route_aux = self._collect_route_aux(self.blocks)
        aux_loss = (
            route_aux
            if route_aux is not None
            else torch.tensor(0.0, device=input_ids.device)
        )
        gate_probs = self._collect_gate_probs(self.blocks)
        return lm, aux_loss, gate_probs

    def ce_from_lm(self, lm: NamedTensor, labels, loss_mask=None,
                   ignore_index=-100, chunk: int = 4096, return_nll: bool = False):
        """Chunked cross-entropy from the pre-logit complex hidden ``lm``.

        The tied head ``lm_r @ E_r.T + lm_i @ E_i.T`` folds into one real matmul
        ``H @ W.T`` with ``H = concat(lm_r, lm_i)`` and ``W = concat(E_r, E_i)``;
        the chunked-CE autograd Function never holds the full ``[N, vocab]``
        logits/softmax. Construction is named; only the final hand-off to the
        Function is raw torch.
        """
        from v13_sempty.fused_ce import fused_linear_cross_entropy
        batch, time = lm.layout[0], lm.layout[1]
        flat = Dim("flat", batch.size * time.size)

        hidden = flatten(
            to_real_concat(lm, into=self.real_imag_feature),
            (batch, time), flat,
        )
        embed_real = named(self.embed.embed_real.weight, (self.vocab, self.model_dim))
        embed_imag = named(self.embed.embed_imag.weight, (self.vocab, self.model_dim))
        weight = cat([embed_real, embed_imag], over=self.model_dim, into=self.real_imag_feature)

        out = fused_linear_cross_entropy(
            hidden.raw(flat, self.real_imag_feature),  # named-exit: chunked-CE Function takes raw torch
            weight.raw(self.vocab, self.real_imag_feature),  # named-exit: ditto
            labels.reshape(-1),  # named-exit: raw int tensor from the dataloader
            mask=(loss_mask.reshape(-1) if loss_mask is not None else None),  # named-exit: raw mask tensor
            chunk=chunk, ignore_index=ignore_index, return_nll=return_nll,
        )
        if return_nll:
            return out, getattr(out, '_nll', None).reshape(batch.size, time.size)  # named-exit: raw nll from the CE Function
        return out

    def fused_ce_loss(self, input_ids, labels, loss_mask=None, ignore_index=-100,
                      chunk: int = 4096):
        """Convenience eager path: hidden stack + chunked CE (exact == forward+CE)."""
        lm, aux_loss, _gate_probs = self._hidden_to_lm(input_ids)
        main = self.ce_from_lm(lm, labels, loss_mask=loss_mask,
                               ignore_index=ignore_index, chunk=chunk)
        return main, aux_loss

    def compile_blocks(self, mode: str = 'default'):
        """Compile each V13Block.forward for use *inside* gradient checkpoint.

        Compiling `_hidden_to_lm` as a whole (v7 trainer `--compile`) wraps the
        checkpoint wrapper, so inductor cannot fuse a block. Compiling the
        block itself (measured +28% at B8/T2048, 2026-08-23) lets inductor
        see the PAM chunk. `--compile_blocks` in v13.train selects this path
        and skips the whole-model compile. Do NOT detach the checkpoint input.
        """
        for block in self.blocks:
            block._compiled_fwd = torch.compile(block.forward, mode=mode, dynamic=False)
        return self

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
        compiled = getattr(block, '_compiled_fwd', None)

        def run(z_in):
            if compiled is not None:
                return compiled(z_in, pam_state=None, step_offset=step_offset)
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



__all__ = [
    "V13Config",
    "V13LM",
    "V13PAMLayer",
    "V13Block",
    "get_config",
    "PRESETS",
]
