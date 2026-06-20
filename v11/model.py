"""
V11 model: V7 PAM core + new memory dynamics (E1/E2/E3).

Reuses the stable V7 complex primitives (ComplexLinear, ComplexNorm, CGU,
ModSwish/ModReLU, ComplexEmbed, RoPE) unchanged. The only architectural change
lives in `V11PAMLayer`, which dispatches on three flags:

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
import copy
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

# Reuse the proven, stable V7 primitives unchanged -- "same core".
from v7.model import (
    cmul, cconj, cabs, cnormalize, to_real_concat,
    ComplexLinear, ComplexNorm, ComplexEmbed, ComplexPosEmbed,
    ComplexGatedUnit, build_rope_cache, _build_activation,
)
from v7.triton_kernels import fused_decay_matrix


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class V11Config:
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


# ── Phase-Associative Memory (V11) ──────────────────────────────────────────

class V11PAMLayer(nn.Module):
    r"""Matrix-state memory with complex-conjugate retrieval and pluggable dynamics.

    Baseline:  S_t = gamma_t * S_{t-1} + V_t (x) K_t^* ;  Y_t = S_t * Q_t
    E1:        gamma_t becomes per-key-channel (vector decay).
    E2:        write becomes delta-rule (erase stale assoc for K_t before write).
    E3:        K states with distinct decay; retrieval = sum_k e^{i phi_k} S_k Q.
    """

    def __init__(self, cfg: V11Config, layer_idx: int = 0):
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

        if cfg.use_gsp:
            self.protect_gate = nn.Linear(cfg.dim, cfg.n_heads)
            nn.init.constant_(self.protect_gate.bias, -3.0)

        # E2: delta-rule write strength beta_t in (0, 1) per head.
        if cfg.write_mode == 'delta':
            self.beta_proj = nn.Linear(cfg.dim, cfg.n_heads)
            nn.init.constant_(self.beta_proj.bias, 0.0)

        # E3: per-state decay bias offsets + per-(head,state) retrieval phase.
        if cfg.n_states > 1:
            offs = torch.linspace(-cfg.state_dt_spread, cfg.state_dt_spread, cfg.n_states)
            self.state_dt_offset = nn.Parameter(offs.clone())          # [K]
            self.phase_proj = nn.Linear(cfg.dim, cfg.n_heads * cfg.n_states)
            nn.init.zeros_(self.phase_proj.weight)
            nn.init.zeros_(self.phase_proj.bias)

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

    # ── Projections + position + decay/gate prep (shared) ─────────────────────

    def _project(self, x: torch.Tensor, step_offset: int):
        B, T, _, _ = x.shape
        H, d = self.num_heads, self.head_dim
        if self.fused_qkv:
            qkv = self.qkv_proj(x).view(B, T, 3, H, d, 2)
            q = qkv[:, :, 0].transpose(1, 2).contiguous()
            k = qkv[:, :, 1].transpose(1, 2).contiguous()
            v = qkv[:, :, 2].transpose(1, 2).contiguous()
        else:
            q = self.q_proj(x).view(B, T, H, d, 2).transpose(1, 2).contiguous()
            k = self.k_proj(x).view(B, T, H, d, 2).transpose(1, 2).contiguous()
            v = self.v_proj(x).view(B, T, H, d, 2).transpose(1, 2).contiguous()

        if self.use_rope:
            end = step_offset + T
            if end > self.rope_cache.shape[0]:
                self.register_buffer(
                    'rope_cache',
                    build_rope_cache(end * 2, d).to(x.device),
                    persistent=False,
                )
            pos = self.rope_cache[step_offset:end].to(dtype=x.dtype)
            q = cmul(q, pos)
            k = cmul(k, pos)

        if self.qk_norm:
            q = cnormalize(q)
            k = cnormalize(k)
        return q, k, v

    def _gamma_and_vprime(self, x: torch.Tensor, v: torch.Tensor, state_offset: float = 0.0):
        """Return decay `gamma` and protected value `v_prime`.

        gamma shape: [B,H,T] (head decay) or [B,H,T,d] (per-channel decay).
        """
        B, T = x.shape[0], x.shape[1]
        H, d = self.num_heads, self.head_dim
        x_flat = to_real_concat(x)
        if self.decay_mode == 'per_channel':
            dt = self.dt_proj(x_flat).view(B, T, H, d)             # [B,T,H,d]
            dt = F.softplus(dt + self.dt_bias + state_offset)      # bias [H,d]
            dt = dt.permute(0, 2, 1, 3).contiguous()               # [B,H,T,d]
        else:
            dt = self.dt_proj(x_flat)                              # [B,T,H]
            dt = F.softplus(dt + self.dt_bias + state_offset)
            dt = dt.transpose(1, 2).contiguous()                  # [B,H,T]

        if self.use_gsp:
            p = torch.sigmoid(self.protect_gate(cabs(x))).transpose(1, 2)  # [B,H,T]
            if self.decay_mode == 'per_channel':
                p_e = p.unsqueeze(-1)
                gamma = torch.exp(-dt) * (1 - p_e) + p_e
            else:
                gamma = torch.exp(-dt) * (1 - p) + p
            v_prime = v * (1 - p).unsqueeze(-1).unsqueeze(-1)
        else:
            gamma = torch.exp(-dt)
            v_prime = v
        return gamma, v_prime

    # ── Baseline dual-form block (head scalar decay, additive write) ──────────

    @staticmethod
    def _dual_form_block(q_s, k, v_prime, gamma, causal_mask):
        B, H, T = gamma.shape
        gamma_flat = gamma.reshape(B * H, T)
        D = fused_decay_matrix(gamma_flat, T).reshape(B, H, T, T)
        qr, qi = q_s[..., 0], q_s[..., 1]
        kr, ki = k[..., 0], k[..., 1]
        wr = qr @ kr.transpose(-1, -2) + qi @ ki.transpose(-1, -2)
        wi = qi @ kr.transpose(-1, -2) - qr @ ki.transpose(-1, -2)
        ar, ai = wr * D, wi * D
        vpr, vpi = v_prime[..., 0], v_prime[..., 1]
        yr = ar @ vpr - ai @ vpi
        yi = ar @ vpi + ai @ vpr
        y = torch.stack([yr, yi], dim=-1)
        D_last = D[:, :, -1, :]
        wv_r = vpr * D_last.unsqueeze(-1)
        wv_i = vpi * D_last.unsqueeze(-1)
        sr = wv_r.transpose(-1, -2) @ kr + wv_i.transpose(-1, -2) @ ki
        si = wv_i.transpose(-1, -2) @ kr - wv_r.transpose(-1, -2) @ ki
        S_block = torch.stack([sr, si], dim=-1)
        return y, S_block

    def _forward_chunked_head(self, q, k, v_prime, gamma, d):
        B, H, T = q.shape[:3]
        C = self.chunk_size
        scale = d ** -0.5
        q_s = q * scale
        S = q.new_zeros(B, H, d, d, 2)
        outputs = []
        for start in range(0, T, C):
            end = min(start + C, T)
            Tc = end - start
            q_c, k_c = q_s[:, :, start:end], k[:, :, start:end]
            v_c, g_c = v_prime[:, :, start:end], gamma[:, :, start:end]
            causal = self._causal[:Tc, :Tc]
            y_c, S_chunk = self._dual_form_block(q_c, k_c, v_c, g_c, causal)
            log_g = torch.log(g_c + 1e-6)
            cum_decay = torch.exp(torch.cumsum(log_g, dim=-1))
            if start > 0:
                Sr, Si = S[..., 0], S[..., 1]
                qr_c, qi_c = q_c[..., 0], q_c[..., 1]
                Sq_r = (Sr @ qr_c.transpose(-1, -2) - Si @ qi_c.transpose(-1, -2)).transpose(-1, -2)
                Sq_i = (Sr @ qi_c.transpose(-1, -2) + Si @ qr_c.transpose(-1, -2)).transpose(-1, -2)
                cd = cum_decay.unsqueeze(-1)
                y_c = y_c + torch.stack([Sq_r * cd, Sq_i * cd], dim=-1)
            outputs.append(y_c)
            total_decay = cum_decay[:, :, -1]
            S = S * total_decay[..., None, None, None] + S_chunk
        return torch.cat(outputs, dim=2), S

    # ── E1: per-channel decay (GLA-style fold), chunked ───────────────────────

    def _forward_chunked_perchannel(self, q, k, v_prime, gamma, d):
        """gamma: [B,H,T,d] per key-channel. Stable chunk-local cumulative fold."""
        B, H, T = q.shape[:3]
        C = self.chunk_size
        scale = d ** -0.5
        S = q.new_zeros(B, H, d, d, 2)               # value(i) x key(j)
        outputs = []
        for start in range(0, T, C):
            end = min(start + C, T)
            Tc = end - start
            q_c = q[:, :, start:end]                  # [B,H,Tc,d,2]
            k_c = k[:, :, start:end]
            v_c = v_prime[:, :, start:end]
            g_c = gamma[:, :, start:end]              # [B,H,Tc,d]

            log_g = torch.log(g_c.clamp_min(1e-6)).float()
            P = torch.cumsum(log_g, dim=2)            # inclusive cumsum, [B,H,Tc,d]
            P = P.clamp(min=-30.0)
            alpha = torch.exp(P)                      # prod_{0..t} g  (<=1)
            inv_alpha = torch.exp(-P)                 # 1/alpha        (>=1, bounded)
            P_total = P[:, :, -1:, :]                 # [B,H,1,d]
            decay_tail = torch.exp(P_total - P)       # alpha_total/alpha_s  (<=1)
            alpha = alpha.to(q.dtype)
            inv_alpha = inv_alpha.to(q.dtype)
            decay_tail = decay_tail.to(q.dtype)
            alpha_total = torch.exp(P_total).to(q.dtype)  # [B,H,1,d]

            # Fold decay into q (q*alpha) and k (k/alpha) -> plain conjugate score.
            qf = q_c * alpha.unsqueeze(-1) * scale
            kf = k_c * inv_alpha.unsqueeze(-1)
            qr, qi = qf[..., 0], qf[..., 1]
            kr, ki = kf[..., 0], kf[..., 1]
            wr = qr @ kr.transpose(-1, -2) + qi @ ki.transpose(-1, -2)
            wi = qi @ kr.transpose(-1, -2) - qr @ ki.transpose(-1, -2)
            causal = self._causal[:Tc, :Tc]
            wr, wi = wr * causal, wi * causal
            vpr, vpi = v_c[..., 0], v_c[..., 1]
            yr = wr @ vpr - wi @ vpi
            yi = wr @ vpi + wi @ vpr
            y_c = torch.stack([yr, yi], dim=-1)

            if start > 0:
                # carried state read: y += (S @ (q*alpha))  per channel.
                qg = q_c * alpha.unsqueeze(-1) * scale         # [B,H,Tc,d,2]
                Sr, Si = S[..., 0], S[..., 1]                  # [B,H,d(i),d(j)]
                qgr, qgi = qg[..., 0], qg[..., 1]              # [B,H,Tc,d(j)]
                yr2 = (qgr @ Sr.transpose(-1, -2) - qgi @ Si.transpose(-1, -2))
                yi2 = (qgr @ Si.transpose(-1, -2) + qgi @ Sr.transpose(-1, -2))
                y_c = y_c + torch.stack([yr2, yi2], dim=-1)

            outputs.append(y_c)

            # state update: S_new[i,j] = alpha_total[j]*S[i,j] + sum_s v_s[i] (k_s* decay_tail)[j]
            kd = k_c * decay_tail.unsqueeze(-1)               # [B,H,Tc,d,2]
            kdr, kdi = kd[..., 0], kd[..., 1]
            # outer sum over s: S_chunk[i,j] = sum_s v_s[i] conj(kd_s)[j]
            sr = vpr.transpose(-1, -2) @ kdr + vpi.transpose(-1, -2) @ kdi
            si = vpi.transpose(-1, -2) @ kdr - vpr.transpose(-1, -2) @ kdi
            S_chunk = torch.stack([sr, si], dim=-1)
            at = alpha_total.squeeze(2)                       # [B,H,d(j)]
            S = S * at.unsqueeze(2).unsqueeze(-1) + S_chunk
        return torch.cat(outputs, dim=2), S

    # ── E2: delta-rule write (UT transform), chunked, head scalar decay ───────

    def _forward_delta(self, q, k, v_prime, gamma, beta, d):
        """Gated delta rule via per-chunk UT transform. gamma: [B,H,T] head scalar."""
        B, H, T = q.shape[:3]
        C = self.delta_chunk
        scale = d ** -0.5
        S = q.new_zeros(B, H, d, d, 2)               # value(i) x key(j)
        outputs = []
        eye = torch.eye(C, device=q.device, dtype=torch.float32)
        for start in range(0, T, C):
            end = min(start + C, T)
            Tc = end - start
            q_c = q[:, :, start:end]
            k_c = k[:, :, start:end]
            v_c = v_prime[:, :, start:end]
            g_c = gamma[:, :, start:end]              # [B,H,Tc]
            b_c = beta[:, :, start:end]               # [B,H,Tc]

            gamma_flat = g_c.reshape(B * H, Tc)
            D = fused_decay_matrix(gamma_flat, Tc).reshape(B, H, Tc, Tc)  # D[t,s]=prod_{s+1..t} g
            log_g = torch.log(g_c + 1e-6)
            cum = torch.exp(torch.cumsum(log_g, dim=-1))                  # alpha_t = prod_{0..t} g

            kr, ki = k_c[..., 0], k_c[..., 1]
            qr, qi = q_c[..., 0], q_c[..., 1]
            # K K^dagger : kk[t,s] = <k_t, k_s> = sum k_t conj(k_s)
            kkr = kr @ kr.transpose(-1, -2) + ki @ ki.transpose(-1, -2)
            kki = ki @ kr.transpose(-1, -2) - kr @ ki.transpose(-1, -2)
            strict = torch.tril(torch.ones(Tc, Tc, device=q.device), -1)
            # M[t,s] = beta_t * D[t,s] * <k_t,k_s>, strictly lower
            Dm = D * strict
            Mr = (b_c.unsqueeze(-1) * Dm * kkr)
            Mi = (b_c.unsqueeze(-1) * Dm * kki)
            # Solve (I + M) U = W  for U (complex), via block-real linear solve.
            Wr_state = q_c.new_zeros(B, H, Tc, d)    # placeholder, filled below

            # w_t = beta_t (v_t - (decayed S_prev) k_t)
            #   decayed S_prev contribution at t: (S @ k_t) * alpha_t
            vpr, vpi = v_c[..., 0], v_c[..., 1]
            if start > 0:
                Sr, Si = S[..., 0], S[..., 1]
                # (S k_t)[i] = sum_j S[i,j] conj?  retrieval uses S * q; for write we need S k.
                # S stores V (x) K*, so S k_t = sum_j S[i,j] k_t[j] (no conj; conj already in stored K*).
                Skr = (kr @ Sr.transpose(-1, -2) - ki @ Si.transpose(-1, -2))  # [B,H,Tc,d]
                Ski = (kr @ Si.transpose(-1, -2) + ki @ Sr.transpose(-1, -2))
                Skr = Skr * cum.unsqueeze(-1)
                Ski = Ski * cum.unsqueeze(-1)
                wr = b_c.unsqueeze(-1) * (vpr - Skr)
                wi = b_c.unsqueeze(-1) * (vpi - Ski)
            else:
                wr = b_c.unsqueeze(-1) * vpr
                wi = b_c.unsqueeze(-1) * vpi

            U_r, U_i = _complex_triangular_solve(Mr, Mi, wr, wi, eye[:Tc, :Tc])

            # outputs: y_t = (S_prev decayed) q_t + sum_{s<=t} D[t,s] <k_s,q_t> u_s
            #   <k_s, q_t> = sum k_s conj? score uses q * conj(k): P[t,s]=<q_t,k_s*>? match retrieval
            # retrieval baseline: y = (q . k*) weighted. Use qk[t,s] = q_t . conj(k_s)
            qkr = qr @ kr.transpose(-1, -2) + qi @ ki.transpose(-1, -2)
            qki = qi @ kr.transpose(-1, -2) - qr @ ki.transpose(-1, -2)
            tril_inc = self._causal[:Tc, :Tc]
            Pr = (D * tril_inc) * qkr
            Pi = (D * tril_inc) * qki
            yr = (Pr @ U_r - Pi @ U_i) * scale
            yi = (Pr @ U_i + Pi @ U_r) * scale
            y_c = torch.stack([yr, yi], dim=-1)
            if start > 0:
                Sr, Si = S[..., 0], S[..., 1]
                qg_r = qr * scale * cum.unsqueeze(-1)
                qg_i = qi * scale * cum.unsqueeze(-1)
                yr2 = (qg_r @ Sr.transpose(-1, -2) - qg_i @ Si.transpose(-1, -2))
                yi2 = (qg_r @ Si.transpose(-1, -2) + qg_i @ Sr.transpose(-1, -2))
                y_c = y_c + torch.stack([yr2, yi2], dim=-1)
            outputs.append(y_c)

            # state update: S_new = alpha_T * S_prev + sum_s (alpha_T/alpha_s) u_s k_s*
            cum_total = cum[:, :, -1:]                       # [B,H,1]
            tail = cum_total / (cum + 1e-12)                # alpha_T/alpha_s
            ud_r = U_r * tail.unsqueeze(-1)
            ud_i = U_i * tail.unsqueeze(-1)
            sr = ud_r.transpose(-1, -2) @ kr + ud_i.transpose(-1, -2) @ ki
            si = ud_i.transpose(-1, -2) @ kr - ud_r.transpose(-1, -2) @ ki
            S_chunk = torch.stack([sr, si], dim=-1)
            S = S * cum_total.unsqueeze(-1).unsqueeze(-1) + S_chunk
        return torch.cat(outputs, dim=2), S

    # ── E3: multi-state superposition (loop over states, phase-combine) ───────

    def _forward_multistate(self, x, q, k, v_prime, d):
        B, T = x.shape[0], x.shape[1]
        H, K = self.num_heads, self.n_states
        scale = d ** -0.5
        # phase per (head, state, position)
        phi = self.phase_proj(cabs(x)).view(B, T, H, K).permute(0, 2, 3, 1)  # [B,H,K,T]
        y_sum = None
        S_list = []
        for kdx in range(K):
            gamma_k, vp_k = self._gamma_and_vprime(
                x, v_prime, state_offset=self.state_dt_offset[kdx]
            )
            if self.decay_mode == 'per_channel':
                # E1+E3 combo: per-key-channel decay inside each superposed state.
                y_k, S_k = self._forward_chunked_perchannel(q, k, vp_k, gamma_k, d)
            elif self.chunk_size > 0 and T > self.chunk_size:
                y_k, S_k = self._forward_chunked_head(q, k, vp_k, gamma_k, d)
            else:
                q_s = q * scale
                y_k, S_k = self._dual_form_block(q_s, k, vp_k, gamma_k, self._causal[:T, :T])
            rot = torch.stack([torch.cos(phi[:, :, kdx]), torch.sin(phi[:, :, kdx])], dim=-1)  # [B,H,T,2]
            y_k = cmul(y_k, rot.unsqueeze(-2))                # rotate complex output
            y_sum = y_k if y_sum is None else y_sum + y_k
            S_list.append(S_k)
        return y_sum, torch.stack(S_list, dim=0)             # [K,B,H,d,d,2]

    # ── Main forward ──────────────────────────────────────────────────────────

    def forward(self, x, state=None, step_offset: int = 0):
        B, T, _, _ = x.shape
        H, d = self.num_heads, self.head_dim
        q, k, v = self._project(x, step_offset)

        # Training (parallel form): state is None and T>1.
        if state is None and T > 1:
            if self.n_states > 1:
                y, new_state = self._forward_multistate(x, q, k, v, d)
            elif self.write_mode == 'delta':
                gamma, v_prime = self._gamma_and_vprime(x, v)
                beta = torch.sigmoid(self.beta_proj(cabs(x))).transpose(1, 2)  # [B,H,T]
                y, new_state = self._forward_delta(q, k, v_prime, gamma, beta, d)
            elif self.decay_mode == 'per_channel':
                gamma, v_prime = self._gamma_and_vprime(x, v)
                y, new_state = self._forward_chunked_perchannel(q, k, v_prime, gamma, d)
            else:
                gamma, v_prime = self._gamma_and_vprime(x, v)
                if self.chunk_size > 0 and T > self.chunk_size:
                    y, new_state = self._forward_chunked_head(q, k, v_prime, gamma, d)
                else:
                    q_s = q * (d ** -0.5)
                    y, new_state = self._dual_form_block(q_s, k, v_prime, gamma, self._causal[:T, :T])
        else:
            y, new_state = self._recurrent(x, q, k, v, state, d)

        y = y.transpose(1, 2).contiguous().view(B, T, self.inner_dim, 2)
        out = self.o_proj(y)
        if self.training:
            mask = self.dropout(torch.ones(B, T, self.dim, device=x.device))
            out = out * mask.unsqueeze(-1)
        return out, new_state

    # ── O(1) recurrent inference (covers all modes) ──────────────────────────

    def _recurrent(self, x, q, k, v, state, d):
        B, T = x.shape[0], x.shape[1]
        H, K = self.num_heads, self.n_states
        scale = d ** -0.5
        beta = None
        if self.write_mode == 'delta':
            beta = torch.sigmoid(self.beta_proj(cabs(x))).transpose(1, 2)  # [B,H,T]
        if self.n_states > 1:
            phi = self.phase_proj(cabs(x)).view(B, T, H, K).permute(0, 2, 3, 1)  # [B,H,K,T]

        # init state
        if state is None:
            if self.n_states > 1:
                S = torch.zeros(K, B, H, d, d, 2, device=x.device, dtype=x.dtype)
            else:
                S = torch.zeros(B, H, d, d, 2, device=x.device, dtype=x.dtype)
        else:
            S = state

        y_list = []
        for t in range(T):
            xt = x[:, t:t+1]
            k_t = k[:, :, t]
            q_t = q[:, :, t] * scale
            v_t = v[:, :, t]
            if self.n_states > 1:
                y_acc = None
                S_new = []
                for kdx in range(K):
                    gamma_k, vp_k = self._gamma_and_vprime(
                        xt, v[:, :, t:t+1], state_offset=self.state_dt_offset[kdx]
                    )
                    g = gamma_k[:, :, 0]                       # [B,H]
                    yk, Sk = self._recur_step_additive(S[kdx], g, vp_k[:, :, 0], k_t, q_t)
                    rot = torch.stack([torch.cos(phi[:, :, kdx, t]), torch.sin(phi[:, :, kdx, t])], dim=-1)
                    yk = cmul(yk, rot.unsqueeze(-2))
                    y_acc = yk if y_acc is None else y_acc + yk
                    S_new.append(Sk)
                y_list.append(y_acc)
                S = torch.stack(S_new, dim=0)
                continue

            gamma, v_prime = self._gamma_and_vprime(xt, v[:, :, t:t+1])
            if self.decay_mode == 'per_channel':
                g = gamma[:, :, 0]                            # [B,H,d]
            else:
                g = gamma[:, :, 0]                            # [B,H]
            vp_t = v_prime[:, :, 0]
            if self.write_mode == 'delta':
                yk, S = self._recur_step_delta(S, g, vp_t, k_t, q_t, beta[:, :, t])
            else:
                yk, S = self._recur_step_additive(S, g, vp_t, k_t, q_t)
            y_list.append(yk)

        y = torch.stack(y_list, dim=2)
        return y, S

    def _recur_step_additive(self, S, g, v_t, k_t, q_t):
        """One additive PAM step. g: [B,H] or [B,H,d] (per-channel). Returns y[B,H,d,2], S."""
        k_conj = torch.stack([k_t[..., 0], -k_t[..., 1]], dim=-1).unsqueeze(-3)  # [B,H,1,d,2]
        # outer[i,j] = v_t[i] * conj(k_t)[j]
        outer_r = v_t[..., 0].unsqueeze(-1) * k_conj[..., 0] - v_t[..., 1].unsqueeze(-1) * k_conj[..., 1]
        outer_i = v_t[..., 0].unsqueeze(-1) * k_conj[..., 1] + v_t[..., 1].unsqueeze(-1) * k_conj[..., 0]
        outer = torch.stack([outer_r, outer_i], dim=-1)        # [B,H,d,d,2]
        if g.dim() == S.dim() - 3:   # per-head scalar [B,H]
            gg = g.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:                        # per-channel [B,H,d] -> decay key axis (j)
            gg = g.unsqueeze(-2).unsqueeze(-1)                 # [B,H,1,d,1]
        S = S * gg + outer
        # y[i] = sum_j S[i,j] q_t[j]
        sq_r = S[..., 0] * q_t[..., 0].unsqueeze(-2) - S[..., 1] * q_t[..., 1].unsqueeze(-2)
        sq_i = S[..., 0] * q_t[..., 1].unsqueeze(-2) + S[..., 1] * q_t[..., 0].unsqueeze(-2)
        y = torch.stack([sq_r.sum(dim=-1), sq_i.sum(dim=-1)], dim=-1)
        return y, S

    def _recur_step_delta(self, S, g, v_t, k_t, q_t, beta_t):
        """One gated delta step. g:[B,H], beta_t:[B,H]. S decays, erase, write."""
        # decay
        gg = g.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        S = S * gg
        # predicted value for key: pred[i] = sum_j S[i,j] k_t[j]
        pr = (S[..., 0] * k_t[..., 0].unsqueeze(-2) - S[..., 1] * k_t[..., 1].unsqueeze(-2)).sum(dim=-1)
        pi = (S[..., 0] * k_t[..., 1].unsqueeze(-2) + S[..., 1] * k_t[..., 0].unsqueeze(-2)).sum(dim=-1)
        b = beta_t.unsqueeze(-1)
        u_r = b * (v_t[..., 0] - pr)
        u_i = b * (v_t[..., 1] - pi)
        u = torch.stack([u_r, u_i], dim=-1)                    # [B,H,d]
        k_conj = torch.stack([k_t[..., 0], -k_t[..., 1]], dim=-1)
        outer_r = u[..., 0].unsqueeze(-1) * k_conj[..., 0].unsqueeze(-2) - u[..., 1].unsqueeze(-1) * k_conj[..., 1].unsqueeze(-2)
        outer_i = u[..., 0].unsqueeze(-1) * k_conj[..., 1].unsqueeze(-2) + u[..., 1].unsqueeze(-1) * k_conj[..., 0].unsqueeze(-2)
        S = S + torch.stack([outer_r, outer_i], dim=-1)
        sq_r = S[..., 0] * q_t[..., 0].unsqueeze(-2) - S[..., 1] * q_t[..., 1].unsqueeze(-2)
        sq_i = S[..., 0] * q_t[..., 1].unsqueeze(-2) + S[..., 1] * q_t[..., 0].unsqueeze(-2)
        y = torch.stack([sq_r.sum(dim=-1), sq_i.sum(dim=-1)], dim=-1)
        return y, S


def _complex_triangular_solve(Mr, Mi, wr, wi, eye):
    """Solve (I + M) U = W for complex U, M strictly lower-tri. Block-real solve.

    Mr,Mi: [B,H,C,C]; wr,wi: [B,H,C,d]; eye: [C,C]. Returns U_r,U_i [B,H,C,d].
    """
    C = Mr.shape[-1]
    Ar = (eye + Mr).float()
    Ai = Mi.float()
    # block-real A = [[Ar,-Ai],[Ai,Ar]] ; rhs = [[wr],[wi]]
    top = torch.cat([Ar, -Ai], dim=-1)
    bot = torch.cat([Ai, Ar], dim=-1)
    A = torch.cat([top, bot], dim=-2)                          # [B,H,2C,2C]
    W = torch.cat([wr.float(), wi.float()], dim=-2)            # [B,H,2C,d]
    U = torch.linalg.solve(A, W)
    U_r, U_i = U[..., :C, :], U[..., C:, :]
    return U_r.to(wr.dtype), U_i.to(wi.dtype)


# ── V11 Block ────────────────────────────────────────────────────────────────

class V11Block(nn.Module):
    """Pre-norm residual: CGU (channel mix) + PAM (sequence mix)."""

    def __init__(self, cfg: V11Config, layer_idx: int = 0):
        super().__init__()
        self.norm1 = ComplexNorm(cfg.dim)
        self.cgu = ComplexGatedUnit(cfg.dim, cfg.expand, activation=cfg.activation)
        self.cgu_scale = nn.Parameter(torch.tensor(1.0))
        self.cgu_dropout = nn.Dropout(cfg.dropout)
        self.norm2 = ComplexNorm(cfg.dim)
        self.pam = V11PAMLayer(cfg, layer_idx=layer_idx)
        self.pam_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, pam_state=None, step_offset: int = 0):
        cgu_out = self.cgu(self.norm1(x))
        if self.training:
            drop = self.cgu_dropout(torch.ones(cgu_out.shape[:-1], device=cgu_out.device))
            cgu_out = cgu_out * drop.unsqueeze(-1)
        x = x + cgu_out * self.cgu_scale
        pam_out, new_state = self.pam(self.norm2(x), state=pam_state, step_offset=step_offset)
        x = x + pam_out * self.pam_scale
        return x, new_state


# ── V11 Language Model ──────────────────────────────────────────────────────

class V11LM(nn.Module):
    """ComplexEmbed -> [V11Block] x N -> tied complex LM head."""

    def __init__(self, cfg: V11Config):
        super().__init__()
        self.config = cfg
        self.embed = ComplexEmbed(cfg.vocab_size, cfg.dim)
        self.pos_embed = (
            ComplexPosEmbed(cfg.max_seq_len, cfg.dim) if cfg.use_learned_pos else None
        )
        self.embed_norm = ComplexNorm(cfg.dim)
        self.blocks = nn.ModuleList([V11Block(cfg, layer_idx=i) for i in range(cfg.n_layers)])
        self.output_norm = ComplexNorm(cfg.dim)
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
                nn.init.constant_(module.protect_gate.bias, -3.0)

    def forward(self, input_ids, states=None, step_offset: int = 0, labels=None):
        z = self.embed(input_ids)
        if self.pos_embed is not None:
            z = self.pos_embed(z, step_offset=step_offset)
        z = self.embed_norm(z)
        use_ckpt = self.config.gradient_checkpointing and self.training and states is None
        new_states = []
        for i, block in enumerate(self.blocks):
            s = states[i] if states is not None else None
            if use_ckpt:
                z, new_s = self._ckpt_block(block, z, step_offset)
            else:
                z, new_s = block(z, pam_state=s, step_offset=step_offset)
            new_states.append(new_s)
        z = self.output_norm(z)
        lm = self.lm_head_norm(self.lm_head_proj(z))
        logits = (
            lm[..., 0] @ self.embed.embed_real.weight.T
            + lm[..., 1] @ self.embed.embed_imag.weight.T
        )
        aux_loss = torch.tensor(0.0, device=input_ids.device)
        return logits, new_states, aux_loss

    @staticmethod
    def _ckpt_block(block, z, step_offset):
        def run(z_in):
            return block(z_in, pam_state=None, step_offset=step_offset)
        return grad_checkpoint(run, z, use_reentrant=False)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0,
                 top_k=50, top_p=0.0, repetition_penalty=1.0, eos_token_id=None):
        self.eval()
        generated = input_ids.clone()
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

def _base_flat(**kw) -> V11Config:
    cfg = V11Config(
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
    # E3 K=3: does more superposition keep helping? (head decay, clean K-sweep)
    'v11_e3_k3': _base_flat(n_states=3, state_dt_spread=2.0),
    # E1+E3 combo: per-channel decay inside each of K=2 superposed states.
    'v11_e1e3_combo': _base_flat(decay_mode='per_channel', n_states=2, state_dt_spread=2.0),
    # tiny smoke
    'tiny': V11Config(
        vocab_size=50257, dim=64, n_heads=2, head_dim=32, n_layers=2,
        expand=2, dropout=0.0, max_seq_len=512, chunk_size=64,
        gradient_checkpointing=False,
    ),
    'tiny_e1': V11Config(
        vocab_size=50257, dim=64, n_heads=2, head_dim=32, n_layers=2,
        expand=2, dropout=0.0, max_seq_len=512, chunk_size=64,
        gradient_checkpointing=False, decay_mode='per_channel',
    ),
    'tiny_e2': V11Config(
        vocab_size=50257, dim=64, n_heads=2, head_dim=32, n_layers=2,
        expand=2, dropout=0.0, max_seq_len=512, chunk_size=64,
        gradient_checkpointing=False, write_mode='delta', delta_chunk=32,
    ),
    'tiny_e3': V11Config(
        vocab_size=50257, dim=64, n_heads=2, head_dim=32, n_layers=2,
        expand=2, dropout=0.0, max_seq_len=512, chunk_size=64,
        gradient_checkpointing=False, n_states=2,
    ),
    'tiny_e1e3': V11Config(
        vocab_size=50257, dim=64, n_heads=2, head_dim=32, n_layers=2,
        expand=2, dropout=0.0, max_seq_len=512, chunk_size=64,
        gradient_checkpointing=False, decay_mode='per_channel', n_states=2,
    ),
}


def get_config(preset: str = 'v11_baseline') -> V11Config:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}")
    return copy.deepcopy(PRESETS[preset])
