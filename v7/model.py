"""
V7: Lean Phase-Associative Memory Language Model.

Architecture: neither transformer nor SSM.
- Channel mixing: ComplexGatedUnit (CGU) -- SwiGLU-style, phase-safe
- Sequence mixing: Phase-Associative Memory (PAM) -- matrix state, complex-conjugate retrieval
- Interleaved: [pre-norm CGU + pre-norm PAM] x N blocks

Complex representation: [..., dim, 2] split-real tensors.
DO NOT use torch.complex64/128 (OOM + autograd issues).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class V7Config:
    vocab_size: int = 50257
    dim: int = 384
    n_heads: int = 6
    head_dim: int = 64
    n_layers: int = 16
    expand: int = 3
    dropout: float = 0.1
    max_seq_len: int = 2048
    use_rope: bool = True
    use_gsp: bool = True
    fused_qkv: bool = True
    qk_norm: bool = False
    tie_weights: bool = True
    # Hierarchical timescale: each PAM layer gets a distinct dt_bias
    # controlling its memory span (global -> step).
    # When True + dt_bias_schedule is None, auto-generates linspace(-6.91, 0.0, n_layers).
    hierarchical_dt: bool = True
    dt_bias_schedule: Optional[tuple] = None
    cross_level: bool = False


# ── Complex Arithmetic (split-real: [..., dim, 2]) ───────────────────────────

@torch.jit.script
def cmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(a_r + i·a_i)(b_r + i·b_i)"""
    return torch.stack([
        a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1],
        a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0],
    ], dim=-1)


@torch.jit.script
def cconj(x: torch.Tensor) -> torch.Tensor:
    return torch.stack([x[..., 0], -x[..., 1]], dim=-1)


@torch.jit.script
def cabs(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(x[..., 0].square() + x[..., 1].square() + 1e-8)


@torch.jit.script
def cnormalize(x: torch.Tensor) -> torch.Tensor:
    return x / cabs(x).unsqueeze(-1)


@torch.jit.script
def to_real_concat(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x[..., 0], x[..., 1]], dim=-1)


# ── Complex Modules ──────────────────────────────────────────────────────────

class ModReLU(nn.Module):
    """Phase-preserving activation: threshold on magnitude, phase untouched."""

    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.full((dim,), -0.1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = cabs(z)
        activated = F.relu(mag + self.bias)
        phase = z / (mag.unsqueeze(-1) + 1e-8)
        return phase * activated.unsqueeze(-1)


class ComplexLinear(nn.Module):
    """Complex linear with block-real GEMM and orthogonal init.

    Fuses four real matmuls into one via:
        [W_r  -W_i] [x_r]   [y_r]
        [W_i   W_r] [x_i] = [y_i]
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        gain = 1.0 / math.sqrt(2)
        self.weight_real = nn.Parameter(torch.empty(out_dim, in_dim))
        self.weight_imag = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.orthogonal_(self.weight_real, gain=gain)
        nn.init.orthogonal_(self.weight_imag, gain=gain)
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_dim))
            self.bias_imag = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = torch.cat([x[..., 0], x[..., 1]], dim=-1)
        W = torch.cat([
            torch.cat([self.weight_real, -self.weight_imag], dim=1),
            torch.cat([self.weight_imag, self.weight_real], dim=1),
        ], dim=0)
        b = None
        if self.bias_real is not None:
            b = torch.cat([self.bias_real, self.bias_imag])
        y = F.linear(x_flat, W, b)
        return torch.stack([y[..., :self.out_dim], y[..., self.out_dim:]], dim=-1)


class ComplexNorm(nn.Module):
    """RMSNorm for complex: normalize magnitude, preserve phase."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = cabs(z)
        rms = torch.sqrt(mag.square().mean(dim=-1, keepdim=True) + self.eps)
        scaled = (mag / rms) * self.scale
        phase = z / (mag.unsqueeze(-1) + 1e-8)
        return phase * scaled.unsqueeze(-1)


class ComplexGatedUnit(nn.Module):
    """SwiGLU-style complex gating: magnitude gates how much, phase gates rotation."""

    def __init__(self, dim: int, expand: int = 3):
        super().__init__()
        hidden = dim * expand
        self.gate_proj = ComplexLinear(dim, hidden, bias=False)
        self.up_proj = ComplexLinear(dim, hidden, bias=False)
        self.down_proj = ComplexLinear(hidden, dim, bias=False)
        self.act = ModReLU(hidden)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(z)
        gate_mag = torch.sigmoid(cabs(gate))
        gate_phase = gate / (cabs(gate).unsqueeze(-1) + 1e-8)
        up = self.act(self.up_proj(z))
        gated = cmul(gate_phase, up) * gate_mag.unsqueeze(-1)
        return self.down_proj(gated)


class ComplexEmbed(nn.Module):
    """Embed tokens into complex space: real + imaginary components."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.dim = dim
        self.embed_real = nn.Embedding(vocab_size, dim)
        self.embed_imag = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embed_real.weight, std=0.02)
        nn.init.normal_(self.embed_imag.weight, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.embed_real(ids), self.embed_imag(ids)], dim=-1)


def build_rope_cache(max_len: int, head_dim: int) -> torch.Tensor:
    """Complex RoPE: e^{i·m·theta_k} for positions m and frequency bands k."""
    freqs = 1.0 / (10000.0 ** (torch.arange(head_dim).float() / head_dim))
    positions = torch.arange(max_len).float()
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)


# ── Phase-Associative Memory ─────────────────────────────────────────────────

class PhaseAssociativeLayer(nn.Module):
    r"""
    Matrix-state memory with complex-conjugate retrieval.

        S_t = gamma_t · S_{t-1} + V_t \otimes K_t^*
        Y_t = S_t · Q_t

    Training: O(T^2) dual form (GPU-friendly matmuls, no sequential loop).
    Inference: O(1) per token recurrent form.
    """

    def __init__(self, cfg: V7Config, layer_idx: int = 0):
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

        if cfg.fused_qkv:
            self.qkv_proj = ComplexLinear(cfg.dim, 3 * inner, bias=False)
        else:
            self.q_proj = ComplexLinear(cfg.dim, inner, bias=False)
            self.k_proj = ComplexLinear(cfg.dim, inner, bias=False)
            self.v_proj = ComplexLinear(cfg.dim, inner, bias=False)
        self.o_proj = ComplexLinear(inner, cfg.dim, bias=False)

        # Hierarchical dt_bias: each layer gets its own base decay rate
        if cfg.hierarchical_dt and cfg.dt_bias_schedule is not None:
            base_dt = cfg.dt_bias_schedule[layer_idx]
        else:
            base_dt = -4.0
        self.dt_proj = nn.Linear(cfg.dim * 2, cfg.n_heads)
        self.dt_bias = nn.Parameter(torch.zeros(cfg.n_heads) + base_dt)

        if cfg.use_gsp:
            self.protect_gate = nn.Linear(cfg.dim, cfg.n_heads)
            nn.init.constant_(self.protect_gate.bias, -3.0)

        # Cross-level drift: project higher layer's PAM output into Q-space
        if cfg.cross_level and layer_idx > 0:
            self.drift_proj = ComplexLinear(cfg.dim, inner, bias=False)

        if cfg.use_rope:
            self.register_buffer(
                'rope_cache',
                build_rope_cache(cfg.max_seq_len, cfg.head_dim),
                persistent=False,
            )

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        step_offset: int = 0,
        drift_signal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _, _ = x.shape
        H, d = self.num_heads, self.head_dim

        # 1. Q/K/V projections
        if self.fused_qkv:
            qkv = self.qkv_proj(x).view(B, T, 3, H, d, 2)
            q = qkv[:, :, 0].transpose(1, 2).contiguous()
            k = qkv[:, :, 1].transpose(1, 2).contiguous()
            v = qkv[:, :, 2].transpose(1, 2).contiguous()
        else:
            q = self.q_proj(x).view(B, T, H, d, 2).transpose(1, 2)
            k = self.k_proj(x).view(B, T, H, d, 2).transpose(1, 2)
            v = self.v_proj(x).view(B, T, H, d, 2).transpose(1, 2)

        # 1b. Complex RoPE on Q, K
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

        # 1c. Optional QK normalization
        if self.qk_norm:
            q = cnormalize(q)
            k = cnormalize(k)

        # 1d. Cross-level drift: bias Q toward higher layer's goal
        if drift_signal is not None and hasattr(self, 'drift_proj'):
            drift_q = self.drift_proj(drift_signal).view(B, T, H, d, 2).transpose(1, 2)
            q = q + drift_q

        # 2. Data-dependent decay + GSP
        x_flat = to_real_concat(x)
        dt = F.softplus(self.dt_proj(x_flat) + self.dt_bias).transpose(1, 2)

        if self.use_gsp:
            p = torch.sigmoid(self.protect_gate(cabs(x))).transpose(1, 2)
            gamma = torch.exp(-dt) * (1 - p) + p
            v_prime = v * (1 - p).unsqueeze(-1).unsqueeze(-1)
        else:
            gamma = torch.exp(-dt)
            v_prime = v

        # 3a. Training: Dual Form O(T^2) -- no sequential loop
        if state is None and T > 1:
            log_gamma = torch.log(gamma + 1e-6)
            C = torch.cumsum(-log_gamma, dim=-1)
            log_D = (C.unsqueeze(-1) - C.unsqueeze(-2)).transpose(-1, -2)
            causal = torch.tril(torch.ones(T, T, device=x.device))
            log_D = log_D * causal + (1 - causal) * (-1e4)
            D = torch.exp(log_D.clamp(max=0.0))

            scale = d ** -0.5
            qr, qi = q[..., 0] * scale, q[..., 1] * scale
            kr, ki = k[..., 0], k[..., 1]
            wr = qr @ kr.transpose(-1, -2) + qi @ ki.transpose(-1, -2)
            wi = qi @ kr.transpose(-1, -2) - qr @ ki.transpose(-1, -2)

            ar, ai = wr * D, wi * D
            vpr, vpi = v_prime[..., 0], v_prime[..., 1]
            yr = ar @ vpr - ai @ vpi
            yi = ar @ vpi + ai @ vpr
            y = torch.stack([yr, yi], dim=-1)

            D_last = D[:, :, -1, :]
            wv_r = v_prime[..., 0] * D_last.unsqueeze(-1)
            wv_i = v_prime[..., 1] * D_last.unsqueeze(-1)
            kr, ki = k[..., 0], k[..., 1]
            sr = wv_r.transpose(-1, -2) @ kr + wv_i.transpose(-1, -2) @ ki
            si = wv_i.transpose(-1, -2) @ kr - wv_r.transpose(-1, -2) @ ki
            new_state = torch.stack([sr, si], dim=-1)

        # 3b. Inference: Recurrent Form O(1) per token
        else:
            if state is None:
                state = torch.zeros(B, H, d, d, 2, device=x.device, dtype=x.dtype)
            scale = d ** -0.5
            y_list = []
            S = state
            for t in range(T):
                v_t = v_prime[:, :, t].unsqueeze(-2)
                k_t = k[:, :, t]
                k_conj = torch.stack([k_t[..., 0], -k_t[..., 1]], dim=-1).unsqueeze(-3)

                outer_r = v_t[..., 0] * k_conj[..., 0] - v_t[..., 1] * k_conj[..., 1]
                outer_i = v_t[..., 0] * k_conj[..., 1] + v_t[..., 1] * k_conj[..., 0]
                outer = torch.stack([outer_r, outer_i], dim=-1)

                g = gamma[:, :, t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                S = S * g + outer

                q_t = q[:, :, t].unsqueeze(-3) * scale
                sq_r = S[..., 0] * q_t[..., 0] - S[..., 1] * q_t[..., 1]
                sq_i = S[..., 0] * q_t[..., 1] + S[..., 1] * q_t[..., 0]
                y_list.append(torch.stack([sq_r, sq_i], dim=-1).sum(dim=-2))

            y = torch.stack(y_list, dim=2)
            new_state = S

        # 4. Output projection + dropout
        y = y.transpose(1, 2).contiguous().view(B, T, self.inner_dim, 2)
        out = self.o_proj(y)

        if self.training:
            mask = self.dropout(torch.ones(B, T, self.dim, device=x.device))
            out = out * mask.unsqueeze(-1)

        return out, new_state


# ── V7 Block: Interleaved CGU (channel mix) + PAM (sequence mix) ─────────────

class V7Block(nn.Module):
    """Pre-norm residual: CGU for channel mixing, PAM for sequence mixing."""

    def __init__(self, cfg: V7Config, layer_idx: int = 0):
        super().__init__()
        self.norm1 = ComplexNorm(cfg.dim)
        self.cgu = ComplexGatedUnit(cfg.dim, cfg.expand)
        self.cgu_scale = nn.Parameter(torch.tensor(1.0))
        self.cgu_dropout = nn.Dropout(cfg.dropout)
        self.norm2 = ComplexNorm(cfg.dim)
        self.pam = PhaseAssociativeLayer(cfg, layer_idx=layer_idx)
        self.pam_scale = nn.Parameter(torch.tensor(0.1))
        self._dim = cfg.dim

    def forward(
        self,
        x: torch.Tensor,
        pam_state: Optional[torch.Tensor] = None,
        step_offset: int = 0,
        drift_signal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cgu_out = self.cgu(self.norm1(x))
        if self.training:
            drop_mask = self.cgu_dropout(
                torch.ones(cgu_out.shape[:-1], device=cgu_out.device)
            )
            cgu_out = cgu_out * drop_mask.unsqueeze(-1)
        x = x + cgu_out * self.cgu_scale
        pam_out, new_state = self.pam(
            self.norm2(x), state=pam_state, step_offset=step_offset,
            drift_signal=drift_signal,
        )
        x = x + pam_out * self.pam_scale
        return x, new_state, pam_out


# ── V7 Language Model ────────────────────────────────────────────────────────

class V7LM(nn.Module):
    """
    ComplexEmbed -> [V7Block] x N -> Tied Complex LM Head

    LM head computes Re(z * conj(embed)) = z_r @ e_r^T + z_i @ e_i^T
    """

    def __init__(self, cfg: V7Config):
        super().__init__()
        # Auto-compute hierarchical schedule if enabled but not explicitly set
        if cfg.hierarchical_dt and cfg.dt_bias_schedule is None:
            cfg.dt_bias_schedule = tuple(
                torch.linspace(-6.91, 0.0, cfg.n_layers).tolist()
            )
        self.config = cfg

        self.embed = ComplexEmbed(cfg.vocab_size, cfg.dim)
        self.embed_norm = ComplexNorm(cfg.dim)
        self.blocks = nn.ModuleList([
            V7Block(cfg, layer_idx=i) for i in range(cfg.n_layers)
        ])
        self.output_norm = ComplexNorm(cfg.dim)
        self.lm_head_proj = ComplexLinear(cfg.dim, cfg.dim)
        self.lm_head_norm = ComplexNorm(cfg.dim)

        self._init_weights()

    def _init_weights(self):
        """Match V6 init: normal_(std=0.02) for nn.Linear, zero biases, re-apply customs."""
        embed_embeddings = {self.embed.embed_real, self.embed.embed_imag}
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding) and module not in embed_embeddings:
                nn.init.normal_(module.weight, std=0.02)
        self._reinit_custom_biases()

    def _reinit_custom_biases(self):
        """Re-apply custom bias values that _init_weights zeroed."""
        for name, module in self.named_modules():
            if hasattr(module, 'protect_gate') and isinstance(module.protect_gate, nn.Linear):
                nn.init.constant_(module.protect_gate.bias, -3.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None,
        step_offset: int = 0,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        z = self.embed_norm(self.embed(input_ids))

        new_states = []
        drift = None
        for i, block in enumerate(self.blocks):
            s = states[i] if states is not None else None
            z, new_s, pam_out = block(
                z, pam_state=s, step_offset=step_offset, drift_signal=drift,
            )
            new_states.append(new_s)
            drift = pam_out if self.config.cross_level else None

        z = self.output_norm(z)
        lm = self.lm_head_norm(self.lm_head_proj(z))
        logits = (
            lm[..., 0] @ self.embed.embed_real.weight.T
            + lm[..., 1] @ self.embed.embed_imag.weight.T
        )
        return logits, new_states

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.0,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids.clone()

        logits, states = self.forward(generated)
        step = generated.shape[1]

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1] / temperature

            if repetition_penalty != 1.0:
                score = torch.gather(next_logits, 1, generated)
                score = torch.where(
                    score > 0, score / repetition_penalty,
                    score * repetition_penalty,
                )
                next_logits.scatter_(1, generated, score)

            if top_k > 0:
                v, _ = next_logits.topk(min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, -1:]] = float('-inf')

            if top_p > 0:
                sorted_logits, sorted_idx = next_logits.sort(descending=True)
                cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                remove = cum_probs - sorted_logits.softmax(dim=-1) >= top_p
                sorted_logits[remove] = float('-inf')
                next_logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            next_token = torch.multinomial(next_logits.softmax(dim=-1), 1)
            generated = torch.cat([generated, next_token], dim=1)

            logits, states = self.forward(
                next_token, states=states, step_offset=step,
            )
            step += 1

        return generated

    def count_parameters(self) -> Dict[str, int]:
        embed_p = sum(p.numel() for p in self.embed.parameters())
        block_p = sum(p.numel() for b in self.blocks for p in b.parameters())
        head_p = (
            sum(p.numel() for p in self.lm_head_proj.parameters())
            + sum(p.numel() for p in self.lm_head_norm.parameters())
        )
        norm_p = (
            sum(p.numel() for p in self.embed_norm.parameters())
            + sum(p.numel() for p in self.output_norm.parameters())
        )
        return {
            'embedding (tied)': embed_p,
            'blocks': block_p,
            'norms': norm_p,
            'lm_head': head_p,
            'total': embed_p + block_p + head_p + norm_p,
        }


# ── Presets ───────────────────────────────────────────────────────────────────

PRESETS = {
    'tiny': V7Config(
        vocab_size=50257, dim=64, n_heads=2, head_dim=32,
        n_layers=2, expand=2, dropout=0.0, max_seq_len=512,
        hierarchical_dt=False,
    ),
    'medium': V7Config(
        vocab_size=50257, dim=384, n_heads=6, head_dim=64,
        n_layers=16, expand=3, dropout=0.1, max_seq_len=2048,
        hierarchical_dt=True,
    ),
    # 6-layer hierarchical: one PAM per resolution level
    # global(~1000tok) -> broad(~250) -> mid(~60) -> local(~15) -> fine(~5) -> step(~2)
    'medium_h6': V7Config(
        vocab_size=50257, dim=512, n_heads=8, head_dim=64,
        n_layers=6, expand=4, dropout=0.1, max_seq_len=2048,
        hierarchical_dt=True,
        dt_bias_schedule=(-6.91, -5.52, -4.08, -2.64, -1.39, 0.0),
        cross_level=True,
    ),
    # 16-layer flat: V6-matched shape, no hierarchy — baseline for V7 code verification
    'medium_h16_flat': V7Config(
        vocab_size=50257, dim=384, n_heads=6, head_dim=64,
        n_layers=16, expand=3, dropout=0.1, max_seq_len=2048,
        hierarchical_dt=False,
        cross_level=False,
    ),
    # 16-layer grouped hierarchy: multiple layers per timescale level
    # global(4L) -> broad(3L) -> mid(3L) -> local(3L) -> fine(2L) -> step(1L)
    'medium_h16_grouped': V7Config(
        vocab_size=50257, dim=384, n_heads=6, head_dim=64,
        n_layers=16, expand=3, dropout=0.1, max_seq_len=2048,
        hierarchical_dt=True,
        dt_bias_schedule=(
            -6.91, -6.91, -6.91, -6.91,   # layers 0-3:  global
            -5.52, -5.52, -5.52,            # layers 4-6:  broad
            -4.08, -4.08, -4.08,            # layers 7-9:  mid
            -2.64, -2.64, -2.64,            # layers 10-12: local
            -1.39, -1.39,                   # layers 13-14: fine
             0.0,                           # layer 15:    step
        ),
        cross_level=True,
    ),
}


def get_config(preset: str = 'medium') -> V7Config:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}")
    import copy
    return copy.deepcopy(PRESETS[preset])
