"""Generate runnable Python training code from a project graph."""

import os
import re
import json
from pathlib import Path
from collections import defaultdict

# Modules that map directly to nn.* and don't need wrapper class definitions
PURE_NN_TYPES = {"nn.Linear", "nn.LayerNorm", "nn.Embedding", "nn.Dropout"}

# Complex arithmetic helpers + Triton fallbacks + activations
# Emitted once when any QLLM complex module is used.
COMPLEX_PREAMBLE = '''
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


# ── Fused Kernel Fallbacks (pure PyTorch) ─────────────────────────────────────

def fused_complex_norm(z: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mag = torch.sqrt(z[..., 0].square() + z[..., 1].square() + 1e-8)
    rms = torch.sqrt(mag.square().mean(dim=-1, keepdim=True) + eps)
    scaled = (mag / rms) * scale
    phase = z / (mag.unsqueeze(-1) + 1e-8)
    return phase * scaled.unsqueeze(-1)


def fused_mod_relu(z: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    mag = torch.sqrt(z[..., 0].square() + z[..., 1].square() + 1e-8)
    activated = F.relu(mag + bias)
    phase = z / (mag.unsqueeze(-1) + 1e-8)
    return phase * activated.unsqueeze(-1)


def fused_mod_swish(z: torch.Tensor, bias: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    mag = torch.sqrt(z[..., 0].square() + z[..., 1].square() + 1e-8)
    activated = mag * torch.sigmoid(beta * mag + bias)
    phase = z / (mag.unsqueeze(-1) + 1e-8)
    return phase * activated.unsqueeze(-1)


def fused_cgu_gate(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    gmag = torch.sqrt(gate[..., 0].square() + gate[..., 1].square() + 1e-8)
    gate_mag = torch.sigmoid(gmag)
    phase = gate / (gmag.unsqueeze(-1) + 1e-8)
    pr, pi = phase[..., 0], phase[..., 1]
    ur, ui = up[..., 0], up[..., 1]
    out_r = (pr * ur - pi * ui) * gate_mag
    out_i = (pr * ui + pi * ur) * gate_mag
    return torch.stack([out_r, out_i], dim=-1)


def fused_decay_matrix(gamma: torch.Tensor, T: int) -> torch.Tensor:
    log_gamma = torch.log(gamma + 1e-6)
    C = torch.cumsum(-log_gamma, dim=-1)
    log_D = (C.unsqueeze(-1) - C.unsqueeze(-2)).transpose(-1, -2)
    causal = torch.tril(torch.ones(T, T, device=gamma.device))
    log_D = log_D * causal + (1 - causal) * (-1e4)
    return torch.exp(log_D.clamp(max=0.0))
'''

# Set of all QLLM complex module types that require the preamble
COMPLEX_MODULE_TYPES = {
    "ModReLU", "ModSwish", "PhaseModulatedActivation",
    "ComplexLinear", "ComplexNorm", "ComplexGatedUnit",
    "ComplexEmbed", "PhaseAssociativeLayer", "V7Block",
    "AuxPredHead", "ComplexLMHead", "ComplexSSM",
    "PhaseAttention", "WorkingMemory",
}

# Built-in wrapper class code emitted into the model file when used.
# These are the REAL implementations from v7/model.py with Triton calls
# replaced by the pure-PyTorch fallbacks defined in COMPLEX_PREAMBLE.
BUILTIN_CLASS_CODE: dict[str, str] = {
    "ModReLU": '''
class ModReLU(nn.Module):
    """Phase-preserving activation: threshold on magnitude, phase untouched."""

    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.full((dim,), -0.1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return fused_mod_relu(z, self.bias)
''',
    "ModSwish": '''
class ModSwish(nn.Module):
    """Smooth phase-preserving activation: Swish on magnitude, phase untouched."""

    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.ones(dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return fused_mod_swish(z, self.bias, self.beta)
''',
    "PhaseModulatedActivation": '''
class PhaseModulatedActivation(nn.Module):
    """Activation that couples magnitude and phase."""

    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.ones(dim))
        self.phase_alpha = nn.Parameter(torch.zeros(dim))
        self.phase_beta = nn.Parameter(torch.zeros(dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = cabs(z)
        activated = mag * torch.sigmoid(self.beta * mag + self.bias)
        phase = z / (mag.unsqueeze(-1) + 1e-8)
        theta = self.phase_alpha * mag + self.phase_beta
        rot = torch.stack([theta.cos(), theta.sin()], dim=-1)
        phase = cmul(phase, rot)
        return phase * activated.unsqueeze(-1)
''',
    "ComplexLinear": '''
class ComplexLinear(nn.Module):
    """Complex linear via split real/imag matmuls with orthogonal init."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        scale = (2 / (in_dim + out_dim)) ** 0.5
        self.weight_real = nn.Parameter(torch.empty(out_dim, in_dim))
        self.weight_imag = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.orthogonal_(self.weight_real, gain=scale)
        nn.init.orthogonal_(self.weight_imag, gain=scale)
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_dim))
            self.bias_imag = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = x[..., 0], x[..., 1]
        yr = F.linear(xr, self.weight_real) - F.linear(xi, self.weight_imag)
        yi = F.linear(xr, self.weight_imag) + F.linear(xi, self.weight_real)
        if self.bias_real is not None:
            yr = yr + self.bias_real
            yi = yi + self.bias_imag
        return torch.stack([yr, yi], dim=-1)
''',
    "ComplexNorm": '''
class ComplexNorm(nn.Module):
    """RMSNorm for complex: normalize magnitude, preserve phase."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return fused_complex_norm(z, self.scale, self.eps)
''',
    "ComplexGatedUnit": '''
def _build_activation(name: str, dim: int) -> nn.Module:
    if name == 'swish':
        return ModSwish(dim)
    elif name == 'phase_mod':
        return PhaseModulatedActivation(dim)
    return ModReLU(dim)


class ComplexGatedUnit(nn.Module):
    """SwiGLU-style complex gating: magnitude gates how much, phase gates rotation."""

    def __init__(self, dim: int, expand: int = 3, activation: str = 'modrelu'):
        super().__init__()
        hidden = dim * expand
        self.gate_proj = ComplexLinear(dim, hidden, bias=False)
        self.up_proj = ComplexLinear(dim, hidden, bias=False)
        self.down_proj = ComplexLinear(hidden, dim, bias=False)
        self.act = _build_activation(activation, hidden)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(z)
        up = self.act(self.up_proj(z))
        gated = fused_cgu_gate(gate, up)
        return self.down_proj(gated)
''',
    "ComplexEmbed": '''
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
''',
    "AuxPredHead": '''
class AuxPredHead(nn.Module):
    """Lightweight per-layer prediction head for multi-scale temporal loss."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = ComplexNorm(dim)

    def forward(self, z: torch.Tensor, embed_real_w: torch.Tensor,
                embed_imag_w: torch.Tensor) -> torch.Tensor:
        z = self.norm(z)
        return z[..., 0] @ embed_real_w.T + z[..., 1] @ embed_imag_w.T
''',
    "PhaseAssociativeLayer": '''
class PhaseAssociativeLayer(nn.Module):
    r"""Matrix-state memory with complex-conjugate retrieval.

        S_t = gamma_t * S_{t-1} + V_t (x) K_t^*
        Y_t = S_t * Q_t

    Training: O(T^2) dual form (GPU-friendly matmuls, no sequential loop).
    Inference: O(1) per token recurrent form.
    Chunked dual form: O(T*C) for long sequences.
    """

    def __init__(self, dim: int, n_heads: int = 6, head_dim: int = 64,
                 use_rope: bool = True, use_gsp: bool = True,
                 fused_qkv: bool = True, qk_norm: bool = False,
                 hierarchical_dt: bool = True, dt_bias_init: float = -4.0,
                 chunk_size: int = 256, use_reverse_assoc: bool = True,
                 cross_level: bool = False, layer_idx: int = 0,
                 dropout: float = 0.1, max_seq_len: int = 2048):
        super().__init__()
        self.num_heads = n_heads
        self.head_dim = head_dim
        inner = n_heads * head_dim
        self.inner_dim = inner
        self.dim = dim
        self.fused_qkv = fused_qkv
        self.use_rope = use_rope
        self.use_gsp = use_gsp
        self.qk_norm = qk_norm

        if fused_qkv:
            self.qkv_proj = ComplexLinear(dim, 3 * inner, bias=False)
        else:
            self.q_proj = ComplexLinear(dim, inner, bias=False)
            self.k_proj = ComplexLinear(dim, inner, bias=False)
            self.v_proj = ComplexLinear(dim, inner, bias=False)
        self.o_proj = ComplexLinear(inner, dim, bias=False)

        self.dt_proj = nn.Linear(dim * 2, n_heads)
        self.dt_bias = nn.Parameter(torch.zeros(n_heads) + dt_bias_init)

        if use_gsp:
            self.protect_gate = nn.Linear(dim, n_heads)
            nn.init.constant_(self.protect_gate.bias, -3.0)

        if cross_level and layer_idx > 0:
            self.drift_proj = ComplexLinear(dim, inner, bias=False)

        if use_rope:
            freqs = 1.0 / (10000.0 ** (torch.arange(head_dim).float() / head_dim))
            positions = torch.arange(max_seq_len).float()
            angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
            self.register_buffer('rope_cache',
                torch.stack([angles.cos(), angles.sin()], dim=-1), persistent=False)

        self.dropout = nn.Dropout(dropout)
        self.use_reverse_assoc = use_reverse_assoc
        if use_reverse_assoc:
            self.rev_scale = nn.Parameter(torch.zeros(1))

        self.chunk_size = chunk_size
        _causal_size = chunk_size if chunk_size > 0 else max_seq_len
        self.register_buffer('_causal',
            torch.tril(torch.ones(_causal_size, _causal_size)), persistent=False)

    @staticmethod
    def _dual_form_block(q_s, k, v_prime, gamma, causal_mask, rev_scale=None):
        """Dual form on a single block."""
        B, H, T = gamma.shape
        log_gamma = torch.log(gamma + 1e-6)
        C = torch.cumsum(-log_gamma, dim=-1)
        log_D = (C.unsqueeze(-1) - C.unsqueeze(-2)).transpose(-1, -2)
        D = torch.exp(log_D.clamp(max=0.0)) * causal_mask

        qr, qi = q_s[..., 0], q_s[..., 1]
        kr, ki = k[..., 0], k[..., 1]
        wr = qr @ kr.transpose(-1, -2) + qi @ ki.transpose(-1, -2)
        wi = qi @ kr.transpose(-1, -2) - qr @ ki.transpose(-1, -2)

        ar, ai = wr * D, wi * D
        vpr, vpi = v_prime[..., 0], v_prime[..., 1]
        yr = ar @ vpr - ai @ vpi
        yi = ar @ vpi + ai @ vpr
        y = torch.stack([yr, yi], dim=-1)

        if rev_scale is not None:
            ar_rev = wr.transpose(-1, -2) * D
            ai_rev = wi.transpose(-1, -2) * D
            yr_rev = ar_rev @ vpr - ai_rev @ vpi
            yi_rev = ar_rev @ vpi + ai_rev @ vpr
            y = y + rev_scale * torch.stack([yr_rev, yi_rev], dim=-1)

        D_last = D[:, :, -1, :]
        wv_r = vpr * D_last.unsqueeze(-1)
        wv_i = vpi * D_last.unsqueeze(-1)
        sr = wv_r.transpose(-1, -2) @ kr + wv_i.transpose(-1, -2) @ ki
        si = wv_i.transpose(-1, -2) @ kr - wv_r.transpose(-1, -2) @ ki
        S_block = torch.stack([sr, si], dim=-1)
        return y, S_block

    def forward(self, x: torch.Tensor, state=None, step_offset: int = 0,
                drift_signal=None):
        B, T, _, _ = x.shape
        H, d = self.num_heads, self.head_dim

        if self.fused_qkv:
            qkv = self.qkv_proj(x).view(B, T, 3, H, d, 2)
            q = qkv[:, :, 0].transpose(1, 2).contiguous()
            k = qkv[:, :, 1].transpose(1, 2).contiguous()
            v = qkv[:, :, 2].transpose(1, 2).contiguous()
        else:
            q = self.q_proj(x).view(B, T, H, d, 2).transpose(1, 2)
            k = self.k_proj(x).view(B, T, H, d, 2).transpose(1, 2)
            v = self.v_proj(x).view(B, T, H, d, 2).transpose(1, 2)

        if self.use_rope:
            pos = self.rope_cache[step_offset:step_offset + T].to(dtype=x.dtype)
            q = cmul(q, pos)
            k = cmul(k, pos)

        if self.qk_norm:
            q = cnormalize(q)
            k = cnormalize(k)

        if drift_signal is not None and hasattr(self, 'drift_proj'):
            drift_q = self.drift_proj(drift_signal).view(B, T, H, d, 2).transpose(1, 2)
            q = q + drift_q

        x_flat = to_real_concat(x)
        dt = F.softplus(self.dt_proj(x_flat) + self.dt_bias).transpose(1, 2)

        if self.use_gsp:
            p = torch.sigmoid(self.protect_gate(cabs(x))).transpose(1, 2)
            gamma = torch.exp(-dt) * (1 - p) + p
            v_prime = v * (1 - p).unsqueeze(-1).unsqueeze(-1)
        else:
            gamma = torch.exp(-dt)
            v_prime = v

        _rev = self.rev_scale if self.use_reverse_assoc else None
        scale = d ** -0.5
        q_s = q * scale
        causal = self._causal[:T, :T]
        y, new_state = self._dual_form_block(q_s, k, v_prime, gamma, causal, _rev)

        y = y.transpose(1, 2).contiguous().view(B, T, self.inner_dim, 2)
        out = self.o_proj(y)
        if self.training:
            mask = self.dropout(torch.ones(B, T, self.dim, device=x.device))
            out = out * mask.unsqueeze(-1)
        return out, new_state
''',
    "V7Block": '''
class V7Block(nn.Module):
    """Pre-norm residual: CGU for channel mixing, PAM for sequence mixing."""

    def __init__(self, dim: int, expand: int = 3, activation: str = 'modrelu',
                 dropout: float = 0.1, n_heads: int = 6, head_dim: int = 64,
                 use_rope: bool = True, use_gsp: bool = True,
                 fused_qkv: bool = True, chunk_size: int = 256,
                 use_reverse_assoc: bool = True, max_seq_len: int = 2048,
                 dt_bias_init: float = -4.0, layer_idx: int = 0,
                 cross_level: bool = False):
        super().__init__()
        self.norm1 = ComplexNorm(dim)
        self.cgu = ComplexGatedUnit(dim, expand, activation=activation)
        self.cgu_scale = nn.Parameter(torch.tensor(1.0))
        self.cgu_dropout = nn.Dropout(dropout)
        self.norm2 = ComplexNorm(dim)
        self.pam = PhaseAssociativeLayer(
            dim=dim, n_heads=n_heads, head_dim=head_dim,
            use_rope=use_rope, use_gsp=use_gsp, fused_qkv=fused_qkv,
            chunk_size=chunk_size, use_reverse_assoc=use_reverse_assoc,
            max_seq_len=max_seq_len, dt_bias_init=dt_bias_init,
            layer_idx=layer_idx, cross_level=cross_level, dropout=dropout,
        )
        self.pam_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, pam_state=None, step_offset: int = 0,
                drift_signal=None):
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
''',
    "ComplexLMHead": '''
class ComplexLMHead(nn.Module):
    """Tied complex LM head: logits = z_r @ E_r^T + z_i @ E_i^T"""

    def __init__(self, dim: int, vocab_size: int = 50257):
        super().__init__()
        self.proj = ComplexLinear(dim, dim)
        self.norm = ComplexNorm(dim)
        self.out_proj = nn.Linear(dim * 2, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lm = self.norm(self.proj(x))
        flat = lm.reshape(*lm.shape[:-2], -1)
        return self.out_proj(flat)
''',
    "ComplexSSM": '''
class ComplexSSM(nn.Module):
    """Complex-valued State Space Model."""
    def __init__(self, dim: int, state_dim: int = 64, n_layers: int = 1):
        super().__init__()
        self.proj_in = ComplexLinear(dim, state_dim, bias=False)
        self.proj_out = ComplexLinear(state_dim, dim, bias=False)
        self.norm = ComplexNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(self.norm(x))
        return self.proj_out(h)
''',
    "PhaseAttention": '''
class PhaseAttention(nn.Module):
    """Sparse windowed attention in complex space."""
    def __init__(self, dim: int, n_heads: int = 4, window_size: int = 256):
        super().__init__()
        self.q_proj = ComplexLinear(dim, dim, bias=False)
        self.k_proj = ComplexLinear(dim, dim, bias=False)
        self.v_proj = ComplexLinear(dim, dim, bias=False)
        self.out_proj = ComplexLinear(dim, dim, bias=False)
        self.norm = ComplexNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        q, k, v = self.q_proj(normed), self.k_proj(normed), self.v_proj(normed)
        return self.out_proj(v)
''',
    "WorkingMemory": '''
class WorkingMemory(nn.Module):
    def __init__(self, dim: int, n_slots: int = 32):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(1, n_slots, dim, 2) * 0.02)
        self.query = ComplexLinear(dim, dim, bias=False)
        self.gate = nn.Linear(dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
''',
    # Standard modules
    "TransformerEncoderLayer": '''
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True
        )
    def forward(self, x):
        return self.layer(x)
''',
    "MultiheadAttention": '''
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return out
''',
    "FeedForward": '''
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.w2(F.gelu(self.w1(x))))
''',
    "SwiGLU": '''
class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 0):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
''',
    "RMSNorm": '''
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
''',
    "RotaryEmbedding": '''
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    def forward(self, x):
        return x
''',
    "LMHead": '''
class LMHead(nn.Module):
    def __init__(self, dim: int, vocab_size: int = 50257):
        super().__init__()
        self.proj = nn.Linear(dim, vocab_size, bias=False)
    def forward(self, x):
        return self.proj(x)
''',
    "Residual": '''
class Residual(nn.Module):
    """Residual connection with learnable scale."""
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
    def forward(self, x, residual):
        return residual + x * self.scale
''',
    "Dropout": '''
class DropoutModule(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p)
    def forward(self, x):
        return self.dropout(x)
''',
    "MambaBlock": '''
class MambaBlock(nn.Module):
    def __init__(self, dim: int, state_dim: int = 16, expand: int = 2):
        super().__init__()
        inner_dim = dim * expand
        self.in_proj = nn.Linear(dim, inner_dim * 2, bias=False)
        self.conv1d = nn.Conv1d(inner_dim, inner_dim, kernel_size=4, padding=3, groups=inner_dim)
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)
        x_branch = self.conv1d(x_branch.transpose(1, 2))[..., :x.shape[1]].transpose(1, 2)
        x_branch = F.silu(x_branch)
        z = F.silu(z)
        return self.out_proj(x_branch * z) + residual
''',
    "S4Block": '''
class S4Block(nn.Module):
    def __init__(self, dim: int, state_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.D = nn.Parameter(torch.ones(dim))
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        residual = x
        x = self.norm(x)
        return self.dropout(self.proj(x) * self.D) + residual
''',
    "RepeatNLayers": '''
class RepeatNLayers(nn.Module):
    def __init__(self, n_layers: int = 6):
        super().__init__()
        self.n_layers = n_layers
    def forward(self, x):
        return x
''',
}


def _sanitize_id(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", s)


def _to_snake_case(name: str) -> str:
    s1 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower().replace(".", "_").replace(" ", "_")


def topological_sort(nodes: list[dict], connections: list[dict]) -> list[str]:
    """Sort node IDs in dependency order."""
    graph: dict[str, list[str]] = {n["id"]: [] for n in nodes}
    in_degree: dict[str, int] = {n["id"]: 0 for n in nodes}

    for conn in connections:
        src, tgt = conn["source"], conn["target"]
        if src in graph and tgt in graph:
            graph[src].append(tgt)
            in_degree[tgt] = in_degree.get(tgt, 0) + 1

    queue = [n for n in in_degree if in_degree[n] == 0]
    result = []
    while queue:
        node = queue.pop(0)
        result.append(node)
        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    remaining = [n["id"] for n in nodes if n["id"] not in result]
    return result + remaining


def generate_code(project_data: dict) -> dict:
    """Generate model.py, train.py, config.py from the project graph."""
    project = project_data.get("project", {})
    nodes = project_data.get("nodes", [])
    connections = project_data.get("connections", [])
    custom_modules = project_data.get("customModules", {})
    training = project_data.get("training", {})

    output_dir = project.get("outputDir", "./experiments/default")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "modules").mkdir(exist_ok=True)

    files_written = []

    # --- Write custom modules ---
    for mod_name, mod_def in custom_modules.items():
        code = mod_def.get("code", "")
        filename = f"{_to_snake_case(mod_name)}.py"
        filepath = output_path / "modules" / filename
        module_code = f'"""Auto-generated module: {mod_name}"""\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n{code}\n'
        filepath.write_text(module_code)
        files_written.append(f"modules/{filename}")

    (output_path / "modules" / "__init__.py").write_text("")
    files_written.append("modules/__init__.py")

    # --- Build variable flow map ---
    sorted_ids = topological_sort(nodes, connections)
    node_map = {n["id"]: n for n in nodes}

    # Map: target_node -> {target_port: (source_node, source_port)}
    input_map: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    for conn in connections:
        input_map[conn["target"]][conn.get("targetPort", "x")] = (
            conn["source"],
            conn.get("sourcePort", "out"),
        )

    # Map: node_id -> output variable names
    output_vars: dict[str, dict[str, str]] = {}

    # --- Determine if we need the complex preamble ---
    used_types = {node_map[nid]["type"] for nid in sorted_ids if nid in node_map}
    needs_complex_preamble = bool(used_types & COMPLEX_MODULE_TYPES)

    # --- Generate model.py ---
    imports = [
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        "import math",
    ]
    custom_imports = set()
    for mod_name in custom_modules:
        snake = _to_snake_case(mod_name)
        custom_imports.add(f"from modules.{snake} import {mod_name}")

    init_lines = []
    forward_lines = []

    first_node_id = sorted_ids[0] if sorted_ids else None

    for node_id in sorted_ids:
        node = node_map.get(node_id)
        if not node:
            continue

        safe_id = _sanitize_id(node_id)
        mod_type = node["type"]
        params = node.get("params", {})

        class_name = mod_type

        # Build constructor args
        param_strs = []
        for k, v in params.items():
            if isinstance(v, str):
                param_strs.append(f'{k}="{v}"')
            elif isinstance(v, bool):
                param_strs.append(f"{k}={v}")
            else:
                param_strs.append(f"{k}={v}")

        init_lines.append(f"        self.{safe_id} = {class_name}({', '.join(param_strs)})")

        # Build forward call — use port ordering from node's input port list
        in_ports = input_map.get(node_id, {})

        input_args = []
        if not in_ports:
            if node_id == first_node_id:
                input_args.append("input_ids")
            else:
                input_args.append("x")
        else:
            # Use the node's declared input port order (from inputPorts on the node data)
            node_inputs = node.get("inputPorts", [])
            if node_inputs:
                port_order = [p["name"] if isinstance(p, dict) else p for p in node_inputs]
            else:
                port_order = sorted(in_ports.keys())

            for port_name in port_order:
                if port_name not in in_ports:
                    continue
                src_node, src_port = in_ports[port_name]
                src_safe = _sanitize_id(src_node)
                var_name = output_vars.get(src_node, {}).get(src_port, f"{src_safe}_out")
                input_args.append(var_name)

        out_var = f"{safe_id}_out"
        forward_lines.append(f"        {out_var} = self.{safe_id}({', '.join(input_args)})")

        # Handle modules that return tuples
        node_outputs = node.get("outputPorts", [])
        if node_outputs and len(node_outputs) > 1:
            out_names = [p["name"] if isinstance(p, dict) else p for p in node_outputs]
            tuple_vars = [f"{safe_id}_{name}" for name in out_names]
            forward_lines[-1] = f"        {', '.join(tuple_vars)} = self.{safe_id}({', '.join(input_args)})"
            output_vars[node_id] = {name: var for name, var in zip(out_names, tuple_vars)}
        else:
            output_vars[node_id] = {"out": out_var, "logits": out_var}

    last_node_id = sorted_ids[-1] if sorted_ids else None
    last_var = _sanitize_id(last_node_id) + "_out" if last_node_id else "x"
    # If last node returns tuple, use first output
    if last_node_id and last_node_id in output_vars:
        last_out = output_vars[last_node_id]
        if "logits" in last_out:
            last_var = last_out["logits"]
        elif "out" in last_out:
            last_var = last_out["out"]
        else:
            last_var = list(last_out.values())[0]

    # Collect needed built-in class definitions (order matters for dependencies)
    DEPENDENCY_ORDER = [
        "ModReLU", "ModSwish", "PhaseModulatedActivation",
        "ComplexLinear", "ComplexNorm",
        "ComplexEmbed", "ComplexGatedUnit",
        "PhaseAssociativeLayer", "V7Block", "AuxPredHead",
        "ComplexSSM", "PhaseAttention", "WorkingMemory",
        "ComplexLMHead",
        "MambaBlock", "S4Block", "LMHead", "TransformerEncoderLayer",
        "MultiheadAttention", "FeedForward", "SwiGLU", "RMSNorm", "RotaryEmbedding",
        "Residual", "Dropout", "RepeatNLayers",
    ]

    # Auto-add dependencies for complex modules
    complex_dependents = {
        "ComplexGatedUnit", "PhaseAssociativeLayer", "ComplexSSM",
        "PhaseAttention", "WorkingMemory", "ComplexLMHead", "V7Block",
        "AuxPredHead",
    }
    if used_types & complex_dependents:
        used_types.add("ComplexLinear")
        used_types.add("ComplexNorm")

    # CGU / V7Block need activations
    if {"ComplexGatedUnit", "V7Block"} & used_types:
        used_types.add("ModReLU")
        used_types.add("ModSwish")
        used_types.add("PhaseModulatedActivation")

    # V7Block needs CGU + PAM
    if "V7Block" in used_types:
        used_types.add("ComplexGatedUnit")
        used_types.add("PhaseAssociativeLayer")

    builtin_class_defs = []
    for cls_name in DEPENDENCY_ORDER:
        if cls_name in used_types and cls_name in BUILTIN_CLASS_CODE and cls_name not in PURE_NN_TYPES:
            builtin_class_defs.append(BUILTIN_CLASS_CODE[cls_name])

    preamble_section = COMPLEX_PREAMBLE if needs_complex_preamble else ""

    model_code = f'''{chr(10).join(imports)}
{chr(10).join(sorted(custom_imports))}
{preamble_section}
{"".join(builtin_class_defs)}

class GeneratedModel(nn.Module):
    """Auto-generated model from QLLM Architecture Builder."""

    def __init__(self):
        super().__init__()
{chr(10).join(init_lines) if init_lines else "        pass"}

    def forward(self, input_ids):
{chr(10).join(forward_lines) if forward_lines else "        pass"}
        return {last_var}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
'''
    (output_path / "model.py").write_text(model_code)
    files_written.append("model.py")

    # --- Generate config.py ---
    opt = training.get("optimizer", {})
    sched = training.get("scheduler", {})
    config_dict = {
        "project_name": project.get("name", "experiment"),
        "output_dir": output_dir,
        "checkpoint_dir": project.get("checkpointDir", output_dir + "/checkpoints"),
        "log_dir": project.get("logDir", output_dir + "/logs"),
        "dataset": training.get("dataset", "tinystories"),
        "tokenizer": training.get("tokenizer", "gpt2"),
        "seq_len": training.get("seqLen", 512),
        "batch_size": training.get("batchSize", 32),
        "optimizer": opt.get("type", "AdamW"),
        "lr": opt.get("lr", 3e-4),
        "weight_decay": opt.get("weightDecay", 0.1),
        "betas": opt.get("betas", [0.9, 0.95]),
        "scheduler": sched.get("type", "cosine"),
        "warmup_steps": sched.get("warmupSteps", 500),
        "epochs": training.get("epochs", 50),
        "grad_clip": training.get("gradClip", 1.0),
        "amp": training.get("amp", True),
        "compile": training.get("compile", False),
        "grad_accumulation": training.get("gradAccumulation", 1),
    }
    config_code = '"""Auto-generated training configuration."""\n\nconfig = ' + json.dumps(config_dict, indent=4) + '\n'
    (output_path / "config.py").write_text(config_code)
    files_written.append("config.py")

    # --- Generate train.py ---
    train_code = '''"""Auto-generated training script from QLLM Architecture Builder."""
import os
import sys
import time
import math
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import GeneratedModel
from config import config


class TextDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        return chunk[:-1].long(), chunk[1:].long()


def load_data(cfg):
    """Load and tokenize dataset."""
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_name = cfg["dataset"]
    if dataset_name == "tinystories":
        ds = load_dataset("roneneldan/TinyStories", split="train")
        text_key = "text"
    elif dataset_name == "wikitext103":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        text_key = "text"
    elif dataset_name == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text_key = "text"
    else:
        ds = load_dataset(dataset_name, split="train")
        text_key = list(ds.features.keys())[0]

    all_ids = []
    for example in ds:
        text = example[text_key]
        if text and text.strip():
            ids = tokenizer.encode(text)
            all_ids.extend(ids)
        if len(all_ids) > 50_000_000:
            break

    tokens = torch.tensor(all_ids, dtype=torch.long)
    print(f"Loaded {len(tokens):,} tokens from {dataset_name}")
    return tokens, tokenizer


def train():
    cfg = config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["log_dir"], exist_ok=True)

    tokens, tokenizer = load_data(cfg)
    dataset = TextDataset(tokens, cfg["seq_len"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=2, pin_memory=True)

    model = GeneratedModel().to(device)
    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,}")

    if cfg.get("compile", False):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=tuple(cfg["betas"]),
        weight_decay=cfg["weight_decay"],
    )

    total_steps = len(loader) * cfg["epochs"] // cfg.get("grad_accumulation", 1)
    warmup_steps = cfg["warmup_steps"]

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and device.type == "cuda")
    best_loss = float("inf")
    global_step = 0
    log_path = Path(cfg["log_dir"]) / "training.log"

    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for batch_idx, (input_ids, labels) in enumerate(loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with torch.amp.autocast("cuda", enabled=cfg["amp"] and device.type == "cuda"):
                output = model(input_ids)
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

            loss_val = loss.item()
            epoch_loss += loss_val * input_ids.size(0)
            epoch_tokens += input_ids.size(0) * input_ids.size(1)

            scaler.scale(loss / cfg.get("grad_accumulation", 1)).backward()

            if (batch_idx + 1) % cfg.get("grad_accumulation", 1) == 0:
                if cfg["grad_clip"] > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            if batch_idx % 50 == 0:
                ppl = math.exp(min(loss_val, 20))
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                tok_per_sec = epoch_tokens / max(elapsed, 1e-6)
                msg = f"epoch {epoch} step {batch_idx}/{len(loader)} loss {loss_val:.4f} ppl {ppl:.2f} lr {lr:.2e} tok/s {tok_per_sec:.0f}"
                print(msg, flush=True)
                with open(log_path, "a") as f:
                    f.write(msg + "\\n")

        avg_loss = epoch_loss / max(len(dataset), 1)
        avg_ppl = math.exp(min(avg_loss, 20))
        elapsed = time.time() - t0
        msg = f"=== Epoch {epoch} done: avg_loss={avg_loss:.4f} ppl={avg_ppl:.2f} time={elapsed:.1f}s ==="
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\\n")

        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "config": cfg,
        }

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, os.path.join(cfg["checkpoint_dir"], "best_model.pt"))
            print(f"  Saved best model (loss={best_loss:.4f})")

        torch.save(ckpt, os.path.join(cfg["checkpoint_dir"], f"epoch_{epoch}.pt"))

    print("Training complete!")


if __name__ == "__main__":
    train()
'''
    (output_path / "train.py").write_text(train_code)
    files_written.append("train.py")

    # --- Write project JSON for reference ---
    (output_path / "project.json").write_text(json.dumps(project_data, indent=2))
    files_written.append("project.json")

    # --- Generate run.sh ---
    run_sh = f"""#!/bin/bash
cd "$(dirname "$0")"
python train.py "$@"
"""
    (output_path / "run.sh").write_text(run_sh)
    os.chmod(output_path / "run.sh", 0o755)
    files_written.append("run.sh")

    return {"outputDir": str(output_path), "files": files_written}
