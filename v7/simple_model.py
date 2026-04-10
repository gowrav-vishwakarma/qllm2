"""
Lean PAM: Stripped-down Phase-Associative Memory Language Model.

Keeps ONLY the novel core:
- Complex PAM with conjugate retrieval (the phase-interference mechanism)
- Complex RoPE on Q/K
- GSP (gated state protection)
- SimpleComplexFFN (no gating, just up -> ModSwish -> down)

Removed vs V7:
- CGU (SwiGLU-style complex gating) -> replaced by SimpleComplexFFN
- Hierarchical dt_bias / cross-level drift
- Multi-scale loss / aux heads
- Reverse association
- Triton fused kernels (plain PyTorch only)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict
import copy


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class LeanPAMConfig:
    vocab_size: int = 50257
    dim: int = 384
    n_heads: int = 6
    head_dim: int = 64
    n_layers: int = 16
    expand: int = 4
    dropout: float = 0.1
    max_seq_len: int = 2048
    chunk_size: int = 256
    gradient_checkpointing: bool = True
    soft_state_norm: bool = False
    head_diversity_lambda: float = 0.0


# ── Complex Arithmetic (split-real: [..., dim, 2]) ───────────────────────────

@torch.jit.script
def cmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1],
        a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0],
    ], dim=-1)


@torch.jit.script
def cabs(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(x[..., 0].square() + x[..., 1].square() + 1e-8)


@torch.jit.script
def to_real_concat(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x[..., 0], x[..., 1]], dim=-1)


# ── Complex Modules ──────────────────────────────────────────────────────────

class ComplexLinear(nn.Module):
    """Complex linear via split real/imag matmuls with orthogonal init."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
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


class ModSwish(nn.Module):
    """Smooth phase-preserving activation: Swish on magnitude, phase untouched."""

    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.ones(dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = cabs(z)
        activated = mag * torch.sigmoid(self.beta * mag + self.bias)
        phase = z / (mag.unsqueeze(-1) + 1e-8)
        return phase * activated.unsqueeze(-1)


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
    """Complex RoPE: e^{i*m*theta_k} for positions m and frequency bands k."""
    freqs = 1.0 / (10000.0 ** (torch.arange(head_dim).float() / head_dim))
    positions = torch.arange(max_len).float()
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)


# ── SimpleComplexFFN ─────────────────────────────────────────────────────────

class SimpleComplexFFN(nn.Module):
    """Minimal complex feed-forward: up -> ModSwish -> down. No gating."""

    def __init__(self, dim: int, expand: int = 4):
        super().__init__()
        hidden = dim * expand
        self.up_proj = ComplexLinear(dim, hidden, bias=False)
        self.act = ModSwish(hidden)
        self.down_proj = ComplexLinear(hidden, dim, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(z)))


# ── Decay matrix (plain PyTorch) ─────────────────────────────────────────────

def decay_matrix(gamma: torch.Tensor, T: int) -> torch.Tensor:
    """Causal decay matrix D[t,s] = prod_{j=s+1}^{t} gamma_j for s <= t."""
    log_gamma = torch.log(gamma + 1e-6)
    C = torch.cumsum(-log_gamma, dim=-1)
    log_D = (C.unsqueeze(-1) - C.unsqueeze(-2)).transpose(-1, -2)
    causal = torch.tril(torch.ones(T, T, device=gamma.device))
    log_D = log_D * causal + (1 - causal) * (-1e4)
    return torch.exp(log_D.clamp(max=0.0))


# ── Phase-Associative Memory (lean) ─────────────────────────────────────────

class LeanPAMLayer(nn.Module):
    r"""
    Matrix-state memory with complex-conjugate retrieval.

        S_t = gamma_t * S_{t-1} + V_t \otimes K_t^*
        Y_t = S_t * Q_t

    Always uses: complex RoPE, GSP, fused QKV.
    Never uses: hierarchical dt, cross-level drift, reverse association.
    """

    def __init__(self, cfg: LeanPAMConfig):
        super().__init__()
        self.num_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        inner = cfg.n_heads * cfg.head_dim
        self.inner_dim = inner
        self.dim = cfg.dim
        self.soft_state_norm = cfg.soft_state_norm
        self.head_diversity_lambda = cfg.head_diversity_lambda

        self.qkv_proj = ComplexLinear(cfg.dim, 3 * inner, bias=False)
        self.o_proj = ComplexLinear(inner, cfg.dim, bias=False)

        self.dt_proj = nn.Linear(cfg.dim * 2, cfg.n_heads)
        self.dt_bias = nn.Parameter(torch.zeros(cfg.n_heads) - 4.0)

        self.protect_gate = nn.Linear(cfg.dim, cfg.n_heads)
        nn.init.constant_(self.protect_gate.bias, -3.0)

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

    # ── State normalization ──────────────────────────────────────────────────

    @staticmethod
    def _soft_norm_state(S: torch.Tensor) -> torch.Tensor:
        """Soft normalize: S / (1 + S_rms). Compresses large states, leaves small ones alone."""
        S_mag = torch.sqrt(S[..., 0].square() + S[..., 1].square() + 1e-8)
        S_rms = torch.sqrt(S_mag.square().mean(dim=(-2, -1), keepdim=True) + 1e-6)
        return S / (1.0 + S_rms.unsqueeze(-1))

    # ── Dual form ────────────────────────────────────────────────────────────

    @staticmethod
    def _dual_form_block(
        q_s: torch.Tensor,
        k: torch.Tensor,
        v_prime: torch.Tensor,
        gamma: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, T = gamma.shape
        gamma_flat = gamma.reshape(B * H, T)
        D = decay_matrix(gamma_flat, T).reshape(B, H, T, T)

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

    def _forward_chunked(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v_prime: torch.Tensor,
        gamma: torch.Tensor,
        d: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, T = q.shape[:3]
        C = self.chunk_size
        scale = d ** -0.5
        q_s = q * scale

        S = q.new_zeros(B, H, d, d, 2)
        outputs = []

        for start in range(0, T, C):
            end = min(start + C, T)
            Tc = end - start

            q_c = q_s[:, :, start:end]
            k_c = k[:, :, start:end]
            v_c = v_prime[:, :, start:end]
            g_c = gamma[:, :, start:end]

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
            if self.soft_state_norm:
                S = self._soft_norm_state(S)

        return torch.cat(outputs, dim=2), S

    # ── Main forward ─────────────────────────────────────────────────────────

    def _compute_head_diversity_loss(self, k: torch.Tensor) -> torch.Tensor:
        """Penalize high complex cosine similarity between heads' key vectors.

        k: [B, H, T, d, 2] -- subsample timesteps for efficiency.
        Returns scalar loss.
        """
        stride = max(1, k.shape[2] // 16)
        k_sub = k[:, :, ::stride]  # [B, H, T', d, 2]
        k_mag = torch.sqrt(k_sub[..., 0].square() + k_sub[..., 1].square() + 1e-8)
        k_norm_r = k_sub[..., 0] / (k_mag + 1e-8)
        k_norm_i = k_sub[..., 1] / (k_mag + 1e-8)

        H = k_sub.shape[1]
        loss = torch.tensor(0.0, device=k.device, dtype=k.dtype)
        n_pairs = 0
        for i in range(H):
            for j in range(i + 1, H):
                sim = (k_norm_r[:, i] * k_norm_r[:, j]
                       + k_norm_i[:, i] * k_norm_i[:, j]).mean()
                loss = loss + sim.abs()
                n_pairs += 1
        return loss / max(n_pairs, 1)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        step_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _, _ = x.shape
        H, d = self.num_heads, self.head_dim

        qkv = self.qkv_proj(x).view(B, T, 3, H, d, 2)
        q = qkv[:, :, 0].transpose(1, 2).contiguous()
        k = qkv[:, :, 1].transpose(1, 2).contiguous()
        v = qkv[:, :, 2].transpose(1, 2).contiguous()

        # Complex RoPE on Q, K
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

        # Head diversity loss (training only)
        div_loss = torch.tensor(0.0, device=x.device)
        if self.training and self.head_diversity_lambda > 0:
            div_loss = self._compute_head_diversity_loss(k)

        # Data-dependent decay + GSP
        x_flat = to_real_concat(x)
        dt = F.softplus(self.dt_proj(x_flat) + self.dt_bias).transpose(1, 2)
        p = torch.sigmoid(self.protect_gate(cabs(x))).transpose(1, 2)
        gamma = torch.exp(-dt) * (1 - p) + p
        v_prime = v * (1 - p).unsqueeze(-1).unsqueeze(-1)

        # Training: dual form
        if state is None and T > 1:
            if self.chunk_size > 0 and T > self.chunk_size:
                y, new_state = self._forward_chunked(q, k, v_prime, gamma, d)
            else:
                scale = d ** -0.5
                q_s = q * scale
                causal = self._causal[:T, :T]
                y, new_state = self._dual_form_block(q_s, k, v_prime, gamma, causal)

        # Inference: recurrent form O(1) per token
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
                if self.soft_state_norm:
                    S = self._soft_norm_state(S)

                q_t = q[:, :, t].unsqueeze(-3) * scale
                sq_r = S[..., 0] * q_t[..., 0] - S[..., 1] * q_t[..., 1]
                sq_i = S[..., 0] * q_t[..., 1] + S[..., 1] * q_t[..., 0]
                y_list.append(torch.stack([sq_r, sq_i], dim=-1).sum(dim=-2))

            y = torch.stack(y_list, dim=2)
            new_state = S

        # Output projection + dropout
        y = y.transpose(1, 2).contiguous().view(B, T, self.inner_dim, 2)
        out = self.o_proj(y)

        if self.training:
            mask = self.dropout(torch.ones(B, T, self.dim, device=x.device))
            out = out * mask.unsqueeze(-1)

        return out, new_state, div_loss


# ── Lean PAM Block ───────────────────────────────────────────────────────────

class LeanPAMBlock(nn.Module):
    """Pre-norm residual: SimpleComplexFFN for channel mixing, PAM for sequence mixing."""

    def __init__(self, cfg: LeanPAMConfig):
        super().__init__()
        self.norm1 = ComplexNorm(cfg.dim)
        self.ffn = SimpleComplexFFN(cfg.dim, cfg.expand)
        self.ffn_scale = nn.Parameter(torch.tensor(1.0))
        self.ffn_dropout = nn.Dropout(cfg.dropout)
        self.norm2 = ComplexNorm(cfg.dim)
        self.pam = LeanPAMLayer(cfg)
        self.pam_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        x: torch.Tensor,
        pam_state: Optional[torch.Tensor] = None,
        step_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ffn_out = self.ffn(self.norm1(x))
        if self.training:
            drop_mask = self.ffn_dropout(
                torch.ones(ffn_out.shape[:-1], device=ffn_out.device)
            )
            ffn_out = ffn_out * drop_mask.unsqueeze(-1)
        x = x + ffn_out * self.ffn_scale

        pam_out, new_state, div_loss = self.pam(
            self.norm2(x), state=pam_state, step_offset=step_offset,
        )
        x = x + pam_out * self.pam_scale
        return x, new_state, div_loss


# ── Lean PAM Language Model ──────────────────────────────────────────────────

class LeanPAMLM(nn.Module):
    """
    ComplexEmbed -> [LeanPAMBlock] x N -> Tied Complex LM Head

    LM head: logits = z_r @ e_r^T + z_i @ e_i^T
    """

    def __init__(self, cfg: LeanPAMConfig):
        super().__init__()
        self.config = cfg

        self.embed = ComplexEmbed(cfg.vocab_size, cfg.dim)
        self.embed_norm = ComplexNorm(cfg.dim)
        self.blocks = nn.ModuleList([
            LeanPAMBlock(cfg) for _ in range(cfg.n_layers)
        ])
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
        for name, module in self.named_modules():
            if hasattr(module, 'protect_gate') and isinstance(module.protect_gate, nn.Linear):
                nn.init.constant_(module.protect_gate.bias, -3.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None,
        step_offset: int = 0,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        z = self.embed_norm(self.embed(input_ids))

        use_ckpt = (
            self.config.gradient_checkpointing
            and self.training
            and states is None
        )

        new_states: List[torch.Tensor] = []
        total_div_loss = torch.tensor(0.0, device=input_ids.device)
        for i, block in enumerate(self.blocks):
            s = states[i] if states is not None else None
            if use_ckpt:
                z, new_s, div_loss = self._checkpointed_block(block, z, step_offset)
            else:
                z, new_s, div_loss = block(z, pam_state=s, step_offset=step_offset)
            new_states.append(new_s)
            total_div_loss = total_div_loss + div_loss

        z = self.output_norm(z)
        lm = self.lm_head_norm(self.lm_head_proj(z))
        logits = (
            lm[..., 0] @ self.embed.embed_real.weight.T
            + lm[..., 1] @ self.embed.embed_imag.weight.T
        )
        aux_loss = total_div_loss * self.config.head_diversity_lambda
        return logits, new_states, aux_loss

    @staticmethod
    def _checkpointed_block(
        block: LeanPAMBlock,
        z: torch.Tensor,
        step_offset: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def run_block(z_in):
            return block(z_in, pam_state=None, step_offset=step_offset)
        return grad_checkpoint(run_block, z, use_reentrant=False)

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

        logits, states, _ = self.forward(generated)
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

            logits, states, _ = self.forward(
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
        total = embed_p + block_p + head_p + norm_p
        return {
            'embedding (tied)': embed_p,
            'blocks': block_p,
            'norms': norm_p,
            'lm_head': head_p,
            'total': total,
        }


# ── Presets ───────────────────────────────────────────────────────────────────

LEAN_PRESETS = {
    'lean_tiny': LeanPAMConfig(
        vocab_size=50257, dim=64, n_heads=2, head_dim=32,
        n_layers=2, expand=2, dropout=0.0, max_seq_len=512,
        chunk_size=0, gradient_checkpointing=False,
    ),
    'lean_medium': LeanPAMConfig(
        vocab_size=50257, dim=384, n_heads=6, head_dim=64,
        n_layers=16, expand=4, dropout=0.1, max_seq_len=2048,
        chunk_size=256,
    ),
    'lean_medium_small': LeanPAMConfig(
        vocab_size=50257, dim=384, n_heads=6, head_dim=64,
        n_layers=16, expand=3, dropout=0.1, max_seq_len=2048,
        chunk_size=256,
    ),
}


def get_lean_config(preset: str = 'lean_medium') -> LeanPAMConfig:
    if preset not in LEAN_PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(LEAN_PRESETS.keys())}")
    return copy.deepcopy(LEAN_PRESETS[preset])
