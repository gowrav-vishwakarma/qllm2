"""
Fused Triton kernels for V11 complex operations (vendored from V7).

Stable framework layer: these kernels operate on split-real [..., dim, 2] tensors
and are agnostic to the model architecture details (PAM, CGU, hierarchy, etc.).

Kernels provided:
  Layer 1 (elementwise, memory-bandwidth bound):
    - fused_complex_norm    replaces ComplexNorm.forward
    - fused_mod_swish       replaces ModSwish.forward
    - fused_mod_relu        replaces ModReLU.forward
    - fused_cgu_gate        replaces CGU gating (eliminates redundant cabs)

  Layer 2 (PAM-specific, compute+bandwidth):
    - fused_decay_matrix    replaces log→cumsum→sub→mask→exp chain

All functions fall back to pure PyTorch when Triton is unavailable.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# Triton JIT kernels
# ═══════════════════════════════════════════════════════════════════════════════

if HAS_TRITON:

    # ── ComplexNorm ───────────────────────────────────────────────────────────

    @triton.jit
    def _complex_norm_fwd(
        Z, Out, Scale,
        N, D: tl.constexpr, BLOCK_D: tl.constexpr,
        EPS: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= N:
            return
        base = row * D * 2
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D

        r = tl.load(Z + base + offs * 2, mask=mask, other=0.0).to(tl.float32)
        i = tl.load(Z + base + offs * 2 + 1, mask=mask, other=0.0).to(tl.float32)

        mag = tl.sqrt(r * r + i * i + 1e-8)

        mag_sq = tl.where(mask, mag * mag, 0.0)
        rms = tl.sqrt(tl.sum(mag_sq, axis=0) / D + EPS)

        scale = tl.load(Scale + offs, mask=mask, other=1.0).to(tl.float32)
        scaled = (mag / rms) * scale

        inv_mag = 1.0 / (mag + 1e-8)
        out_r = r * inv_mag * scaled
        out_i = i * inv_mag * scaled

        tl.store(Out + base + offs * 2, out_r, mask=mask)
        tl.store(Out + base + offs * 2 + 1, out_i, mask=mask)

    # ── ModSwish ──────────────────────────────────────────────────────────────

    @triton.jit
    def _mod_swish_fwd(
        Z, Out, Bias, Beta,
        N, D: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= N:
            return
        base = row * D * 2
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D

        r = tl.load(Z + base + offs * 2, mask=mask, other=0.0).to(tl.float32)
        i = tl.load(Z + base + offs * 2 + 1, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(Bias + offs, mask=mask, other=0.0).to(tl.float32)
        beta = tl.load(Beta + offs, mask=mask, other=1.0).to(tl.float32)

        mag = tl.sqrt(r * r + i * i + 1e-8)
        activated = mag * tl.sigmoid(beta * mag + bias)

        inv_mag = 1.0 / (mag + 1e-8)
        out_r = r * inv_mag * activated
        out_i = i * inv_mag * activated

        tl.store(Out + base + offs * 2, out_r, mask=mask)
        tl.store(Out + base + offs * 2 + 1, out_i, mask=mask)

    # ── ModReLU ───────────────────────────────────────────────────────────────

    @triton.jit
    def _mod_relu_fwd(
        Z, Out, Bias,
        N, D: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= N:
            return
        base = row * D * 2
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D

        r = tl.load(Z + base + offs * 2, mask=mask, other=0.0).to(tl.float32)
        i = tl.load(Z + base + offs * 2 + 1, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(Bias + offs, mask=mask, other=0.0).to(tl.float32)

        mag = tl.sqrt(r * r + i * i + 1e-8)
        activated = tl.maximum(mag + bias, 0.0)

        inv_mag = 1.0 / (mag + 1e-8)
        out_r = r * inv_mag * activated
        out_i = i * inv_mag * activated

        tl.store(Out + base + offs * 2, out_r, mask=mask)
        tl.store(Out + base + offs * 2 + 1, out_i, mask=mask)

    # ── CGU Gate ──────────────────────────────────────────────────────────────

    @triton.jit
    def _cgu_gate_fwd(
        Gate, Up, Out,
        N, D: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Fused: gate_mag * cmul(gate_phase, up). Eliminates redundant cabs."""
        row = tl.program_id(0)
        if row >= N:
            return
        base = row * D * 2
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D

        gr = tl.load(Gate + base + offs * 2, mask=mask, other=0.0).to(tl.float32)
        gi = tl.load(Gate + base + offs * 2 + 1, mask=mask, other=0.0).to(tl.float32)
        ur = tl.load(Up + base + offs * 2, mask=mask, other=0.0).to(tl.float32)
        ui = tl.load(Up + base + offs * 2 + 1, mask=mask, other=0.0).to(tl.float32)

        gmag = tl.sqrt(gr * gr + gi * gi + 1e-8)
        gate_mag = tl.sigmoid(gmag)
        inv_gmag = 1.0 / (gmag + 1e-8)
        pr = gr * inv_gmag
        pi = gi * inv_gmag

        out_r = (pr * ur - pi * ui) * gate_mag
        out_i = (pr * ui + pi * ur) * gate_mag

        tl.store(Out + base + offs * 2, out_r, mask=mask)
        tl.store(Out + base + offs * 2 + 1, out_i, mask=mask)

    # ── Decay Matrix ──────────────────────────────────────────────────────────

    @triton.jit
    def _decay_matrix_from_cumsum(
        C, D_out,
        BH, T,
        stride_c_bh: tl.constexpr, stride_c_t: tl.constexpr,
        stride_d_bh: tl.constexpr, stride_d_row: tl.constexpr,
        stride_d_col: tl.constexpr,
        BLOCK_ROW: tl.constexpr, BLOCK_COL: tl.constexpr,
    ):
        """D[t,s] = exp(clamp(C[s] - C[t], max=0)) for s <= t, else 0."""
        bh = tl.program_id(0)
        rb = tl.program_id(1)
        cb = tl.program_id(2)

        rows = rb * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
        cols = cb * BLOCK_COL + tl.arange(0, BLOCK_COL)
        rmask = rows < T
        cmask = cols < T

        c_r = tl.load(C + bh * stride_c_bh + rows * stride_c_t,
                       mask=rmask, other=0.0).to(tl.float32)
        c_c = tl.load(C + bh * stride_c_bh + cols * stride_c_t,
                       mask=cmask, other=0.0).to(tl.float32)

        log_d = c_c[None, :] - c_r[:, None]
        causal = cols[None, :] <= rows[:, None]
        log_d = tl.where(causal, tl.minimum(log_d, 0.0), -1e4)
        d = tl.exp(log_d)

        out_offs = (bh * stride_d_bh
                    + rows[:, None] * stride_d_row
                    + cols[None, :] * stride_d_col)
        tl.store(D_out + out_offs, d, mask=rmask[:, None] & cmask[None, :])


# ═══════════════════════════════════════════════════════════════════════════════
# Autograd wrappers (Triton forward, PyTorch backward)
# ═══════════════════════════════════════════════════════════════════════════════

class _FusedComplexNormFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, z: Tensor, scale: Tensor, eps: float) -> Tensor:
        orig_shape = z.shape
        feature_dim = orig_shape[-2]
        num_rows = z.numel() // (feature_dim * 2)
        # .contiguous() so Triton sees a dense [num_rows, 2*feature_dim] layout
        z_flat = z.contiguous().reshape(num_rows, feature_dim * 2)
        out_flat = torch.empty_like(z_flat)
        block_feature_dim = _next_pow2(feature_dim)

        _complex_norm_fwd[(num_rows,)](
            z_flat, out_flat, scale, num_rows, feature_dim, block_feature_dim, eps,
        )
        ctx.save_for_backward(z, scale)
        ctx.eps = eps
        return out_flat.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        z, scale = ctx.saved_tensors
        eps = ctx.eps
        grad_output_f = grad_output.float()
        z_f = z.float()
        scale_f = scale.float()
        real_part, imag_part = z_f[..., 0], z_f[..., 1]
        magnitude = torch.sqrt(real_part * real_part + imag_part * imag_part + 1e-8)
        rms = torch.sqrt(magnitude.square().mean(dim=-1, keepdim=True) + eps)
        inv_magnitude = 1.0 / (magnitude + 1e-8)
        inv_rms = 1.0 / rms
        unit_real, unit_imag = real_part * inv_magnitude, imag_part * inv_magnitude
        scaled_magnitude = magnitude * inv_rms * scale_f

        grad_out_real, grad_out_imag = grad_output_f[..., 0], grad_output_f[..., 1]

        grad_scaled = grad_out_real * unit_real + grad_out_imag * unit_imag

        grad_scale = (grad_scaled * magnitude * inv_rms).sum(
            dim=tuple(range(len(z.shape) - 2))
        )

        feature_dim = z.shape[-2]
        grad_norm_magnitude = grad_scaled * scale_f
        grad_magnitude = grad_norm_magnitude * inv_rms
        grad_rms = -(grad_norm_magnitude * magnitude * inv_rms * inv_rms).sum(-1, keepdim=True)
        grad_magnitude = grad_magnitude + grad_rms * magnitude / (feature_dim * rms)

        grad_phase_real = grad_out_real * scaled_magnitude
        grad_phase_imag = grad_out_imag * scaled_magnitude
        phase_dot = grad_phase_real * unit_real + grad_phase_imag * unit_imag
        grad_z_real = (grad_phase_real - unit_real * phase_dot) * inv_magnitude + grad_magnitude * real_part / magnitude
        grad_z_imag = (grad_phase_imag - unit_imag * phase_dot) * inv_magnitude + grad_magnitude * imag_part / magnitude

        return torch.stack([grad_z_real, grad_z_imag], dim=-1).to(grad_output.dtype), grad_scale.to(scale.dtype), None


class _FusedModSwishFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, z: Tensor, bias: Tensor, beta: Tensor) -> Tensor:
        orig_shape = z.shape
        D = orig_shape[-2]
        N = z.numel() // (D * 2)
        z_flat = z.contiguous().reshape(N, D * 2)
        out_flat = torch.empty_like(z_flat)
        BLOCK_D = _next_pow2(D)

        _mod_swish_fwd[(N,)](
            z_flat, out_flat, bias, beta, N, D, BLOCK_D,
        )

        ctx.save_for_backward(z, bias, beta)
        return out_flat.reshape(orig_shape)

    @staticmethod
    def backward(ctx, d_out: Tensor):
        z, bias, beta = ctx.saved_tensors
        d_out_f = d_out.float()
        z_f = z.float()
        bias_f, beta_f = bias.float(), beta.float()
        zr, zi = z_f[..., 0], z_f[..., 1]
        mag = torch.sqrt(zr * zr + zi * zi + 1e-8)
        inv_mag = 1.0 / (mag + 1e-8)
        pr, pi = zr * inv_mag, zi * inv_mag

        sig = torch.sigmoid(beta_f * mag + bias_f)
        activated = mag * sig
        do_r, do_i = d_out_f[..., 0], d_out_f[..., 1]

        d_activated = do_r * pr + do_i * pi
        sig_deriv = sig * (1.0 - sig)
        d_mag = d_activated * (sig + mag * beta_f * sig_deriv)

        batch_dims = tuple(range(len(z.shape) - 2))
        d_bias = (d_activated * mag * sig_deriv).sum(dim=batch_dims)
        d_beta = (d_activated * mag * mag * sig_deriv).sum(dim=batch_dims)

        d_phase_r = do_r * activated
        d_phase_i = do_i * activated
        dot = d_phase_r * pr + d_phase_i * pi
        dz_r = (d_phase_r - pr * dot) * inv_mag + d_mag * zr / mag
        dz_i = (d_phase_i - pi * dot) * inv_mag + d_mag * zi / mag

        return torch.stack([dz_r, dz_i], dim=-1).to(d_out.dtype), d_bias.to(bias.dtype), d_beta.to(beta.dtype)


class _FusedModReluFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, z: Tensor, bias: Tensor) -> Tensor:
        orig_shape = z.shape
        D = orig_shape[-2]
        N = z.numel() // (D * 2)
        z_flat = z.contiguous().reshape(N, D * 2)
        out_flat = torch.empty_like(z_flat)
        BLOCK_D = _next_pow2(D)

        _mod_relu_fwd[(N,)](
            z_flat, out_flat, bias, N, D, BLOCK_D,
        )

        ctx.save_for_backward(z, bias)
        return out_flat.reshape(orig_shape)

    @staticmethod
    def backward(ctx, d_out: Tensor):
        z, bias = ctx.saved_tensors
        d_out_f = d_out.float()
        z_f = z.float()
        bias_f = bias.float()
        zr, zi = z_f[..., 0], z_f[..., 1]
        mag = torch.sqrt(zr * zr + zi * zi + 1e-8)
        inv_mag = 1.0 / (mag + 1e-8)
        pr, pi = zr * inv_mag, zi * inv_mag

        activated = F.relu(mag + bias_f)
        active_mask = (mag + bias_f > 0).float()
        do_r, do_i = d_out_f[..., 0], d_out_f[..., 1]

        d_activated = do_r * pr + do_i * pi
        d_mag = d_activated * active_mask
        d_bias = (d_activated * active_mask).sum(
            dim=tuple(range(len(z.shape) - 2))
        )

        d_phase_r = do_r * activated
        d_phase_i = do_i * activated
        dot = d_phase_r * pr + d_phase_i * pi
        dz_r = (d_phase_r - pr * dot) * inv_mag + d_mag * zr / mag
        dz_i = (d_phase_i - pi * dot) * inv_mag + d_mag * zi / mag

        return torch.stack([dz_r, dz_i], dim=-1).to(d_out.dtype), d_bias.to(bias.dtype)


class _FusedCGUGateFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, gate: Tensor, up: Tensor) -> Tensor:
        orig_shape = gate.shape
        D = orig_shape[-2]
        N = gate.numel() // (D * 2)
        g_flat = gate.contiguous().reshape(N, D * 2)
        u_flat = up.contiguous().reshape(N, D * 2)
        out_flat = torch.empty_like(g_flat)
        BLOCK_D = _next_pow2(D)

        _cgu_gate_fwd[(N,)](
            g_flat, u_flat, out_flat, N, D, BLOCK_D,
        )
        ctx.save_for_backward(gate, up)
        return out_flat.reshape(orig_shape)

    @staticmethod
    def backward(ctx, d_out: Tensor):
        gate, up = ctx.saved_tensors
        d_out_f = d_out.float()
        gate_f, up_f = gate.float(), up.float()
        gr, gi = gate_f[..., 0], gate_f[..., 1]
        ur, ui = up_f[..., 0], up_f[..., 1]
        do_r, do_i = d_out_f[..., 0], d_out_f[..., 1]

        gmag = torch.sqrt(gr * gr + gi * gi + 1e-8)
        gate_mag = torch.sigmoid(gmag)
        inv_gmag = 1.0 / (gmag + 1e-8)
        pr, pi = gr * inv_gmag, gi * inv_gmag

        cmul_r = pr * ur - pi * ui
        cmul_i = pr * ui + pi * ur

        d_gm = do_r * cmul_r + do_i * cmul_i
        d_gmag_from_sig = d_gm * gate_mag * (1.0 - gate_mag)

        dc_r = do_r * gate_mag
        dc_i = do_i * gate_mag

        du_r = dc_r * pr + dc_i * pi
        du_i = -dc_r * pi + dc_i * pr

        dp_r = dc_r * ur + dc_i * ui
        dp_i = -dc_r * ui + dc_i * ur

        dot = dp_r * pr + dp_i * pi
        dg_r = (dp_r - pr * dot) * inv_gmag + d_gmag_from_sig * gr / gmag
        dg_i = (dp_i - pi * dot) * inv_gmag + d_gmag_from_sig * gi / gmag

        d_gate = torch.stack([dg_r, dg_i], dim=-1).to(d_out.dtype)
        d_up = torch.stack([du_r, du_i], dim=-1).to(d_out.dtype)
        return d_gate, d_up


class _FusedDecayMatrixFn(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, decay_gamma: Tensor, seq_len: int, tile_size: int = 64,
    ) -> Tensor:
        decay_gamma_f = decay_gamma.float()
        log_gamma = torch.log(decay_gamma_f + 1e-6)
        cum_neg_log_gamma = torch.cumsum(-log_gamma, dim=-1)

        batch_heads = cum_neg_log_gamma.shape[0]
        decay_matrix = cum_neg_log_gamma.new_zeros(batch_heads, seq_len, seq_len)
        cum_neg_log_gamma_contig = cum_neg_log_gamma.contiguous()
        decay_matrix_contig = decay_matrix.contiguous()

        grid = (batch_heads, (seq_len + tile_size - 1) // tile_size, (seq_len + tile_size - 1) // tile_size)
        _decay_matrix_from_cumsum[grid](
            cum_neg_log_gamma_contig, decay_matrix_contig, batch_heads, seq_len,
            cum_neg_log_gamma_contig.stride(0), cum_neg_log_gamma_contig.stride(1),
            decay_matrix_contig.stride(0), decay_matrix_contig.stride(1), decay_matrix_contig.stride(2),
            tile_size, tile_size,
        )
        ctx.save_for_backward(cum_neg_log_gamma, decay_gamma)
        ctx.seq_len = seq_len
        ctx.tile_size = tile_size
        return decay_matrix.to(decay_gamma.dtype)

    @staticmethod
    def backward(ctx, grad_decay_matrix: Tensor):
        cum_neg_log_gamma, decay_gamma = ctx.saved_tensors
        seq_len = ctx.seq_len
        tile_size = ctx.tile_size
        grad_decay_matrix_f = grad_decay_matrix.float()
        decay_gamma_f = decay_gamma.float()

        # Recompute decay_matrix from saved cum_neg_log_gamma (cheap: [BH,T] -> [BH,T,T])
        batch_heads = cum_neg_log_gamma.shape[0]
        decay_matrix = cum_neg_log_gamma.new_zeros(batch_heads, seq_len, seq_len)
        cum_neg_log_gamma_contig = cum_neg_log_gamma.contiguous()
        decay_matrix_contig = decay_matrix.contiguous()
        grid = (batch_heads, (seq_len + tile_size - 1) // tile_size, (seq_len + tile_size - 1) // tile_size)
        _decay_matrix_from_cumsum[grid](
            cum_neg_log_gamma_contig, decay_matrix_contig, batch_heads, seq_len,
            cum_neg_log_gamma_contig.stride(0), cum_neg_log_gamma_contig.stride(1),
            decay_matrix_contig.stride(0), decay_matrix_contig.stride(1), decay_matrix_contig.stride(2),
            tile_size, tile_size,
        )

        cum_col = cum_neg_log_gamma.unsqueeze(-2)
        cum_row = cum_neg_log_gamma.unsqueeze(-1)
        log_decay = cum_col - cum_row
        causal = torch.tril(torch.ones(seq_len, seq_len, device=grad_decay_matrix.device, dtype=torch.bool))
        not_clamped = causal & (log_decay <= 0)
        grad_log_decay = grad_decay_matrix_f * decay_matrix * not_clamped.float()

        grad_cum = grad_log_decay.sum(dim=-2) - grad_log_decay.sum(dim=-1)

        grad_neg_log_gamma = grad_cum.flip(-1).cumsum(-1).flip(-1)

        grad_decay_gamma = -grad_neg_log_gamma / (decay_gamma_f + 1e-6)
        return grad_decay_gamma.to(decay_gamma.dtype), None, None


# ═══════════════════════════════════════════════════════════════════════════════
# Public API — falls back to PyTorch when Triton unavailable
# ═══════════════════════════════════════════════════════════════════════════════

# ── PyTorch fallback implementations ──────────────────────────────────────────

def _pt_complex_norm(z: Tensor, scale: Tensor, eps: float = 1e-6) -> Tensor:
    mag = torch.sqrt(z[..., 0].square() + z[..., 1].square() + 1e-8)
    rms = torch.sqrt(mag.square().mean(dim=-1, keepdim=True) + eps)
    scaled = (mag / rms) * scale
    phase = z / (mag.unsqueeze(-1) + 1e-8)
    return phase * scaled.unsqueeze(-1)


def _pt_mod_swish(z: Tensor, bias: Tensor, beta: Tensor) -> Tensor:
    mag = torch.sqrt(z[..., 0].square() + z[..., 1].square() + 1e-8)
    activated = mag * torch.sigmoid(beta * mag + bias)
    phase = z / (mag.unsqueeze(-1) + 1e-8)
    return phase * activated.unsqueeze(-1)


def _pt_mod_relu(z: Tensor, bias: Tensor) -> Tensor:
    mag = torch.sqrt(z[..., 0].square() + z[..., 1].square() + 1e-8)
    activated = F.relu(mag + bias)
    phase = z / (mag.unsqueeze(-1) + 1e-8)
    return phase * activated.unsqueeze(-1)


def _pt_cgu_gate(gate: Tensor, up: Tensor) -> Tensor:
    gmag = torch.sqrt(gate[..., 0].square() + gate[..., 1].square() + 1e-8)
    gate_mag = torch.sigmoid(gmag)
    phase = gate / (gmag.unsqueeze(-1) + 1e-8)
    pr, pi = phase[..., 0], phase[..., 1]
    ur, ui = up[..., 0], up[..., 1]
    out_r = (pr * ur - pi * ui) * gate_mag
    out_i = (pr * ui + pi * ur) * gate_mag
    return torch.stack([out_r, out_i], dim=-1)


def _pt_decay_matrix(decay_gamma: Tensor, seq_len: int) -> Tensor:
    log_gamma = torch.log(decay_gamma + 1e-6)
    cum_neg_log_gamma = torch.cumsum(-log_gamma, dim=-1)
    log_decay = (cum_neg_log_gamma.unsqueeze(-1) - cum_neg_log_gamma.unsqueeze(-2)).transpose(-1, -2)
    causal = torch.tril(torch.ones(seq_len, seq_len, device=decay_gamma.device))
    log_decay = log_decay * causal + (1 - causal) * (-1e4)
    return torch.exp(log_decay.clamp(max=0.0))


# ── Public functions (auto-dispatch) ─────────────────────────────────────────

_use_triton = HAS_TRITON


def set_triton_enabled(enabled: bool) -> None:
    """Toggle Triton kernels on/off at runtime (for debugging/benchmarking)."""
    global _use_triton
    _use_triton = enabled and HAS_TRITON


def triton_enabled() -> bool:
    return _use_triton


def fused_complex_norm(z: Tensor, scale: Tensor, eps: float = 1e-6) -> Tensor:
    if _use_triton and z.is_cuda and not torch.compiler.is_compiling():
        return _FusedComplexNormFn.apply(z, scale, eps)
    return _pt_complex_norm(z, scale, eps)


def fused_mod_swish(z: Tensor, bias: Tensor, beta: Tensor) -> Tensor:
    if _use_triton and z.is_cuda and not torch.compiler.is_compiling():
        return _FusedModSwishFn.apply(z, bias, beta)
    return _pt_mod_swish(z, bias, beta)


def fused_mod_relu(z: Tensor, bias: Tensor) -> Tensor:
    if _use_triton and z.is_cuda and not torch.compiler.is_compiling():
        return _FusedModReluFn.apply(z, bias)
    return _pt_mod_relu(z, bias)


def fused_cgu_gate(gate: Tensor, up: Tensor) -> Tensor:
    if _use_triton and gate.is_cuda and not torch.compiler.is_compiling():
        return _FusedCGUGateFn.apply(gate, up)
    return _pt_cgu_gate(gate, up)


def fused_decay_matrix(decay_gamma: Tensor, seq_len: int, block: int = 64) -> Tensor:
    if _use_triton and decay_gamma.is_cuda and not torch.compiler.is_compiling():
        return _FusedDecayMatrixFn.apply(decay_gamma, seq_len, block)
    return _pt_decay_matrix(decay_gamma, seq_len)


# ═══════════════════════════════════════════════════════════════════════════════
# Correctness tests
# ═══════════════════════════════════════════════════════════════════════════════

def _test_correctness(device: str = 'cuda', atol: float = 1e-4, rtol: float = 1e-3):
    """Compare Triton kernels against PyTorch reference. Run with:
        uv run python -c "from v11.triton_kernels import _test_correctness; _test_correctness()"
    """
    import time

    torch.manual_seed(42)
    B, T, D = 3, 128, 384
    D_hidden = D * 3

    print("=" * 60)
    print("Triton kernel correctness tests")
    print(f"Device: {device}, B={B}, T={T}, D={D}, D_hidden={D_hidden}")
    print("=" * 60)

    def _check(name, triton_fn, pt_fn, *args, grad_inputs=None):
        # Forward
        out_t = triton_fn(*args)
        set_triton_enabled(False)
        out_p = pt_fn(*args)
        set_triton_enabled(True)
        max_diff = (out_t - out_p).abs().max().item()
        ok = torch.allclose(out_t, out_p, atol=atol, rtol=rtol)
        status = "PASS" if ok else "FAIL"
        print(f"  {name:25s}  fwd max_diff={max_diff:.2e}  [{status}]")

        # Backward (if grad_inputs provided)
        if grad_inputs is not None:
            # Triton backward
            loss_t = out_t.sum()
            loss_t.backward(retain_graph=False)
            grads_t = []
            for inp in grad_inputs:
                g = inp.grad
                grads_t.append(g.clone() if g is not None else None)
                if g is not None:
                    inp.grad = None

            # PyTorch backward (fresh forward)
            set_triton_enabled(False)
            out_p2 = pt_fn(*args)
            set_triton_enabled(True)
            loss_p = out_p2.sum()
            loss_p.backward()
            grads_p = []
            for inp in grad_inputs:
                g = inp.grad
                grads_p.append(g.clone() if g is not None else None)
                if g is not None:
                    inp.grad = None

            for j, (gt, gp) in enumerate(zip(grads_t, grads_p)):
                if gt is None or gp is None:
                    print(f"  {' ':25s}  bwd[{j}] SKIPPED (no grad)")
                    continue
                gd = (gt - gp).abs().max().item()
                gok = torch.allclose(gt, gp, atol=atol, rtol=rtol)
                gs = "PASS" if gok else "FAIL"
                print(f"  {' ':25s}  bwd[{j}] max_diff={gd:.2e}  [{gs}]")

    # 1. ComplexNorm
    print("\n1. ComplexNorm")
    z = torch.randn(B, T, D, 2, device=device, requires_grad=True)
    scale = torch.ones(D, device=device, requires_grad=True)
    _check("fused_complex_norm", fused_complex_norm, _pt_complex_norm,
           z, scale, 1e-6, grad_inputs=[z, scale])
    z.grad = None
    scale.grad = None

    # 2. ModSwish
    print("\n2. ModSwish")
    z2 = torch.randn(B, T, D_hidden, 2, device=device, requires_grad=True)
    bias = torch.zeros(D_hidden, device=device, requires_grad=True)
    beta = torch.ones(D_hidden, device=device, requires_grad=True)
    _check("fused_mod_swish", fused_mod_swish, _pt_mod_swish,
           z2, bias, beta, grad_inputs=[z2, bias, beta])
    z2.grad = None
    bias.grad = None
    beta.grad = None

    # 3. ModReLU
    print("\n3. ModReLU")
    z3 = torch.randn(B, T, D_hidden, 2, device=device, requires_grad=True)
    bias_r = torch.full((D_hidden,), -0.1, device=device, requires_grad=True)
    _check("fused_mod_relu", fused_mod_relu, _pt_mod_relu,
           z3, bias_r, grad_inputs=[z3, bias_r])
    z3.grad = None
    bias_r.grad = None

    # 4. CGU Gate
    print("\n4. CGU Gate")
    gate = torch.randn(B, T, D_hidden, 2, device=device, requires_grad=True)
    up = torch.randn(B, T, D_hidden, 2, device=device, requires_grad=True)
    _check("fused_cgu_gate", fused_cgu_gate, _pt_cgu_gate,
           gate, up, grad_inputs=[gate, up])
    gate.grad = None
    up.grad = None

    # 5. Decay Matrix
    print("\n5. Decay Matrix")
    H = 6
    Tc = 256
    gamma = torch.sigmoid(torch.randn(B * H, Tc, device=device)).detach().requires_grad_(True)
    _check("fused_decay_matrix",
           lambda g, T_: fused_decay_matrix(g, T_),
           lambda g, T_: _pt_decay_matrix(g, T_),
           gamma, Tc, grad_inputs=[gamma])

    print("\n" + "=" * 60)
    print("Done.")


def _benchmark(device: str = 'cuda', warmup: int = 50, rep: int = 200):
    """Benchmark Triton vs PyTorch. Run with:
        uv run python -c "from v11.triton_kernels import _benchmark; _benchmark()"
    """
    import time
    torch.manual_seed(42)

    configs = [
        ("ComplexNorm [3,2048,384,2]", 3, 2048, 384),
        ("ModSwish [3,2048,1152,2]", 3, 2048, 1152),
        ("CGU Gate [3,2048,1152,2]", 3, 2048, 1152),
    ]

    print("=" * 70)
    print(f"Benchmark: Triton vs PyTorch  (warmup={warmup}, rep={rep})")
    print("=" * 70)

    for name, B, T, D in configs:
        z = torch.randn(B, T, D, 2, device=device)
        scale = torch.ones(D, device=device)
        bias = torch.zeros(D, device=device)
        beta = torch.ones(D, device=device)
        up = torch.randn(B, T, D, 2, device=device)

        if "Norm" in name:
            fn_t = lambda: fused_complex_norm(z, scale)
            set_triton_enabled(False)
            fn_p = lambda: _pt_complex_norm(z, scale)
        elif "Swish" in name:
            fn_t = lambda: fused_mod_swish(z, bias, beta)
            set_triton_enabled(False)
            fn_p = lambda: _pt_mod_swish(z, bias, beta)
        else:
            fn_t = lambda: fused_cgu_gate(z, up)
            set_triton_enabled(False)
            fn_p = lambda: _pt_cgu_gate(z, up)
        set_triton_enabled(True)

        for _ in range(warmup):
            fn_t()
            fn_p()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(rep):
            fn_t()
        torch.cuda.synchronize()
        dt_triton = (time.perf_counter() - t0) / rep * 1000

        t0 = time.perf_counter()
        for _ in range(rep):
            fn_p()
        torch.cuda.synchronize()
        dt_pytorch = (time.perf_counter() - t0) / rep * 1000

        speedup = dt_pytorch / dt_triton if dt_triton > 0 else float('inf')
        print(f"  {name:35s}  Triton={dt_triton:.3f}ms  PyTorch={dt_pytorch:.3f}ms  speedup={speedup:.2f}x")

    # Decay matrix benchmark
    H, Tc = 6, 256
    gamma = torch.sigmoid(torch.randn(3 * H, Tc, device=device))

    fn_t = lambda: fused_decay_matrix(gamma, Tc)
    set_triton_enabled(False)
    fn_p = lambda: _pt_decay_matrix(gamma, Tc)
    set_triton_enabled(True)

    for _ in range(warmup):
        fn_t()
        fn_p()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(rep):
        fn_t()
    torch.cuda.synchronize()
    dt_t = (time.perf_counter() - t0) / rep * 1000

    t0 = time.perf_counter()
    for _ in range(rep):
        fn_p()
    torch.cuda.synchronize()
    dt_p = (time.perf_counter() - t0) / rep * 1000

    sp = dt_p / dt_t if dt_t > 0 else float('inf')
    print(f"  {'DecayMatrix [18,256,256]':35s}  Triton={dt_t:.3f}ms  PyTorch={dt_p:.3f}ms  speedup={sp:.2f}x")

    print("=" * 70)
