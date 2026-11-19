"""v17 Two-Organ O(1) Architecture in ``sempyt`` (POWLM: Phase-Operator Workspace LM).

Core principle (Gowrav, 2026-11-19):
  Intelligence and Memory are two different organs. Stuffing both into a single
  1-pass key-value outer-product state ``S_t = sum v_t k_t^T`` forces them to fight:

  1. Organ 1 — Dual-Phase Synchrony Memory (O(1) state):
     Separates phase into two orthogonal coordinates on ``(S^1)^P``:
     - WHEN: Chrono positive scalar clock ``tau_t * omega_p`` (episodic order &
       beyond-window extrapolation).
     - WHOSE / WHICH SCOPE: Unit-circle synchrony carrier phasor ``z_t in (S^1)^P``,
       integrated in ``R^2`` via ``sempyt.scan`` then normalized onto ``S^1`` so
       intra-clause filler causes ZERO phase spin (fixing v14 N9s's angle-decay bug).
     Under ``SplitComplex(rot_pair)``, ``rot_full = rot_time * z_unit`` binds
     ``(role (x) entity/scope)`` multiplicatively: matching entities phase-lock
     (``dphi = 0``) at any distance while same-role lures cancel (``<cos dphi> ~ 0``).

  2. Organ 2 — Active Intelligence Workspace (O(1) state):
     - Organ 2A (Iterative SRAM Bounces): ``R`` sequential query-refine-requery
       bounces over ``S_t`` per layer via ``pam_shared_read`` (zero WY re-solve,
       zero extra state bytes). Each bounce re-locks its own synchrony phasor
       ``z^(b)_t`` onto the hop-(b-1) entity retrieved in ``h_work``, turning
       1-hop lookup into compositional multi-hop reasoning.
     - Organ 2B (Causal Phase-Operator Engine): Maps tokens onto ``M`` learned 2D
       phase rings under ``SplitComplex(rot_pair)``. An operand latch ``xi_t``
       holds the left-hand symbol across delimiters; when outcome ``x_t`` arrives,
       the complex quotient ``r_t = zeta_out * xi_{t-1}.conj()`` extracts the
       token-invariant transformation into a slow operator state ``Omega_t``, and
       ``om_unit * zeta_q`` applies that operator to unseen inputs or retrieved facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sempyt import (
    Dim,
    NamedTensor,
    ProductDim,
    SplitComplex,
    as_complex,
    cat,
    clamp,
    cos,
    cumsum,
    exp,
    l2_normalize,
    log,
    named,
    ones,
    repeat_along,
    scan,
    select,
    shift,
    sigmoid,
    silu,
    sin,
    softplus,
    split,
    stack,
    zeros,
)
from sempyt.nn import Embed, Linear, RMSNorm, SwiGLU, TiedHead

from v13_sempty.fused_ce import fused_linear_cross_entropy
from v13_sempty.triton_kernels import pam_delta_batched, pam_shared_read


# ── Pure sempyt helpers ──────────────────────────────────────────────────────

def as_float(t: NamedTensor) -> NamedTensor:
    return NamedTensor(t.data.float(), t.layout, t.policy, bind=False)


def as_dtype(t: NamedTensor, dtype: torch.dtype) -> NamedTensor:
    return NamedTensor(t.data.to(dtype), t.layout, t.policy, bind=False)


def stateful_scan(
    u: NamedTensor,
    decay: NamedTensor,
    over: Dim,
    carry: Optional[NamedTensor] = None,
    chunk: int = 64,
) -> tuple[NamedTensor, NamedTensor]:
    """Exact named causal scan ``y_t = decay_t * y_{t-1} + u_t`` with ``carry``.

    All math runs in fp32 over named axes via ``sempyt.scan``.
    Returns ``(y, y_last)`` where ``y`` keeps ``u.layout`` and ``y_last`` omits ``over``.
    """
    orig_dtype = u.dtype
    uf = as_float(u)
    df = clamp(as_float(decay), min=1e-12, max=1.0)
    T = uf.size(over)
    rest = tuple(d for d in uf.layout if d is not over)

    if T == 1 and carry is not None:
        cf = as_float(carry).to(*rest)
        d0 = select(df, over, 0)
        u0 = select(uf, over, 0)
        y_last = (d0 * cf + u0).to(*rest)
        y_seq = stack([y_last], into=over).to(*uf.layout)
        return as_dtype(y_seq, orig_dtype), y_last

    y_zero = scan(uf, df, over=over, mode="parallel", chunk=chunk).to(*uf.layout)
    if carry is not None:
        cf = as_float(carry).to(*rest)
        cum_decay = exp(cumsum(log(df), over=over))
        y = (y_zero + cum_decay * cf).to(*uf.layout)
    else:
        y = y_zero
    y_last = select(y, over, T - 1).to(*rest)
    return as_dtype(y, orig_dtype), y_last


def unit_circle(z: NamedTensor, pair: Dim, cplx: SplitComplex, eps: float = 1e-6) -> NamedTensor:
    """Project a real or complex ``(..., pair)`` NamedTensor onto ``S^1`` under ``SplitComplex``."""
    lead = tuple(d for d in z.layout if d is not pair)
    zr = NamedTensor(z.to(*lead, pair).data.float(), lead + (pair,))
    normed = l2_normalize(zr, over=pair, eps=eps * eps)
    return NamedTensor(normed.data, lead + (pair,), cplx, bind=False)


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class POWConfig:
    vocab_size: int = 50257
    dim: int = 384
    n_layers: int = 4
    n_heads: int = 6
    head_dim: int = 64
    mlp_hidden: int = 1024
    # Organ 1: Dual-Phase Synchrony Memory
    use_rope: bool = True
    chrono: bool = True
    sync_phasor: bool = True
    sync_only_frac: float = 0.5       # fraction of planes with omega_p = 0 (pure distance-free synchrony)
    warp_min: float = -4.5            # clk3 finding: wider slow-clock floor
    warp_max: float = 3.0
    short_conv_k: int = 4             # local token-pair awareness on qkv (0 = off)
    # Organ 2A: Iterative SRAM Workspace bounces per layer (1 = standard 1-pass)
    n_bounces: int = 3
    # Organ 2B: Causal Phase-Operator Engine
    operator_engine: bool = True
    op_rings: int = 192               # number of 2D relational phase rings
    op_heads: int = 6
    echo: bool = True                 # causal token-transition memory
    # Anti-laziness MLP scale init (seed-44 autopsy: smaller init prevents static MLP shortcutting)
    mlp_scale_init: float = 0.4


# ── Organ 1 + Organ 2A: Dual-Phase Memory + Iterative SRAM Workspace ─────────

class DualPhaseWorkspaceLayer(nn.Module):
    """Delta associative memory with Dual-Phase (Chrono Time + S^1 Synchrony)
    and R-bounce iterative query-refinement workspace in SRAM, written in ``sempyt``.

    Carry state tuple per layer (strictly O(1)):
      ``(S_nb, clock_nt, z_carry_nt, conv_buf_raw)``
    """

    def __init__(self, cfg: POWConfig, model_dim: Dim, heads: Dim,
                 head_feature: Dim, head_pair: Dim, rot_pair: Dim):
        super().__init__()
        self.cfg = cfg
        self.model_dim = model_dim
        self.heads = heads
        self.head_feature = head_feature
        self.head_pair = head_pair
        self.rot_pair = rot_pair
        self.cplx = SplitComplex(rot_pair)

        self.head_row = Dim("head_row", cfg.head_dim)
        self.head_col = Dim("head_col", cfg.head_dim)
        self.qkv_slot = Dim("qkv_slot", 3)
        self.inner = ProductDim(heads, head_feature)
        self.qkv_out = ProductDim(self.qkv_slot, heads, head_feature)
        self.sync_planes = ProductDim(heads, head_pair)

        self.qkv = Linear(model_dim, self.qkv_out, bias=False)
        self.conv_k = cfg.short_conv_k
        if self.conv_k > 0:
            C = self.qkv_out.size
            self.qkv_conv = nn.Conv1d(C, C, kernel_size=self.conv_k, groups=C, bias=True)
            nn.init.zeros_(self.qkv_conv.weight)
            nn.init.zeros_(self.qkv_conv.bias)

        # Delta retention & write/erase gates
        self.dt_proj = Linear(model_dim, heads, bias=True)
        self.dt_bias = nn.Parameter(torch.linspace(-4.2, -2.5, cfg.n_heads))
        nn.init.zeros_(self.dt_proj.weight)
        nn.init.zeros_(self.dt_proj.bias)

        self.write_proj = Linear(model_dim, heads, bias=True)
        self.erase_proj = Linear(model_dim, heads, bias=True)
        nn.init.zeros_(self.write_proj.bias)
        nn.init.zeros_(self.erase_proj.bias)

        # Chrono clock (WHEN)
        self.use_rope = cfg.use_rope
        self.chrono = cfg.chrono and cfg.use_rope
        P = cfg.head_dim // 2
        p_sync = int(round(P * cfg.sync_only_frac)) if cfg.sync_phasor else 0
        p_time = P - p_sync
        if not cfg.use_rope:
            inv = torch.zeros(P)
        else:
            inv_time = (1.0 / (10000.0 ** (torch.arange(p_time).float() / max(1, p_time)))
                        if p_time > 0 else torch.zeros(0))
            inv = torch.cat([torch.zeros(p_sync), inv_time], dim=0)
        self.register_buffer("inv_freq", inv, persistent=False)

        if self.chrono:
            self.warp_proj = Linear(model_dim, heads, bias=True)
            nn.init.zeros_(self.warp_proj.weight)
            nn.init.zeros_(self.warp_proj.bias)

        # S^1 Synchrony Carrier Phasor (WHOSE / WHICH SCOPE)
        self.sync = cfg.sync_phasor
        if self.sync:
            self.sync_alpha = Linear(model_dim, self.sync_planes, bias=False)
            nn.init.zeros_(self.sync_alpha.weight)                          # identity (1, 0) at init
            self.sync_gate = Linear(model_dim, heads, bias=True)
            nn.init.normal_(self.sync_gate.weight, std=0.02)
            nn.init.constant_(self.sync_gate.bias, -1.0)

        # Bounce-1 output gate & projection
        self.out_gate = Linear(model_dim, heads, bias=True)
        nn.init.zeros_(self.out_gate.weight)
        nn.init.constant_(self.out_gate.bias, 1.2785)                       # silu(1.2785) = 1.0
        self.o_proj = Linear(self.inner, model_dim, bias=False)

        # Organ 2A: Iterative SRAM Workspace Bounces (b = 2 .. n_bounces)
        self.n_bounces = max(1, int(cfg.n_bounces))
        if self.n_bounces > 1:
            nb = self.n_bounces - 1
            self.bounce_norms = nn.ModuleList([RMSNorm(model_dim) for _ in range(nb)])
            self.bounce_q = nn.ModuleList([Linear(model_dim, self.inner, bias=False) for _ in range(nb)])
            self.bounce_gate = nn.ModuleList([Linear(model_dim, heads, bias=True) for _ in range(nb)])
            self.bounce_out = nn.ModuleList([Linear(self.inner, model_dim, bias=False) for _ in range(nb)])
            if self.sync:
                self.bounce_alpha = nn.ModuleList([
                    Linear(model_dim, self.sync_planes, bias=False) for _ in range(nb)
                ])
            for bi in range(nb):
                nn.init.zeros_(self.bounce_gate[bi].weight)
                nn.init.constant_(self.bounce_gate[bi].bias, 1.2785)
                nn.init.normal_(self.bounce_out[bi].weight, std=0.01)
                if self.sync:
                    nn.init.zeros_(self.bounce_alpha[bi].weight)

    def _turn_by_phasor(self, x: NamedTensor, rot: NamedTensor,
                        batch: Dim, time: Dim) -> NamedTensor:
        """Rotate ``x (batch, heads, time, head_feature)`` by ``SplitComplex`` phasor ``rot``."""
        xp = x.to(batch, self.heads, time, self.head_pair, self.rot_pair)
        xc = NamedTensor(xp.data.float(), xp.layout, self.cplx, bind=False)
        turned = xc * rot.to(batch, self.heads, time, self.head_pair, self.rot_pair)
        real_pair = NamedTensor(turned.data.to(x.dtype), xp.layout)
        return real_pair.to(batch, self.heads, time, self.head_feature)

    def _compute_rotations(self, x: NamedTensor, batch: Dim, time: Dim,
                           clk_in: Optional[NamedTensor],
                           z_in: Optional[NamedTensor]):
        T = x.size(time)
        dev = x.device
        if self.chrono:
            w = clamp(as_float(self.warp_proj(x)), min=self.cfg.warp_min, max=self.cfg.warp_max)
            g_clk = exp(w).to(batch, self.heads, time)
            run = cumsum(g_clk, over=time)
            if clk_in is not None:
                run = run + clk_in.alias(clk_in.layout[0], batch)
            tau = run - g_clk                                               # exclusive causal clock
            clk_out = select(run, time, T - 1)
        else:
            steps = named(torch.arange(T, device=dev, dtype=torch.float32), (time,))
            if clk_in is None:
                z_clk = zeros(batch, self.heads, device=dev, dtype=torch.float32)
                tau = z_clk + steps
                clk_out = z_clk + float(T)
            else:
                c_aligned = clk_in.alias(clk_in.layout[0], batch)
                tau = c_aligned + steps
                clk_out = c_aligned + float(T)

        inv_nt = named(self.inv_freq.to(dev), (self.head_pair,))
        phi_time = tau * inv_nt                                             # (batch, heads, time, head_pair)
        rot_time = as_complex(cos(phi_time), sin(phi_time), self.rot_pair)

        if not self.sync:
            return rot_time, rot_time, clk_out, None

        alpha = as_float(self.sync_alpha(x)).to(batch, self.heads, time, self.head_pair)
        u_real = stack([cos(alpha), sin(alpha)], into=self.rot_pair)        # real carrier increment on S^1
        s = 0.02 + 0.93 * sigmoid(as_float(self.sync_gate(x)).to(batch, self.heads, time))
        inc = (s * u_real).to(batch, self.heads, time, self.head_pair, self.rot_pair)
        keep = 1.0 - s

        if z_in is None:
            c0 = ones(batch, self.heads, self.head_pair, device=dev, dtype=torch.float32)
            s0 = zeros(batch, self.heads, self.head_pair, device=dev, dtype=torch.float32)
            z_carry_in = stack([c0, s0], into=self.rot_pair)
        else:
            z_carry_in = z_in.alias(z_in.layout[0], batch)

        z_raw, z_last = stateful_scan(inc, keep, over=time, carry=z_carry_in)
        z_unit = unit_circle(z_raw, self.rot_pair, self.cplx)
        rot_full = rot_time.to(batch, self.heads, time, self.head_pair, self.rot_pair) * z_unit
        return rot_full, rot_time, clk_out, z_last

    def forward(self, x: NamedTensor, batch: Dim, time: Dim, state: Optional[tuple] = None):
        B, T = batch.size, time.size
        H, K = self.heads.size, self.head_feature.size
        S_in, clk_in, z_in, conv_in = state if state is not None else (None, None, None, None)

        qkv = self.qkv(x)                                                   # (batch, time, qkv_out)
        conv_out = None
        if self.conv_k > 0:
            # Declared raw-torch boundary: causal depthwise Conv1d along time
            xr = qkv.raw(batch, time, self.qkv_out).transpose(1, 2)         # [B, C, T]
            xp = F.pad(xr, (self.conv_k - 1, 0)) if conv_in is None else torch.cat([conv_in.to(xr.dtype), xr], dim=2)
            conv_out = xp[:, :, -(self.conv_k - 1):].detach() if self.conv_k > 1 else None
            conv_y = F.silu(self.qkv_conv(xp)).transpose(1, 2)
            qkv = qkv + named(conv_y, (batch, time, self.qkv_out))

        qkv_u = qkv.to(batch, time, self.qkv_slot, self.heads, self.head_feature)
        q = select(qkv_u, self.qkv_slot, 0).to(batch, self.heads, time, self.head_feature)
        k = select(qkv_u, self.qkv_slot, 1).to(batch, self.heads, time, self.head_feature)
        v = select(qkv_u, self.qkv_slot, 2).to(batch, self.heads, time, self.head_feature)

        rot_full, rot_time, clk_out, z_out = self._compute_rotations(x, batch, time, clk_in, z_in)
        q1 = self._turn_by_phasor(q, rot_full, batch, time)
        kr = self._turn_by_phasor(k, rot_full, batch, time)
        kn = as_dtype(l2_normalize(as_float(kr), over=self.head_feature, eps=1e-12), x.dtype)

        dt = softplus(as_float(self.dt_proj(x)) + named(self.dt_bias.float(), (self.heads,)))
        retention = exp(-dt).to(batch, self.heads, time)
        bw = sigmoid(as_float(self.write_proj(x))).to(batch, self.heads, time)
        be = clamp(sigmoid(as_float(self.erase_proj(x)) - 3.0), max=0.95).to(batch, self.heads, time)

        # Declared raw-torch boundary: Triton delta kernel / single-step fp32 recurrence
        q1_raw = q1.raw(batch, self.heads, time, self.head_feature)
        kn_raw = kn.raw(batch, self.heads, time, self.head_feature)
        v_raw = v.raw(batch, self.heads, time, self.head_feature)
        ret_raw = retention.raw(batch, self.heads, time)
        bw_raw = bw.raw(batch, self.heads, time)
        be_raw = be.raw(batch, self.heads, time)

        if T == 1:
            S = (S_in.alias(S_in.layout[0], batch).raw(batch, self.heads, self.head_row, self.head_col).float()
                 if S_in is not None else torch.zeros(B, H, K, K, device=x.device, dtype=torch.float32))
            kt, vt, qt = kn_raw[:, :, 0].float(), v_raw[:, :, 0].float(), q1_raw[:, :, 0].float()
            Sk = torch.einsum("bhvk,bhk->bhv", S, kt)
            erased = S - be_raw[:, :, 0, None, None] * Sk.unsqueeze(-1) * kt.unsqueeze(-2)
            S_raw = ret_raw[:, :, 0, None, None] * erased + bw_raw[:, :, 0, None, None] * vt.unsqueeze(-1) * kt.unsqueeze(-2)
            r1_raw = torch.einsum("bhvk,bhk->bhv", S_raw, qt).unsqueeze(2) * (K ** -0.5)
            write_cache = None
        else:
            carry = (S_in.alias(S_in.layout[0], batch).raw(batch, self.heads, self.head_row, self.head_col)
                     .reshape(B * H, K, K).float() if S_in is not None else None)
            need_write = self.n_bounces > 1
            res = pam_delta_batched(
                q1_raw.reshape(B * H, T, K),
                kn_raw.reshape(B * H, T, K),
                v_raw.reshape(B * H, T, K),
                ret_raw.reshape(B * H, T).to(x.dtype),
                bw_raw.reshape(B * H, T).to(x.dtype),
                be_raw.reshape(B * H, T).to(x.dtype),
                carry,
                return_write=need_write,
            )
            if need_write:
                read_flat, S_flat, write_cache = res
            else:
                read_flat, S_flat = res
                write_cache = None
            r1_raw = read_flat.view(B, H, T, K).float() * (K ** -0.5)
            S_raw = S_flat.view(B, H, K, K).float()

        S_out = named(S_raw, (batch, self.heads, self.head_row, self.head_col))
        r1 = named(r1_raw, (batch, self.heads, time, self.head_feature))
        g1 = silu(as_float(self.out_gate(x))).to(batch, self.heads, time)
        y1 = as_dtype(r1 * g1, x.dtype).to(batch, time, self.inner)
        h_work = x + self.o_proj(y1)

        # Organ 2A: Iterative Multi-Bounce Reasoning over S_t in SRAM
        for bi in range(self.n_bounces - 1):
            hb = self.bounce_norms[bi](h_work)
            qb = self.bounce_q[bi](hb).to(batch, self.heads, time, self.head_feature)
            if self.sync:
                ab = as_float(self.bounce_alpha[bi](hb)).to(batch, self.heads, time, self.head_pair)
                zb = as_complex(cos(ab), sin(ab), self.rot_pair)
                rot_b = rot_time.to(batch, self.heads, time, self.head_pair, self.rot_pair) * zb
            else:
                rot_b = rot_time
            qb = self._turn_by_phasor(qb, rot_b, batch, time)
            qb_raw = qb.raw(batch, self.heads, time, self.head_feature)
            if T == 1 and write_cache is None:
                rb_raw = torch.einsum("bhvk,bhk->bhv", S_raw, qb_raw[:, :, 0].float()).unsqueeze(2) * (K ** -0.5)
            else:
                rb_raw = pam_shared_read(qb_raw.reshape(B * H, T, K), write_cache).view(B, H, T, K).float() * (K ** -0.5)
            rb = named(rb_raw, (batch, self.heads, time, self.head_feature))
            gb = silu(as_float(self.bounce_gate[bi](hb))).to(batch, self.heads, time)
            yb = as_dtype(rb * gb, x.dtype).to(batch, time, self.inner)
            h_work = h_work + self.bounce_out[bi](yb)

        return h_work - x, (S_out, clk_out, z_out, conv_out)


# ── Organ 2B: Causal Phase-Operator Engine + Echo Transition Head ────────────

class CausalOperatorEngine(nn.Module):
    """Extracts token-invariant transformations on ``M`` 2D phase rings using
    ``sempyt`` ``SplitComplex`` algebra: ``r_t = zeta_out * xi_{t-1}.conj()``,
    accumulated into ``Omega_t`` and applied via ``om_unit * zeta_q``.

    Carry state tuple (strictly O(1)):
      ``(xi_last, om_last, h_last, S_echo_out)``
    """

    def __init__(self, cfg: POWConfig, model_dim: Dim, heads: Dim,
                 head_feature: Dim, rot_pair: Dim):
        super().__init__()
        self.cfg = cfg
        self.model_dim = model_dim
        self.heads = heads
        self.head_feature = head_feature
        self.rot_pair = rot_pair
        self.cplx = SplitComplex(rot_pair)

        self.dual_dim = Dim("dual_dim", 2 * cfg.dim)
        self.op_heads = Dim("op_heads", cfg.op_heads)
        assert cfg.op_rings % cfg.op_heads == 0
        self.op_sub = Dim("op_sub", cfg.op_rings // cfg.op_heads)
        self.op_rings_flat = ProductDim(self.op_heads, self.op_sub, rot_pair)

        if cfg.operator_engine:
            self.in_coord = Linear(self.dual_dim, self.op_rings_flat, bias=False)
            self.out_coord = Linear(model_dim, self.op_rings_flat, bias=False)
            self.q_coord = Linear(self.dual_dim, self.op_rings_flat, bias=False)
            self.latch_gate = Linear(self.dual_dim, self.op_heads, bias=True)
            nn.init.constant_(self.latch_gate.bias, 0.5)

            self.gate_slot3 = Dim("gate_slot3", 3)
            self.op_gate_out = ProductDim(self.gate_slot3, self.op_heads)
            self.op_gates = Linear(self.dual_dim, self.op_gate_out, bias=True)
            with torch.no_grad():
                H_op = cfg.op_heads
                self.op_gates.bias[:H_op].fill_(-5.5)
                self.op_gates.bias[H_op:2 * H_op].fill_(-1.5)
                self.op_gates.bias[2 * H_op:].fill_(-3.5)

            self.op_read_gate = Linear(model_dim, self.op_heads, bias=True)
            nn.init.constant_(self.op_read_gate.bias, 0.5)
            self.op_out = Linear(self.op_rings_flat, model_dim, bias=False)
            nn.init.normal_(self.op_out.weight, std=0.01)

        self.echo = cfg.echo
        if self.echo:
            self.qk_slot = Dim("qk_slot", 2)
            self.echo_qk_out = ProductDim(self.qk_slot, heads, head_feature)
            self.echo_qk = Linear(model_dim, self.echo_qk_out, bias=False)
            self.echo_slot3 = Dim("echo_slot3", 3)
            self.echo_gate_out = ProductDim(self.echo_slot3, heads)
            self.echo_gates = Linear(model_dim, self.echo_gate_out, bias=True)
            with torch.no_grad():
                H = cfg.n_heads
                self.echo_gates.bias[:H].fill_(-5.0)
                self.echo_gates.bias[H:2 * H].fill_(-0.5)
                self.echo_gates.bias[2 * H:].fill_(0.0)
            self.echo_inner = ProductDim(heads, head_feature)
            self.echo_out = Linear(self.echo_inner, model_dim, bias=False)
            nn.init.zeros_(self.echo_out.weight)
            self.echo_row = Dim("echo_row", cfg.head_dim)
            self.echo_col = Dim("echo_col", cfg.head_dim)

    def forward(self, h: NamedTensor, e_tok: NamedTensor, batch: Dim, time: Dim,
                state: Optional[tuple] = None):
        B, T = batch.size, time.size
        dev = h.device
        xi_in, om_in, h_prev_in, S_echo_in = state if state is not None else (None, None, None, None)

        if h_prev_in is None:
            h_prev = shift(h, over=time, by=1, fill=0.0)
            m_raw = torch.ones(B, T, device=dev, dtype=torch.float32)
            m_raw[:, 0] = 0.0
            first_mask = named(m_raw, (batch, time))
        else:
            hp0 = as_dtype(h_prev_in.alias(h_prev_in.layout[0], batch), h.dtype)
            if T == 1:
                h_prev = stack([hp0], into=time).to(batch, time, self.model_dim)
            else:
                t1, t_tail = Dim("time", 1), Dim("time", T - 1)
                hp_step = stack([hp0], into=t1).to(batch, t1, self.model_dim)
                h_rest = named(h.raw(batch, time, self.model_dim)[:, :T - 1, :], (batch, t_tail, self.model_dim))
                h_prev = cat([hp_step, h_rest], over="time", into=time)
            first_mask = ones(batch, time, device=dev, dtype=torch.float32)

        total = zeros(batch, time, self.model_dim, device=dev, dtype=h.dtype)
        xi_last, om_last = None, None

        if self.cfg.operator_engine:
            he = cat([h, e_tok], over=self.model_dim, into=self.dual_dim)
            ring_layout = (batch, time, self.op_heads, self.op_sub, self.rot_pair)

            zeta_in_c = unit_circle(self.in_coord(he).to(*ring_layout), self.rot_pair, self.cplx)
            zeta_in_r = NamedTensor(zeta_in_c.data, ring_layout)            # real view for linear scan
            eta = 0.02 + 0.96 * sigmoid(as_float(self.latch_gate(he)))      # (batch, time, op_heads)

            if xi_in is None:
                c0 = ones(batch, self.op_heads, self.op_sub, device=dev, dtype=torch.float32)
                s0 = zeros(batch, self.op_heads, self.op_sub, device=dev, dtype=torch.float32)
                xi0 = stack([c0, s0], into=self.rot_pair)
            else:
                xi0 = xi_in.alias(xi_in.layout[0], batch)

            xi_raw, xi_last = stateful_scan((eta * zeta_in_r).to(*ring_layout), 1.0 - eta, over=time, carry=xi0)
            xi_unit = unit_circle(xi_raw, self.rot_pair, self.cplx)         # SplitComplex on S^1

            # Strictly causal: pair current outcome e_t with PREVIOUS latched operand xi_{t-1}
            xi0_unit = unit_circle(xi0, self.rot_pair, self.cplx)
            if T == 1:
                xi_prev_unit = NamedTensor(
                    stack([NamedTensor(xi0_unit.data, xi0.layout)], into=time).to(*ring_layout).data,
                    ring_layout, self.cplx, bind=False)
            else:
                xi_sh = shift(NamedTensor(xi_unit.data, ring_layout), over=time, by=1, fill=0.0)
                t1, t_tail = Dim("time", 1), Dim("time", T - 1)
                head0 = stack([NamedTensor(xi0_unit.data, xi0.layout)], into=t1).to(
                    batch, t1, self.op_heads, self.op_sub, self.rot_pair)
                tail_d = named(
                    xi_sh.raw(batch, time, self.op_heads, self.op_sub, self.rot_pair)[:, 1:],
                    (batch, t_tail, self.op_heads, self.op_sub, self.rot_pair))
                merged = cat([head0, tail_d], over="time", into=time).to(*ring_layout)
                xi_prev_unit = NamedTensor(merged.data, ring_layout, self.cplx, bind=False)

            zeta_out = unit_circle(self.out_coord(e_tok).to(*ring_layout), self.rot_pair, self.cplx)
            # Named complex quotient: r_t = zeta_out(e_t) * conj(xi_{t-1})!
            r_quot = zeta_out * xi_prev_unit.conj()
            r_quot_r = NamedTensor(r_quot.data, ring_layout)

            trans_in = cat([h_prev, e_tok], over=self.model_dim, into=self.dual_dim)
            gates_u = as_float(self.op_gates(trans_in)).to(batch, time, self.gate_slot3, self.op_heads)
            dt_op = select(gates_u, self.gate_slot3, 0)
            bw_op = select(gates_u, self.gate_slot3, 1)
            be_op = select(gates_u, self.gate_slot3, 2)

            ret_op = exp(-softplus(dt_op))
            w_op = sigmoid(bw_op) * first_mask
            e_op = clamp(sigmoid(be_op), max=0.90) * first_mask
            keep_op = ret_op * (1.0 - e_op)

            om0 = (zeros(batch, self.op_heads, self.op_sub, self.rot_pair, device=dev, dtype=torch.float32)
                   if om_in is None else om_in.alias(om_in.layout[0], batch))
            om_raw, om_last = stateful_scan((w_op * r_quot_r).to(*ring_layout), keep_op, over=time, carry=om0)
            om_unit = unit_circle(om_raw, self.rot_pair, self.cplx)

            # Named complex operator application: pred = Omega_t * zeta_query(h_t, e_t)!
            zeta_q = unit_circle(self.q_coord(he).to(*ring_layout), self.rot_pair, self.cplx)
            pred_c = om_unit * zeta_q
            pred_r = NamedTensor(pred_c.data, ring_layout)
            rg = silu(as_float(self.op_read_gate(h)))                       # (batch, time, op_heads)
            op_feat = as_dtype(pred_r * rg, h.dtype).to(batch, time, self.op_rings_flat)
            total = total + self.op_out(op_feat)

        S_echo_out = None
        if self.echo:
            H, K = self.heads.size, self.head_feature.size
            qk_now = self.echo_qk(h).to(batch, time, self.qk_slot, self.heads, self.head_feature)
            qk_prv = self.echo_qk(h_prev).to(batch, time, self.qk_slot, self.heads, self.head_feature)
            qh = select(qk_now, self.qk_slot, 0).to(batch, self.heads, time, self.head_feature)
            kh_raw_nt = select(qk_prv, self.qk_slot, 1) * as_dtype(first_mask, h.dtype)
            kh = as_dtype(
                l2_normalize(as_float(kh_raw_nt), over=self.head_feature, eps=1e-12),
                h.dtype,
            ).to(batch, self.heads, time, self.head_feature)
            vh = e_tok.to(batch, time, self.heads, self.head_feature).to(batch, self.heads, time, self.head_feature)

            eg = as_float(self.echo_gates(h)).to(batch, time, self.echo_slot3, self.heads)
            ret_e = exp(-softplus(select(eg, self.echo_slot3, 0))).to(batch, self.heads, time)
            bw_e = (sigmoid(select(eg, self.echo_slot3, 1)) * first_mask).to(batch, self.heads, time)
            be_e = (clamp(sigmoid(select(eg, self.echo_slot3, 2) - 3.0), max=0.95) * first_mask).to(batch, self.heads, time)

            # Declared raw-torch boundary: Triton delta kernel / single-step fp32 recurrence
            qh_r = qh.raw(batch, self.heads, time, self.head_feature)
            kh_r = kh.raw(batch, self.heads, time, self.head_feature)
            vh_r = vh.raw(batch, self.heads, time, self.head_feature)
            ret_r = ret_e.raw(batch, self.heads, time)
            bw_r = bw_e.raw(batch, self.heads, time)
            be_r = be_e.raw(batch, self.heads, time)

            if T == 1:
                Se = (S_echo_in.alias(S_echo_in.layout[0], batch).raw(batch, self.heads, self.echo_row, self.echo_col).float()
                      if S_echo_in is not None else torch.zeros(B, H, K, K, device=dev, dtype=torch.float32))
                kt, vt, qt = kh_r[:, :, 0].float(), vh_r[:, :, 0].float(), qh_r[:, :, 0].float()
                Sk = torch.einsum("bhvk,bhk->bhv", Se, kt)
                erased = Se - be_r[:, :, 0, None, None] * Sk.unsqueeze(-1) * kt.unsqueeze(-2)
                Se_raw = ret_r[:, :, 0, None, None] * erased + bw_r[:, :, 0, None, None] * vt.unsqueeze(-1) * kt.unsqueeze(-2)
                re_raw = torch.einsum("bhvk,bhk->bhv", Se_raw, qt).unsqueeze(2)
            else:
                carry_e = (S_echo_in.alias(S_echo_in.layout[0], batch).raw(batch, self.heads, self.echo_row, self.echo_col)
                           .reshape(B * H, K, K).float() if S_echo_in is not None else None)
                re_flat, Se_flat = pam_delta_batched(
                    qh_r.reshape(B * H, T, K),
                    kh_r.reshape(B * H, T, K),
                    vh_r.reshape(B * H, T, K),
                    ret_r.reshape(B * H, T).to(h.dtype),
                    bw_r.reshape(B * H, T).to(h.dtype),
                    be_r.reshape(B * H, T).to(h.dtype),
                    carry_e,
                )
                re_raw = re_flat.view(B, H, T, K).float()
                Se_raw = Se_flat.view(B, H, K, K).float()

            S_echo_out = named(Se_raw, (batch, self.heads, self.echo_row, self.echo_col))
            re_norm = F.rms_norm(re_raw, (K,), eps=1e-6).to(h.dtype)
            re_nt = named(re_norm, (batch, self.heads, time, self.head_feature)).to(batch, time, self.echo_inner)
            total = total + self.echo_out(re_nt)

        h_last = select(h, time, T - 1)
        return total, (xi_last, om_last, h_last, S_echo_out)


# ── Full POWLM Body in sempyt ────────────────────────────────────────────────

class POWBlock(nn.Module):
    def __init__(self, cfg: POWConfig, model_dim: Dim, mlp_dim: Dim,
                 heads: Dim, head_feature: Dim, head_pair: Dim, rot_pair: Dim):
        super().__init__()
        self.norm1 = RMSNorm(model_dim)
        self.mlp = SwiGLU(model_dim, mlp_dim)
        self.mlp_scale = nn.Parameter(torch.tensor(float(cfg.mlp_scale_init)))
        self.norm2 = RMSNorm(model_dim)
        self.mem_work = DualPhaseWorkspaceLayer(cfg, model_dim, heads, head_feature, head_pair, rot_pair)
        self.mem_scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, x: NamedTensor, batch: Dim, time: Dim, state: Optional[tuple] = None):
        x = x + self.mlp(self.norm1(x)) * self.mlp_scale.to(x.dtype)
        dm, new_state = self.mem_work(self.norm2(x), batch, time, state)
        x = x + dm * self.mem_scale.to(x.dtype)
        return x, new_state


class POWLM(nn.Module):
    """Two-Organ Phase-Operator Workspace Language Model in ``sempyt`` (O(1) state)."""

    def __init__(self, cfg: POWConfig):
        super().__init__()
        self.cfg = cfg
        self.vocab_dim = Dim("vocab", cfg.vocab_size)
        self.model_dim = Dim("model_dim", cfg.dim)
        self.mlp_dim = Dim("mlp_dim", cfg.mlp_hidden)
        self.heads = Dim("heads", cfg.n_heads)
        self.head_feature = Dim("head_feature", cfg.head_dim)
        self.head_pair = Dim("head_pair", cfg.head_dim // 2)
        self.rot_pair = Dim("rot_pair", 2)

        assert cfg.dim == cfg.n_heads * cfg.head_dim, "dim must equal n_heads * head_dim"
        self.embed = Embed(self.vocab_dim, self.model_dim)
        self.blocks = nn.ModuleList([
            POWBlock(cfg, self.model_dim, self.mlp_dim, self.heads,
                     self.head_feature, self.head_pair, self.rot_pair)
            for _ in range(cfg.n_layers)
        ])
        self.norm = RMSNorm(self.model_dim)
        self.unit_embed_norm = RMSNorm(self.model_dim)
        with torch.no_grad():
            self.unit_embed_norm.weight.fill_(1.0)
        self.unit_embed_norm.weight.requires_grad_(False)
        self.op_engine = CausalOperatorEngine(cfg, self.model_dim, self.heads, self.head_feature, self.rot_pair)
        self.lm_head = TiedHead(self.embed)

    def forward_named(self, input_ids: torch.Tensor, states: Optional[list] = None) -> tuple[NamedTensor, list]:
        B, T = input_ids.shape
        batch = Dim("batch", B)
        time = Dim("time", T)
        e_raw = self.embed(input_ids, batch, time)
        h = e_raw
        layer_in = states[:-1] if states is not None else [None] * len(self.blocks)
        op_in = states[-1] if states is not None else None
        new_states = []
        for blk, st in zip(self.blocks, layer_in):
            h, nst = blk(h, batch, time, st)
            new_states.append(nst)
        h = self.norm(h)
        e_unit = self.unit_embed_norm(e_raw)
        op_delta, op_st = self.op_engine(h, e_unit, batch, time, op_in)
        h_out = h + op_delta
        new_states.append(op_st)
        return h_out, new_states

    def _hidden_to_lm(self, input_ids: torch.Tensor, step_offset: int = 0):
        h_nt, _ = self.forward_named(input_ids, states=None)
        return h_nt, torch.zeros((), device=input_ids.device)

    def forward(self, input_ids: torch.Tensor, states: Optional[list] = None):
        h_nt, new_states = self.forward_named(input_ids, states=states)
        logits_nt = self.lm_head(h_nt)
        batch, time = h_nt.layout[0], h_nt.layout[1]
        return logits_nt.raw(batch, time, self.vocab_dim), new_states

    def ce_from_lm(self, lm, labels: torch.Tensor,
                   loss_mask: Optional[torch.Tensor] = None,
                   ignore_index: int = -100, chunk: int = 4096,
                   return_nll: bool = False, gemm_dtype=None):
        raw_lm = lm.raw(*lm.layout) if isinstance(lm, NamedTensor) else lm
        B, T, D = raw_lm.shape
        out = fused_linear_cross_entropy(
            raw_lm.reshape(B * T, D), self.embed.weight, labels.reshape(-1),
            mask=(loss_mask.reshape(-1) if loss_mask is not None else None),
            chunk=chunk, ignore_index=ignore_index, return_nll=return_nll,
            gemm_dtype=gemm_dtype)
        if return_nll:
            return out, getattr(out, "_nll", None).reshape(B, T)
        return out

    def state_bytes(self, batch_size: int = 1) -> int:
        cfg = self.cfg
        H, K, P, M = cfg.n_heads, cfg.head_dim, cfg.head_dim // 2, cfg.op_rings
        per_layer = H * K * K + H + (H * P * 2 if cfg.sync_phasor else 0) + (3 * H * K * max(0, cfg.short_conv_k - 1))
        top = (M * 2 * 2 if cfg.operator_engine else 0) + cfg.dim + (H * K * K if cfg.echo else 0)
        return 4 * batch_size * (cfg.n_layers * per_layer + top)


# ── Param-matched arm factory (exact parameter parity across all 6 arms!) ────

def _solve_mlp_hidden(build_fn, target_params: int) -> int:
    with torch.random.fork_rng(devices=[]):
        p64 = sum(p.numel() for p in build_fn(64).parameters() if p.requires_grad)
        p128 = sum(p.numel() for p in build_fn(128).parameters() if p.requires_grad)
    slope = (p128 - p64) / 64.0
    raw_h = 64.0 + (target_params - p64) / max(1.0, slope)
    return max(64, int(round(raw_h / 16.0) * 16))


def build_arm(arm: str, vocab_size: int = 50257, dim: int = 384,
              n_layers: int = 4, n_heads: int = 6, head_dim: int = 64,
              base_mlp: int = 1024):
    """Build parameter-matched arms anchored to ``pow_v17``'s trainable parameter count."""
    ref_cfg = POWConfig(
        vocab_size=vocab_size, dim=dim, n_layers=n_layers,
        n_heads=n_heads, head_dim=head_dim, mlp_hidden=base_mlp,
    )
    with torch.random.fork_rng(devices=[]):
        target = sum(p.numel() for p in POWLM(ref_cfg).parameters() if p.requires_grad)

    if arm == "pow_v17":
        return POWLM(ref_cfg)

    if arm == "tpp":
        from v13_sempty.baseline_transformer import TransformerConfig, TransformerLM
        h = _solve_mlp_hidden(
            lambda mh: TransformerLM(TransformerConfig(
                vocab_size=vocab_size, d_model=dim, n_layer=n_layers, n_head=n_heads, d_ff=mh)),
            target)
        return TransformerLM(TransformerConfig(
            vocab_size=vocab_size, d_model=dim, n_layer=n_layers, n_head=n_heads, d_ff=h))

    arm_overrides = {
        "pam_v2": dict(use_rope=True, chrono=True, sync_phasor=False, short_conv_k=0,
                       n_bounces=1, operator_engine=False, echo=True,
                       warp_min=-3.0, mlp_scale_init=1.0),
        "gdn_like": dict(use_rope=False, chrono=False, sync_phasor=False, short_conv_k=4,
                         n_bounces=1, operator_engine=False, echo=True,
                         mlp_scale_init=1.0),
        "pow_sync_only": dict(use_rope=True, chrono=True, sync_phasor=True, short_conv_k=4,
                              n_bounces=1, operator_engine=False, echo=True,
                              mlp_scale_init=0.4),
        "pow_work_only": dict(use_rope=True, chrono=True, sync_phasor=False, short_conv_k=4,
                              n_bounces=3, operator_engine=True, echo=True,
                              mlp_scale_init=0.4),
    }
    if arm not in arm_overrides:
        raise ValueError(f"Unknown arm {arm!r}")
    ov = arm_overrides[arm]
    h = _solve_mlp_hidden(
        lambda mh: POWLM(POWConfig(vocab_size=vocab_size, dim=dim, n_layers=n_layers,
                                   n_heads=n_heads, head_dim=head_dim, mlp_hidden=mh, **ov)),
        target)
    return POWLM(POWConfig(vocab_size=vocab_size, dim=dim, n_layers=n_layers,
                           n_heads=n_heads, head_dim=head_dim, mlp_hidden=h, **ov))
