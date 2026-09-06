"""Fused real-PAM scan: the notebook recurrence in chunked linear-attention form.

The real PAM notebook per (batch, head) is ``S_t = r_t S_{t-1} + v_t (x) k_t``
and the read is ``y_t = S_t . q_t``.  ``model._stable_notebook`` realises this
by materialising the notebook at EVERY position -- ``[B*H, w, K, K]`` per
window -- which at K=98 is ~470 MB of fp32 per layer per window and ~100x the
memory traffic the algebra needs.  Expanding the read instead,

    y_s = a_s (S_in . q_s) + sum_{t<=s} (a_s / a_t) (q_s . k_t) v_t

(``a_s = prod_{i<=s} r_i``), the whole window is one ``[w, w]`` score matrix
``(Q K^T) * M`` applied to ``V`` plus one ``[K, K]`` state -- the same closed
form the chunked complex arm uses, without the per-position notebook.  The
state is carried exactly as before, so decode (``_stepwise``) is untouched.

Two implementations of the same math live here:

  * ``pam_scan_torch``      pure torch (CPU / Triton unavailable / kill switch)
  * ``_PamScanFn``          Triton forward AND backward (CUDA), tile BT=64:
        _pam_fwd_h    sequential state scan, one [K,K] state per tile
        _pam_fwd_o    intra-tile scores + inter-tile state read
        _pam_bwd_dh   reverse state-gradient scan
        _pam_bwd_dqk  dq, dk and the per-position log-decay gradient
        _pam_bwd_dv   dv
    The decay enters as ``g_t = log(r_t + 1e-6) <= 0`` with in-tile cumsum
    ``G``; ``dg_i = sum_{s>=i} (q_s.dq_s - k_s.dk_s) + [last] <S_out, dS_out>``.

Both agree with the eager oracle (``pam_kernel_test.py``) forward and
backward, fp32 to 3e-5 and bf16 to 3e-2.  Public contract:

    fused_real_pam_read(q, k, v, retention, carry, chunk_size)
      -> (read [B*H, T, K] in q.dtype, carry_out [B*H, K, K] fp32)

    q/k/v      [B*H, T, K]   (RoPE'd, NOT scaled -- the caller applies K**-0.5)
    retention  [B*H, T]      in (0, 1)
    carry      [B*H, K, K]   fp32, rows = value channel, cols = key channel
                             (or None for "nothing carried in")

This module is a declared raw-torch boundary (``check_torch_layout.SKIP_FILES``).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False

_EPS = 1e-6
_BT = 64          # positions per tile
_BK = 64          # channels per block (K <= 128 -> at most 2 blocks)
_MAX_K = 128
_WARPS = 8         # 4 warps spills every tile kernel (measured 130-440 spills)

_ENABLED = os.environ.get("V13S_KERNEL", "1") == "1"


def set_kernel_enabled(on: bool) -> None:
    """Runtime kill switch for the Triton path (falls back to pam_scan_torch)."""
    global _ENABLED
    _ENABLED = bool(on)


def kernel_enabled() -> bool:
    return _ENABLED and HAS_TRITON


# ── pure torch: chunked linear-attention form ────────────────────────────────

def pam_scan_torch(q, k, v, retention, carry, window):
    """Reference / fallback. Same math as the kernel, windows of ``window``.

    Decay math is fp32; the score/value matmuls follow the input dtype (bf16
    under autocast) with the decay-weighted scores cast back before the
    value matmul, mirroring the kernel's tensor-core path.
    """
    BH, T, K = q.shape
    dt = q.dtype
    r = retention.float()
    state = carry.float() if carry is not None else None      # [BH, K(v), K(k)]
    reads = []
    for start in range(0, T, window):
        w = min(window, T - start)
        qw, kw, vw = q[:, start:start + w], k[:, start:start + w], v[:, start:start + w]
        C = torch.cumsum(-torch.log(r[:, start:start + w] + _EPS), dim=-1)      # >= 0
        M = torch.exp(torch.clamp(C.unsqueeze(-2) - C.unsqueeze(-1), max=0.0))
        M = M * torch.tril(torch.ones(w, w, device=q.device, dtype=M.dtype))
        scores = torch.bmm(qw, kw.transpose(1, 2))                              # [BH, w, w]
        read = torch.bmm((scores.float() * M).to(dt), vw)                       # [BH, w, K]
        if state is not None:
            inter = torch.bmm(qw.float(), state.transpose(1, 2))                # q . S^T
            read = read + (torch.exp(-C).unsqueeze(-1) * inter).to(dt)
            state = torch.exp(-C[:, -1])[:, None, None] * state
        else:
            state = 0.0
        e = torch.exp(C - C[:, -1:])                                            # <= 1
        state = state + torch.bmm((vw.float() * e.unsqueeze(-1)).transpose(1, 2), kw.float())
        reads.append(read)
    return torch.cat(reads, dim=1), state


# ── Triton kernels ───────────────────────────────────────────────────────────

if HAS_TRITON:

    @triton.jit
    def _dot(a, b, IEEE: tl.constexpr):
        # fp32 inputs must not silently go through tf32 (the 3e-5 fp32
        # parity bar); bf16 inputs take the tensor-core path.
        if IEEE:
            c = tl.dot(a, b, input_precision="ieee")
        else:
            c = tl.dot(a, b)
        return c

    @triton.jit
    def _pam_fwd_h(k_ptr, v_ptr, g_ptr, h0_ptr, h_ptr,
                   T, K, NT,
                   HAS_H0: tl.constexpr, IEEE: tl.constexpr,
                   BT: tl.constexpr, BK: tl.constexpr):
        """h[bh, c] = state BEFORE tile c (internal layout [K(k), K(v)]);
        h[bh, NT] = final state.  One program per (k-block, v-block, bh)."""
        i_k = tl.program_id(0)
        i_v = tl.program_id(1)
        i_bh = tl.program_id(2)
        offs_k = i_k * BK + tl.arange(0, BK)
        offs_v = i_v * BK + tl.arange(0, BK)
        m_k = offs_k < K
        m_v = offs_v < K
        m_kv = m_k[:, None] & m_v[None, :]
        ar = tl.arange(0, BT)
        hoffs = offs_k[:, None] * K + offs_v[None, :]

        if HAS_H0:
            b_h = tl.load(h0_ptr + i_bh * K * K + hoffs, mask=m_kv, other=0.0)
        else:
            b_h = tl.zeros([BK, BK], dtype=tl.float32)

        h_base = h_ptr + i_bh * (NT + 1) * K * K
        kv_base = i_bh * T * K
        for c in range(0, NT):
            tl.store(h_base + c * K * K + hoffs, b_h, mask=m_kv)
            offs_t = c * BT + ar
            m_t = offs_t < T
            b_g = tl.load(g_ptr + i_bh * T + offs_t, mask=m_t, other=0.0)
            b_G = tl.cumsum(b_g, axis=0)
            G_w = tl.sum(tl.where(ar == BT - 1, b_G, 0.0), axis=0)
            e = tl.exp(tl.minimum(G_w - b_G, 0.0))
            b_k = tl.load(k_ptr + kv_base + offs_t[:, None] * K + offs_k[None, :],
                          mask=m_t[:, None] & m_k[None, :], other=0.0)
            b_v = tl.load(v_ptr + kv_base + offs_t[:, None] * K + offs_v[None, :],
                          mask=m_t[:, None] & m_v[None, :], other=0.0)
            b_ke = (b_k.to(tl.float32) * e[:, None]).to(b_k.dtype)
            b_h = tl.exp(G_w) * b_h + _dot(tl.trans(b_ke), b_v, IEEE)
        tl.store(h_base + NT * K * K + hoffs, b_h, mask=m_kv)

    @triton.jit
    def _pam_fwd_o(q_ptr, k_ptr, v_ptr, g_ptr, h_ptr, o_ptr,
                   T, K, NT, NK,
                   IEEE: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr):
        """o = ((Q K^T) * M) V + exp(G) (Q h_c).  One program per (tile, v-block, bh)."""
        i_t = tl.program_id(0)
        i_v = tl.program_id(1)
        i_bh = tl.program_id(2)
        ar = tl.arange(0, BT)
        offs_t = i_t * BT + ar
        m_t = offs_t < T
        offs_v = i_v * BK + tl.arange(0, BK)
        m_v = offs_v < K
        qkv_base = i_bh * T * K

        b_g = tl.load(g_ptr + i_bh * T + offs_t, mask=m_t, other=0.0)
        b_G = tl.cumsum(b_g, axis=0)

        b_A = tl.zeros([BT, BT], dtype=tl.float32)
        b_o = tl.zeros([BT, BK], dtype=tl.float32)
        h_base = h_ptr + (i_bh * (NT + 1) + i_t) * K * K
        for i_k in range(0, NK):
            offs_k = i_k * BK + tl.arange(0, BK)
            m_k = offs_k < K
            b_q = tl.load(q_ptr + qkv_base + offs_t[:, None] * K + offs_k[None, :],
                          mask=m_t[:, None] & m_k[None, :], other=0.0)
            b_k = tl.load(k_ptr + qkv_base + offs_t[:, None] * K + offs_k[None, :],
                          mask=m_t[:, None] & m_k[None, :], other=0.0)
            b_A += _dot(b_q, tl.trans(b_k), IEEE)
            b_h = tl.load(h_base + offs_k[:, None] * K + offs_v[None, :],
                          mask=m_k[:, None] & m_v[None, :], other=0.0)
            b_o += _dot(b_q, b_h.to(b_q.dtype), IEEE)
        b_o = b_o * tl.exp(b_G)[:, None]

        causal = ar[:, None] >= ar[None, :]
        d = tl.where(causal, tl.minimum(b_G[:, None] - b_G[None, :], 0.0), -1e30)
        b_A = b_A * tl.exp(d)
        b_v = tl.load(v_ptr + qkv_base + offs_t[:, None] * K + offs_v[None, :],
                      mask=m_t[:, None] & m_v[None, :], other=0.0)
        b_o += _dot(b_A.to(b_v.dtype), b_v, IEEE)
        tl.store(o_ptr + qkv_base + offs_t[:, None] * K + offs_v[None, :],
                 b_o.to(o_ptr.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])

    @triton.jit
    def _pam_bwd_dh(q_ptr, do_ptr, g_ptr, dhf_ptr, dh_ptr, dh0_ptr,
                    T, K, NT,
                    HAS_DHF: tl.constexpr, IEEE: tl.constexpr,
                    BT: tl.constexpr, BK: tl.constexpr):
        """dh[bh, c] = dL/d(state AFTER tile c); dh0 = dL/d(carry in).
        Reverse scan: dS_in = exp(G_w) dS_out + sum_s exp(G_s) q_s (x) do_s."""
        i_k = tl.program_id(0)
        i_v = tl.program_id(1)
        i_bh = tl.program_id(2)
        offs_k = i_k * BK + tl.arange(0, BK)
        offs_v = i_v * BK + tl.arange(0, BK)
        m_k = offs_k < K
        m_v = offs_v < K
        m_kv = m_k[:, None] & m_v[None, :]
        ar = tl.arange(0, BT)
        hoffs = offs_k[:, None] * K + offs_v[None, :]

        if HAS_DHF:
            b_dh = tl.load(dhf_ptr + i_bh * K * K + hoffs, mask=m_kv, other=0.0)
        else:
            b_dh = tl.zeros([BK, BK], dtype=tl.float32)

        dh_base = dh_ptr + i_bh * NT * K * K
        qo_base = i_bh * T * K
        for cc in range(0, NT):
            c = NT - 1 - cc
            tl.store(dh_base + c * K * K + hoffs, b_dh, mask=m_kv)
            offs_t = c * BT + ar
            m_t = offs_t < T
            b_g = tl.load(g_ptr + i_bh * T + offs_t, mask=m_t, other=0.0)
            b_G = tl.cumsum(b_g, axis=0)
            G_w = tl.sum(tl.where(ar == BT - 1, b_G, 0.0), axis=0)
            b_q = tl.load(q_ptr + qo_base + offs_t[:, None] * K + offs_k[None, :],
                          mask=m_t[:, None] & m_k[None, :], other=0.0)
            b_do = tl.load(do_ptr + qo_base + offs_t[:, None] * K + offs_v[None, :],
                           mask=m_t[:, None] & m_v[None, :], other=0.0)
            b_qe = (b_q.to(tl.float32) * tl.exp(b_G)[:, None]).to(b_q.dtype)
            b_dh = tl.exp(G_w) * b_dh + _dot(tl.trans(b_qe), b_do, IEEE)
        tl.store(dh0_ptr + i_bh * K * K + hoffs, b_dh, mask=m_kv)

    @triton.jit
    def _pam_bwd_dqk(q_ptr, k_ptr, v_ptr, do_ptr, g_ptr, h_ptr, dh_ptr,
                     dq_ptr, dk_ptr, dgp_ptr,
                     T, K, NT, NV, BH,
                     IEEE: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr):
        """dq, dk for one k-block, plus the partial log-decay gradient
        dgp[k-block, bh, t] = q_t.dq_t - k_t.dk_t (summed over k-blocks by the caller)."""
        i_t = tl.program_id(0)
        i_k = tl.program_id(1)
        i_bh = tl.program_id(2)
        ar = tl.arange(0, BT)
        offs_t = i_t * BT + ar
        m_t = offs_t < T
        offs_k = i_k * BK + tl.arange(0, BK)
        m_k = offs_k < K
        qkv_base = i_bh * T * K

        b_g = tl.load(g_ptr + i_bh * T + offs_t, mask=m_t, other=0.0)
        b_G = tl.cumsum(b_g, axis=0)
        G_w = tl.sum(tl.where(ar == BT - 1, b_G, 0.0), axis=0)

        b_dA = tl.zeros([BT, BT], dtype=tl.float32)
        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_dks = tl.zeros([BT, BK], dtype=tl.float32)
        h_base = h_ptr + (i_bh * (NT + 1) + i_t) * K * K
        dh_base = dh_ptr + (i_bh * NT + i_t) * K * K
        for i_v in range(0, NV):
            offs_v = i_v * BK + tl.arange(0, BK)
            m_v = offs_v < K
            b_do = tl.load(do_ptr + qkv_base + offs_t[:, None] * K + offs_v[None, :],
                           mask=m_t[:, None] & m_v[None, :], other=0.0)
            b_v = tl.load(v_ptr + qkv_base + offs_t[:, None] * K + offs_v[None, :],
                          mask=m_t[:, None] & m_v[None, :], other=0.0)
            b_dA += _dot(b_do, tl.trans(b_v), IEEE)
            b_h = tl.load(h_base + offs_k[:, None] * K + offs_v[None, :],
                          mask=m_k[:, None] & m_v[None, :], other=0.0)
            b_dq += _dot(b_do, tl.trans(b_h.to(b_do.dtype)), IEEE)
            b_dh = tl.load(dh_base + offs_k[:, None] * K + offs_v[None, :],
                           mask=m_k[:, None] & m_v[None, :], other=0.0)
            b_dks += _dot(b_v, tl.trans(b_dh.to(b_v.dtype)), IEEE)
        b_dq = b_dq * tl.exp(b_G)[:, None]

        causal = ar[:, None] >= ar[None, :]
        d = tl.where(causal, tl.minimum(b_G[:, None] - b_G[None, :], 0.0), -1e30)
        b_dA = b_dA * tl.exp(d)
        b_q = tl.load(q_ptr + qkv_base + offs_t[:, None] * K + offs_k[None, :],
                      mask=m_t[:, None] & m_k[None, :], other=0.0)
        b_k = tl.load(k_ptr + qkv_base + offs_t[:, None] * K + offs_k[None, :],
                      mask=m_t[:, None] & m_k[None, :], other=0.0)
        b_dq += _dot(b_dA.to(b_k.dtype), b_k, IEEE)
        b_dk = _dot(tl.trans(b_dA).to(b_q.dtype), b_q, IEEE) \
            + b_dks * tl.exp(tl.minimum(G_w - b_G, 0.0))[:, None]

        m_tk = m_t[:, None] & m_k[None, :]
        tl.store(dq_ptr + qkv_base + offs_t[:, None] * K + offs_k[None, :],
                 b_dq.to(dq_ptr.dtype.element_ty), mask=m_tk)
        tl.store(dk_ptr + qkv_base + offs_t[:, None] * K + offs_k[None, :],
                 b_dk.to(dk_ptr.dtype.element_ty), mask=m_tk)
        b_dg = tl.sum(b_q.to(tl.float32) * b_dq, axis=1) - tl.sum(b_k.to(tl.float32) * b_dk, axis=1)
        tl.store(dgp_ptr + (i_k * BH + i_bh) * T + offs_t, b_dg, mask=m_t)

    @triton.jit
    def _pam_bwd_dv(q_ptr, k_ptr, do_ptr, g_ptr, dh_ptr, dv_ptr,
                    T, K, NT, NK,
                    IEEE: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr):
        """dv = A^T do + exp(G_w - G) (K dS_out) for one v-block."""
        i_t = tl.program_id(0)
        i_v = tl.program_id(1)
        i_bh = tl.program_id(2)
        ar = tl.arange(0, BT)
        offs_t = i_t * BT + ar
        m_t = offs_t < T
        offs_v = i_v * BK + tl.arange(0, BK)
        m_v = offs_v < K
        qkv_base = i_bh * T * K

        b_g = tl.load(g_ptr + i_bh * T + offs_t, mask=m_t, other=0.0)
        b_G = tl.cumsum(b_g, axis=0)
        G_w = tl.sum(tl.where(ar == BT - 1, b_G, 0.0), axis=0)

        b_A = tl.zeros([BT, BT], dtype=tl.float32)
        b_dvs = tl.zeros([BT, BK], dtype=tl.float32)
        dh_base = dh_ptr + (i_bh * NT + i_t) * K * K
        for i_k in range(0, NK):
            offs_k = i_k * BK + tl.arange(0, BK)
            m_k = offs_k < K
            b_q = tl.load(q_ptr + qkv_base + offs_t[:, None] * K + offs_k[None, :],
                          mask=m_t[:, None] & m_k[None, :], other=0.0)
            b_k = tl.load(k_ptr + qkv_base + offs_t[:, None] * K + offs_k[None, :],
                          mask=m_t[:, None] & m_k[None, :], other=0.0)
            b_A += _dot(b_q, tl.trans(b_k), IEEE)
            b_dh = tl.load(dh_base + offs_k[:, None] * K + offs_v[None, :],
                           mask=m_k[:, None] & m_v[None, :], other=0.0)
            b_dvs += _dot(b_k, b_dh.to(b_k.dtype), IEEE)

        causal = ar[:, None] >= ar[None, :]
        d = tl.where(causal, tl.minimum(b_G[:, None] - b_G[None, :], 0.0), -1e30)
        b_A = b_A * tl.exp(d)
        b_do = tl.load(do_ptr + qkv_base + offs_t[:, None] * K + offs_v[None, :],
                       mask=m_t[:, None] & m_v[None, :], other=0.0)
        b_dv = _dot(tl.trans(b_A).to(b_do.dtype), b_do, IEEE) \
            + b_dvs * tl.exp(tl.minimum(G_w - b_G, 0.0))[:, None]
        tl.store(dv_ptr + qkv_base + offs_t[:, None] * K + offs_v[None, :],
                 b_dv.to(dv_ptr.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])


# ── autograd Function around the kernels ─────────────────────────────────────

def _launch_fwd_h(k, v, g, h0):
    BH, T, K = k.shape
    NT, NK = triton.cdiv(T, _BT), triton.cdiv(K, _BK)
    h = torch.empty(BH, NT + 1, K, K, device=k.device, dtype=torch.float32)
    _pam_fwd_h[(NK, NK, BH)](
        k, v, g, h0 if h0 is not None else h, h, T, K, NT,
        HAS_H0=h0 is not None, IEEE=k.dtype == torch.float32, BT=_BT, BK=_BK,
        num_warps=_WARPS, num_stages=1,
    )
    return h


class _PamScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, retention, carry):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        BH, T, K = q.shape
        NT, NK = triton.cdiv(T, _BT), triton.cdiv(K, _BK)
        g = torch.log(retention.float() + _EPS).contiguous()
        # internal state layout is [K(k), K(v)]; the public one is [v, k]
        h0 = carry.float().transpose(1, 2).contiguous() if carry is not None else None
        h = _launch_fwd_h(k, v, g, h0)
        o = torch.empty_like(q)
        _pam_fwd_o[(NT, NK, BH)](
            q, k, v, g, h, o, T, K, NT, NK,
            IEEE=q.dtype == torch.float32, BT=_BT, BK=_BK, num_warps=_WARPS, num_stages=1,
        )
        carry_out = h[:, NT].transpose(1, 2).contiguous()
        ctx.save_for_backward(q, k, v, retention, g, h0)
        ctx.set_materialize_grads(False)
        return o, carry_out

    @staticmethod
    def backward(ctx, do, dcarry_out):
        q, k, v, retention, g, h0 = ctx.saved_tensors
        BH, T, K = q.shape
        NT, NK = triton.cdiv(T, _BT), triton.cdiv(K, _BK)
        ieee = q.dtype == torch.float32
        do = torch.zeros_like(q) if do is None else do.contiguous()
        dhf = (dcarry_out.float().transpose(1, 2).contiguous()
               if dcarry_out is not None else None)

        h = _launch_fwd_h(k, v, g, h0)                      # recompute, never saved
        dh = torch.empty(BH, NT, K, K, device=q.device, dtype=torch.float32)
        dh0 = torch.empty(BH, K, K, device=q.device, dtype=torch.float32)
        _pam_bwd_dh[(NK, NK, BH)](
            q, do, g, dhf if dhf is not None else dh, dh, dh0, T, K, NT,
            HAS_DHF=dhf is not None, IEEE=ieee, BT=_BT, BK=_BK, num_warps=_WARPS, num_stages=1,
        )
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        dgp = torch.empty(NK, BH, T, device=q.device, dtype=torch.float32)
        _pam_bwd_dqk[(NT, NK, BH)](
            q, k, v, do, g, h, dh, dq, dk, dgp, T, K, NT, NK, BH,
            IEEE=ieee, BT=_BT, BK=_BK, num_warps=_WARPS, num_stages=1,
        )
        _pam_bwd_dv[(NT, NK, BH)](
            q, k, do, g, dh, dv, T, K, NT, NK,
            IEEE=ieee, BT=_BT, BK=_BK, num_warps=_WARPS, num_stages=1,
        )

        # dG_s = q_s.dq_s - k_s.dk_s, plus <S_out, dS_out> at each tile's last
        # valid position; g_i feeds every G_s with s >= i inside its tile.
        dG = dgp.sum(0)                                                     # [BH, T]
        hd = (h[:, 1:] * dh).sum(dim=(-2, -1))                              # [BH, NT]
        last = (torch.arange(1, NT + 1, device=q.device) * _BT).clamp(max=T) - 1
        dG.index_add_(1, last, hd)
        dG = F.pad(dG, (0, NT * _BT - T)).view(BH, NT, _BT)
        dg = dG.flip(-1).cumsum(-1).flip(-1).reshape(BH, NT * _BT)[:, :T]
        dret = (dg / (retention.float() + _EPS)).to(retention.dtype)
        dcarry = dh0.transpose(1, 2) if ctx.needs_input_grad[4] else None
        return dq, dk, dv, dret, dcarry


# ── public entry ─────────────────────────────────────────────────────────────

def _use_triton(q) -> bool:
    return (kernel_enabled() and q.is_cuda and q.shape[-1] <= _MAX_K
            and not torch.compiler.is_compiling())


def _common_dtype(q, k, v):
    if q.dtype == k.dtype == v.dtype:
        return q.dtype
    # Under autocast the RoPE multiply promotes q/k to fp32 while v stays
    # bf16; the scan runs in the autocast dtype like every other matmul.
    if torch.is_autocast_enabled(q.device.type):
        return torch.get_autocast_dtype(q.device.type)
    return torch.promote_types(torch.promote_types(q.dtype, k.dtype), v.dtype)


def fused_real_pam_read(q, k, v, retention, carry, chunk_size):
    """Chunked real-PAM read + carried state.  See the module docstring."""
    dt = _common_dtype(q, k, v)
    q, k, v = q.to(dt), k.to(dt), v.to(dt)
    if _use_triton(q):
        return _PamScanFn.apply(q, k, v, retention, carry)
    return pam_scan_torch(q, k, v, retention, carry, chunk_size)


def fused_complex_pam_read(q, k, v, retention, carry, chunk_size):
    r"""Complex additive PAM read + carried state, on the real kernel.

    Same recurrence as the complex ``PAMLayer``:

        S_t = g_t S_{t-1} + v_t (x) conj(k_t),   y_s = S_s . q_s

    with a *real* per-(token) retention ``g_t``.  Everything is split-real:
    ``q, k, v`` are ``[BH, T, K, 2]`` (last axis = real, imag), ``retention``
    is ``[BH, T]``, ``carry`` is ``[BH, K, K, 2]`` ([value, key] rows/cols) or
    ``None``.  Returns ``(read [BH, T, K, 2], carry_out [BH, K, K, 2])``.

    Trick (exact, autograd-complete, reuses ``fused_real_pam_read`` unchanged):
    stack real/imag into width-2K real q~/k~/v~ so a real dot reproduces the
    conjugate score ``<q_s, k_t> = q_r.k_r + q_i.k_i + i(q_i.k_r - q_r.k_i)``.
    Two real scans (with q~ and the rotated q~'=[q_i;-q_r]) give the real and
    imaginary weighted value sums; recombine.  See EXPERIMENTS_SEMPY "Complex
    at kernel speed" for the block algebra.
    """
    BH, T, K, _ = q.shape
    qr, qi = q[..., 0], q[..., 1]
    kr, ki = k[..., 0], k[..., 1]
    vr, vi = v[..., 0], v[..., 1]
    ktil = torch.cat([kr, ki], dim=-1)             # [BH, T, 2K]
    vtil = torch.cat([vr, vi], dim=-1)
    qtil = torch.cat([qr, qi], dim=-1)             # score real part  A = q.k* real
    qtilp = torch.cat([qi, -qr], dim=-1)           # score imag part  B = q.k* imag

    carry_til = None
    if carry is not None:
        ReS, ImS = carry[..., 0], carry[..., 1]    # [BH, K(v), K(k)]
        carry_til = q.new_zeros(BH, 2 * K, 2 * K, dtype=torch.float32)
        carry_til[:, :K, :K] = ReS                 # v_r rows, k_r cols  (S~_rr)
        carry_til[:, K:, :K] = ImS                 # v_i rows, k_r cols  (S~_ir)

    A, Sout = fused_real_pam_read(qtil, ktil, vtil, retention, carry_til, chunk_size)
    B, _ = fused_real_pam_read(qtilp, ktil, vtil, retention, carry_til, chunk_size)

    y_r = A[..., :K] - B[..., K:]
    y_i = A[..., K:] + B[..., :K]
    read = torch.stack([y_r, y_i], dim=-1)         # [BH, T, K, 2]

    ReSo = Sout[:, :K, :K] + Sout[:, K:, K:]       # S~_rr + S~_ii
    ImSo = Sout[:, K:, :K] - Sout[:, :K, K:]       # S~_ir - S~_rk
    carry_out = torch.stack([ReSo, ImSo], dim=-1)  # [BH, K, K, 2]
    return read, carry_out


def pam_delta_torch(q, k, v, retention, beta_w, beta_e, carry, chunk):
    r"""Chunked delta erase/write (A3), torch WY form on top of the additive scan.

    Recurrence (real, [value, key] state; read ``y = S q``):

        S_t = g_t S_{t-1} (I - b_e,t k_t k_t^T) + b_w,t v_t k_t^T

    with unit-norm keys ``k``.  Within a chunk this equals the additive scan
    with *pseudo-values* ``W`` in place of ``v``:

        w_t = b_w,t v_t - b_e,t a_t (S_0 k_t) - b_e,t sum_{j<t}(a_t/a_j)(k_j.k_t) w_j
        (I + P) W = diag(b_w) V - diag(b_e a) (K S_0^T),  P[t,j]=b_e,t (a_t/a_j)(k_t.k_j)

    (derivation in EXPERIMENTS_SEMPY "A3").  ``P`` is strictly lower-triangular,
    so ``(I+P)`` is unit lower-triangular and ``W`` is one triangular solve
    (fp32); then ``fused_real_pam_read(q, k, W, g, S_0)`` gives the read and the
    next carry.  ``q,k,v [BH,T,K]``; ``retention/beta_w/beta_e [BH,T]``;
    ``carry [BH,K,K]`` or None.
    """
    BH, T, K = q.shape
    dt = q.dtype
    r = retention.float()
    bw = beta_w.float()
    be = beta_e.float()
    S = carry.float() if carry is not None else None
    reads = []
    for start in range(0, T, chunk):
        w = min(chunk, T - start)
        sl = slice(start, start + w)
        qc, kc, vc = q[:, sl].float(), k[:, sl].float(), v[:, sl].float()
        gc = r[:, sl]                                                   # [BH, w]
        bwc, bec = bw[:, sl], be[:, sl]                                 # [BH, w]
        G = torch.cumsum(torch.log(gc + _EPS), dim=-1)                  # [BH, w]
        a = torch.exp(G)                                               # cumulative decay
        low = torch.tril(torch.ones(w, w, device=q.device), -1)        # strictly lower
        Gamma = torch.exp(torch.clamp(G.unsqueeze(-1) - G.unsqueeze(-2), max=0.0))  # a_i/a_j
        KK = torch.bmm(kc, kc.transpose(1, 2))                          # [BH, w, w]
        P = (bec.unsqueeze(-1) * (KK * Gamma)) * low                    # [BH, w, w]
        A = P + torch.eye(w, device=q.device)                          # unit lower-tri
        rhs = bwc.unsqueeze(-1) * vc                                    # [BH, w, K]
        if S is not None:
            KS = torch.bmm(kc, S.transpose(1, 2))                       # S_0 k_t : [BH, w, K(v)]
            rhs = rhs - (bec * a).unsqueeze(-1) * KS
        W = torch.linalg.solve_triangular(A, rhs, upper=False, unitriangular=True)
        read, S = fused_real_pam_read(qc.to(dt), kc.to(dt), W.to(dt), gc, S, w)
        reads.append(read)
    return torch.cat(reads, dim=1), S


def pam_delta_batched(q, k, v, retention, beta_w, beta_e, carry, sub=_BT):
    r"""Same delta recurrence as ``pam_delta_torch``, restructured for speed.

    Why (2026-09-06 profile, 100M B6 T2048): ``pam_delta_torch`` costs 2x the
    additive path, and none of it is the scan kernel -- it is the per-chunk
    Python loop (T/256 iterations x 16 layers): eight fp32 triangular solves +
    their backward, fp32 SIMT GEMMs and ``[w, w]`` elementwise traffic at
    w=256, eight scan launches, eight slice-backward nodes.

    Key identity: the pseudo-values are LINEAR in the carried state.  For a
    sub-chunk c with start state ``S`` (rows value, cols key):

        W_c = A_c^{-1} diag(b_w) V_c  -  A_c^{-1} diag(b_e a) K_c  S^T
            =        W0_c            -           U_c              S^T

    ``W0`` and ``U`` do not depend on ``S``, so ONE batched unit-triangular
    solve over all sub-chunks (rhs ``[bw V | diag(be a) K]``) gives both.  The
    chunk-end state then obeys a linear matrix recurrence in ``S`` alone:

        S_c = a_last S_{c-1} + (E o W_c)^T K_c
            = S_{c-1} (a_last I - (E o U_c)^T K_c) + (E o W0_c)^T K_c
            = S_{c-1} M_c + N_c                     (E_t = a_last / a_t)

    with ``M_c, N_c`` precomputed batched -- T/64 small ``[K, K]`` bmm steps.
    Then ``W = W0 - U S_prev^T`` for every sub-chunk at once, and a SINGLE
    ``fused_real_pam_read(q, k, W, retention)`` over the full sequence returns
    the read and the final carry (its internal BT=64 tiles line up with the
    sub-chunks, and its states equal ``S_c`` by construction).

    Sub-chunk 64 (not 256) makes the solve and all ``[w, w]`` work 4x smaller
    per token.  Ragged ``T`` is zero-padded (retention 1, gates 0 => W = 0).
    Autograd flows through plain torch ops.  Parity with ``pam_delta_torch``:
    ``v13_sempty/tmp/delta_parity.py``.
    """
    BH, T, K = q.shape
    dt = q.dtype
    Tp = ((T + sub - 1) // sub) * sub
    if Tp != T:
        pad = Tp - T
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        retention = F.pad(retention, (0, pad), value=1.0)
        beta_w = F.pad(beta_w, (0, pad), value=0.0)
        beta_e = F.pad(beta_e, (0, pad), value=0.0)
    S0 = carry.float() if carry is not None else torch.zeros(BH, K, K, device=q.device)
    W = _delta_prep_fn()(k, v, retention, beta_w, beta_e, S0, sub)
    read, S_out = fused_real_pam_read(q, k, W, retention, carry, sub)
    if Tp != T:
        read = read[:, :T]
    return read, S_out


def _delta_prep(k, v, retention, beta_w, beta_e, S0, sub: int):
    """Pseudo-values ``W [BH, Tp, K]`` (scan dtype) for the delta recurrence.

    Batched per-sub-chunk WY solve -> recurrence coefficients -> sequential
    chunk-state recurrence -> ``W = W0 - U S_prev^T``.  Pure torch; this is
    the region ``torch.compile`` fuses (eager: ~60 ms/step of un-fused fp32
    elementwise + casts, plus ~2000 tiny dispatches for the recurrence, at
    100M B6 T2048).
    """
    BH, Tp, K = k.shape
    dt = k.dtype
    NC = Tp // sub
    kf = k.float().reshape(BH * NC, sub, K)
    vf = v.float().reshape(BH * NC, sub, K)
    g = retention.float().reshape(BH * NC, sub)
    bw = beta_w.float().reshape(BH * NC, sub)
    be = beta_e.float().reshape(BH * NC, sub)

    G = torch.cumsum(torch.log(g + _EPS), dim=-1)                       # [BHN, w]
    a = torch.exp(G)                                                    # decay from chunk start
    a_last = a[:, -1]                                                   # [BHN]
    E = torch.exp(G[:, -1:] - G)                                        # a_last / a_t  <= 1
    low = torch.tril(torch.ones(sub, sub, device=k.device), -1)
    Gamma = torch.exp(torch.clamp(G.unsqueeze(-1) - G.unsqueeze(-2), max=0.0))   # a_i / a_j
    KK = torch.bmm(kf, kf.transpose(1, 2))                              # [BHN, w, w]
    A = (be.unsqueeze(-1) * (KK * Gamma)) * low + torch.eye(sub, device=k.device)
    rhs = torch.cat([bw.unsqueeze(-1) * vf, (be * a).unsqueeze(-1) * kf], dim=-1)  # [BHN, w, 2K]
    sol = torch.linalg.solve_triangular(A, rhs, upper=False, unitriangular=True)
    W0, U = sol.split(K, dim=-1)                                        # [BHN, w, K] each

    # Chunk recurrence coefficients: S_c = S_{c-1} M_c + N_c.  These GEMMs run
    # in the scan dtype (bf16 under autocast, fp32 accumulate) -- the precision
    # the additive kernel itself uses for W and k.  The [K,K] recurrence in
    # the caller stays fp32.
    kd = k.reshape(BH * NC, sub, K)
    EU = (E.unsqueeze(-1) * U).to(dt)
    EW0 = (E.unsqueeze(-1) * W0).to(dt)
    B_ = torch.bmm(EU.transpose(1, 2), kd).float()                      # (E o U)^T K  [BHN, K, K]
    N_ = torch.bmm(EW0.transpose(1, 2), kd).float()                     # (E o W0)^T K
    eye = torch.eye(K, device=k.device)
    M_ = (a_last.view(-1, 1, 1) * eye - B_).view(BH, NC, K, K)
    N_ = N_.view(BH, NC, K, K)

    # Sequential over sub-chunks: NC-1 baddbmm steps of [BH,K,K]x[K,K].  (A
    # log-depth scan was tried: 5x the fp32 K^3 FLOPs + cat traffic, slower.)
    # unbind/stack keep autograd to one node each -- per-step select/slice
    # backward zero-fills the whole [BH,NC,K,K] tensor every time.
    Ms, Ns = M_.unbind(1), N_.unbind(1)
    S = S0
    starts = [S]
    for c in range(NC - 1):                                             # states at sub-chunk starts
        S = torch.baddbmm(Ns[c], S, Ms[c])
        starts.append(S)
    S_prev = torch.stack(starts, dim=1)                                 # [BH, NC, K, K]

    W = W0.to(dt).view(BH, NC, sub, K) - torch.matmul(
        U.to(dt).view(BH, NC, sub, K), S_prev.to(dt).transpose(-1, -2))
    return W.reshape(BH, Tp, K)


_DELTA_COMPILE = os.environ.get("V13S_DELTA_COMPILE", "1") == "1"
_delta_prep_compiled = None


def _delta_prep_fn():
    """``_delta_prep`` compiled once (lazily); eager if disabled/unavailable."""
    global _delta_prep_compiled
    if not _DELTA_COMPILE or torch.compiler.is_compiling():
        return _delta_prep
    if _delta_prep_compiled is None:
        try:
            _delta_prep_compiled = torch.compile(_delta_prep, dynamic=False)
        except Exception:  # pragma: no cover
            _delta_prep_compiled = _delta_prep
    return _delta_prep_compiled


__all__ = ["fused_real_pam_read", "fused_complex_pam_read", "pam_scan_torch",
           "pam_delta_torch", "pam_delta_batched", "set_kernel_enabled",
           "kernel_enabled", "HAS_TRITON"]
