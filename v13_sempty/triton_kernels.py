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
            b_dq += _dot(b_do, tl.trans(b_h).to(b_do.dtype), IEEE)
            b_dh = tl.load(dh_base + offs_k[:, None] * K + offs_v[None, :],
                           mask=m_k[:, None] & m_v[None, :], other=0.0)
            b_dks += _dot(b_v, tl.trans(b_dh).to(b_v.dtype), IEEE)
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
        num_warps=4, num_stages=1,
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
            IEEE=q.dtype == torch.float32, BT=_BT, BK=_BK, num_warps=4, num_stages=1,
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
            HAS_DHF=dhf is not None, IEEE=ieee, BT=_BT, BK=_BK, num_warps=4, num_stages=1,
        )
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        dgp = torch.empty(NK, BH, T, device=q.device, dtype=torch.float32)
        _pam_bwd_dqk[(NT, NK, BH)](
            q, k, v, do, g, h, dh, dq, dk, dgp, T, K, NT, NK, BH,
            IEEE=ieee, BT=_BT, BK=_BK, num_warps=4, num_stages=1,
        )
        _pam_bwd_dv[(NT, NK, BH)](
            q, k, do, g, dh, dv, T, K, NT, NK,
            IEEE=ieee, BT=_BT, BK=_BK, num_warps=4, num_stages=1,
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


def fused_real_pam_read(q, k, v, retention, carry, chunk_size):
    """Chunked real-PAM read + carried state.  See the module docstring."""
    if _use_triton(q):
        return _PamScanFn.apply(q, k, v, retention, carry)
    return pam_scan_torch(q, k, v, retention, carry, chunk_size)


__all__ = ["fused_real_pam_read", "pam_scan_torch", "set_kernel_enabled",
           "kernel_enabled", "HAS_TRITON"]
