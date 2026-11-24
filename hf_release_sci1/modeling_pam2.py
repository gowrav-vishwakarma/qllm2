"""
SCI-1: a PAM-2 (Phase-Associative Memory, real arm) language model, self-contained.

Not a transformer and not Mamba: every "notebook" layer keeps one fixed-size
K x K matrix per head (a delta-rule associative memory), so inference holds
O(1) state per token, independent of context length. No KV cache.

This file is a pure-PyTorch re-implementation of exactly the inference path
of the qllm2 ``v13_sempty`` model (``v13_sempty/model.py``) for the
configuration of the SCI-1 checkpoint. It has no dependency on the qllm2
repo, on ``sempyt`` or on Triton: only ``torch``. Parameter names match the
repo's ``model_state_dict`` one to one, so a repo checkpoint loads with
``load_state_dict(strict=True)`` and no key mapping.

Architecture (SCI-1, 386M params)
---------------------------------
  embed (tied head) -> RMSNorm
  24 blocks, each pre-norm residual:
      x = x + cgu_scale * SwiGLU(RMSNorm(x))                  (every block)
      notebook blocks (3, 7, 11, 15, 19, 23):
      x = x + pam_scale * PAM(RMSNorm(x))                      (delta notebook)
      blocks above the first notebook without one (4-6, 8-10, ...):
      x = x + SharedRead(RMSNorm(x), nearest notebook below)   (M2 shared read)
  hidden = RMSNorm(x)
  lm     = RMSNorm(lm_head_proj(hidden)) + EchoMemory(hidden, ids)
  logits = lm @ embed.weight^T

PAM notebook layer (per head h, key/value/query of width K = 128):
    q, k, v      = split(W_qkv x)                         (slot-major: q | k | v)
    q, k         = rotate(q, k, Chrono clock)             (content-warped RoPE)
    g_t          = exp(-softplus(W_dt x + b_dt + dt_bias))   retention in (0, 1)
    b_w          = sigmoid(W_w x),  b_e = min(sigmoid(W_e x - 3), 0.95)
    k            = k / |k|
    S_t          = g_t S_{t-1} (I - b_e k_t k_t^T) + b_w v_t k_t^T
    y_t          = silu(W_gate x) * (S_t q_t) / sqrt(K)
    out          = W_o y
  Chrono clock: w_t = exp(clamp(W_warp x, -3, 3)) per head, tau_t = sum_{j<t} w_j,
  angle of pair i = tau_t / 10000^(i / (K/2)). With w = 1 it is plain RoPE.
  reset_at_eos: at the separator token (50256) the retention is 0 (fresh notebook).

Shared read (M2): y_t = silu(W_g x) * (S^w_t . rot_w(W_q x)) / sqrt(K), out = W_o y,
  where S^w_t is the writer notebook after step t and rot_w its Chrono rotation.

Echo memory (E1): a delta memory at the top of the stack:
    key_t = unit(W_k hidden_{t-1}) (0 at t = 0), value_t = rms_unit(embed[x_t]),
    query_t = W_q hidden_t, gates from hidden_t, read rms-normed per head, out = W_echo r.
  reset_at_eos: retention 0 and key 0 at the separator.

Two execution modes, same algebra:
  * ``forward`` (prefill / training form): chunked closed form, chunks of 64,
    the delta rule turned into an additive scan through WY pseudo-values
    (one unit-lower-triangular solve per chunk). This is the same math as the
    repo's ``triton_kernels.pam_delta_batched`` (its pure-torch path).
  * ``step`` (decode): the plain recurrence, one token, O(1) per token.
``forward_recurrent`` runs ``step`` over a whole sequence (a slow reference).
Both agree to float round-off (see test_parity.py / PARITY.md).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-6     # log-retention epsilon of the chunked scan (as in the repo kernels)
_SUB = 64       # chunk width of the chunked delta / additive scan (repo: _BT = 64)


def _up(t: torch.Tensor) -> torch.Tensor:
    """Memory math runs in at least fp32 (fp16/bf16 -> fp32; fp32/fp64 kept)."""
    return t if t.dtype in (torch.float32, torch.float64) else t.float()


# ── configuration ────────────────────────────────────────────────────────────


@dataclass
class PAM2Config:
    """The fields of the repo's ``PAMConfig`` that this checkpoint uses.

    Unknown keys in a checkpoint config are ignored by ``from_dict``; every
    option that changes the math is checked by ``validate`` and only the
    SCI-1 setting is accepted.
    """
    vocab_size: int = 50261
    dim: int = 1024
    n_heads: int = 8
    head_dim: int = 128
    n_layers: int = 24
    expand: int = 3
    mlp_hidden: int = 3704
    max_seq_len: int = 4096          # training length (informational; no position table)
    chunk_size: int = 256            # unused by the delta path (it always uses 64-wide chunks)
    base_dt_bias: float = -4.0       # init only
    echo_dt_bias: float = -6.0       # init only
    pam_layers: Tuple[int, ...] = (3, 7, 11, 15, 19, 23)
    reset_at_eos: bool = True
    reset_token_id: int = 50256
    # switches that must take the SCI-1 value (asserted in validate)
    use_rope: bool = True
    tie_weights: bool = True
    activation: str = 'swish'
    is_complex: bool = False
    n_states: int = 1
    vault: bool = False
    delta: bool = True
    cond_mem: bool = False
    chrono: bool = True
    chrono_heads: int = -1
    out_gate: bool = True
    short_conv: bool = False
    echo: bool = True
    share_reads: bool = True
    swa_layers: Tuple[int, ...] = ()
    dropout: float = 0.0

    @classmethod
    def from_dict(cls, raw: dict) -> 'PAM2Config':
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in raw.items() if k in names}
        for k in ('pam_layers', 'swa_layers'):
            if k in kw and kw[k] is not None:
                kw[k] = tuple(kw[k])
        cfg = cls(**kw)
        cfg.validate()
        return cfg

    def to_dict(self) -> dict:
        d = asdict(self)
        d['pam_layers'] = list(self.pam_layers)
        d['swa_layers'] = list(self.swa_layers)
        return d

    def validate(self) -> None:
        need = dict(use_rope=True, tie_weights=True, activation='swish', is_complex=False,
                    n_states=1, vault=False, delta=True, cond_mem=False, chrono=True,
                    out_gate=True, short_conv=False, echo=True, share_reads=True)
        for k, v in need.items():
            if getattr(self, k) != v:
                raise NotImplementedError(
                    f'modeling_pam2 supports only the SCI-1 configuration: {k}={getattr(self, k)!r} '
                    f'(supported: {v!r})')
        if self.chrono_heads not in (-1, self.n_heads):
            raise NotImplementedError(f'chrono_heads={self.chrono_heads}: only all heads clocked (-1)')
        if tuple(self.swa_layers):
            raise NotImplementedError('swa_layers (W1 window reader) is not supported')
        if not self.pam_layers:
            raise NotImplementedError('pam_layers must be set (sparse notebooks + shared reads)')
        if self.dim != self.n_heads * self.head_dim:
            raise NotImplementedError('echo needs dim == n_heads * head_dim')
        if self.head_dim % 2:
            raise ValueError('head_dim must be even (rotary pairs)')


# ── small primitives ─────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    """``x * rsqrt(mean(x^2) + eps) * weight`` (the repo's sempyt RMSNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class GatedUnit(nn.Module):
    """SwiGLU: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Embed(nn.Module):
    """Holds ``embed.embed.weight`` (the repo's RealEmbed key layout)."""

    def __init__(self, vocab: int, dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embed(ids)


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate channel pairs (2i, 2i+1) of ``x [B,H,T,K]`` by ``cos/sin [B,H,T,K/2]``."""
    B, H, T, K = x.shape
    xe, xo = x.reshape(B, H, T, K // 2, 2).unbind(-1)
    return torch.stack((xe * cos - xo * sin, xe * sin + xo * cos), dim=-1).reshape(B, H, T, K)


# ── chunked closed forms (prefill) ───────────────────────────────────────────


def additive_scan(q, k, w, g, S0, chunk: int = _SUB):
    """Read ``y_t = S_t q_t`` of the additive notebook ``S_t = g_t S_{t-1} + w_t k_t^T``.

    ``q, k, w [N, T, K]``, ``g [N, T]`` in [0, 1], ``S0 [N, K(v), K(k)]`` or None.
    Chunked linear-attention form (fp32): inside a chunk
    ``y_s = a_s (S_in q_s) + sum_{t<=s} (a_s/a_t) (q_s.k_t) w_t`` with
    ``a_s / a_t = exp(C_t - C_s)``, ``C = cumsum(-log(g + 1e-6))``.
    Returns ``(y [N, T, K], S_end [N, K, K])``.
    """
    N, T, K = q.shape
    q, k, w, g = _up(q), _up(k), _up(w), _up(g)
    S = _up(S0) if S0 is not None else None
    reads = []
    for s0 in range(0, T, chunk):
        n = min(chunk, T - s0)
        qc, kc, wc = q[:, s0:s0 + n], k[:, s0:s0 + n], w[:, s0:s0 + n]
        C = torch.cumsum(-torch.log(g[:, s0:s0 + n] + _EPS), dim=-1)                 # [N, n] >= 0
        M = torch.exp(torch.clamp(C.unsqueeze(-2) - C.unsqueeze(-1), max=0.0))
        M = M * torch.tril(torch.ones(n, n, device=q.device, dtype=M.dtype))
        y = torch.bmm(torch.bmm(qc, kc.transpose(1, 2)) * M, wc)                    # intra-chunk
        if S is not None:
            y = y + torch.exp(-C).unsqueeze(-1) * torch.bmm(qc, S.transpose(1, 2))   # carried notebook
            S = torch.exp(-C[:, -1])[:, None, None] * S
        else:
            S = torch.zeros(N, K, K, device=q.device, dtype=q.dtype)
        e = torch.exp(C - C[:, -1:])                                                # a_end / a_t <= 1
        S = S + torch.bmm((wc * e.unsqueeze(-1)).transpose(1, 2), kc)
        reads.append(y)
    return torch.cat(reads, dim=1), S


def delta_pseudo_values(k, v, g, bw, be, S0, chunk: int = _SUB):
    """WY pseudo-values ``w`` that turn the delta rule into the additive scan.

    Delta rule (unit keys ``k``):  ``S_t = g_t S_{t-1} (I - b_e,t k_t k_t^T) + b_w,t v_t k_t^T``.
    Inside a chunk with start state ``S``, ``S_t = g-decayed S + sum_j (a_t/a_j) w_j k_j^T`` with

        (I + P) W = diag(b_w) V - diag(b_e a) K S^T,   P[t, j] = b_e,t (a_t/a_j) (k_t . k_j), j < t

    (``a`` = cumulative retention from the chunk start). ``P`` is strictly lower
    triangular, so ``W`` is one unit-triangular solve per chunk. Returns
    ``(W [N, T, K], S_end [N, K, K])`` (fp32).
    """
    N, T, K = k.shape
    k, v, g, bw, be = _up(k), _up(v), _up(g), _up(bw), _up(be)
    S = _up(S0) if S0 is not None else torch.zeros(N, K, K, device=k.device, dtype=k.dtype)
    Ws = []
    for s0 in range(0, T, chunk):
        n = min(chunk, T - s0)
        kc, vc = k[:, s0:s0 + n], v[:, s0:s0 + n]
        gc, bwc, bec = g[:, s0:s0 + n], bw[:, s0:s0 + n], be[:, s0:s0 + n]
        G = torch.cumsum(torch.log(gc + _EPS), dim=-1)                              # [N, n] <= 0
        a = torch.exp(G)
        Gamma = torch.exp(torch.clamp(G.unsqueeze(-1) - G.unsqueeze(-2), max=0.0))  # a_t / a_j
        low = torch.tril(torch.ones(n, n, device=k.device), -1)
        A = (bec.unsqueeze(-1) * (torch.bmm(kc, kc.transpose(1, 2)) * Gamma)) * low \
            + torch.eye(n, device=k.device)
        rhs = bwc.unsqueeze(-1) * vc - (bec * a).unsqueeze(-1) * torch.bmm(kc, S.transpose(1, 2))
        W = torch.linalg.solve_triangular(A, rhs, upper=False, unitriangular=True)
        # chunk-end state: S_end = a_end S + sum_t (a_end / a_t) w_t k_t^T
        E = torch.exp(G[:, -1:] - G)
        S = a[:, -1][:, None, None] * S + torch.bmm((W * E.unsqueeze(-1)).transpose(1, 2), kc)
        Ws.append(W)
    return torch.cat(Ws, dim=1), S


def delta_step(S, q, k, v, g, bw, be):
    """One delta-rule step (fp32): fade, erase along ``k``, write, then read.

    ``S [N, K(v), K(k)]``, ``q, k, v [N, K]``, ``g, bw, be [N]``. Returns ``(S_new, S_new q)``.
    """
    Sk = torch.einsum('nvk,nk->nv', S, k)
    erased = S - be[:, None, None] * Sk.unsqueeze(-1) * k.unsqueeze(-2)
    S = g[:, None, None] * erased + bw[:, None, None] * v.unsqueeze(-1) * k.unsqueeze(-2)
    return S, torch.einsum('nvk,nk->nv', S, q)


# ── layers ───────────────────────────────────────────────────────────────────


class PAMLayer(nn.Module):
    """Delta notebook with Chrono clock, read-out gate and reset at the separator."""

    def __init__(self, cfg: PAM2Config):
        super().__init__()
        D, H, K = cfg.dim, cfg.n_heads, cfg.head_dim
        self.H, self.K = H, K
        self.qkv_proj = nn.Linear(D, 3 * H * K, bias=False)
        self.o_proj = nn.Linear(H * K, D, bias=False)
        self.dt_proj = nn.Linear(D, H)
        self.dt_bias = nn.Parameter(torch.full((H,), cfg.base_dt_bias))
        self.erase_proj = nn.Linear(D, H)
        self.write_proj = nn.Linear(D, H)
        self.warp_proj = nn.Linear(D, H)
        self.gate_proj = nn.Linear(D, H)
        P = K // 2
        self.register_buffer('chrono_inv_freq', 1.0 / (10000.0 ** (torch.arange(P).float() / P)),
                             persistent=False)

    def _inputs(self, x, keep, clock):
        """q (rotated), unit k (rotated), v, retention, gates, rotation, clock out."""
        B, T, _ = x.shape
        H, K = self.H, self.K
        q, k, v = self.qkv_proj(x).reshape(B, T, 3, H, K).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))                         # [B,H,T,K]
        # Chrono clock: per-head warp, exclusive cumulative sum (continues `clock`).
        w = torch.exp(_up(self.warp_proj(x).transpose(1, 2)).clamp(-3.0, 3.0))  # [B,H,T]
        run = torch.cumsum(w, dim=2)
        if clock is not None:
            run = run + clock.to(run.dtype).unsqueeze(-1)
        tau = run - w
        clock_out = run[:, :, -1]
        phi = tau.unsqueeze(-1) * self.chrono_inv_freq.view(1, 1, 1, -1)          # [B,H,T,P]
        cos, sin = torch.cos(phi).to(q.dtype), torch.sin(phi).to(q.dtype)
        q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        k = F.normalize(k, dim=-1)
        g = torch.exp(-F.softplus(self.dt_proj(x) + self.dt_bias)).transpose(1, 2)  # [B,H,T]
        if keep is not None:
            g = g * keep[:, None, :].to(g.dtype)
        be = torch.sigmoid(self.erase_proj(x) - 3.0).clamp(max=0.95).transpose(1, 2)
        bw = torch.sigmoid(self.write_proj(x)).transpose(1, 2)
        return q, k, v, g, bw, be, (cos, sin), clock_out

    def _out(self, x, read):
        B, T, _ = x.shape
        y = read * (self.K ** -0.5)                                               # [B,H,T,K]
        y = y * F.silu(self.gate_proj(x)).transpose(1, 2).unsqueeze(-1)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.H * self.K))

    def forward(self, x, keep, state):
        """Chunked prefill over ``x [B,T,D]``. ``state`` = (S [B,H,K,K], clock [B,H]) or None.
        Returns (out, shared-write dict for readers, new state)."""
        B, T, _ = x.shape
        H, K = self.H, self.K
        S0, clock = state if state is not None else (None, None)
        q, k, v, g, bw, be, rot, clock_out = self._inputs(x, keep, clock)
        flat = lambda t: t.reshape(B * H, T, *t.shape[3:])
        S0f = S0.reshape(B * H, K, K) if S0 is not None else None
        W, S_end = delta_pseudo_values(flat(k), flat(v), flat(g), flat(bw), flat(be), S0f)
        read, _ = additive_scan(flat(q), flat(k), W, flat(g), S0f)
        read = read.reshape(B, H, T, K).to(x.dtype)
        shared = {'k': flat(k), 'W': W, 'g': flat(g), 'S0': S0f, 'rot': rot}
        return self._out(x, read), shared, (S_end.reshape(B, H, K, K), clock_out)

    def step(self, x, keep, state):
        """One-token decode, ``x [B,1,D]``: the plain recurrence on the carried notebook."""
        B = x.shape[0]
        H, K = self.H, self.K
        S0, clock = state if state is not None else (None, None)
        q, k, v, g, bw, be, rot, clock_out = self._inputs(x, keep, clock)
        S = _up(S0.reshape(B * H, K, K)) if S0 is not None else \
            torch.zeros(B * H, K, K, device=x.device, dtype=_up(x).dtype)
        one = lambda t: _up(t[:, :, 0].reshape(B * H, *t.shape[3:]))
        S, read = delta_step(S, one(q), one(k), one(v), one(g), one(bw), one(be))
        read = read.reshape(B, H, 1, K).to(x.dtype)
        shared = {'S': S, 'rot': rot}
        return self._out(x, read), shared, (S.reshape(B, H, K, K), clock_out)


class SharedRead(nn.Module):
    """M2 reader of the nearest notebook below: own query, gate and out-projection."""

    def __init__(self, cfg: PAM2Config):
        super().__init__()
        D, H, K = cfg.dim, cfg.n_heads, cfg.head_dim
        self.H, self.K = H, K
        self.q_proj = nn.Linear(D, H * K, bias=False)
        self.gate = nn.Linear(D, H)
        self.o_proj = nn.Linear(H * K, D, bias=False)

    def forward(self, x, shared):
        B, T, _ = x.shape
        H, K = self.H, self.K
        q = self.q_proj(x).reshape(B, T, H, K).transpose(1, 2)
        q = _rotate(q, *shared['rot']).reshape(B * H, T, K)
        if 'S' in shared:                                                          # decode: S_t q_t
            read = torch.einsum('nvk,ntk->ntv', shared['S'], _up(q))
        else:
            read, _ = additive_scan(q, shared['k'], shared['W'], shared['g'], shared['S0'])
        read = read.to(x.dtype) * (K ** -0.5)
        gate = F.silu(self.gate(x)).transpose(1, 2).reshape(B * H, T, 1)
        y = (read * gate).reshape(B, H, T, K).transpose(1, 2).reshape(B, T, H * K)
        return self.o_proj(y)


class Block(nn.Module):
    def __init__(self, cfg: PAM2Config, idx: int):
        super().__init__()
        D = cfg.dim
        self.norm1 = RMSNorm(D)
        self.cgu = GatedUnit(D, cfg.mlp_hidden or D * cfg.expand)
        self.cgu_scale = nn.Parameter(torch.tensor(1.0))
        self.has_pam = idx in tuple(cfg.pam_layers)
        self.pam = None
        self.reader = None
        if self.has_pam:
            self.norm2 = RMSNorm(D)
            self.pam = PAMLayer(cfg)
            self.pam_scale = nn.Parameter(torch.tensor(0.1))
        elif cfg.share_reads and idx > min(cfg.pam_layers):
            self.norm2 = RMSNorm(D)
            self.reader = SharedRead(cfg)


# ── state ────────────────────────────────────────────────────────────────────


@dataclass
class PAM2State:
    """Everything the model carries between calls: O(1) in context length.

    ``layers[i]`` = (notebook S [B,H,K,K] fp32, Chrono clock [B,H] fp32) for a
    notebook block, None otherwise; ``echo`` = (S [B,H,K,K] fp32, last hidden [B,D]).
    ``n_tokens`` counts the tokens consumed (informational).
    """
    layers: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = field(default_factory=list)
    echo: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    n_tokens: int = 0

    def nbytes(self) -> int:
        ts = [t for p in self.layers if p is not None for t in p]
        if self.echo is not None:
            ts += list(self.echo)
        return sum(t.numel() * t.element_size() for t in ts)

    def clone(self) -> 'PAM2State':
        cl = lambda p: None if p is None else tuple(t.clone() for t in p)
        return PAM2State([cl(p) for p in self.layers], cl(self.echo), self.n_tokens)


# ── language model ───────────────────────────────────────────────────────────


class PAM2LM(nn.Module):
    def __init__(self, cfg: PAM2Config):
        super().__init__()
        cfg.validate()
        self.config = cfg
        D, H, K = cfg.dim, cfg.n_heads, cfg.head_dim
        self.embed = _Embed(cfg.vocab_size, D)
        self.embed_norm = RMSNorm(D)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.output_norm = RMSNorm(D)
        self.lm_head_proj = nn.Linear(D, D)
        self.lm_head_norm = RMSNorm(D)
        self.echo_qk = nn.Linear(D, 2 * D, bias=False)
        self.echo_gates = nn.Linear(D, 3 * H)        # dt | write | erase
        self.echo_out = nn.Linear(D, D, bias=False)

    # -- echo memory --------------------------------------------------------

    def _echo(self, hidden, ids, state, recurrent: bool):
        cfg = self.config
        B, T, D = hidden.shape
        H, K = cfg.n_heads, cfg.head_dim
        q, k = self.echo_qk(hidden).chunk(2, dim=-1)
        if state is None:
            k = F.pad(k, (0, 0, 1, 0))[:, :T]                                   # k_t from h_{t-1}; k_0 = 0
            S0 = None
        else:
            k_prev = self.echo_qk(state[1].to(hidden.dtype)).chunk(2, dim=-1)[1]
            k = torch.cat([k_prev[:, None], k[:, :T - 1]], dim=1)
            S0 = _up(state[0].reshape(B * H, K, K))
        v = F.embedding(ids, self.embed.embed.weight).to(hidden.dtype)
        v = v * torch.rsqrt(_up(v).pow(2).mean(-1, keepdim=True) + 1e-6).to(hidden.dtype)
        heads = lambda t: t.reshape(B, T, H, K).transpose(1, 2).reshape(B * H, T, K)
        per_head = lambda t: t.transpose(1, 2).reshape(B * H, T)
        dt, wr, er = self.echo_gates(hidden).split(H, dim=-1)
        g = per_head(torch.exp(-F.softplus(dt)))
        bw = per_head(torch.sigmoid(wr))
        be = per_head(torch.sigmoid(er - 3.0).clamp(max=0.95))
        kn = F.normalize(heads(k), dim=-1)
        if cfg.reset_at_eos:
            sep = ids == cfg.reset_token_id
            g = g * (~sep).to(g.dtype)[:, None, :].expand(B, H, T).reshape(B * H, T)
            kn = kn * (~sep).to(kn.dtype)[:, None, :, None].expand(B, H, T, 1).reshape(B * H, T, 1)
        qh, vh = heads(q), heads(v)
        if recurrent:
            S = S0 if S0 is not None else torch.zeros(B * H, K, K, device=hidden.device, dtype=_up(hidden).dtype)
            reads = []
            for t in range(T):
                S, r = delta_step(S, _up(qh[:, t]), _up(kn[:, t]), _up(vh[:, t]),
                                  _up(g[:, t]), _up(bw[:, t]), _up(be[:, t]))
                reads.append(r)
            read = torch.stack(reads, dim=1)
        else:
            W, S = delta_pseudo_values(kn, vh, g, bw, be, S0)
            read, _ = additive_scan(qh, kn, W, g, S0)
        read = _up(read)
        read = (read * torch.rsqrt(read.pow(2).mean(-1, keepdim=True) + 1e-6)).to(hidden.dtype)
        read = read.reshape(B, H, T, K).transpose(1, 2).reshape(B, T, D)
        return self.echo_out(read), (S.reshape(B, H, K, K), _up(hidden[:, -1].detach()))

    # -- the stack ----------------------------------------------------------

    def _run(self, input_ids, state: Optional[PAM2State], recurrent: bool):
        cfg = self.config
        B, T = input_ids.shape
        if recurrent and T != 1:
            raise ValueError('step() takes one token per call')
        x = self.embed_norm(self.embed(input_ids))
        keep = (input_ids != cfg.reset_token_id).float() if cfg.reset_at_eos else None
        new_layers: List[Optional[tuple]] = []
        shared = None
        for i, blk in enumerate(self.blocks):
            x = x + blk.cgu(blk.norm1(x)) * blk.cgu_scale
            if blk.pam is not None:
                st = state.layers[i] if state is not None else None
                fn = blk.pam.step if recurrent else blk.pam.forward
                out, shared, st_new = fn(blk.norm2(x), keep, st)
                x = x + out * blk.pam_scale
                new_layers.append(st_new)
            else:
                if blk.reader is not None:
                    x = x + blk.reader(blk.norm2(x), shared)
                new_layers.append(None)
        hidden = self.output_norm(x)
        lm = self.lm_head_norm(self.lm_head_proj(hidden))
        echo_out, echo_state = self._echo(hidden, input_ids, state.echo if state is not None else None,
                                          recurrent)
        lm = lm + echo_out
        logits = F.linear(lm, self.embed.embed.weight)
        n0 = state.n_tokens if state is not None else 0
        return logits, PAM2State(new_layers, echo_state, n0 + T)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, state: Optional[PAM2State] = None):
        """Chunked prefill. ``input_ids [B, T]`` -> (logits [B, T, V] fp32-or-model-dtype, new state).
        A ``state`` from an earlier call continues that sequence exactly."""
        return self._run(input_ids, state, recurrent=False)

    @torch.no_grad()
    def step(self, input_ids: torch.Tensor, state: Optional[PAM2State] = None):
        """One-token decode, ``input_ids [B, 1]`` -> (logits [B, 1, V], new state). O(1) per token."""
        return self._run(input_ids, state, recurrent=True)

    @torch.no_grad()
    def forward_recurrent(self, input_ids: torch.Tensor, state: Optional[PAM2State] = None):
        """Reference: ``step`` over every token (slow; for checking the chunked form)."""
        outs = []
        for t in range(input_ids.shape[1]):
            lg, state = self.step(input_ids[:, t:t + 1], state)
            outs.append(lg)
        return torch.cat(outs, dim=1), state

    # -- generation ---------------------------------------------------------

    @staticmethod
    def _pick(logits, generated, temperature, top_k, top_p, repetition_penalty, generator=None):
        """Next token from ``logits [B, V]`` (same rules as the repo's LM.generate)."""
        logits = logits.float()
        if temperature > 0.0:
            logits = logits / temperature
        if repetition_penalty != 1.0 and generated is not None and generated.numel():
            score = torch.gather(logits, 1, generated)
            score = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            logits = logits.scatter(1, generated, score)
        if temperature <= 0.0:
            return logits.argmax(-1, keepdim=True)
        if top_k > 0:
            v, _ = logits.topk(min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < v[:, -1:], float('-inf'))
        if top_p > 0:
            sl, si = logits.sort(descending=True)
            p = sl.softmax(dim=-1)
            sl = sl.masked_fill(p.cumsum(dim=-1) - p >= top_p, float('-inf'))
            logits = sl.scatter(1, si, sl)
        return torch.multinomial(logits.softmax(dim=-1), 1, generator=generator)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100, temperature: float = 1.0,
                 top_k: int = 50, top_p: float = 0.0, repetition_penalty: float = 1.0,
                 eos_token_id=None, state: Optional[PAM2State] = None,
                 return_state: bool = False, penalty_context: Optional[torch.Tensor] = None,
                 generator: Optional[torch.Generator] = None, streamer=None):
        """Prefill ``input_ids`` (continuing ``state`` if given), then decode O(1)/token.

        ``temperature <= 0`` = greedy. ``eos_token_id`` (int or list of ints) stops a row.
        Returns ``input_ids`` + new tokens; with ``return_state=True`` also the state,
        which has consumed every returned token EXCEPT the last generated one (feed
        that one first when continuing, e.g. the next chat turn). ``penalty_context`` (ids) is
        what the repetition penalty looks at; default = ``input_ids`` + generated.
        ``streamer(token_id_tensor)`` is called after each new token.
        """
        self.eval()
        eos = set([eos_token_id] if isinstance(eos_token_id, int) else (eos_token_id or []))
        logits, state = self.forward(input_ids, state)
        generated = input_ids
        seen = penalty_context if penalty_context is not None else input_ids
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for i in range(max_new_tokens):
            nxt = self._pick(logits[:, -1], seen, temperature, top_k, top_p, repetition_penalty,
                             generator)
            generated = torch.cat([generated, nxt], dim=1)
            seen = torch.cat([seen, nxt], dim=1)
            if streamer is not None:
                streamer(nxt)
            if eos:
                finished |= torch.isin(nxt.squeeze(1), torch.tensor(sorted(eos), device=nxt.device))
                if bool(finished.all()):
                    break
            if i == max_new_tokens - 1:
                break
            logits, state = self.step(nxt, state)
        return (generated, state) if return_state else generated

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── loading ──────────────────────────────────────────────────────────────────


def map_repo_state_dict(sd: dict) -> dict:
    """Repo ``model_state_dict`` -> this model's keys. Identity, except a
    ``torch.compile`` prefix ``_orig_mod.`` is stripped if present."""
    return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}


def load_model(path: str, device='cpu', dtype: torch.dtype = torch.float32,
               weights_only: bool = True) -> PAM2LM:
    """Load a release file holding ``config`` + ``model_state_dict``.

    ``weights_only=True`` (default) unpickles tensors and plain types only (safe for
    downloaded files). A full repo training save also carries RNG / optimizer
    objects; load such a TRUSTED file with ``weights_only=False``.
    """
    ckpt = torch.load(path, map_location='cpu', weights_only=weights_only, mmap=True)
    cfg = PAM2Config.from_dict(ckpt['config'])
    model = PAM2LM(cfg)
    model.load_state_dict(map_repo_state_dict(ckpt['model_state_dict']), strict=True)
    model.to(device=device, dtype=dtype).eval()
    model.meta = {k: ckpt.get(k) for k in ('global_step', 'global_tokens') if k in ckpt}
    return model


class TokenStreamer:
    """Print generated tokens as they arrive (incremental BPE decode: print the new suffix)."""

    def __init__(self, decode, stop_ids=(), out=None):
        import sys
        self.decode, self.stop_ids, self.out = decode, set(stop_ids), out or sys.stdout
        self.ids, self.shown = [], ''

    def __call__(self, nxt: torch.Tensor):
        t = int(nxt.view(-1)[0])
        if t in self.stop_ids:
            return
        self.ids.append(t)
        text = self.decode(self.ids)
        if text.endswith('\ufffd'):          # incomplete multi-byte character: wait for the next token
            return
        self.out.write(text[len(self.shown):])
        self.out.flush()
        self.shown = text


IM_START, IM_END, THINK_START, THINK_END = '<|im_start|>', '<|im_end|>', '<think>', '</think>'


def load_tokenizer(name: str = 'gpt2'):
    """GPT-2 BPE + 4 added special tokens = vocab 50261, exactly as training used
    (qllm2 ``v7/data.get_chat_tokenizer``): ids 50257 ``<|im_start|>``, 50258
    ``<|im_end|>``, 50259 ``<think>``, 50260 ``</think>``; 50256 ``<|endoftext|>``
    is the document separator (and pad). Needs ``transformers``."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.add_special_tokens({'additional_special_tokens': [IM_START, IM_END, THINK_START, THINK_END]})
    tok.pad_token = tok.eos_token
    want = {IM_START: 50257, IM_END: 50258, THINK_START: 50259, THINK_END: 50260}
    got = {t: tok.convert_tokens_to_ids(t) for t in want}
    if got != want or len(tok) != 50261:
        raise RuntimeError(f'tokenizer ids {got} / len {len(tok)} do not match the model (vocab 50261)')
    return tok


__all__ = ['PAM2Config', 'PAM2LM', 'PAM2State', 'TokenStreamer', 'load_model', 'load_tokenizer',
           'map_repo_state_dict',
           'additive_scan', 'delta_pseudo_values', 'delta_step']
