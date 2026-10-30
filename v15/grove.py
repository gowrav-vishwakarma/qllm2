"""Grove (v15): a sequence organ made of living units under selection.

A unit ("tree") is not a weight and a bias.  It is an organism with
  * a canopy share m_i in [0, 1]: its share of a conserved budget (the shares
    of the K trees of a grove sum to 1),
  * a root r_i in R^n: what it grows from (a prototype it moves toward the
    inputs it wins),
  * a fruit u_i in R^n: what it yields (a prototype of what FOLLOWED the
    inputs it won: succession),
  * innate traits (learned by backprop): its initial root and fruit and an
    innate fitness bias b_i.

The recurrence is selection.  Each token gives every tree a fitness
f_i(t) = temp * cos(x_t, r_i) + b_i; a tree's evidence accumulates
l_i(t) = lam * l_i(t-1) + f_i(t), and its share is the softmax of the
evidence over the grove: m(t) = softmax_i l_i(t).  With lam = 1 this is
exactly the Bayesian posterior over K hypotheses given the tokens so far;
lam < 1 is a forgetful posterior.  Forgetting is RELATIVE: a tree loses share
only when others gain.  There is no decay constant on the state.

Growth happens in seasons (every C tokens).  Inside a season all reads use
the roots and fruits of the season's start, so the season form is exactly
parallel (a C x C matrix per grove for the evidence scan, one softmax, one
matmul for the read); at the season's end every tree moves its root toward
the mass-weighted mean of the inputs it won, and its fruit toward the
mass-weighted mean of what followed them, by a step that grows with the mass
it won and never overshoots.  Decode keeps a season buffer and applies the
same growth every C tokens, so prefill == decode to round-off (selftest).

The read is selective by construction: o(t) = sum_i m_i(t) u_i, a softmax
mixture over K trees, never a linear superposition.  State per grove is
K * (2n + 2) numbers: O(1) in the sequence length.

Plain torch on purpose (new lane, prototype); dims are written in comments
as [B, T, G, K, n] = batch, time, groves, trees per grove, root width.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GroveConfig:
    dim: int = 512            # model width d
    groves: int = 8           # G
    trees: int = 32           # K
    root: int = 64            # n
    season: int = 32          # C tokens between growths
    lam_init: tuple = (0.1, 0.9)   # evidence retention, spread across groves (sigmoid param)
    alpha_init: float = 2.0        # growth rate: step = 1 - exp(-alpha * mass won) (softplus param)
    temp_init: float = 10.0        # fitness = temp * cos(x, root) (log param)
    conv: int = 4             # depthwise causal conv on the soil path (0 = off)


def _inv_sigmoid(p: float) -> float:
    return math.log(p / (1 - p))


class GroveState:
    """Per-sequence state of one GroveLayer (all tensors [B, G, ...])."""

    def __init__(self, ev, roots, fruits, m_last, buf=None):
        self.ev = ev            # [B, G, K] accumulated evidence
        self.roots = roots      # [B, G, K, n]
        self.fruits = fruits    # [B, G, K, n]
        self.m_last = m_last    # [B, G, K] share at the last token (for succession)
        # decode buffer: season sums (Sx, gx, Sy, gy, count) and the conv tail
        self.buf = buf

    def detach(self):
        return GroveState(self.ev.detach(), self.roots.detach(), self.fruits.detach(),
                          self.m_last.detach(), None if self.buf is None else
                          {k: v.detach() for k, v in self.buf.items()})


class GroveLayer(nn.Module):
    def __init__(self, cfg: GroveConfig):
        super().__init__()
        self.cfg = cfg
        d, G, K, n = cfg.dim, cfg.groves, cfg.trees, cfg.root
        self.soil = nn.Linear(d, G * n, bias=False)      # x: what a tree is matched on
        self.sap = nn.Linear(d, G * n, bias=False)       # y: what a tree stores as fruit
        self.out = nn.Linear(G * n, d, bias=False)
        nn.init.zeros_(self.out.weight)                  # the block starts as identity
        self.root0 = nn.Parameter(torch.randn(G, K, n) / math.sqrt(n))
        self.fruit0 = nn.Parameter(torch.randn(G, K, n) / math.sqrt(n))
        self.bias = nn.Parameter(torch.zeros(G, K))
        lo, hi = cfg.lam_init
        lam0 = torch.linspace(lo, hi, G) if G > 1 else torch.tensor([hi])
        self.lam_p = nn.Parameter(torch.log(lam0 / (1 - lam0)))
        self.alpha_p = nn.Parameter(torch.full((G,), math.log(math.expm1(cfg.alpha_init))))
        self.temp_p = nn.Parameter(torch.full((G,), math.log(cfg.temp_init)))
        self.conv = nn.Conv1d(G * n, G * n, cfg.conv, groups=G * n) if cfg.conv > 1 else None

    # ---- helpers -------------------------------------------------------------------
    @property
    def lam(self):
        return torch.sigmoid(self.lam_p)                 # [G]

    @property
    def alpha(self):
        return F.softplus(self.alpha_p)                  # [G]

    @property
    def temp(self):
        return self.temp_p.exp()                         # [G]

    def _fitness(self, x, roots):
        """x [B, (C,) G, n], roots [B, G, K, n] -> temp * cosine + innate bias."""
        xn = F.normalize(x, dim=-1)
        rn = F.normalize(roots, dim=-1)
        if x.dim() == 4:
            f = torch.einsum('bcgn,bgkn->bcgk', xn, rn)
        else:
            f = torch.einsum('bgn,bgkn->bgk', xn, rn)
        return self.temp[:, None] * f + self.bias

    def init_state(self, B: int, device=None) -> GroveState:
        G, K, n = self.cfg.groves, self.cfg.trees, self.cfg.root
        z = torch.zeros(B, G, K, device=device, dtype=self.root0.dtype)
        return GroveState(z.clone(), self.root0.expand(B, G, K, n).clone(),
                          self.fruit0.expand(B, G, K, n).clone(), torch.full_like(z, 1.0 / K))

    def _soil(self, h: torch.Tensor, tail: torch.Tensor | None):
        """h [B, T, d] -> x [B, T, G, n] after the causal conv, and the new conv
        tail (the last k-1 raw columns, so decode continues exactly)."""
        B, T, _ = h.shape
        G, n = self.cfg.groves, self.cfg.root
        x = self.soil(h)                                                  # [B, T, G*n]
        new_tail = None
        if self.conv is not None:
            k = self.cfg.conv
            xt = x.transpose(1, 2)                                        # [B, G*n, T]
            if tail is None:
                tail = xt.new_zeros(B, G * n, k - 1)
            full = torch.cat([tail, xt], dim=2)                           # [B, G*n, k-1+T]
            y = F.conv1d(full, self.conv.weight, self.conv.bias, groups=G * n)   # valid: [B, G*n, T]
            new_tail = full[..., -(k - 1):]
            x = y.transpose(1, 2)
        return x.reshape(B, T, G, n), new_tail

    @staticmethod
    def _grow(proto, S, g, alpha):
        """Move prototypes toward the mass-weighted mean of what they won.
        proto [B,G,K,n], S [B,G,K,n] = sum_t m x, g [B,G,K] = sum_t m, alpha [G]."""
        step = 1 - torch.exp(-alpha[None, :, None] * g)                      # in [0, 1)
        mean = S / g.clamp_min(1e-6).unsqueeze(-1)
        return proto + step.unsqueeze(-1) * (mean - proto)

    # ---- season (parallel) form ----------------------------------------------------
    def forward(self, h: torch.Tensor, state: GroveState | None = None):
        """h [B, T, d] -> o [B, T, d], new state.  A trailing partial season is
        padded with zeros (causal: the real outputs are unchanged); its growth
        and the returned state are then those of the padded sequence, so a
        state to continue from needs T to be a multiple of the season."""
        B, T0, _ = h.shape
        cfg = self.cfg
        G, K, n, C = cfg.groves, cfg.trees, cfg.root, cfg.season
        if T0 % C:
            h = F.pad(h, (0, 0, 0, C - T0 % C))
        T = h.shape[1]
        if state is None:
            state = self.init_state(B, h.device)
        tail = None if state.buf is None else state.buf.get('tail')
        x, tail = self._soil(h, tail)                                    # [B, T, G, n]
        y = self.sap(h).view(B, T, G, n)
        # the economy runs in fp32 whatever the autocast dtype: evidence sums
        # reach ~temp * C / (1 - lam) and the state must not drift to bf16
        if x.dtype in (torch.bfloat16, torch.float16):
            x, y = x.float(), y.float()
        lam, alpha = self.lam, self.alpha
        # evidence scan inside a season: L[g, t, s] = lam_g^(t-s) for s <= t
        idx = torch.arange(C, device=h.device)
        expo = (idx[:, None] - idx[None, :]).clamp_min(0).to(h.dtype)    # [C, C]
        Lmat = torch.where(idx[:, None] >= idx[None, :], lam[:, None, None] ** expo, 0.0)  # [G,C,C]
        carry = lam[:, None] ** (idx + 1).to(h.dtype)                     # [G, C]
        ev, roots, fruits, m_last = state.ev, state.roots, state.fruits, state.m_last
        outs = []
        ac = torch.autocast(device_type=h.device.type, enabled=False)
        ac.__enter__()
        for c in range(T // C):
            xc = x[:, c * C:(c + 1) * C]                                  # [B, C, G, n]
            yc = y[:, c * C:(c + 1) * C]
            f = self._fitness(xc, roots)
            evc = torch.einsum('gts,bsgk->btgk', Lmat, f) + carry.T[None, :, :, None] * ev[:, None]
            m = torch.softmax(evc, dim=-1)                                # [B, C, G, K]
            outs.append(torch.einsum('bcgk,bgkn->bcgn', m, fruits))
            # season's end: growth
            m_prev = torch.cat([m_last[:, None], m[:, :-1]], dim=1)      # share at t-1
            Sx = torch.einsum('bcgk,bcgn->bgkn', m, xc)
            Sy = torch.einsum('bcgk,bcgn->bgkn', m_prev, yc)
            roots = self._grow(roots, Sx, m.sum(1), alpha)
            fruits = self._grow(fruits, Sy, m_prev.sum(1), alpha)
            ev, m_last = evc[:, -1], m[:, -1]
        ac.__exit__(None, None, None)
        o = torch.cat(outs, dim=1).reshape(B, T, G * n)[:, :T0]
        buf = {'tail': tail} if tail is not None else None
        return self.out(o.to(h.dtype)), GroveState(ev, roots, fruits, m_last, buf)

    # ---- token-by-token (decode) form ----------------------------------------------
    def step(self, h_t: torch.Tensor, state: GroveState | None = None):
        """h_t [B, d] -> o_t [B, d], state.  Applies growth every `season` tokens."""
        B, _ = h_t.shape
        cfg = self.cfg
        G, K, n, C = cfg.groves, cfg.trees, cfg.root, cfg.season
        if state is None:
            state = self.init_state(B, h_t.device)
        buf = state.buf or {}
        if 'count' not in buf:
            z = torch.zeros(B, G, K, device=h_t.device, dtype=h_t.dtype)
            buf.update(Sx=torch.zeros(B, G, K, n, device=h_t.device, dtype=h_t.dtype),
                       Sy=torch.zeros_like(state.roots), gx=z.clone(), gy=z.clone(),
                       count=torch.zeros((), device=h_t.device))
        x, tail = self._soil(h_t[:, None], buf.get('tail'))
        x = x[:, 0]                                                       # [B, G, n]
        y = self.sap(h_t).view(B, G, n)
        f = self._fitness(x, state.roots)
        ev = self.lam[None, :, None] * state.ev + f
        m = torch.softmax(ev, dim=-1)                                     # [B, G, K]
        o = torch.einsum('bgk,bgkn->bgn', m, state.fruits).reshape(B, G * n)
        buf['Sx'] = buf['Sx'] + m.unsqueeze(-1) * x.unsqueeze(2)
        buf['gx'] = buf['gx'] + m
        buf['Sy'] = buf['Sy'] + state.m_last.unsqueeze(-1) * y.unsqueeze(2)
        buf['gy'] = buf['gy'] + state.m_last
        buf['count'] = buf['count'] + 1
        if tail is not None:
            buf['tail'] = tail
        roots, fruits = state.roots, state.fruits
        if int(buf['count']) == C:
            roots = self._grow(roots, buf['Sx'], buf['gx'], self.alpha)
            fruits = self._grow(fruits, buf['Sy'], buf['gy'], self.alpha)
            for k in ('Sx', 'Sy', 'gx', 'gy'):
                buf[k] = torch.zeros_like(buf[k])
            buf['count'] = torch.zeros_like(buf['count'])
        return self.out(o), GroveState(ev, roots, fruits, m, buf)


# ---- a minimal LM around the organ (for selftests and micro-benches) ----------------

@dataclass
class GroveLMConfig:
    vocab_size: int = 50257
    dim: int = 512
    n_layers: int = 4
    mlp_hidden: int = 1536
    grove: GroveConfig = field(default_factory=GroveConfig)


class Block(nn.Module):
    def __init__(self, cfg: GroveLMConfig):
        super().__init__()
        g = GroveConfig(**{**cfg.grove.__dict__, 'dim': cfg.dim})
        self.n1 = nn.RMSNorm(cfg.dim)
        self.grove = GroveLayer(g)
        self.n2 = nn.RMSNorm(cfg.dim)
        self.up = nn.Linear(cfg.dim, 2 * cfg.mlp_hidden, bias=False)
        self.down = nn.Linear(cfg.mlp_hidden, cfg.dim, bias=False)

    def _mlp(self, h):
        a, b = self.up(h).chunk(2, dim=-1)
        return self.down(F.silu(a) * b)

    def forward(self, h, state=None):
        o, state = self.grove(self.n1(h), state)
        h = h + o
        return h + self._mlp(self.n2(h)), state

    def step(self, h, state=None):
        o, state = self.grove.step(self.n1(h), state)
        h = h + o
        return h + self._mlp(self.n2(h)), state


class GroveLM(nn.Module):
    def __init__(self, cfg: GroveLMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.RMSNorm(cfg.dim)

    def forward(self, ids, states=None, labels=None):
        h = self.embed(ids)
        states = states or [None] * len(self.blocks)
        new = []
        for blk, st in zip(self.blocks, states):
            h, st = blk(h, st)
            new.append(st)
        logits = F.linear(self.norm(h), self.embed.weight)
        if labels is None:
            return logits, new
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                               labels[:, 1:].reshape(-1), ignore_index=-100)
        return loss, logits, new

    @torch.no_grad()
    def step(self, ids_t, states=None):
        h = self.embed(ids_t)
        states = states or [None] * len(self.blocks)
        new = []
        for blk, st in zip(self.blocks, states):
            h, st = blk.step(h, st)
            new.append(st)
        return F.linear(self.norm(h), self.embed.weight), new

    def state_bytes(self, B: int = 1) -> int:
        g = self.cfg.grove
        per_layer = g.groves * g.trees * (2 * g.root + 2) * 4
        return B * per_layer * self.cfg.n_layers
