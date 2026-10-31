"""Grove II (v15): a sequence organ made of living units under selection.

A unit ("tree") is not a weight and a bias.  It is an organism in a niche:
  * a root r_i in R^n: what it eats (the running mean of every token it ate),
  * a fruit u_i in R^n: what it yields (the running mean of what FOLLOWED the
    tokens it ate: succession),
  * use_i: how much it has eaten lately (starvation when it stops),
  * innate traits (learned by backprop): initial root and fruit, a fitness
    bias, the soil/sap/out projections; per grove a reign lam, a surrogate
    temperature and a fixed novelty threshold theta.

Three laws of ecology give the recurrence:
  competitive exclusion  each token is eaten by exactly ONE tree of a grove:
                         the argmax of accumulated evidence
                         l_i(t) = lam * l_i(t-1) + temp * cos(x_t, r_i) + b_i
                         (gradient through a straight-through softmax);
  colonisation           a token no root fits (max cos < theta), and that is
                         the first of its kind this season, founds a tree
                         (root = the token) and is eaten by it; the newborn's
                         evidence starts at its founding fitness (a brief reign);
  starvation             at the season's end newborns take the niches with the
                         smallest use (use <- rho * use + tokens eaten).
The read is the winner's fruit, ripened by the season's credits so far
(a value is readable the token after it is written).  Fast groves with a
strict theta are exact instance caches; slow groves with a loose theta hold
prototypes and context.

Seasons (every C tokens) make it a GPU model: inside a season roots are
fixed, so births are a strict-causal mask on the season's C x C cosine
matrix, evidence is a C x C retention matrix, the landing one argmax over
K + C columns, the reads C x C ripening matrices, and placement one scatter.
Decode keeps a season buffer and mirrors it token by token; prefill ==
decode to round-off (selftest).  State per grove is K * (2n + 3) numbers:
O(1) in the sequence length.

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
    trees: int = 32           # K niches per grove
    root: int = 64            # n
    season: int = 32          # C tokens between growths / placements
    lam_init: tuple = (0.1, 0.9)   # evidence retention (the reign), spread across groves
    temp_init: float = 10.0        # fitness = temp * cos(x, root): shapes the surrogate gradient
    theta_init: tuple = (0.95, 0.2) # novelty threshold on cosine, spread across groves, paired
                                    # with lam: fast groves strict (instance caches), slow groves
                                    # loose (prototypes / context)
    use_decay: float = 0.75        # starvation: use <- use_decay * use + wins, per season
    use_init: float = 1.0          # innate experience of a dawn tree (its root0 counts this much)
    conv: int = 4             # depthwise causal conv on the soil path (0 = off)


def _inv_sigmoid(p: float) -> float:
    return math.log(p / (1 - p))


class GroveState:
    """Per-sequence state of one GroveLayer (all tensors [B, G, ...])."""

    def __init__(self, ev, roots, fruits, m_last, use, buf=None):
        self.ev = ev            # [B, G, K] accumulated evidence
        self.roots = roots      # [B, G, K, n]
        self.fruits = fruits    # [B, G, K, n]
        self.m_last = m_last    # [B, G, K] one-hot of the last winner (for succession credit)
        self.use = use          # [B, G, K] decayed count of tokens eaten (starvation)
        self.buf = buf          # decode: this season's buffers (see GroveLayer.step)

    def detach(self):
        return GroveState(self.ev.detach(), self.roots.detach(), self.fruits.detach(),
                          self.m_last.detach(), self.use.detach(), None if self.buf is None else
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
        self.fruit0_born = nn.Parameter(torch.zeros(G, n))   # a newborn's fruit before it is fed
        self.bias = nn.Parameter(torch.zeros(G, K))
        self.bias_born = nn.Parameter(torch.zeros(G))
        lo, hi = cfg.lam_init
        lam0 = torch.linspace(lo, hi, G) if G > 1 else torch.tensor([hi])
        self.lam_p = nn.Parameter(torch.log(lam0 / (1 - lam0)))
        self.temp_p = nn.Parameter(torch.full((G,), math.log(cfg.temp_init)))
        t_lo, t_hi = cfg.theta_init
        self.register_buffer('theta', torch.linspace(t_lo, t_hi, G) if G > 1 else torch.tensor([t_hi]))
        self.conv = nn.Conv1d(G * n, G * n, cfg.conv, groups=G * n, bias=False) if cfg.conv > 1 else None
        if self.conv is not None:                        # identity on the current token at init:
            with torch.no_grad():                        # a seed's identity is the token; context is learned
                self.conv.weight.zero_()
                self.conv.weight[:, 0, -1] = 1.0
        self._record = None                              # diagnostics: list of (m_hard, born) per season

    # ---- helpers -------------------------------------------------------------------
    @property
    def lam(self):
        return torch.sigmoid(self.lam_p)                 # [G]

    @property
    def temp(self):
        return self.temp_p.exp()                         # [G]

    def init_state(self, B: int, device=None) -> GroveState:
        G, K, n = self.cfg.groves, self.cfg.trees, self.cfg.root
        z = torch.zeros(B, G, K, device=device, dtype=self.root0.dtype)
        return GroveState(z.clone(), self.root0.expand(B, G, K, n).clone(),
                          self.fruit0.expand(B, G, K, n).clone(), z.clone(),
                          torch.full_like(z, self.cfg.use_init))

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
            y = F.conv1d(full, self.conv.weight, None, groups=G * n)   # valid: [B, G*n, T]
            new_tail = full[..., -(k - 1):]
            x = y.transpose(1, 2)
        return x.reshape(B, T, G, n), new_tail

    @staticmethod
    def _grow(proto, S, g, base):
        """A tree is the running mean of what it ate: proto counts for `base`
        tokens (its decayed use), the new food for g.  proto [..., J, n],
        S [..., J, n] = sum m x, g [..., J] = sum m, base [..., J] (0 for a newborn)."""
        step = g / (base + g).clamp_min(1e-6)                                 # in [0, 1]
        mean = S / g.clamp_min(1e-6).unsqueeze(-1)
        return proto + step.unsqueeze(-1) * (mean - proto)

    def _place(self, roots, fruits, use, ev, bornT, rootb, fruitb, useb, evb_end, me_last):
        """Season's end: born trees (in birth order) take the niches with the
        smallest use.  bornT [B,G,C] bool; rootb/fruitb [B,G,C,n]; useb/evb_end
        [B,G,C]; me_last [B,G,K+C] the eating mass at the last token.
        Returns new roots, fruits, use, ev and m_last [B,G,K] (that mass carried
        into the niches: a replaced dawn tree's and a dropped newborn's are lost)."""
        B, G, K, n = roots.shape
        C = bornT.shape[-1]
        rank = bornT.long().cumsum(-1) - 1                                   # [B,G,C]
        keep = bornT & (rank < K)
        order = use.argsort(dim=-1)                                          # ascending use
        slot = torch.gather(order, -1, rank.clamp(0, K - 1))                 # [B,G,C]
        idx = torch.where(keep, slot, torch.full_like(slot, K))              # K = the dust bin
        def put(base, new):                                                  # base [B,G,K,...], new [B,G,C,...]
            pad = torch.cat([base, torch.zeros_like(base[:, :, :1])], dim=2)
            ix = idx.view(B, G, C, *([1] * (new.dim() - 3))).expand_as(new)
            return pad.scatter(2, ix, new)[:, :, :K]
        roots = put(roots, rootb)
        fruits = put(fruits, fruitb)
        use = put(use, useb)
        ev = put(ev, evb_end)
        replaced = torch.zeros(B, G, K + 1, dtype=torch.bool, device=roots.device)
        replaced = replaced.scatter(2, idx, torch.ones_like(idx, dtype=torch.bool))[:, :, :K]
        m_last = me_last[:, :, :K] * (~replaced).to(roots.dtype) + put(torch.zeros_like(use), me_last[:, :, K:])
        return roots, fruits, use, ev, m_last

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
        # the economy runs in fp32 whatever the autocast dtype
        if x.dtype in (torch.bfloat16, torch.float16):
            x, y = x.float(), y.float()
        dt = x.dtype
        lam, temp, theta, rho = self.lam, self.temp, self.theta, cfg.use_decay
        idx = torch.arange(C, device=h.device)
        expo = (idx[:, None] - idx[None, :]).clamp_min(0).to(dt)
        Lmat = torch.where(idx[:, None] >= idx[None, :], lam[:, None, None] ** expo, 0.0)  # [G,C,C]
        carry = lam[:, None] ** (idx + 1).to(dt)                          # [G, C]
        strict = idx[:, None] > idx[None, :]                              # [C, C]  t > s
        strictf = strict.to(dt)
        eyeC = torch.eye(C, dtype=torch.bool, device=h.device)
        neg = torch.finfo(dt).min
        ev, roots, fruits, m_last, use = state.ev, state.roots, state.fruits, state.m_last, state.use
        outs = []
        ac = torch.autocast(device_type=h.device.type, enabled=False)
        ac.__enter__()
        for c in range(T // C):
            xc = x[:, c * C:(c + 1) * C]                                  # [B, C, G, n]
            yc = y[:, c * C:(c + 1) * C]
            xn = F.normalize(xc, dim=-1)
            cosR = torch.einsum('bcgn,bgkn->bcgk', xn, F.normalize(roots, dim=-1))   # [B,C,G,K]
            f = temp[:, None] * cosR + self.bias
            evd = torch.einsum('gts,bsgk->btgk', Lmat, f) + carry.T[None, :, :, None] * ev[:, None]
            # births: novel = no dawn root fits; born = first of its kind this season
            nov = cosR.max(-1).values < theta                             # [B,C,G]
            cosX = torch.einsum('btgn,bsgn->btsg', xn, xn)                # [B,C(t),C(s),G]
            dup = ((cosX >= theta) & nov[:, None] & strict[None, :, :, None]).any(2)
            born = nov & ~dup                                             # [B,C,G]
            valid = strict[None, :, :, None] & born[:, None]              # [B,C(t),C(s),G]
            # the newborn's evidence starts at its founding fitness (cos 1): a brief reign
            self_land = eyeC[None, :, :, None] & born[:, None]            # [B,C(t),C(s),G]
            fb = torch.where(valid | self_land, temp * cosX + self.bias_born,
                             torch.zeros((), dtype=dt, device=h.device))
            evb = torch.einsum('gtr,brsg->btsg', Lmat, fb)                # [B,C,C(s),G]
            evb_m = torch.where(valid, evb, torch.full((), neg, dtype=dt, device=h.device))
            score = torch.cat([evd, evb_m.permute(0, 1, 3, 2)], dim=-1)   # [B,C,G,K+C]
            # eating is soft (every tree grows by its share), except that a novel token
            # is eaten by its own newborn alone
            m_eat = torch.softmax(score, dim=-1)
            newborn = F.one_hot(K + idx, K + C).to(dt)                     # [C, K+C]
            me = torch.where(born[..., None], newborn[None, :, None], m_eat)   # [B,C,G,K+C]
            # succession credit: y_t goes to whoever ate x_{t-1}, by share
            m_last_ext = torch.cat([m_last, torch.zeros(B, G, C, dtype=dt, device=h.device)], -1)
            m_prev = torch.cat([m_last_ext[:, None], me[:, :-1]], dim=1)  # [B,C,G,K+C]
            gy_all = m_prev.cumsum(1)
            gy = gy_all - m_prev                                          # credits before t
            # canopy: the READ is a soft mixture over the trees that BEAR FRUIT (what you
            # see); a sapling (a newborn nobody has credited yet) is not readable
            sapling = torch.cat([torch.zeros(B, C, G, K, dtype=torch.bool, device=h.device),
                                 gy[..., K:] <= 0], dim=-1)
            m = torch.softmax(torch.where(sapling, torch.full((), neg, dtype=dt, device=h.device), score), dim=-1)
            if self._record is not None:
                self._record.append((me.detach(), born.detach(), m.detach()))
            prior = torch.cat([fruits, self.fruit0_born[None, :, None].expand(B, G, C, n)], dim=2)
            base = torch.cat([rho * use, torch.zeros(B, G, C, dtype=dt, device=h.device)], -1)
            step = gy / (base[:, None] + gy).clamp_min(1e-6)              # ripening = running mean
            wr = m * step / gy.clamp_min(1e-6)
            A = torch.einsum('btgj,bsgj->bgts', wr, m_prev) * strictf
            outs.append(torch.einsum('btgj,bgjn->btgn', m * (1 - step), prior)
                        + torch.einsum('bgts,bsgn->btgn', A, yc))
            # season's end: growth, starvation, placement
            wins = me.sum(1)                                              # [B,G,K+C] mass eaten
            massx = wins
            Sx = torch.einsum('btgj,btgn->bgjn', me, xc)
            Sy = torch.einsum('btgj,btgn->bgjn', m_prev, yc)
            gyS = gy_all[:, -1]
            roots_n = self._grow(roots, Sx[:, :, :K], massx[:, :, :K], rho * use)
            fruits_n = self._grow(fruits, Sy[:, :, :K], gyS[:, :, :K], rho * use)
            use_n = rho * use + wins[:, :, :K]
            zero = torch.zeros(B, G, C, dtype=dt, device=h.device)
            rootb = self._grow(xc.permute(0, 2, 1, 3), Sx[:, :, K:], massx[:, :, K:], zero)
            fruitb = self._grow(prior[:, :, K:], Sy[:, :, K:], gyS[:, :, K:], zero)
            useb = wins[:, :, K:]
            roots, fruits, use, ev, m_last = self._place(
                roots_n, fruits_n, use_n, evd[:, -1], born.permute(0, 2, 1), rootb, fruitb,
                useb, torch.where(born.permute(0, 2, 1), evb[:, -1].permute(0, 2, 1),
                                  torch.full((), neg, dtype=dt, device=h.device)),
                me[:, -1])
        ac.__exit__(None, None, None)
        o = torch.cat(outs, dim=1).reshape(B, T, G * n)[:, :T0]
        buf = {'tail': tail} if tail is not None else None
        return self.out(o.to(h.dtype)), GroveState(ev, roots, fruits, m_last, use, buf)

    # ---- token-by-token (decode) form ----------------------------------------------
    @torch.no_grad()
    def step(self, h_t: torch.Tensor, state: GroveState | None = None):
        """h_t [B, d] -> o_t [B, d], state.  Mirrors the season form token by
        token; births wait in the season buffer and are placed every C tokens."""
        B, _ = h_t.shape
        cfg = self.cfg
        G, K, n, C = cfg.groves, cfg.trees, cfg.root, cfg.season
        if state is None:
            state = self.init_state(B, h_t.device)
        buf = state.buf or {}
        dt = state.roots.dtype
        dev = h_t.device
        if 'count' not in buf:
            buf.update(Sx=torch.zeros(B, G, K + C, n, device=dev, dtype=dt),
                       Sy=torch.zeros(B, G, K + C, n, device=dev, dtype=dt),
                       gy=torch.zeros(B, G, K + C, device=dev, dtype=dt),
                       massx=torch.zeros(B, G, K + C, device=dev, dtype=dt),
                       wins=torch.zeros(B, G, K + C, device=dev, dtype=dt),
                       xs=torch.zeros(B, G, C, n, device=dev, dtype=dt),       # this season's soil
                       nov=torch.zeros(B, G, C, dtype=torch.bool, device=dev),
                       born=torch.zeros(B, G, C, dtype=torch.bool, device=dev),
                       evb=torch.zeros(B, G, C, device=dev, dtype=dt),
                       credit=torch.cat([state.m_last, torch.zeros(B, G, C, device=dev, dtype=dt)], -1),
                       count=torch.zeros((), device=dev))
        p = int(buf['count'])
        lam, temp, theta, rho = self.lam, self.temp, self.theta, cfg.use_decay
        x, tail = self._soil(h_t[:, None], buf.get('tail'))
        x = x[:, 0].to(dt)                                                # [B, G, n]
        y = self.sap(h_t).view(B, G, n).to(dt)
        xn = F.normalize(x, dim=-1)
        cosR = torch.einsum('bgn,bgkn->bgk', xn, F.normalize(state.roots, dim=-1))
        f = temp[:, None] * cosR + self.bias
        ev = lam[None, :, None] * state.ev + f                             # [B, G, K]
        pos = torch.arange(C, device=dev)
        earlier = pos < p                                                  # [C]
        cosX = torch.einsum('bgn,bgsn->bgs', xn, F.normalize(buf['xs'], dim=-1))   # [B,G,C]
        nov = cosR.max(-1).values < theta                                  # [B, G]
        dup = ((cosX >= theta[None, :, None]) & buf['nov'] & earlier).any(-1)
        born_now = nov & ~dup
        valid = buf['born'] & earlier                                      # born trees that compete
        fb = torch.where(valid, temp[:, None] * cosX + self.bias_born[:, None], torch.zeros((), dtype=dt, device=dev))
        evb = lam[None, :, None] * buf['evb'] + fb
        evb[:, :, p] = torch.where(born_now, temp[None, :] + self.bias_born[None, :], evb[:, :, p])
        neg = torch.finfo(dt).min
        scb = torch.where(valid, evb, torch.full((), neg, dtype=dt, device=dev))
        score = torch.cat([ev, scb], -1)
        m_eat = torch.softmax(score, dim=-1)
        mh = torch.where(born_now[..., None], F.one_hot(torch.tensor(K + p, device=dev), K + C).to(dt)[None, None], m_eat)  # eating mass
        sapling = torch.cat([torch.zeros(B, G, K, dtype=torch.bool, device=dev), buf['gy'][:, :, K:] <= 0], -1)
        m = torch.softmax(torch.where(sapling, torch.full((), neg, dtype=dt, device=dev), score), dim=-1)   # soft read
        # read: the winner's ripening fruit (credits before this token)
        prior = torch.cat([state.fruits, self.fruit0_born[None, :, None].expand(B, G, C, n)], 2)
        base = torch.cat([rho * state.use, torch.zeros(B, G, C, dtype=dt, device=dev)], -1)
        ripe = self._grow(prior, buf['Sy'], buf['gy'], base)               # [B,G,K+C,n]
        o = torch.einsum('bgj,bgjn->bgn', m, ripe).reshape(B, G * n)
        # credit this token's sap to the winner at t-1 (+ founding), then eat
        credit = buf['credit']
        buf['Sy'] = buf['Sy'] + credit.unsqueeze(-1) * y.unsqueeze(2)
        buf['gy'] = buf['gy'] + credit
        buf['Sx'] = buf['Sx'] + mh.unsqueeze(-1) * x.unsqueeze(2)
        buf['massx'] = buf['massx'] + mh
        buf['wins'] = buf['wins'] + mh
        buf['xs'][:, :, p] = x
        buf['nov'][:, :, p] = nov
        buf['born'][:, :, p] = born_now
        buf['evb'] = evb
        buf['credit'] = mh.clone()
        if tail is not None:
            buf['tail'] = tail
        buf['count'] = buf['count'] + 1
        roots, fruits, use, m_last = state.roots, state.fruits, state.use, mh[:, :, :K]
        if p + 1 == C:
            roots_n = self._grow(roots, buf['Sx'][:, :, :K], buf['massx'][:, :, :K], rho * use)
            fruits_n = self._grow(fruits, buf['Sy'][:, :, :K], buf['gy'][:, :, :K], rho * use)
            use_n = rho * use + buf['wins'][:, :, :K]
            zero = torch.zeros(B, G, C, dtype=dt, device=dev)
            rootb = self._grow(buf['xs'], buf['Sx'][:, :, K:], buf['massx'][:, :, K:], zero)
            fruitb = self._grow(prior[:, :, K:], buf['Sy'][:, :, K:], buf['gy'][:, :, K:], zero)
            useb = buf['wins'][:, :, K:]
            evb_end = torch.where(buf['born'], evb, torch.full((), neg, dtype=dt, device=dev))
            roots, fruits, use, ev, m_last = self._place(
                roots_n, fruits_n, use_n, ev, buf['born'], rootb, fruitb, useb, evb_end, mh)
            keep_tail = buf.get('tail')
            buf = {} if keep_tail is None else {'tail': keep_tail}
        return self.out(o.to(h_t.dtype)), GroveState(ev, roots, fruits, m_last, use, buf)


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
        per_layer = g.groves * g.trees * (2 * g.root + 3) * 4
        return B * per_layer * self.cfg.n_layers
