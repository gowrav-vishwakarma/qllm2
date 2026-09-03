"""A4: Engram-style conditional memory (the n-gram substitute).

A dedicated hashed n-gram lookup table read, gated by the hidden state and
added residually after an early block. Distinct from the PAM outer-product
memory: this is content-addressable *by surface n-gram*, giving O(1) recall of
locally-bound associations (the read-side gap the probes exposed).

Design (DeepSeek Engram / DSE, ACL 2026, adapted):
  * canonical token ids (NFKC + lowercase + whitespace-collapse) so surface
    variants of a word share a slot; precomputed once and cached.
  * suffix n-grams n in {2, 3}, two hash heads per order -> 4 lookups into one
    shared table [M, d_m] (multiplicative hashing, wrap-around mod 2^k).
  * concat -> W_V (value, to D) and W_K (key, to D).
  * gate a = sigmoid(RMSNorm(h) . RMSNorm(W_K e) / sqrt(D)); u = a * (W_V e).
  * y = SiLU(dwConv1d(RMSNorm(u), k=4, dilation=3)) + u; residual add.
  * table: lr x5, no weight decay (handled by the trainer param groups); conv
    zero-init so the local mixing starts as a pass-through of u.

This module is a declared raw-torch boundary (integer hashing / Conv1d);
``check_torch_layout.SKIP_FILES`` lists it.
"""
from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from sempyt.tensor import named

_CACHE = Path(__file__).resolve().parent.parent / ".cache" / "cond_mem"


def _canon_string(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.strip().lower()
    return " ".join(s.split())


def build_canonical_ids(vocab_size: int, tokenizer=None) -> torch.Tensor:
    """[vocab] -> dense canonical id; surface variants of a word collapse.

    Cached under .cache/cond_mem/ keyed by vocab size.
    """
    _CACHE.mkdir(parents=True, exist_ok=True)
    path = _CACHE / f"canon_gpt2_{vocab_size}.pt"
    if path.exists():
        return torch.load(path)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    canon = torch.zeros(vocab_size, dtype=torch.long)
    table: dict[str, int] = {"": 0}
    nxt = 1
    for tid in range(vocab_size):
        try:
            s = _canon_string(tokenizer.decode([tid]))
        except Exception:  # noqa: BLE001
            s = ""
        cid = table.get(s)
        if cid is None:
            cid = nxt
            table[s] = cid
            nxt += 1
        canon[tid] = cid
    torch.save(canon, path)
    return canon


class ConditionalMemory(nn.Module):
    # four odd multipliers < 2^63 (splitmix / FNV-ish); int64 overflow on the
    # key*mult product wraps two's-complement, which is fine for hashing.
    _MULTS = (0x2545F4914F6CDD1D, 0x27D4EB2F165667C5,
              0x1E3779B97F4A7C15, 0x165667B19E3779F9)

    def __init__(self, cfg, model_dim, tokenizer=None):
        super().__init__()
        self.model_dim = model_dim
        self.D = cfg.dim
        self.M = int(cfg.cond_mem_slots)      # power of two
        assert self.M & (self.M - 1) == 0, "cond_mem_slots must be a power of two"
        self.dm = int(cfg.cond_mem_dim)
        self.n_lookups = 4
        self.table = nn.Embedding(self.M, self.dm)
        self.w_v = nn.Linear(self.n_lookups * self.dm, self.D, bias=False)
        self.w_k = nn.Linear(self.n_lookups * self.dm, self.D, bias=False)
        self._dil = 3
        self._k = 4
        self.conv = nn.Conv1d(self.D, self.D, kernel_size=self._k, groups=self.D,
                              dilation=self._dil, bias=True)
        self.norm_h = nn.Parameter(torch.ones(self.D))
        self.norm_e = nn.Parameter(torch.ones(self.D))
        self.norm_u = nn.Parameter(torch.ones(self.D))
        canon = build_canonical_ids(cfg.vocab_size, tokenizer)
        self.register_buffer("canon", canon, persistent=False)
        self.Vc = int(canon.max().item()) + 1
        self.register_buffer("mults", torch.tensor(self._MULTS, dtype=torch.long),
                             persistent=False)
        # inits: conv zero (pass-through of u at start), table/W small.
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        nn.init.normal_(self.table.weight, std=0.02)
        nn.init.normal_(self.w_v.weight, std=0.02)
        nn.init.normal_(self.w_k.weight, std=0.02)

    def is_table_param(self, param) -> bool:
        return param is self.table.weight

    def _rms(self, x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w

    def _lookup(self, key, head):
        idx = (key * self.mults[head]) & (self.M - 1)
        return self.table(idx)                                  # [B, T, dm]

    def forward(self, z_named, input_ids):
        batch, time = z_named.layout[0], z_named.layout[1]
        h = z_named.raw(batch, time, self.model_dim)            # [B, T, D]
        B, T, D = h.shape
        c = self.canon.to(input_ids.device)[input_ids]         # [B, T] canonical ids
        Vc = self.Vc
        c1 = F.pad(c, (1, 0))[:, :T]                            # c_{t-1}
        c2 = F.pad(c, (2, 0))[:, :T]                            # c_{t-2}
        bikey = c + c1 * Vc                                     # wrap ok
        trikey = c + c1 * Vc + c2 * (Vc * Vc)
        e = torch.cat([self._lookup(bikey, 0), self._lookup(bikey, 1),
                       self._lookup(trikey, 2), self._lookup(trikey, 3)], dim=-1)
        ev = self.w_v(e)                                        # [B, T, D]
        ek = self.w_k(e)
        gate = torch.sigmoid(
            (self._rms(h, self.norm_h) * self._rms(ek, self.norm_e)).sum(-1, keepdim=True)
            / (D ** 0.5))                                       # [B, T, 1]
        u = gate * ev
        un = self._rms(u, self.norm_u).transpose(1, 2)         # [B, D, T]
        pad = (self._k - 1) * self._dil
        uc = self.conv(F.pad(un, (pad, 0))).transpose(1, 2)    # causal dilated
        y = F.silu(uc) + u
        return named(h + y, (batch, time, self.model_dim))     # named-exit: cond-mem boundary
