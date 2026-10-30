r"""N7: in-context n-gram memory (exact copy that does not fade, O(1)).

Why (EXPERIMENTS_V14 "N7"): PAM's in-context lookup fails by *key
collisions*. Echo files every label under a template key ("<noun> is"),
and the delta erase lets the last write win (v13 label_exact: PAM 0.54-0.73
vs T++ 0.86-0.92). N7 does echo's job ("what followed the last moment like
this one") with exact hash keys, so different contexts cannot share a key.

Per sequence and per order n there is a table of S slots. Reading token x_t:

    write   slot(g_{t-1}) <- (fp(g_{t-1}), x_t)      g_t = (x_{t-n+1} .. x_t)
    read    (fp, v) = table[slot(g_t)];  found_t = (fp == fp(g_t));  value_t = v

slot() and fp() are two independent integer hashes of the n-gram; a write
overwrites its slot (bounded memory), and the fingerprint turns a slot
collision into "not found" instead of a wrong answer. Memory is S slots per
order per sequence, whatever the length: O(1) state, O(1) time per token.

Training / prefill computes the same thing for all positions at once:
value_t comes from the latest t' < t whose n-gram hashed to the same slot
(the last write to that slot before t), found if its fingerprint matches.
``lookup_parallel`` == ``lookup_step`` run token by token (selftest).

The read into the body (``NGramReader``): per injection block and order,

    r = found * sigmoid(w . rms(h) + b) * (W E[value] + f)

added to the residual stream. W and f are zero at init, so a model with N7
on starts bit-identical to the model without it.

Raw-torch boundary: integer hashing, sort and gather (declared in
``check_layout``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Fixed odd 64-bit multipliers (deterministic, no RNG): one set for the slot
# hash and one for the fingerprint, indexed by the position inside the n-gram.
_SLOT_MULT = (0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9,
              0x27D4EB2F165667C5, 0x85EBCA77C2B2AE63, 0xFF51AFD7ED558CCD)
_FP_MULT = (0xD6E8FEB86659FD93, 0xA0761D6478BD642F, 0xE7037ED1A0B428DB,
            0x8EBC6AF09C88C6E3, 0x589965CC75374CC3, 0x1D8E4E27C47D124F)
MAX_ORDER = len(_SLOT_MULT)


def _i64(c: int) -> int:
    """An unsigned 64-bit constant as the signed int64 torch stores."""
    return c - (1 << 64) if c >= (1 << 63) else c


def _hash(grams: torch.Tensor, mults) -> torch.Tensor:
    """grams [..., n] int64 -> [...] int64; multiply-add wraps mod 2^64."""
    h = torch.zeros(grams.shape[:-1], dtype=torch.int64, device=grams.device)
    for i in range(grams.shape[-1]):
        h = h * 31 + grams[..., i] * _i64(mults[i])
    return h ^ (h >> 29)


def _grams(ids: torch.Tensor, n: int) -> torch.Tensor:
    """ids [B, T] -> [B, T, n]: the n-gram ending at each t (left-padded with -1)."""
    pad = torch.full((ids.shape[0], n - 1), -1, dtype=torch.int64, device=ids.device)
    x = torch.cat([pad, ids.long()], dim=1)
    return x.unfold(1, n, 1)                                     # [B, T, n]


def lookup_parallel(ids: torch.Tensor, n: int, slots: int, want_table: bool = True):
    """All positions at once. Returns (value [B, T] long, found [B, T] bool,
    table) where table = (fp [B, S], val [B, S], used [B, S]) after the last
    write, i.e. the decode state after this prefix (the n-gram ending at the
    last token is written only when the next token arrives)."""
    B, T = ids.shape
    dev = ids.device
    grams = _grams(ids, n)
    valid = torch.arange(T, device=dev) >= n - 1                 # a full n-gram ends here
    slot = torch.remainder(_hash(grams, _SLOT_MULT), slots)      # [B, T]
    fp = _hash(grams, _FP_MULT)
    # positions without a full n-gram get their own slot beyond the table
    t_idx = torch.arange(T, device=dev).expand(B, T)
    slot = torch.where(valid, slot, slots + t_idx)
    # last earlier position with the same slot = predecessor in (slot, t) order
    order = torch.argsort(slot * T + t_idx, dim=1)
    s_sorted = torch.gather(slot, 1, order)
    prev = torch.roll(order, 1, dims=1)
    same = torch.roll(s_sorted, 1, dims=1) == s_sorted
    same[:, 0] = False
    src = torch.full_like(order, -1)
    src.scatter_(1, order, torch.where(same, prev, torch.full_like(prev, -1)))
    has = src >= 0                                               # a write to this slot before t
    srcc = src.clamp(min=0)
    found = has & valid & (torch.gather(fp, 1, srcc) == fp)
    value = torch.gather(ids.long(), 1, (srcc + 1).clamp(max=T - 1))
    value = torch.where(found, value, torch.zeros_like(value))
    if not want_table:                                           # training: no decode state needed
        return value, found, None
    # decode state: slot <- (fp, next token) for every t <= T-2 with a full n-gram,
    # the latest write winning
    w_ok = valid.clone()
    w_ok[T - 1:] = False                                         # last n-gram has no next token yet
    t_w = torch.where(w_ok, t_idx, torch.full_like(t_idx, -1))
    last = torch.full((B, slots), -1, dtype=torch.int64, device=dev)
    last.scatter_reduce_(1, torch.where(w_ok, slot, torch.zeros_like(slot)), t_w,
                         reduce='amax', include_self=True)
    used = last >= 0
    lc = last.clamp(min=0)
    tab_fp = torch.where(used, torch.gather(fp, 1, lc), torch.zeros_like(lc))
    tab_val = torch.where(used, torch.gather(ids.long(), 1, (lc + 1).clamp(max=T - 1)),
                          torch.zeros_like(lc))
    return value, found, (tab_fp, tab_val, used)


def empty_state(batch: int, n: int, slots: int, device) -> dict:
    z = torch.zeros(batch, slots, dtype=torch.int64, device=device)
    return {'fp': z.clone(), 'val': z.clone(), 'used': torch.zeros(batch, slots, dtype=torch.bool, device=device),
            'tail': torch.full((batch, n), -1, dtype=torch.int64, device=device)}


def state_from_prefix(ids: torch.Tensor, n: int, slots: int) -> dict:
    """The decode state after reading ``ids`` from an empty table."""
    _, _, (fp, val, used) = lookup_parallel(ids, n, slots)
    tail = _grams(ids, n)[:, -1]                                 # last n tokens (-1 padded)
    return {'fp': fp, 'val': val, 'used': used, 'tail': tail}


def lookup_step(state: dict, x: torch.Tensor, n: int, slots: int):
    """One token x [B] -> (value [B], found [B], new_state). Writes the pending
    n-gram (ending at the previous token) -> x, then reads the n-gram ending at x."""
    fp_t, val_t, used_t, tail = state['fp'].clone(), state['val'].clone(), state['used'].clone(), state['tail']
    rows = torch.arange(x.shape[0], device=x.device)
    prev_ok = (tail >= 0).all(dim=1)                             # a full n-gram ended at t-1
    s_prev = torch.remainder(_hash(tail, _SLOT_MULT), slots)
    f_prev = _hash(tail, _FP_MULT)
    fp_t[rows[prev_ok], s_prev[prev_ok]] = f_prev[prev_ok]
    val_t[rows[prev_ok], s_prev[prev_ok]] = x.long()[prev_ok]
    used_t[rows[prev_ok], s_prev[prev_ok]] = True
    tail = torch.cat([tail[:, 1:], x.long()[:, None]], dim=1)
    ok = (tail >= 0).all(dim=1)
    s = torch.remainder(_hash(tail, _SLOT_MULT), slots)
    f = _hash(tail, _FP_MULT)
    found = ok & used_t[rows, s] & (fp_t[rows, s] == f)
    value = torch.where(found, val_t[rows, s], torch.zeros_like(x.long()))
    return value, found, {'fp': fp_t, 'val': val_t, 'used': used_t, 'tail': tail}


def lookup_orders(ids: torch.Tensor, orders: tuple, slots: int, state=None,
                  want_state: bool = False):
    """All orders: -> value [k, B, T], found [k, B, T], states (list per order or
    None). ``state`` None = from an empty table (parallel); else step token by
    token from the carried tables (decode)."""
    values, founds, states = [], [], []
    for j, n in enumerate(orders):
        if state is None:
            v, f, _ = lookup_parallel(ids, n, slots, want_table=False)
            if want_state:
                states.append(state_from_prefix(ids, n, slots))
        else:
            st, vs, fs = state[j], [], []
            for t in range(ids.shape[1]):
                v_t, f_t, st = lookup_step(st, ids[:, t], n, slots)
                vs.append(v_t)
                fs.append(f_t)
            v, f = torch.stack(vs, 1), torch.stack(fs, 1)
            states.append(st)
        values.append(v)
        founds.append(f)
    return torch.stack(values), torch.stack(founds), (states if states else None)


class NGramReader(nn.Module):
    """r = sum_n found_n * sigmoid(w_n . rms(h) + b_n) * (W_n E[value_n] + f_n), W, f zero at init."""

    def __init__(self, dim: int, orders: tuple):
        super().__init__()
        self.orders = tuple(orders)
        k = len(self.orders)
        self.proj = nn.Parameter(torch.zeros(k, dim, dim))       # W_n (zero: off at init)
        self.found_bias = nn.Parameter(torch.zeros(k, dim))      # f_n
        self.gate_w = nn.Parameter(torch.zeros(k, dim))
        self.gate_b = nn.Parameter(torch.zeros(k))
        self.proj._zero_init = True
        self.found_bias._zero_init = True
        self.gate_w._zero_init = True

    def forward(self, h: torch.Tensor, embedded: torch.Tensor, found: torch.Tensor) -> torch.Tensor:
        """h [B, T, D]; embedded [k, B, T, D] = E[value_n]; found [k, B, T] -> r [B, T, D]."""
        hn = h.float() * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        gate = torch.sigmoid(torch.einsum('btd,kd->kbt', hn, self.gate_w.float())
                             + self.gate_b.float()[:, None, None])
        val = torch.einsum('kbtd,ked->kbte', embedded.float(), self.proj.float()) \
            + self.found_bias.float()[:, None, None, :]
        r = (found.float() * gate)[..., None] * val
        return r.sum(0).to(h.dtype)
