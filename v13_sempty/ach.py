"""ACh-1 (cu2b): a trained answer mode. After a tag, the plain logits are distilled toward the model's
own evidence contrast, so the query-time twin (cd2 / A9) is not needed and the gate exists by
construction (the tag never appears in free text; cd3).

Side batch per step, built from the cu2 / cu1 generators (TRAINING splits only):
  short items  rows + question + TAG + answer        (evidence = the rows; twin = question + TAG + answer)
               kinds: fld right-row questions, fsq-style definitions
  long items   skill pack + filler + query + TAG + answer (evidence = the pack; twin = filler + query + ...)
               the filler is a slice of the CURRENT training window (training-split web text)
Only answerable items (the evidence holds the answer). Negatives keep plain CE in the main stream (vfy),
because a contrast with a WRONG row would amplify copying it.

Loss on the answer tokens (teacher-forced, positions aligned from the end):
  student = log_softmax(l_ev)                       (grad)
  teacher = softmax(l_ev.detach() + a (l_ev.detach() - l_noev))   (l_noev: no-grad twin)
  aux = KL(teacher || student) on the answer positions where the teacher's top token is the gold token
        (summed, divided by all answer positions); total += lambda * aux
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

import v7.data as D

TAG = ' [answer]'


def _short_item(rng):
    """(evidence text, question text, answer text) for one answerable short item."""
    nat = rng.random() < 0.6
    if rng.random() < 0.7:                       # fld: one field of a key's row among 1-3 rows
        keys = []
        while len(keys) < rng.randint(1, 3):
            k = D._fld_key(rng, nat)
            if k not in keys:
                keys.append(k)
        rows = [D._fld_row(rng, k, rng.randint(2, 4)) for k in keys]
        i = rng.randrange(len(keys)); f, v = rng.choice(rows[i][1])
        return ''.join(r for r, _ in rows), rng.choice(D._FLD_QS).format(f=f, k=keys[i]), f' {v}.'
    keys = []                                    # fsq-style definitions
    while len(keys) < rng.randint(2, 5):
        k = D._fsq_key(rng)
        if k not in keys:
            keys.append(k)
    vals = [rng.choice(D._FSQ_VALUES) for _ in keys]
    i = rng.randrange(len(keys))
    return ''.join(f' Definition: {k} refers to {v}.' for k, v in zip(keys, vals)), f' Query: {keys[i]} refers to', f' {vals[i]}'


def _long_item(rng, tok, filler_ids, max_fill):
    """B2 base task (trained values) as a skill pack read once, a slice of training text, the query."""
    from v14.rules_ood import build_example
    from memory_probes.spread import split_units
    ex = build_example(tok, rng.choice(('shift_wide', 'affine_map')), 8, 20_000_000 + rng.randrange(10 ** 6))
    header, units = split_units(ex)
    n = rng.randint(max_fill // 4, max_fill)
    st = rng.randrange(0, max(1, len(filler_ids) - n))
    fill = filler_ids[st:st + n]
    return header + ''.join(units[:-1]), fill, units[-1], ex.target_text


class AChBatcher:
    def __init__(self, tok, seed: int, n_short: int = 16, n_long: int = 1, max_fill: int = 2048):
        self.tok, self.rng = tok, random.Random(seed * 7919 + 4242)
        self.n_short, self.n_long, self.max_fill = n_short, n_long, max_fill
        self.tag = tok.encode(TAG, add_special_tokens=False)

    def build(self, window_ids):
        """List of (ev_ids, noev_ids, n_answer). Answer tokens are the last n_answer tokens of both."""
        enc = lambda s: self.tok.encode(s, add_special_tokens=False)  # noqa: E731
        items = []
        for _ in range(self.n_short):
            evd, q, a = _short_item(self.rng)
            ans = enc(a); qa = enc(q) + self.tag + ans
            items.append((enc(evd) + qa, qa, len(ans)))
        filler = window_ids.tolist() if hasattr(window_ids, 'tolist') else list(window_ids)
        for _ in range(self.n_long):
            pack, fill, query, a = _long_item(self.rng, self.tok, filler, self.max_fill)
            ans = enc(' ' + a.strip()); tail = list(fill) + enc(query) + self.tag + ans
            items.append((enc(pack) + tail, tail, len(ans)))
        return items


def _answer_logits(model, seqs, n_ans, device, pad_to=None):
    """Logits at the positions that predict the last n_ans[i] tokens of each (right-padded) sequence.
    Same path as scripts/run_skillpack_probe.Body.last_logits (checked equal to model(x)). pad_to fixes the
    shape (torch.compile recompiles on every new shape otherwise)."""
    if pad_to is not None:
        seqs = [s[-pad_to:] for s in seqs]            # keep the end (question + tag + answer) if ever longer
    L = pad_to or max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), 50256, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s, device=device)
    z, b, t = model._stem(x, 0)
    z, _ = model._run_blocks(z, b, t, None, 0, x)
    lm, _ = model._lm_vector(z, x, b, t)
    H = lm.raw(b, t, model.model_dim)
    rows = [H[i, torch.arange(len(s) - 1 - n, len(s) - 1, device=device)] for i, (s, n) in enumerate(zip(seqs, n_ans))]
    tgt = torch.cat([torch.tensor(s[-n:], device=device) for s, n in zip(seqs, n_ans)])
    return torch.cat(rows, 0) @ model.embed.embed.weight.T, tgt


SHORT_LEN = 192


def ach_loss(model, items, alpha: float, device, long_len: int = 2048 + 256):
    # short and long items run as separate forwards at FIXED lengths (no per-step recompiles), so short
    # ones are not padded to a long item's length
    groups = [([it for it in items if len(it[0]) <= SHORT_LEN], SHORT_LEN),
              ([it for it in items if len(it[0]) > SHORT_LEN], long_len)]
    l_ev, tgt, l_no = [], [], []
    for g, plen in groups:
        if not g:
            continue
        le, tg = _answer_logits(model, [a for a, _, _ in g], [c for _, _, c in g], device, pad_to=plen)
        with torch.no_grad():
            ln, _ = _answer_logits(model, [b for _, b, _ in g], [c for _, _, c in g], device, pad_to=plen)
        l_ev.append(le); tgt.append(tg); l_no.append(ln)
    l_ev, tgt, l_no = torch.cat(l_ev), torch.cat(tgt), torch.cat(l_no)
    with torch.no_grad():
        teacher = F.softmax(l_ev.detach().float() + alpha * (l_ev.detach().float() - l_no.float()), dim=-1)
    student = F.log_softmax(l_ev.float(), dim=-1)
    # trust the teacher only where it is RIGHT (its top token is the gold answer token): a contrast of a weak,
    # noisy evidence effect amplifies noise (smoke: KL 2.88 -> 0.20 while tagged accuracy fell to 0.00)
    ok = (teacher.argmax(-1) == tgt)
    if ok.any():
        kl = (F.kl_div(student[ok], teacher[ok], reduction='none').sum(-1)).sum() / max(1, len(tgt))
    else:
        kl = student.sum() * 0.0
    acc = (l_ev.argmax(-1) == tgt).float().mean()
    return kl, float(acc), float(ok.float().mean())
