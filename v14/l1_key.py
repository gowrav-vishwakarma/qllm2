"""L1: a learned library key over event spans, on the FROZEN mind (v14/EXPERIMENTS_V14.md "L").

    key(span) = normalize( sum_i softmax_i(a . h_i) * W h_i )      h_i = the mind's pre-logit vector

One head for rows and queries. Two row forms (same head, same bars, trained separately):
  fwd : a burst starting at token f -> span [f, f + SPAN)           (amendment A3)
  evt : the event between two bursts -> span [f_prev, f), <= 128    (amendment A4)
A burst = consecutive tokens with surprisal > tau (gap <= 2 merged); the gate is the harness's own
(echo-included surprisal), tau = 10 nats (amendment A5). The query key is the head applied to the
span that starts at the question's key token.

Train:  .venv/bin/python -m v14.l1_key train --docs <dir> --val <dir> --form fwd --out <file.pt>
Use in the harness:  L1_CKPT=<file.pt> python v16/bench.py ... --key_fn v14.l1_key:key_fn
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

DIM, KEY_DIM, SPAN, EVT_MAX = 588, 128, 24, 128   # DIM = the C1-ctl mind width
TAU = 10.0


class SpanKey(nn.Module):
    def __init__(self, dim=DIM, key_dim=KEY_DIM):
        super().__init__()
        self.score = nn.Linear(dim, 1)
        self.proj = nn.Linear(dim, key_dim, bias=False)
        self.log_temp = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """h [N, S, D] span vectors, mask [N, S] bool -> unit keys [N, key_dim]."""
        a = self.score(h).squeeze(-1).masked_fill(~mask, float('-inf'))
        w = torch.softmax(a, dim=-1)
        return F.normalize((w.unsqueeze(-1) * self.proj(h)).sum(1), dim=-1)


def bursts(nll: torch.Tensor, tau: float = TAU, gap: int = 2) -> list[tuple[int, int]]:
    """[(start, end)) of runs of tokens with nll > tau, firings <= gap apart merged."""
    idx = (nll.float() > tau).nonzero().flatten().tolist()
    out = []
    for i in idx:
        if out and i - out[-1][1] <= gap:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    return [tuple(b) for b in out]


def row_spans(T: int, nll: torch.Tensor, form: str, tau: float = TAU) -> list[tuple[int, int]]:
    b = bursts(nll, tau)
    if form == 'fwd':
        return [(s, min(T, s + SPAN)) for s, _ in b]
    starts = [0] + [s for s, _ in b]
    return [(max(p, s - EVT_MAX), s) for p, s in zip(starts[:-1], starts[1:]) if s > p] if b else []


def pool_spans(head: SpanKey, vecs: torch.Tensor, spans: list[tuple[int, int]], max_len: int) -> torch.Tensor:
    """keys for spans of one document; vecs [T, D]."""
    if not spans:
        return vecs.new_zeros(0, KEY_DIM)
    n = len(spans)
    idx = torch.zeros(n, max_len, dtype=torch.long, device=vecs.device)
    mask = torch.zeros(n, max_len, dtype=torch.bool, device=vecs.device)
    for r, (s, e) in enumerate(spans):
        L = min(max_len, e - s)
        idx[r, :L] = torch.arange(s, s + L, device=vecs.device)
        mask[r, :L] = True
    return head(vecs[idx], mask)


def positives(spans, target, tlen):
    t0, t1 = target, target + tlen
    return [r for r, (s, e) in enumerate(spans) if s <= t0 < e or s < t1 <= e]


def load_docs(path):
    return sorted(glob.glob(os.path.join(path, 'doc_*.pt')))


def batch_loss(head, docs, form, device):
    """InfoNCE over every row of every document in the batch (in-doc rows are the hard negatives)."""
    all_keys, owner, row_pos, q_keys, q_doc = [], [], [], [], []
    max_len = SPAN if form == 'fwd' else EVT_MAX
    base = 0
    n_q = n_skip = 0
    for di, d in enumerate(docs):
        v = d['vecs'].to(device).float()
        spans = row_spans(v.shape[0], d['nll'], form)
        pos = positives(spans, d['target'], d['tlen'])
        if not pos:
            n_skip += 1
            continue
        k = pool_spans(head, v, spans, max_len)
        qs, qe = d['qpos'], min(v.shape[0], d['qpos'] + SPAN)
        q = pool_spans(head, v, [(qs, qe)], SPAN)
        all_keys.append(k)
        row_pos.append([base + p for p in pos])
        q_keys.append(q)
        base += k.shape[0]
        n_q += 1
    if not q_keys:
        return None, 0, n_skip
    K = torch.cat(all_keys)
    Q = torch.cat(q_keys)
    logits = head.log_temp.exp() * (Q @ K.t())
    loss = 0.0
    for i, pos in enumerate(row_pos):
        lse_all = torch.logsumexp(logits[i], -1)
        lse_pos = torch.logsumexp(logits[i, pos], -1)
        loss = loss + (lse_all - lse_pos)
    return loss / n_q, n_q, n_skip


@torch.no_grad()
def evaluate(head, files, form, device, tau_list=(TAU,)):
    """top-1 / top-5 of the right row among all rows of the document; event recall (a row holds the target)."""
    res = dict(n=0, hit1=0, hit5=0, has_row=0, rows=0, toks=0)
    for f in files:
        d = torch.load(f)
        v = d['vecs'].to(device).float()
        spans = row_spans(v.shape[0], d['nll'], form)
        res['n'] += 1
        res['rows'] += len(spans)
        res['toks'] += v.shape[0]
        pos = positives(spans, d['target'], d['tlen'])
        if not pos:
            continue
        res['has_row'] += 1
        k = pool_spans(head, v, spans, SPAN if form == 'fwd' else EVT_MAX)
        q = pool_spans(head, v, [(d['qpos'], min(v.shape[0], d['qpos'] + SPAN))], SPAN)
        top = (q @ k.t()).squeeze(0).topk(min(5, k.shape[0])).indices.tolist()
        res['hit1'] += int(top[0] in pos)
        res['hit5'] += int(any(t in pos for t in top))
    n = max(1, res['n'])
    return dict(top1=res['hit1'] / n, top5=res['hit5'] / n, event_recall=res['has_row'] / n,
                rows_per_16=16 * res['rows'] / max(1, res['toks']), n=res['n'])


def train(args):
    device = torch.device('cuda')
    if os.environ.get('L1_MEM_FRAC'):                      # hard cap when sharing a busy card
        torch.cuda.set_per_process_memory_fraction(float(os.environ['L1_MEM_FRAC']))
    torch.manual_seed(args.seed)
    files = load_docs(args.docs)
    val = load_docs(args.val)
    print(f"{len(files)} train docs, {len(val)} val docs, form {args.form}", flush=True)
    head = SpanKey().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    rng = random.Random(args.seed)
    cache = {}

    def get(f):
        if f not in cache:
            cache[f] = torch.load(f)
        return cache[f]

    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g['lr'] = args.lr * min(1.0, step / 100) * (0.5 * (1 + math.cos(math.pi * step / args.steps)))
        docs = [get(f) for f in rng.sample(files, args.batch)]
        loss, n_q, n_skip = batch_loss(head, docs, args.form, device)
        if loss is None:
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0 or step == 1:
            print(f"step {step:5d} loss {float(loss):.3f} queries {n_q} skipped {n_skip} "
                  f"temp {float(head.log_temp.exp()):.1f}", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            r = evaluate(head, val, args.form, device)
            print(f"  val: top1 {r['top1']:.3f} top5 {r['top5']:.3f} event_recall {r['event_recall']:.3f} "
                  f"rows/16 {r['rows_per_16']:.2f} (n {r['n']})", flush=True)
    torch.save({'state': head.state_dict(), 'form': args.form, 'tau': TAU, 'span': SPAN, 'args': vars(args)}, args.out)
    print('saved', args.out, flush=True)


_HEAD = {}


def key_fn(vecs: torch.Tensor, ids, bursts_=None) -> torch.Tensor:
    """Harness hook. vecs [T, D] -> keys [T, KEY_DIM] (the key of the span starting at each position).
    With ``bursts_`` (list of (start, end)) -> one key per burst span instead."""
    if 'head' not in _HEAD:
        ck = torch.load(os.environ['L1_CKPT'], map_location='cpu')
        h = SpanKey()
        h.load_state_dict(ck['state'])
        _HEAD['head'], _HEAD['form'] = h.eval(), ck['form']
    head = _HEAD['head'].to(vecs.device)
    v = vecs.float()
    T = v.shape[0]
    with torch.no_grad():
        if bursts_ is not None:
            return pool_spans(head, v, [(s, e) for s, e in bursts_], EVT_MAX)
        a = head.score(v).squeeze(-1)
        pv = head.proj(v)
        out = torch.empty(T, KEY_DIM, device=v.device)
        pad_a = F.pad(a, (0, SPAN - 1), value=float('-inf'))
        pad_p = F.pad(pv, (0, 0, 0, SPAN - 1))
        for s in range(0, T, 32768):
            e = min(T, s + 32768)
            ua = pad_a[s:e + SPAN - 1].unfold(0, SPAN, 1)             # [n, SPAN]
            up = pad_p[s:e + SPAN - 1].unfold(0, SPAN, 1)             # [n, KEY_DIM, SPAN]
            w = torch.softmax(ua, dim=-1)
            out[s:e] = F.normalize(torch.einsum('ns,nks->nk', w, up), dim=-1)
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--docs', required=True)
    t.add_argument('--val', required=True)
    t.add_argument('--form', choices=['fwd', 'evt'], default='fwd')
    t.add_argument('--out', required=True)
    t.add_argument('--steps', type=int, default=1500)
    t.add_argument('--batch', type=int, default=48)
    t.add_argument('--lr', type=float, default=2e-3)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--log_every', type=int, default=50)
    t.add_argument('--eval_every', type=int, default=250)
    args = ap.parse_args()
    if args.cmd == 'train':
        train(args)


if __name__ == '__main__':
    main()
