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
    """arch 'full': learned projection dim -> key_dim (default; overfits on small data).
    'w': no projection, only learned token weights over the span (589 numbers).
    'lr': identity plus a rank-r residual (small capacity)."""

    def __init__(self, dim=DIM, key_dim=KEY_DIM, arch='full', rank=32, drop=0.0):
        super().__init__()
        self.arch, self.drop = arch, drop
        self.score = nn.Linear(dim, 1)
        if arch == 'full':
            self.proj = nn.Linear(dim, key_dim, bias=False)
        elif arch == 'lr':
            self.down = nn.Linear(dim, rank, bias=False)
            self.up = nn.Linear(rank, dim, bias=False)
            nn.init.zeros_(self.up.weight)
        self.log_temp = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def project(self, h):
        if self.arch == 'full':
            return self.proj(h)
        h = F.normalize(h, dim=-1)
        return h if self.arch == 'w' else h + self.up(self.down(h))

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """h [N, S, D] span vectors, mask [N, S] bool -> unit keys."""
        if self.training and self.drop > 0:
            h = F.dropout(h, self.drop)
        a = self.score(h).squeeze(-1).masked_fill(~mask, float('-inf'))
        w = torch.softmax(a, dim=-1)
        return F.normalize((w.unsqueeze(-1) * self.project(h)).sum(1), dim=-1)


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


def span_index(spans, max_len):
    """(idx [n, max_len] long, mask [n, max_len] bool), CPU."""
    n = len(spans)
    idx = torch.zeros(n, max_len, dtype=torch.long)
    mask = torch.zeros(n, max_len, dtype=torch.bool)
    for r, (s, e) in enumerate(spans):
        L = min(max_len, e - s)
        idx[r, :L] = torch.arange(s, s + L)
        mask[r, :L] = True
    return idx, mask


def pool_spans(head: SpanKey, vecs: torch.Tensor, spans: list[tuple[int, int]], max_len: int) -> torch.Tensor:
    """keys for spans of one document; vecs [T, D]."""
    if not spans:
        return vecs.new_zeros(0, KEY_DIM)
    idx, mask = span_index(spans, max_len)
    return head(vecs[idx.to(vecs.device)], mask.to(vecs.device))


def anchor(d):
    """(query position, target position) at the LAST token of the key: there the mind has read the whole key.
    (v14/l1_data.py stored the first token and the key length; the first token alone is a word fragment.)"""
    k = max(1, int(d['tlen'])) - 1
    return d['qpos'] + k, d['target'] + k


def prep(d, form):
    """Per-document cache: row spans, their index tensors, positives, the query span's index."""
    T = d['vecs'].shape[0]
    qpos, target = anchor(d)
    spans = row_spans(T, d['nll'], form)
    pos = positives(spans, target, 1)
    ml = SPAN if form == 'fwd' else EVT_MAX
    idx, mask = span_index(spans, ml) if spans else (None, None)
    qi, qm = span_index([(qpos, min(T, qpos + SPAN))], SPAN)
    return dict(vecs=d['vecs'], spans=spans, pos=pos, idx=idx, mask=mask, qi=qi, qm=qm, kind=d['kind'])


def positives(spans, target, tlen):
    t0, t1 = target, target + tlen
    return [r for r, (s, e) in enumerate(spans) if s <= t0 < e or s < t1 <= e]


def load_docs(path):
    return sorted(glob.glob(os.path.join(path, 'doc_*.pt')))


def batch_loss(head, docs, device):
    """InfoNCE over every row of every (prepped) document in the batch; in-doc rows are the hard negatives."""
    all_keys, row_pos, q_keys = [], [], []
    base = n_skip = 0
    for p in docs:
        if not p['pos']:
            n_skip += 1
            continue
        v = p['vecs'].to(device).float()
        k = head(v[p['idx'].to(device)], p['mask'].to(device))
        q = head(v[p['qi'].to(device)], p['qm'].to(device))
        all_keys.append(k)
        row_pos.append([base + x for x in p['pos']])
        q_keys.append(q)
        base += k.shape[0]
    if not q_keys:
        return None, 0, n_skip
    K, Q = torch.cat(all_keys), torch.cat(q_keys)
    logits = head.log_temp.exp() * (Q @ K.t())
    loss = 0.0
    for i, pos in enumerate(row_pos):
        loss = loss + torch.logsumexp(logits[i], -1) - torch.logsumexp(logits[i, pos], -1)
    return loss / len(q_keys), len(q_keys), n_skip


@torch.no_grad()
def evaluate(head, preps, device):
    """top-1 / top-5 of the right row among all rows of the document; event recall (a row holds the target)."""
    head.eval()
    n = hit1 = hit5 = has = rows = toks = 0
    for p in preps:
        v = p['vecs'].to(device).float()
        n += 1
        rows += len(p['spans'])
        toks += v.shape[0]
        if not p['pos']:
            continue
        has += 1
        k = head(v[p['idx'].to(device)], p['mask'].to(device))
        q = head(v[p['qi'].to(device)], p['qm'].to(device))
        top = (q @ k.t()).squeeze(0).topk(min(5, k.shape[0])).indices.tolist()
        hit1 += int(top[0] in p['pos'])
        hit5 += int(any(t in p['pos'] for t in top))
    head.train()
    n = max(1, n)
    return dict(top1=hit1 / n, top5=hit5 / n, event_recall=has / n, rows_per_16=16 * rows / max(1, toks), n=n)


def train(args):
    device = torch.device('cuda')
    if os.environ.get('L1_MEM_FRAC'):                      # hard cap when sharing a busy card
        torch.cuda.set_per_process_memory_fraction(float(os.environ['L1_MEM_FRAC']))
    torch.manual_seed(args.seed)
    files = [f for d in args.docs.split(',') for f in load_docs(d)]
    vals = {os.path.basename(d.rstrip('/')): [prep(torch.load(f), args.form) for f in load_docs(d)]
            for d in args.val.split(',')}
    print(f"{len(files)} train docs, val sets {[(k, len(v)) for k, v in vals.items()]}, form {args.form}, "
          f"arch {args.arch}", flush=True)
    head = SpanKey(arch=args.arch, rank=args.rank, drop=args.drop).to(device)
    print(f"head parameters: {sum(p.numel() for p in head.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    rng = random.Random(args.seed)
    cache = {}
    best = (-1.0, None)
    key_set = args.select
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g['lr'] = args.lr * min(1.0, step / 100) * (0.5 * (1 + math.cos(math.pi * step / args.steps)))
        batch = []
        for f in rng.sample(files, args.batch):
            if f not in cache:
                if len(cache) > args.cache_docs:
                    cache.pop(next(iter(cache)))
                cache[f] = prep(torch.load(f), args.form)
            batch.append(cache[f])
        loss, n_q, n_skip = batch_loss(head, batch, device)
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
            line, score = [], 0.0
            for name, ps in vals.items():
                r = evaluate(head, ps, device)
                line.append(f"{name}: top1 {r['top1']:.3f} top5 {r['top5']:.3f} ev {r['event_recall']:.2f}")
                if name == key_set:
                    score = r['top1']
            print("  val | " + " | ".join(line), flush=True)
            if score > best[0]:
                best = (score, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
                print(f"  best so far on {key_set}: {score:.3f} (step {step})", flush=True)
    torch.save({'state': best[1], 'form': args.form, 'tau': TAU, 'span': SPAN, 'arch': args.arch,
                'rank': args.rank, 'select': key_set, 'best_top1': best[0], 'args': vars(args)}, args.out)
    print('saved', args.out, 'best', best[0], flush=True)


_HEAD = {}


def key_fn(vecs: torch.Tensor, ids, bursts_=None) -> torch.Tensor:
    """Harness hook. vecs [T, D] -> keys [T, KEY_DIM] (the key of the span starting at each position).
    With ``bursts_`` (list of (start, end)) -> one key per burst span instead."""
    if 'head' not in _HEAD:
        ck = torch.load(os.environ['L1_CKPT'], map_location='cpu')
        h = SpanKey(arch=ck.get('arch', 'full'), rank=ck.get('rank', 32))
        h.load_state_dict(ck['state'])
        _HEAD['head'], _HEAD['form'] = h.eval(), ck['form']
    head = _HEAD['head'].to(vecs.device)
    v = vecs.float()
    T = v.shape[0]
    with torch.no_grad():
        if bursts_ is not None:
            return pool_spans(head, v, [(s, e) for s, e in bursts_], EVT_MAX)
        a = head.score(v).squeeze(-1)
        pv = head.project(v)
        out = torch.empty(T, pv.shape[-1], device=v.device)
        pad_a = F.pad(a, (0, SPAN - 1), value=float('-inf'))
        pad_p = F.pad(pv, (0, 0, 0, SPAN - 1))
        for s in range(0, T, 32768):
            e = min(T, s + 32768)
            ua = pad_a[s:e + SPAN - 1].unfold(0, SPAN, 1)             # [n, SPAN]
            up = pad_p[s:e + SPAN - 1].unfold(0, SPAN, 1)             # [n, key dim, SPAN]
            w = torch.softmax(ua, dim=-1)
            out[s:e] = F.normalize(torch.einsum('ns,nks->nk', w, up), dim=-1)
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--docs', required=True, help='comma-separated dirs')
    t.add_argument('--val', required=True, help='comma-separated dirs; the first is NOT special, --select names the one that picks the step')
    t.add_argument('--form', choices=['fwd', 'evt'], default='fwd')
    t.add_argument('--out', required=True)
    t.add_argument('--steps', type=int, default=1500)
    t.add_argument('--batch', type=int, default=48)
    t.add_argument('--lr', type=float, default=2e-3)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--arch', choices=['full', 'w', 'lr'], default='full')
    t.add_argument('--rank', type=int, default=32)
    t.add_argument('--drop', type=float, default=0.0)
    t.add_argument('--wd', type=float, default=0.01)
    t.add_argument('--select', default='val_indist', help='validation set (dir name) that picks the best step')
    t.add_argument('--cache_docs', type=int, default=400)
    t.add_argument('--log_every', type=int, default=50)
    t.add_argument('--eval_every', type=int, default=250)
    args = ap.parse_args()
    if args.cmd == 'train':
        train(args)


if __name__ == '__main__':
    main()
