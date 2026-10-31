"""Text-first library packs and the install tool (v14 lane; Lead's order item 2, 2026-10-31).

A PACK is plain text with provenance and no keys, so it never goes stale across mind versions:

    line 1 : {"format": "qllm-pack/v1", "pack_id", "title", "license", "created"}
    line 2+: {"id", "text", "source", "sha256", "time", "trust"}           (one entry per line)

INSTALL reads the pack with the user's mind once and writes a local LIBRARY (a cache, rebuilt per mind
version):
  rows   : one per surprise burst (the mind's own output surprisal > tau, firings <= 2 apart merged), kept
           inside its entry, at least one per entry; span [start, start + 24)
  keys   : the L1 head on the mind's block-8 state, centred by the reference mean that ships with the head
           (a session query and the library come from different texts, so per-document centring would not
           line up), fp16
  organ  : sorted 64-bit hashes of every 4-gram of the library text + the position after it (exact lookup)
  tokens : the library text as GPT-2 token ids (u16), for re-reading a fetched span
  meta   : mind checkpoint, head, key version, tau, layer, counts, MB and install seconds per 1M tokens

    python -m v14.pack install --pack <pack.jsonl> --out <dir>      (mind / head default to the L1 ones)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from v14 import l1_key as L  # noqa: E402

MIND = '/media/gowrav/d_drive/qllm2/checkpoints_v13_sempty/c1ctlb_main_7ae6f1f/best_model.pt'
HEAD = '/media/gowrav/d_drive/qllm_ckpt/v14/l1p_fwd_L8c_99e6803.pt'
MEAN = '/media/gowrav/d_drive/qllm_ckpt/v14/l1p_fwd_L8c_99e6803.mean.pt'
LAYER, TAU, NGRAM, CHUNK = 8, 10.0, 4, 4096
_MASK = (1 << 61) - 1


def write_pack(path, header, entries):
    with open(path, 'w') as f:
        f.write(json.dumps({'format': 'qllm-pack/v1', **header}) + '\n')
        for e in entries:
            e = dict(e)
            e.setdefault('sha256', hashlib.sha256(e['text'].encode()).hexdigest())
            f.write(json.dumps(e) + '\n')


def read_pack(path):
    lines = [json.loads(x) for x in open(path) if x.strip()]
    assert lines[0].get('format') == 'qllm-pack/v1', 'not a qllm-pack/v1 file'
    return lines[0], lines[1:]


def gram_hash(g):
    h = 1469598103934665603
    for t in g:
        h = ((h ^ int(t)) * 1099511628211) & _MASK
    return h


class Mind:
    """The frozen mind: block-k state and output surprisal for a token window."""

    def __init__(self, ckpt=MIND, dev='cuda'):
        from run_memory_behavioral import _load_v13_sempty
        self.dev = torch.device(dev)
        self.m, self.tok, _ = _load_v13_sempty(ckpt, 'baseline_real_pm', self.dev)
        self.m.eval()
        self.E = self.m.embed.embed.weight.float()

    @torch.no_grad()
    def read(self, ids, layer=LAYER):
        m = self.m
        x = torch.tensor([ids], device=self.dev)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.dev.type == 'cuda'):
            z, b, t = m._stem(x, 0)
            shared, mid = None, None
            for i, blk in enumerate(m.blocks):
                if m.config.share_reads:
                    z, _ = blk(z, pam_state=None, step_offset=0, shared=shared)
                    if blk.pam is not None:
                        shared = blk.pam._shared
                else:
                    z, _ = blk(z, pam_state=None, step_offset=0)
                if i + 1 == layer:
                    mid = z.raw(b, t, m.model_dim)[0].float()
            lm, _ = m._lm_vector(z, x, b, t)
            v = lm.raw(b, t, m.model_dim)[0].float()
        nll = torch.zeros(len(ids), device=self.dev)
        for s in range(0, len(ids) - 1, 1024):
            lg = v[s:min(s + 1024, len(ids) - 1)] @ self.E.t()
            nll[s + 1:s + 1 + lg.shape[0]] = torch.logsumexp(lg, -1) - lg.gather(-1, x[0, s + 1:s + 1 + lg.shape[0], None]).squeeze(-1)
        return mid, nll


class KeyHead:
    def __init__(self, head=HEAD, mean=MEAN, dev='cuda'):
        ck = torch.load(head, map_location='cpu')
        self.h = L.SpanKey(key_dim=ck.get('key_dim', L.KEY_DIM), arch=ck.get('arch', 'full'), rank=ck.get('rank', 32))
        self.h.load_state_dict(ck['state'])
        self.h.eval().to(dev)
        self.mu = torch.load(mean, map_location='cpu')['mean'].float().to(dev)
        self.version = f"{Path(head).stem}+{Path(mean).stem}"

    @torch.no_grad()
    def keys(self, state, spans):
        return L.pool_spans(self.h, state - self.mu, spans, L.SPAN)


def install(pack_path, out_dir, mind=None, head=None, tau=TAU, log=print):
    t0 = time.time()
    hdr, entries = read_pack(pack_path)
    mind = mind or Mind()
    head = head or KeyHead()
    tok = mind.tok
    sep = tok.encode('\n\n', add_special_tokens=False)
    toks, offs = [], []                 # library text and each entry's [start, end) in it
    for e in entries:
        ids = sep + tok.encode(e['text'], add_special_tokens=False)
        offs.append((len(toks), len(toks) + len(ids)))
        toks += ids
    # chunks of whole entries (an entry longer than CHUNK gets its own chunks)
    chunks, cur = [], []
    for i, (s, e) in enumerate(offs):
        if cur and offs[cur[-1]][1] - offs[cur[0]][0] + (e - s) > CHUNK:
            chunks.append(cur)
            cur = []
        cur.append(i)
    if cur:
        chunks.append(cur)
    keys, rows = [], []
    for ch in chunks:
        c0, c1 = offs[ch[0]][0], offs[ch[-1]][1]
        for w0 in range(c0, c1, CHUNK):
            w1 = min(c1, w0 + CHUNK)
            state, nll = mind.read(toks[w0:w1])
            bursts = L.bursts(nll.cpu(), tau)
            for i in ch:
                s, e = offs[i]
                if e <= w0 or s >= w1:
                    continue
                ls, le = max(s, w0) - w0, min(e, w1) - w0
                spans = [(b0, min(le, b0 + L.SPAN)) for b0, _ in bursts if ls <= b0 < le]
                if not spans:                                     # every entry stays findable
                    spans = [(ls, min(le, ls + L.SPAN))]
                k = head.keys(state, spans)
                keys.append(k.half().cpu())
                rows += [(i, w0 + a, b - a) for a, b in spans]
    K = torch.cat(keys)
    R = torch.tensor(rows, dtype=torch.int32)
    T = np.asarray(toks, dtype=np.uint16)
    hs = np.array([gram_hash(T[j:j + NGRAM]) for j in range(len(T) - NGRAM)], dtype=np.int64)
    order = np.argsort(hs, kind='stable')
    organ_h, organ_p = hs[order], (order + NGRAM).astype(np.int32)
    secs = time.time() - t0
    os.makedirs(out_dir, exist_ok=True)
    torch.save({'keys': K, 'rows': R, 'tokens': torch.from_numpy(T.astype(np.int32)),
                'offsets': torch.tensor(offs, dtype=torch.int32),
                'organ_hash': torch.from_numpy(organ_h), 'organ_pos': torch.from_numpy(organ_p)},
               os.path.join(out_dir, 'library.pt'))
    n = len(toks)
    mb = dict(keys=K.numel() * 2, row_pointers=R.numel() * 4, text=2 * n, organ=hs.size * 12)
    per_m = {k: round(v * 1e6 / max(1, n) / 1e6, 2) for k, v in mb.items()}
    meta = dict(pack=hdr, entries=len(entries), tokens=n, rows=int(R.shape[0]),
                rows_per_16_tokens=round(16 * R.shape[0] / max(1, n), 3), key_version=head.version,
                mind=str(getattr(mind, 'ckpt', MIND)), layer=LAYER, tau=tau,
                mb_per_1M_tokens=per_m, mb_per_1M_total=round(sum(per_m.values()), 2),
                install_seconds=round(secs, 1), install_seconds_per_1M_tokens=round(secs * 1e6 / max(1, n), 1))
    json.dump(meta, open(os.path.join(out_dir, 'meta.json'), 'w'), indent=1)
    log(json.dumps({k: meta[k] for k in ('entries', 'tokens', 'rows', 'rows_per_16_tokens', 'mb_per_1M_tokens',
                                         'mb_per_1M_total', 'install_seconds', 'install_seconds_per_1M_tokens')}))
    return meta


class Library:
    def __init__(self, out_dir, dev='cuda'):
        d = torch.load(os.path.join(out_dir, 'library.pt'))
        self.meta = json.load(open(os.path.join(out_dir, 'meta.json')))
        self.K = torch.nn.functional.normalize(d['keys'].float().to(dev), dim=-1)
        self.R, self.T, self.offs = d['rows'], d['tokens'], d['offsets']
        self.oh, self.op = d['organ_hash'].numpy(), d['organ_pos'].numpy()

    def by_key(self, qkey, k=5):
        sim = self.K @ torch.nn.functional.normalize(qkey.float(), dim=-1)
        top = sim.topk(min(k, sim.numel()))
        return [(int(self.R[i, 0]), int(self.R[i, 1]), int(self.R[i, 2]), float(s))
                for s, i in zip(top.values.tolist(), top.indices.tolist())]

    def by_organ(self, last_ids):
        """exact match of the last NGRAM tokens -> (entry, position after the match) of the LATEST copy."""
        h = gram_hash(last_ids[-NGRAM:])
        lo, hi = np.searchsorted(self.oh, h, 'left'), np.searchsorted(self.oh, h, 'right')
        if lo == hi:
            return None
        p = int(self.op[lo:hi].max())
        e = int(np.searchsorted(self.offs[:, 0].numpy(), p, 'right') - 1)
        return e, p


def query_key(mind, head, ids, qpos):
    """the session's key for a question: the head on the span starting at the query's key token."""
    state, _ = mind.read(ids)
    return head.keys(state, [(qpos, min(len(ids), qpos + L.SPAN))])[0]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    i = sub.add_parser('install')
    i.add_argument('--pack', required=True)
    i.add_argument('--out', required=True)
    i.add_argument('--tau', type=float, default=TAU)
    a = ap.parse_args()
    if a.cmd == 'install':
        install(a.pack, a.out, tau=a.tau)


if __name__ == '__main__':
    main()
