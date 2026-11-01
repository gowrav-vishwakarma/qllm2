"""The PACKAGE fetch as one 3-arg key_fn for v16/bench.py (Lead's request, 2026-10-31 21:40 IST).

    python v16/bench.py ... --vec_layer 8 --centre --centre_ref <mean_raw.pt> --key_fn v14.package_fetch:key_fn

Two-organ rule (L2, pre-registered 21:12 IST):
  1. EXACT ORGAN: if the query's last NGRAM tokens occur in the document (before doc_end), the sentence around
     the latest match is added as a row whose key IS the query key, so it ranks first (similarity 1).
  2. else the LEARNED KEY: the key head (PACKAGE_HEAD; the L1 head until L3 lands) on the mind's block-8 state,
     centred by the shipped reference mean (PACKAGE_MEAN), one row per surprise burst.
  3. ABSTAIN: when the organ misses and the best learned-key similarity is below PACKAGE_T, info['abstain'] = 1;
     with PACKAGE_STRICT=1, abstain whenever the organ misses (strict mode, a rule, not a threshold).
Rows are FIELD-AWARE: from the burst start to the end of its sentence (the next '.'), at most 64 tokens.
Returns (row keys [n, d], query key [d], row spans [(start, end), ...]); info gets 'organ_hit', 'abstain',
'best_sim' for the harness to record.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from v14 import l1_key as L
from v14 import norm_organ as N

HEAD = os.environ.get('PACKAGE_HEAD', '/media/gowrav/d_drive/qllm_ckpt/v14/l1p_fwd_L8c_99e6803.pt')
MEAN = os.environ.get('PACKAGE_MEAN', '/media/gowrav/d_drive/qllm_ckpt/v14/l1p_fwd_L8c_99e6803.mean.pt')
T_ABSTAIN = float(os.environ.get('PACKAGE_T', '0.655151'))
STRICT = os.environ.get('PACKAGE_STRICT', '0') == '1'      # strict mode: answer only on an exact-organ hit
NORM = os.environ.get('PACKAGE_NORM', '0') == '1'          # A4: the normalised-word organ after the token organ
NGRAM, MAX_ROW, DOT = 4, 64, 13
_S = {}


def _load(dev):
    if 'head' not in _S:
        ck = torch.load(HEAD, map_location='cpu')
        h = L.SpanKey(key_dim=ck.get('key_dim', L.KEY_DIM), arch=ck.get('arch', 'full'), rank=ck.get('rank', 32))
        h.load_state_dict(ck['state'])
        m = torch.load(MEAN, map_location='cpu')
        _S['head'] = h.eval()
        _S['mean'] = (m['mean'] if isinstance(m, dict) else m).float()
    return _S['head'].to(dev), _S['mean'].to(dev)


def sentence_end(ids, start, end):
    """index just after the first '.' at or after start (exclusive bound), capped at start + MAX_ROW and end."""
    for j in range(start, min(end, start + MAX_ROW)):
        if ids[j] == DOT:
            return j + 1
    return min(end, start + MAX_ROW)


def _strs():
    if 'strs' not in _S:
        from v7.data import get_chat_tokenizer
        _S['strs'] = N.tok_strings(get_chat_tokenizer())
    return _S['strs']


def row_around(ids, a, b, end):
    """the sentence holding tokens [a, b): from after the previous '.' (<= 64 back) to the next '.' at or after b."""
    s0 = max(0, a - 16)
    for j in range(a - 1, max(0, a - 64), -1):
        if ids[j] == DOT:
            s0 = j + 1
            break
    return s0, max(a + 1, sentence_end(ids, b, end))


def norm_hit(ids, end, q0):
    """A4: (step, n, row start, row end) of the normalised-word organ, or None."""
    strs = _strs()
    h = N.find(N.words(strs, ids, 0, end), N.words(strs, ids, q0, len(ids)))
    if h is None:
        return None
    step, n, a, b = h
    return (step, n) + row_around(ids, a, b, end)


def text_fetch(ids, doc_end):
    """The TEXT-ONLY library, body-agnostic (no mind keys; Lead's "every body gets the same library" arm).

    Token organ (the question's last NGRAM tokens, latest exact match), then the normalised-word organ on the
    WHOLE question (A4); strict abstain = both miss. ids = document + question, doc_end = where the question
    starts. Returns (row or None, info): row = (start, end) token span of the fetched sentence; info has
    organ_hit, step (0 = token organ, 1-3 = normalised organ step) and abstain. Step 3 (a bigram holding a word
    seen <= 2 times) assumes a real stream: in a very short document a frame bigram ("refers to") can be rare.
    Measured never-stored false hits at 32k: 0.00 / 0.025 (A4)."""
    end = int(doc_end)
    hit = organ(ids, end, len(ids))
    if hit is not None:
        return row_around(ids, hit, hit, end), dict(organ_hit=1, step=0, abstain=0)
    nh = norm_hit(ids, end, end)
    if nh:
        return tuple(nh[2:]), dict(organ_hit=1, step=int(nh[0]), abstain=0)
    return None, dict(organ_hit=0, step=None, abstain=1)


_TOOL_ROW = None


def field_cut(row_text, q_text):
    """A19 (Lead): cut a fetched TOOL row down to the asked field. row_text holds "Result: job: K; f1: v1; ...;
    fn: vn." (possibly after prose); the asked field is the row field name that appears as a word in the question
    (the last one if several). Returns " Result: job: K; f: v." (re-rendered from the row's own text) or None when
    the row is not a tool row or no field name matches. Text only; a field SYNONYM in the question does not match."""
    import re
    global _TOOL_ROW
    if _TOOL_ROW is None:
        _TOOL_ROW = re.compile(r'Result: job: (.+?); (.+?)\.')
    m = _TOOL_ROW.search(row_text)
    if not m:
        return None
    fields = [f.split(': ', 1) for f in m.group(2).split('; ') if ': ' in f]
    qwords = re.findall(r"[A-Za-z]+", q_text.lower())
    hits = [(qwords.index(f.lower()) if f.lower() in qwords else -1, f, v) for f, v in fields]
    hits = [h for h in hits if h[0] >= 0]
    if not hits:
        return None
    _, f, v = max(hits)
    return f" Result: job: {m.group(1)}; {f}: {v}."


def organ(ids, doc_end, qend):
    """position (the token after the match) of the LATEST exact match of the query's last NGRAM tokens."""
    g = tuple(ids[qend - NGRAM:qend])
    hit = None
    for i in range(NGRAM, doc_end):
        if tuple(ids[i - NGRAM:i]) == g:
            hit = i
    return hit


def key_fn(keys: torch.Tensor, ids, info):
    head, mu = _load(keys.device)
    v = keys.float() - mu
    T, end, q0 = v.shape[0], int(info['doc_end']), int(info['query_pos'])
    spans = []
    for b0, _ in info['bursts']:
        b0 = int(b0)
        if b0 < end:
            spans.append((b0, max(b0 + 1, sentence_end(ids, b0, end))))
    with torch.no_grad():
        kb = L.pool_spans(head, v, spans, MAX_ROW) if spans else v.new_zeros(0, L.KEY_DIM)
        kq = L.pool_spans(head, v, [(q0, min(T, q0 + L.SPAN))], L.SPAN)[0]
    hit = organ(ids, end, len(ids))
    row = row_around(ids, hit, hit, end) if hit is not None else None
    if row is None and NORM:                  # A4: the normalised-word organ
        nh = norm_hit(ids, end, end)       # the whole question (info query_pos = the key's last token)
        info['norm_step'] = nh[0] if nh else 0
        row = nh[2:] if nh else None
    info['organ_hit'] = int(row is not None)
    if row is not None:                       # the organ's sentence, ranked first by construction
        spans = [row] + spans
        kb = torch.cat([kq[None], kb])
        info['best_sim'] = 1.0
        info['abstain'] = 0
    else:
        best = float((F.normalize(kb, dim=-1) @ F.normalize(kq, dim=-1)).max()) if kb.numel() else -1.0
        info['best_sim'] = round(best, 4)
        info['abstain'] = 1 if STRICT else int(best < T_ABSTAIN)
    return kb, kq, spans
