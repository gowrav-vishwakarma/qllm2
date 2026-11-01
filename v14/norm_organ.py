"""A4: the normalised-word organ inside the session (pre-registered 2026-11-01 01:04 IST; PLUMBING, no claim).

SL2's normaliser (shared_fetch.NormOrgan) applied to the session stream, used only after the token organ misses:
words = runs of letters / digits in the decoded tokens, lower-cased, "a / an / the" dropped, one trailing "s"
stripped when the word has >= 4 letters. Then, over the stream words before doc_end:
  (1) the longest query SUFFIX word n-gram, n = 6..3;
  (2) else the longest ANY-position query word n-gram, n = 6..3;
  (3) else any query word bigram holding a word seen <= RARE times in the stream.
  (4) A20, only with `bg` (background counts) and tried BEFORE (3): the question word seen 1-2 times in the stream
      and <= bg_max times in the background (key-like: a name, a made-up word; never a frame word).
The latest stream occurrence wins. find() returns (step, n, first token, last token + 1) of the match, or None.
"""
from __future__ import annotations

import bisect
import re
from collections import Counter

ART = {'a', 'an', 'the'}
WORD = re.compile(r'[A-Za-z0-9]+')
NMIN, NMAX, RARE = 3, 6, 2
_STR: dict = {}


def tok_strings(tok):
    """decoded text of every token id (cached per tokenizer)."""
    k = id(tok)
    if k not in _STR:
        _STR[k] = [tok.decode([i]) for i in range(len(tok))]
    return _STR[k]


def norm(w: str):
    w = w.lower()
    if w in ART:
        return None
    if len(w) >= 4 and w.endswith('s'):
        w = w[:-1]
    return w


def words(strs, ids, lo, hi):
    """[(normalised word, first token, last token)] for tokens [lo, hi)."""
    offs, parts, n = [], [], 0
    for j in range(lo, hi):
        s = strs[int(ids[j])]
        offs.append(n); parts.append(s); n += len(s)
    out = []
    for m in WORD.finditer(''.join(parts)):
        w = norm(m.group())
        if w is not None:
            out.append((w, lo + bisect.bisect_right(offs, m.start()) - 1, lo + bisect.bisect_right(offs, m.end() - 1) - 1))
    return out


def _latest(sw, grams, n):
    """latest stream start index of any n-gram in `grams` -> (start, gram) or None."""
    best = None
    for i in range(len(sw) - n + 1):
        g = tuple(sw[i:i + n])
        if g in grams:
            best = (i, g)
    return best


def find(stream, query, bg=None, bg_max=3):
    sw = [w for w, _, _ in stream]
    qw = [w for w, _, _ in query]
    hit = None
    for n in range(NMAX, NMIN - 1, -1):                          # (1) suffix
        if len(qw) >= n:
            r = _latest(sw, {tuple(qw[-n:])}, n)
            if r:
                hit = (1, n, r[0]); break
    if hit is None:
        for n in range(NMAX, NMIN - 1, -1):                      # (2) any position
            r = _latest(sw, {tuple(qw[i:i + n]) for i in range(len(qw) - n + 1)}, n)
            if r:
                hit = (2, n, r[0]); break
    if hit is None and bg is not None:                           # (4) A20: rare in the stream AND rare in English
        cnt = Counter(sw)
        cands = [(bg.get(w, 0), cnt[w], -i, w) for i, w in enumerate(qw) if 1 <= cnt[w] <= RARE and bg.get(w, 0) <= bg_max]
        if cands:
            w = min(cands)[3]
            last = max(i for i, x in enumerate(sw) if x == w)
            hit = (4, 1, last)
    if hit is None:                                              # (3) rare-word bigram
        cnt = Counter(sw)
        grams = {tuple(qw[i:i + 2]) for i in range(len(qw) - 1)}
        grams = {g for g in grams if min(cnt[g[0]], cnt[g[1]]) <= RARE}
        r = _latest(sw, grams, 2)
        if r:
            hit = (3, 2, r[0])
    if hit is None:
        return None
    step, n, i = hit
    return step, n, stream[i][1], stream[i + n - 1][2] + 1
