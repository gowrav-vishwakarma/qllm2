"""A20: the shipped BACKGROUND word counts (how common a normalised word is in ordinary English text).

Normalised words (v14/norm_organ.py: lower-case, "a / an / the" dropped, one plural "s" stripped) counted over the v14
FineWeb train pool (6M tokens, held out from the eval prose), saved as JSON {word: count} on the d_drive. Used by the
word organ's step 4: a question word rare in the stream AND rare here is key-like (a name, a made-up word), never a
question's frame word ("forgot", "remind", "stand").

    python -m v14.bg_counts
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

POOL = '/home/gowrav/Development/qllm2/.cache/v14_train_pool_fineweb014.pt'
OUT = '/media/gowrav/d_drive/qllm_ckpt/v14/bg_wordcounts_fineweb014.json'
WORD = re.compile(r'[A-Za-z0-9]+')


def main():
    import torch
    from v7.data import get_chat_tokenizer
    from v14.norm_organ import norm
    tok = get_chat_tokenizer()
    toks = torch.load(POOL, map_location='cpu', weights_only=False)['tokens']
    toks = [int(t) for t in toks]
    c = Counter()
    for s in range(0, len(toks), 200_000):
        for m in WORD.finditer(tok.decode(toks[s:s + 200_000])):
            w = norm(m.group())
            if w is not None:
                c[w] += 1
    json.dump(dict(c), open(OUT, 'w'))
    print(f'{len(toks):,} tokens -> {len(c):,} distinct words, {sum(c.values()):,} words; '
          f'{OUT} = {os.path.getsize(OUT) / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
