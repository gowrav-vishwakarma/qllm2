"""A5: the SYNONYM cell. A fact is stored under one phrase and asked with a synonym phrase that shares no
content word, so token / normalised organs and BM25 miss by construction; only a key that carries MEANING can
find the row. Same shape as v16/bench.py build(): returns (ids, answer token, query length, meta).

    syn     8 facts " Definition: {A} refers to {v}." in held-out prose; asked " Query: {B} refers to" where
            (A, B) is a synonym pair (which side is stored is random); meta key = the stored phrase A.
    synneg  the same 8 facts; asked with one side of a pair that was NOT stored (abstain is right).
"""
from __future__ import annotations

PAIRS = [
    ('old ship', 'ancient vessel'), ('quiet harbour', 'silent port'), ('big house', 'large home'),
    ('fast horse', 'quick steed'), ('dark forest', 'gloomy woods'), ('angry dog', 'furious hound'),
    ('happy child', 'joyful kid'), ('cold river', 'chilly stream'), ('rich merchant', 'wealthy trader'),
    ('loud bell', 'noisy chime'), ('hidden door', 'secret entrance'), ('tall tower', 'high spire'),
    ('empty room', 'vacant chamber'), ('narrow street', 'cramped alley'), ('huge mountain', 'enormous peak'),
    ('clever student', 'smart pupil'), ('sick sailor', 'ill seaman'), ('brave soldier', 'bold warrior'),
    ('calm lake', 'still pond'), ('busy market', 'crowded bazaar'), ('bright lamp', 'shining lantern'),
    ('heavy stone', 'weighty rock'), ('strange man', 'odd fellow'), ('famous singer', 'renowned vocalist'),
    ('wide valley', 'broad dale'), ('wet field', 'damp meadow'), ('sharp knife', 'keen blade'),
    ('lazy cat', 'idle kitten'), ('polite waiter', 'courteous server'), ('sad poet', 'sorrowful bard'),
    ('tiny insect', 'minuscule bug'), ('aged priest', 'elderly cleric'), ('ugly building', 'hideous structure'),
    ('wise king', 'sage monarch'), ('young girl', 'youthful maiden'), ('thick book', 'hefty tome'),
    ('hot desert', 'scorching wasteland'), ('deep cave', 'bottomless cavern'), ('shy boy', 'timid lad'),
    ('sly fox', 'cunning vixen'), ('rotten apple', 'spoiled fruit'), ('stormy sea', 'turbulent ocean'),
    ('cheap hotel', 'inexpensive inn'), ('sacred text', 'holy scripture'), ('icy road', 'frosty highway'),
    ('dirty coat', 'filthy jacket'), ('strong bull', 'powerful ox'), ('gentle nurse', 'tender caregiver'),
    ('silly clown', 'foolish jester'), ('tired farmer', 'weary peasant'), ('nervous pilot', 'anxious aviator'),
    ('grand palace', 'magnificent castle'), ('lost traveller', 'missing wanderer'), ('small village', 'little hamlet'),
    ('steep hill', 'sheer slope'), ('elegant lady', 'graceful dame'), ('rude guest', 'impolite visitor'),
    ('rowdy mob', 'unruly throng'), ('tough meat', 'chewy beef'), ('smooth silk', 'sleek satin'),
    ('cruel tyrant', 'brutal despot'), ('honest judge', 'truthful magistrate'), ('fierce lion', 'ferocious beast'),
    ('broken vase', 'shattered urn'), ('sweet cake', 'sugary pastry'), ('shiny coin', 'gleaming penny'),
    ('wild garden', 'untamed orchard'), ('covert plot', 'clandestine scheme'), ('messy desk', 'untidy table'),
    ('starry sky', 'twinkling heavens'), ('sleepy town', 'drowsy borough'), ('skilled doctor', 'expert physician'),
    ('blind beggar', 'sightless pauper'), ('merry party', 'cheerful celebration'), ('dangerous path', 'perilous trail'),
    ('strict teacher', 'stern instructor'), ('humble cottage', 'modest hut'), ('giant whale', 'colossal leviathan'),
    ('cozy cabin', 'snug lodge'), ('sour lemon', 'tart citrus'), ('brief letter', 'short note'),
    ('final chapter', 'last section'),
]


def _check():
    from v7.data import _RECALL_VALUE_NOUNS
    from v14.norm_organ import norm
    seen = set()
    for a, b in PAIRS:
        wa, wb = {norm(w) for w in a.split()}, {norm(w) for w in b.split()}
        assert not (wa & wb), (a, b)
        for w in wa | wb:
            assert w not in seen, w
            assert w not in _RECALL_VALUE_NOUNS, w
            seen.add(w)
    return len(PAIRS), len(seen)


def build_syn(rng, tok, filler, L, n=8, neg=False):
    from v7.data import _RECALL_VALUE_NOUNS
    def enc(s): return tok.encode(s, add_special_tokens=False)
    start = rng.randrange(0, len(filler) - L - 1)
    body = filler[start:start + L].tolist()
    picks = rng.sample(range(len(PAIRS)), n + 1)                 # the last pair is never stored (synneg)
    stored, asked = [], []
    for i in picks:
        a, b = PAIRS[i]
        if rng.random() < 0.5:
            a, b = b, a
        stored.append(a); asked.append(b)
    vals = [rng.choice(_RECALL_VALUE_NOUNS) for _ in range(n)]
    inserts = [enc(f" Definition: {k} refers to {v}.") for k, v in zip(stored[:n], vals)]
    if neg:
        key, qkey, val = stored[n], asked[n], rng.choice(_RECALL_VALUE_NOUNS)
    else:
        qi = rng.randrange(n); key, qkey, val = stored[qi], asked[qi], vals[qi]
    query = enc(f" Query: {qkey} refers to")
    pos = sorted(rng.sample(range(0, max(1, L - 64)), n))
    out, cur = [], 0
    for p, d in zip(pos, inserts):
        out += body[cur:p]; out += d; cur = p
    out += body[cur:]
    ids = out + query
    ans = tok.encode(f" {val}", add_special_tokens=False)[0]
    meta = dict(key=key, val=val, hop_key=None, query_key=qkey)
    if neg:
        meta['negative'] = True
    return ids, ans, len(query), meta


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    print('pairs, distinct words:', _check())
