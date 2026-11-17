"""proto_far2 tests (CPU, ~1 min): python -m v14.test_proto_far2
1. proto_far (v1) is bit-identical to the 2026-11-07 stream: hash of 60 docs from Random(5), recorded before the v2 refactor.
2. v2 modes: own docs state the asked quantity's protocol and answer with the convention; none docs carry no protocol sentence
   and answer the raw value (unit family: bare number, then '.'); other docs state the OTHER quantity's protocol and answer raw.
3. v2 streams are deterministic; v2's iterator stream differs from v1's (own rng offset); doc length rules hold.
"""
from __future__ import annotations

import hashlib
import random
import re

from v14 import protocol_docs as P

V1_HASH = '68a702f3fe73c0f8'          # sha256[:16] of 60 build_far_doc docs from Random(5), taken at 9d5f43de before the refactor


def test_v1_unchanged():
    rng = random.Random(5); h = hashlib.sha256()
    for _ in range(60):
        h.update(P.build_far_doc(rng).encode())
    assert h.hexdigest()[:16] == V1_HASH, h.hexdigest()[:16]
    print('  v1 stream bit-identical')


def _answers(d):
    return [a.strip() for a in d.split(P.ANSWER_MARK)[1::2]]


def test_v2_modes(n=300):
    rng = random.Random(11); seen = {}
    all_protos = {w.split('{')[0] for w in P.UNIT_W + P.ROUND_W + P.CONVERT_W if not w.startswith('{')}
    for _ in range(n):
        d, m = P.build_far_doc_v2(rng, with_meta=True)
        seen[m['mode']] = seen.get(m['mode'], 0) + 1
        ans = _answers(d); assert ans, 'no marked answer'
        assert P._ntok(d) <= P.FAR_MAX
        if m['mode'] == 'own':
            assert m['protocol'] and m['parts'][0] in d
            if m['family'] == 'unit':
                assert all(a.endswith(' ' + m['sub']) for a in ans), (ans, m['sub'])
        elif m['mode'] == 'none':
            assert not m['protocol'] and m['other_q'] is None
            assert not any(len(p) > 8 and p in d for p in all_protos), 'a protocol sentence leaked into a none doc'
            if m['family'] == 'unit':
                assert all(re.fullmatch(r'\d+\.\d', a) for a in ans), ans          # bare value
                assert all(P.ANSWER_MARK + ' ' + a + P.ANSWER_MARK + '.' in d for a in ans)  # then '.'
        else:
            assert m['family'] in ('unit', 'convert') and m['other_q'] not in (None, m['quantity'])
            pl = m['parts'][0].lower()
            assert m['parts'][0] in d and (m['other_q'] in pl or P.PLURAL.get(m['other_q'], '#') in pl), m['parts'][0]
            if m['family'] == 'unit':
                assert all(re.fullmatch(r'\d+\.\d', a) for a in ans), ans
            else:                                                                  # convert: the row's own value + unit, unconverted
                u1 = m['sub'][0]; assert all(a.endswith(' ' + u1) for a in ans), ans
    assert set(seen) == {'own', 'none', 'other'}, seen
    print(f'  modes over {n}: {seen}')


def test_streams():
    a = list(zip(range(3), P.protocol_text_iter(seed=3, far=True, far_version=2)))
    b = list(zip(range(3), P.protocol_text_iter(seed=3, far=True, far_version=2)))
    c = list(zip(range(3), P.protocol_text_iter(seed=3, far=True)))
    assert a == b and a != c
    print('  v2 deterministic, own stream')


def main():
    for t in (test_v1_unchanged, test_v2_modes, test_streams):
        print(t.__name__ + ':'); t()
    print('ALL PASS')


if __name__ == '__main__':
    main()
