"""proto_far2 tests (CPU, ~1 min): python -m v14.test_proto_far2
1. proto_far (v1) is bit-identical to the 2026-11-07 stream: hash of 60 docs from Random(5), recorded before the v2 refactor.
2. v2 modes: own docs state the asked quantity's protocol and answer with the convention; none docs carry no protocol sentence
   and answer the raw value (unit family: bare number, then '.'); other docs state the OTHER quantity's protocol and answer raw.
3. v2 streams are deterministic; v2's iterator stream differs from v1's (own rng offset); doc length rules hold.
4. SCOPE sources (11-17): build_doc (proto / proto4k) and build_far_doc_v2 (proto_far2, armed for 3e) stay bit-identical (hashes
   taken before the scope refactor); proto_far3 and proto4k2 'other' docs state the OTHER quantity's protocol and answer raw;
   realized shares near their targets; all five iterator streams are distinct.
"""
from __future__ import annotations

import hashlib
import random
import re

from v14 import protocol_docs as P

V1_HASH = '68a702f3fe73c0f8'
V2_HASH = '884452eec7730950'          # 60 build_far_doc_v2 docs, Random(5), after the held-out-direction patch (11-17, Lead (b));
                                     # the pre-patch (armed until Mind re-checks) v2 hash was 1295aa3bf38d0450 at e670df5a
DOC_HASH = '62ae4850c80d73fc'         # 400 build_doc(max_gap=300) docs, Random(5), at e670df5a          # sha256[:16] of 60 build_far_doc docs from Random(5), taken at 9d5f43de before the refactor


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
            assert not (m['family'] == 'unit' and m['quantity'] == P.SCOPE_HELD_ASKED), 'held-out direction in proto_far2'
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


def _h(f, n):
    rng = random.Random(5); h = hashlib.sha256()
    for _ in range(n):
        h.update(f(rng).encode())
    return h.hexdigest()[:16]


def test_old_streams_unchanged():
    assert _h(P.build_far_doc_v2, 60) == V2_HASH
    assert _h(lambda r: P.build_doc(r, max_gap=300), 400) == DOC_HASH
    print('  proto_far2 and build_doc bit-identical')


def _check_other(d, m):
    pl = m['parts'][0].lower()
    assert m['family'] in ('unit', 'convert') and m['other_q'] not in (None, m['quantity'])
    assert not (m['family'] == 'unit' and m['quantity'] == P.SCOPE_HELD_ASKED), 'held-out scope direction in training'
    assert m['parts'][0] in d and (m['other_q'] in pl or P.PLURAL.get(m['other_q'], '#') in pl), m['parts'][0]
    ans = _answers(d)
    if m['family'] == 'unit':
        assert all(re.fullmatch(r'\d+\.\d', a) for a in ans), ans
        assert all(P.ANSWER_MARK + ' ' + a + P.ANSWER_MARK + '.' in d for a in ans)
    else:
        assert all(a.endswith(' ' + m['sub'][0]) for a in ans), ans


def test_scope_sources(n_far=300, n_doc=2000):
    rng = random.Random(13); seen = {}
    for _ in range(n_far):
        d, m = P.build_far_doc_v3(rng, with_meta=True); seen[m['mode']] = seen.get(m['mode'], 0) + 1
        assert P._ntok(d) <= P.FAR_MAX
        if m['mode'] == 'other':
            _check_other(d, m)
    assert abs(seen['other'] / n_far - 0.20) < 0.06, seen
    print(f'  proto_far3 modes over {n_far}: {seen}')
    rng = random.Random(17); seen = {}
    for _ in range(n_doc):
        d, m = P.build_doc_v2(rng, with_meta=True); seen[m['mode']] = seen.get(m['mode'], 0) + 1
        if m['mode'] == 'other':
            _check_other(d, m)
    assert abs(seen['other'] / n_doc - 0.20) < 0.03, seen
    print(f'  proto4k2 modes over {n_doc}: {seen}')
    it = lambda **k: [x for _, x in zip(range(2), P.protocol_text_iter(seed=3, **k))]  # noqa: E731
    streams = [it(), it(scope_version=2, max_gap=300), it(far=True), it(far=True, far_version=2), it(far=True, far_version=3)]
    assert len({tuple(x) for x in streams}) == 5
    print('  five iterator streams distinct')


def main():
    for t in (test_v1_unchanged, test_old_streams_unchanged, test_v2_modes, test_streams, test_scope_sources):
        print(t.__name__ + ':'); t()
    print('ALL PASS')


if __name__ == '__main__':
    main()
