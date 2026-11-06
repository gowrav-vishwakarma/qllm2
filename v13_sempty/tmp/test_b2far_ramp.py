"""Tests for the R2 ramp in v7/data.py `_blend_interleave_text_iters` and the b2s_far source (2026-11-07).
  1. no ramp: the blend's pick sequence is bit-identical to the committed (HEAD) function
  2. ramp: realized shares move linearly from weights to ramp_weights; no burst after a weight change
  3. b2s_far: 600 docs: token lengths (all <= 3,200 incl. the answer), share >= 2,048, layout, no `_new` task, resume-exact
    python v13_sempty/tmp/test_b2far_ramp.py [--skip_b2far]
"""
import subprocess, sys, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]
from v7 import data as D  # noqa: E402


def dummy(name, n_chars):
    i = 0
    while True:
        i += 1
        yield f'{name}{i} ' + 'x' * n_chars


def picks(fn, n, **kw):
    tc = {'a': 0, 'b': 0, 'c': 0}
    it = fn([('a', dummy('a', 400)), ('b', dummy('b', 4000)), ('c', dummy('c', 1200))], [0.6, 0.3, 0.1], token_counters=tc, **kw)
    out = []
    for _ in range(n):
        name, text = next(it)
        tc[name] += int(len(text) / D._CHARS_PER_TOKEN)   # the dataset's counters, approximated
        out.append(name)
    return out, tc


def test_identical():
    src = subprocess.run(['git', 'show', 'HEAD:v7/data.py'], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    mod = types.ModuleType('old_data'); mod.__dict__['__file__'] = str(ROOT / 'v7/data.py')
    code = src[src.index('def _blend_interleave_text_iters('):src.index('_CHARS_PER_TOKEN = 3.9')]
    exec('from typing import *\n' + code, mod.__dict__); mod._CHARS_PER_TOKEN = D._CHARS_PER_TOKEN
    old, _ = picks(mod._blend_interleave_text_iters, 20000)
    new, _ = picks(D._blend_interleave_text_iters, 20000)
    assert old == new, 'no-ramp blend changed'
    print('1. no ramp: 20,000 picks identical to HEAD: OK')


def test_ramp():
    # realistic ratio: ramp >> deficit window (the run: 2B vs 4M; here 8M vs 100k), and run past the ramp's end
    n = 80000
    seq, tc = picks(D._blend_interleave_text_iters, n, ramp_weights=[0.3, 0.3, 0.4], ramp_from_tokens=1_000_000, ramp_tokens=8_000_000,
                    ramp_window=100_000.0)
    # realized token share of c in consecutive windows of 2,000 picks, and its longest run of consecutive picks
    sz = {'a': 400 / 3.9, 'b': 4000 / 3.9, 'c': 1200 / 3.9}
    shares = []
    for k in range(0, n, 2000):
        w = seq[k:k + 2000]; tot = sum(sz[x] for x in w)
        shares.append(round(sum(sz[x] for x in w if x == 'c') / tot, 3))
    run = best = 0
    for x in seq:
        run = run + 1 if x == 'c' else 0; best = max(best, run)
    print('2. ramp: share of c per 2,000 picks:', shares[:3], '...', shares[-3:], '| longest c run', best,
          '| total tokens', f'{sum(tc.values()):,}')
    assert abs(shares[0] - 0.1) < 0.03 and abs(shares[-1] - 0.4) < 0.03, 'ramp did not reach its targets'
    assert best <= 6, 'burst: a long run of the ramped source'
    print('   OK (starts at 0.1, ends at 0.4, no burst)')


def test_b2far():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')
    it = D._b2far_text_iter(seed=43)
    docs = [next(it) for _ in range(600)]
    lens = sorted(len(tok.encode(D.strip_answer_marks(d), add_special_tokens=False)) for d in docs)
    q = lambda p: lens[int(p * (len(lens) - 1))]
    print(f'3. b2s_far 600 docs: tokens min {lens[0]} median {q(0.5)} p90 {q(0.9)} max {lens[-1]}; '
          f'>= 2,048: {sum(l >= 2048 for l in lens)} / 600; >= 4,096: {sum(l >= 4096 for l in lens)}')
    assert lens[-1] <= 3200, 'a doc exceeds 3,200 tokens'
    heads = {d.split('\n')[0][:13] for d in docs}
    print('   headers:', sorted(heads))
    assert all(d.count(D.ANSWER_MARK) == 2 for d in docs), 'answer marks'
    it2 = D._b2far_text_iter(seed=43, skip_docs=599)
    assert next(it2) == docs[599], 'resume (skip_docs) is not exact'
    print('   resume-exact at skip_docs=599: OK')
    print('   sample (head / tail):', repr(docs[0][:160]), '...', repr(docs[0][-80:]))


if __name__ == '__main__':
    test_identical(); test_ramp()
    if '--skip_b2far' not in sys.argv:
        test_b2far()
    print('ALL OK')
