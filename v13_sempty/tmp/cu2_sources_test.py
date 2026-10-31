# cu2 sources (fld, vfy): determinism, resume, answer correctness, held-out split, orphan rate at 2048.
import itertools, os, re, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import v7.data as D
M = D.ANSWER_MARK
for src, it_fn in (('fld', D._fld_text_iter), ('vfy', D._vfy_text_iter)):
    docs = list(itertools.islice(it_fn(seed=42), 3000)); again = list(itertools.islice(it_fn(seed=42), 3000))
    assert docs == again, 'not deterministic'
    cnt = {}; skip = list(itertools.islice(it_fn(seed=42, skip_docs=500, doc_counters=cnt), 10))
    assert skip == docs[500:510] and cnt == {src: 10}, 'resume'
    n_items = n_unk = bad = held = probe_fmt = 0
    for d in docs:
        for part in d.rstrip('\n').split('\n\n'):
            assert part.count(M) == 2, part
            body, ans = part.split(M)[0], part.split(M)[1]
            n_items += 1; n_unk += ans == ' unknown.'
            assert ans.endswith('.') and ans.startswith(' '), ans
            # find the question (one of the trained formats) and check the answer against the rows
            q = next(((qi, m) for qi, qf in enumerate(D._FLD_QS)
                      for m in [re.search(re.escape(qf).replace(r'\{f\}', r'(?P<f>\w+)').replace(r'\{k\}', r'(?P<k>[\w ]+?)') + '$', body)] if m), None)
            assert q is not None, ('no trained question', body[-80:])
            f, k = q[1].group('f'), q[1].group('k')
            rows = body[:q[1].start()]
            # the value of field f in the row of key k (any trained row format), else unknown
            want = ' unknown.'
            for rf in (rf' Record: {re.escape(k)} \|([^.]*)\.', rf' {re.escape(k)}:([^.]*)\.', rf' For {re.escape(k)},([^.]*)\.'):
                for m in re.finditer(rf, rows):
                    seg = m.group(1)
                    mm = re.search(rf'(?:{f}: |{f} | the {f} is )([^|;,]+?)(?: \||;| and |,|$)', seg)
                    if mm and want == ' unknown.':
                        want = f' {mm.group(1).strip()}.'
            bad += want != ans
            for v in re.findall(r'(?:colour|city|code|owner|room|tool)(?::| is|) ([A-Za-z0-9-]+(?: [A-Za-z]+(?: \d+)?)?)', rows):
                pass
            probe_fmt += bool(re.search(r'What \w+ does .+ have\? Answer:|Asked for the \w+ of|\] \w+=', body))
    # held-out values: record every value the generator emits (direct, no parsing)
    seen = []; orig = D._fld_value
    D._fld_value = lambda rng, field, held=False: seen.append(orig(rng, field, held)) or seen[-1]
    list(itertools.islice(it_fn(seed=7), 500)); D._fld_value = orig
    held = sum(D._fld_heldout(v) for v in seen); vals = seen
    print(f'{src}: {len(docs)} docs, {n_items} items, unknown {n_unk / n_items:.1%}, wrong answers {bad}, '
          f'held-out values emitted {held} of {len(vals)}, probe formats {probe_fmt}')
    assert bad == 0 and held == 0 and probe_fmt == 0
# orphan rate at 2048 when interleaved with web-like docs (ANSWER_IN_WINDOW decides what it costs)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('gpt2'); os.environ['ANSWER_IN_WINDOW'] = '0'
import random; rng = random.Random(0); W = 'the of and to in a is that for it as was with be by on'.split()
for src, it_fn in (('fld', D._fld_text_iter), ('vfy', D._vfy_text_iter)):
    mix = [x for d in itertools.islice(it_fn(seed=1), 200) for x in (d, ' '.join(rng.choice(W) for _ in range(900)))]
    res = []
    for on in ('0', '1'):
        os.environ['ANSWER_IN_WINDOW'] = on
        res.append(sum(int((s['loss_mask'] > 1).sum()) for s in D.StreamingTokenChunkDataset(iter(mix), tok, 2048, shuffle_buffer=0, answer_weight=100.0)))
    print(f'{src}: answer tokens {res[0]}, orphaned at SEQ 2048 {res[0] - res[1]} ({(res[0] - res[1]) / res[0]:.1%})')
print('CU2 SOURCES TEST PASS')
