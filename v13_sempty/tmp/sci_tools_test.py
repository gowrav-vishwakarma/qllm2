# SCI-1 tools / think / pysci test: rendered samples, counts, train / holdout disjoint, skip_docs resume exact.
import sys, itertools, hashlib
sys.path.insert(0, '/home/gowrav/Development/qllm2')
from v7 import data as D
h = lambda t: hashlib.sha1(t.encode()).hexdigest()[:10]
for name in ('tools', 'think', 'pysci'):
    sp = D.SOURCE_REGISTRY[name]
    op = lambda skip=0, c=None: D._open_source_iter(name, sp, edu_score_min=3, fineweb_name='sample-10BT', exclude_holdout=True,
                                                    holdout_pct=5, skip=skip, doc_counters=c if c is not None else {}, mix_seed=43)
    tr = list(itertools.islice(op(), 60)); r = next(op(skip=40))
    print(f'== {name}: 60 train docs, mean {sum(map(len, tr)) // 60} chars; all train-bucket {not any(D.pretrain_holdout_bucket(t, 5) for t in tr)}; '
          f'skip 40 -> doc 41 identical {h(r) == h(tr[40])}')
    print('   sample:', repr(tr[1][:500]))
    if name == 'tools':
        print('   with <tool_call>:', sum('<tool_call>' in t for t in tr), '/ 60; with <tool_response>:', sum('<tool_response>' in t for t in tr),
              '; library.fetch:', sum('library.fetch' in t for t in tr))
    if name == 'think':
        print('   with <think>:', sum('<think>' in t for t in tr), '/ 60; max chars', max(map(len, tr)))
