# SCI-1 science sources test: (1) each streams text; (2) train / holdout disjoint (the shared hash); (3) exact cursor resume:
# read 30 docs, keep the cursor, reopen -> the next doc equals doc 31 of an uninterrupted stream; (4) source_holdout_tokens.
import sys, hashlib, itertools
sys.path.insert(0, '/home/gowrav/Development/qllm2')
from v7 import data as D
from transformers import AutoTokenizer
h = lambda t: hashlib.sha1(t.encode()).hexdigest()[:10]
for name in ('arxiv', 'pubmed', 'pes2o', 'cosmo'):
    tr = list(itertools.islice(D._sci_text_iter(name, exclude_holdout=True, counter_key='t'), 40))
    ho = list(itertools.islice(D._sci_text_iter(name, only_holdout=True, counter_key='h'), 5))
    overlap = {h(t) for t in tr} & {h(t) for t in ho}
    all_hold = all(D.pretrain_holdout_bucket(t, 5) for t in ho) and not any(D.pretrain_holdout_bucket(t, 5) for t in tr)
    c = {}; a = list(itertools.islice(D._sci_text_iter(name, exclude_holdout=True, doc_counters=c, counter_key=name), 30))
    cur = {k: v for k, v in c.items()}
    b = next(D._sci_text_iter(name, exclude_holdout=True, doc_counters=dict(cur), counter_key=name))
    print(f'{name}: train docs {len(tr)} (mean {sum(map(len, tr)) // len(tr)} chars), holdout {len(ho)}; overlap {len(overlap)}; buckets right {all_hold}; '
          f'cursor {cur}; resume == uninterrupted doc 31: {h(b) == h(tr[30])}')
tok = AutoTokenizer.from_pretrained('gpt2')
t = D.source_holdout_tokens('arxiv', tok, n_tokens=4096)
print('source_holdout_tokens arxiv', tuple(t.shape), repr(tok.decode(t[:30].tolist())[:120]))
