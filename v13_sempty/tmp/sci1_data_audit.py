# SCI-1 data audit for the pre-registration header: per source, N docs through the real source iterator (train split):
# decontamination drop rate (v7/decontam index), within-source duplicate rate (sha1 of the first 1,000 normalised chars), mean chars;
# plus the kept fraction of our own filters (pysci: python files with a scientific import; pes2o: arXiv-sourced docs dropped).
import sys, itertools, hashlib, json, re, gzip
sys.path.insert(0, '/home/gowrav/Development/qllm2')
from v7 import data as D
from v7.decontam import DecontamFilter
F = DecontamFilter('/home/gowrav/Development/qllm2/.cache/decontam/v1.npy'); N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
out = {'index_sha256': F.sha}
for name in ('dclm', 'fineweb', 'arxiv', 'pubmed', 'pes2o', 'cosmo', 'finemath', 'smoltalk2_mid', 'chatmt', 'tools', 'think', 'pysci',
             'b2s', 'recall4k', 'proto4k'):
    it = D._open_source_iter(name, D.SOURCE_REGISTRY[name], edu_score_min=3, fineweb_name='sample-10BT', exclude_holdout=True,
                             holdout_pct=5, skip=0, doc_counters={}, mix_seed=43)
    seen, dup, hit, chars = set(), 0, 0, 0
    docs = list(itertools.islice(it, N))
    for d in docs:
        k = hashlib.sha1(re.sub(r'\s+', ' ', d[:1000]).strip().lower().encode()).hexdigest()
        dup += k in seen; seen.add(k); hit += F.hit(d); chars += len(d)
    out[name] = {'n': len(docs), 'decon_drop': round(hit / len(docs), 4), 'dup_rate': round(dup / len(docs), 4), 'mean_chars': chars // max(1, len(docs))}
    print(name, out[name], flush=True)
# our own filters' kept fractions (raw rows, before holdout)
from huggingface_hub import HfFileSystem
fs = HfFileSystem()
def raw_rows(repo, pat, n):
    f = D._hf_glob(repo, pat)[0]
    with fs.open(f, 'rb') as fh:
        g = gzip.GzipFile(fileobj=fh)
        return [json.loads(g.readline()) for _ in range(n)]
rows = raw_rows('common-pile/peS2o_filtered', 'peS2o-*.json.gz', 5000)
out['pes2o_arxiv_dropped_fraction'] = round(sum(str((r.get('metadata') or {}).get('pdf_src', '')).lower() == 'arxiv' for r in rows) / len(rows), 4)
from v7.sci_tools import SCI_IMPORT
py = [f for f in D._hf_glob('common-pile/stackv2_edu_filtered', 'stack-edu-*.json.gz')]
tot = keep = 0
for f in py:
    with fs.open(f, 'rb') as fh:
        g = gzip.GzipFile(fileobj=fh); r = json.loads(g.readline())
    if (r.get('metadata') or {}).get('language') != 'Python':
        continue
    with fs.open(f, 'rb') as fh:
        g = gzip.GzipFile(fileobj=fh)
        for _ in range(5000):
            r = json.loads(g.readline()); tot += 1; keep += bool(SCI_IMPORT.search(r.get('text') or ''))
    break
out['pysci_kept_fraction_of_python'] = round(keep / max(1, tot), 4)
print(json.dumps(out, indent=1))
json.dump(out, open('/home/gowrav/Development/qllm2/logs/sci1/v13_sempty_sci1_data_audit.json', 'w'), indent=1)
