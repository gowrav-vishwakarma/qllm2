# Share of answer-weighted tokens that the packer puts in a window where their document did
# not start (= the context the answer needs may be in an earlier window), per synthetic source,
# at the real mix: each source's docs interleaved 1 : k with web-like docs so offsets are random.
import itertools, os, random, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from transformers import AutoTokenizer
import v7.data as D
tok = AutoTokenizer.from_pretrained('gpt2')
rng = random.Random(1)
W = 'the of and to in a is that for it as was with be by on not he this are or his from at which'.split()
web = lambda: ' '.join(rng.choice(W) for _ in range(rng.randint(200, 1500)))
def docs_of(src, n):
    it = D._open_source_iter(src, D.SOURCE_REGISTRY[src], edu_score_min=3, fineweb_name='sample-10BT',
                             exclude_holdout=True, holdout_pct=5, skip=0, doc_counters=None, mix_seed=42)
    out = []
    for x in it:
        t = x[1] if isinstance(x, tuple) else x
        if D.ANSWER_MARK in t: out.append(t)
        if len(out) >= n: break
    return out
def rate(ds, seq):
    tot = kept = 0
    for on in (False, True):
        os.environ['ANSWER_IN_WINDOW'] = '1' if on else '0'
        n = sum(int((s['loss_mask'] > 1).sum()) for s in D.StreamingTokenChunkDataset(iter(ds), tok, seq, shuffle_buffer=0, answer_weight=100.0))
        if on: kept = n
        else: tot = n
    return tot, kept
for src in sys.argv[1].split(','):
    try:
        d = docs_of(src, 300)
    except Exception as e:
        print(src, 'skipped:', type(e).__name__, str(e)[:100]); continue
    lens = sorted(len(tok.encode(D.strip_answer_marks(x))) for x in d)
    mix = [y for x in d for y in (x, web())]
    for seq in (2048, 8192):
        tot, kept = rate(mix, seq)
        print(f'{src:7s} SEQ {seq}: docs {len(d)}, tokens median {lens[len(lens)//2]} p90 {lens[int(.9*len(lens))]}; answer tokens {tot}, orphaned {tot-kept} ({(tot-kept)/max(tot,1):.1%})', flush=True)
