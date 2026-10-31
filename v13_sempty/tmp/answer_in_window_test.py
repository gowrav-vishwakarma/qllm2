# ANSWER_IN_WINDOW test: (1) off = old weights exactly; (2) on = answer weight only where the
# token's document starts inside the window, checked against an independent recount;
# (3) the share of orphaned answer tokens on a real long-doc source (the size of the fix).
import itertools, os, random, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from transformers import AutoTokenizer
import v7.data as D
tok = AutoTokenizer.from_pretrained('gpt2')
M = D.ANSWER_MARK
rng = random.Random(0)
words = 'alpha beta gamma delta river stone light cloud paper window'.split()
def doc():
    n = rng.randint(1, 60)
    body = ' '.join(rng.choice(words) for _ in range(n))
    return f"{body} -> {M} {rng.choice(words)}{M}\n" * rng.randint(1, 3)
docs = [doc() for _ in range(400)]
def run(on, seq, it):
    os.environ['ANSWER_IN_WINDOW'] = '1' if on else '0'
    ds = D.StreamingTokenChunkDataset(iter(it), tok, seq, shuffle_buffer=0, answer_weight=100.0)
    return [(s['input_ids'], s['labels'], s['loss_mask']) for s in ds]
# independent recount
ids, w, start, dn = [], [], [], []
for k, d in enumerate(docs):
    a, b = D.encode_marked(tok, d, 100.0); a.append(tok.eos_token_id); b.append(1.0)
    s0 = len(ids); ids += a; w += b; start += [s0] * len(a)
seq = 64; step = seq + 1
off, on = run(False, seq, docs), run(True, seq, docs)
assert len(off) == len(on) == len(ids) // step
n_orph = n_ans = 0
for c, ((_, _, mo), (_, lab, mn)) in enumerate(zip(off, on)):
    for j in range(seq):
        g = c * step + j + 1
        assert mo[j] == w[g], 'off changed the weights'
        exp = w[g] if start[g] >= c * step else min(w[g], 1.0)
        assert mn[j] == exp, (c, j, float(mn[j]), exp)
        n_ans += w[g] > 1; n_orph += (w[g] > 1) and start[g] < c * step
print(f'recount OK: {len(on)} windows, answer tokens {n_ans}, orphaned {n_orph} ({n_orph / n_ans:.1%})')
# (3) size of the fix on a real long-doc source. The b2gap source this was first measured on
# (44 % orphaned at SEQ 8192) was removed after LAW-1 / lg8r, and srd (22 % at 2048) after cu1;
# recall is the long-doc answer-weighted source left (15.9 % at SEQ 2048, orphan_rate.py).
it = D._open_source_iter('recall', D.SOURCE_REGISTRY['recall'], edu_score_min=3, fineweb_name='sample-10BT',
                         exclude_holdout=True, holdout_pct=5, skip=0, doc_counters=None, mix_seed=42)
gd = [t for t in (x[1] if isinstance(x, tuple) else x for x in itertools.islice(it, 600)) if D.ANSWER_MARK in t][:300]
for name, mix in (('recall 1 : 1 with web-like docs', [x for g in gd for x in (g, ' '.join(rng.choice(words) for _ in range(900)))]),):
    a, b = run(False, 2048, mix), run(True, 2048, mix)
    tot = sum(int((m > 1).sum()) for _, _, m in a); kept = sum(int((m > 1).sum()) for _, _, m in b)
    print(f'SEQ 2048, {name}: answer tokens {tot}, orphaned {tot - kept} ({(tot - kept) / tot:.1%})')
print('ANSWER_IN_WINDOW TEST PASS')
