# ANSWER_W_SRC: per-source answer weight (b2 at 25 while recall keeps 100); unset = unchanged.
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import v7.data as D
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('gpt2'); M = D.ANSWER_MARK
docs = [('b2', f'Letter rule. a -> b x ->{M} y{M}\n'), ('recall', f'Fact: key is{M} gold{M}\n')] * 40
def weights(env):
    os.environ['ANSWER_W_SRC'] = env
    ds = D.StreamingTokenChunkDataset(iter(docs), tok, 64, shuffle_buffer=0, answer_weight=100.0)
    return sorted({float(w) for s in ds for w in s['loss_mask'].tolist()})
print('unset   :', weights('')); print('b2:25   :', weights('b2:25'))
assert weights('') == [1.0, 100.0] and weights('b2:25') == [1.0, 25.0, 100.0]
# parity: every listed source at x100 must reproduce the unset packing bit-for-bit (ids and weights)
def samples(env):
    os.environ['ANSWER_W_SRC'] = env
    return [(s['input_ids'].tolist(), s['loss_mask'].tolist()) for s in D.StreamingTokenChunkDataset(iter(docs), tok, 64, shuffle_buffer=0, answer_weight=100.0)]
assert samples('') == samples('b2:100,recall:100'), 'x100 must equal the old packing'
print('parity: x100 everywhere == unset, bit-for-bit on', len(samples('')), 'windows')
print('ANSWER_W_SRC TEST PASS')
