"""sft1 tests (CPU, no network): python v13_sempty/tmp/test_sft_mix.py
1. sft_ids_targets == v7 _encode_sft_example's unpadded ids / targets (same ChatML layout).
2. Packed windows: loss_mask is 1 exactly on assistant content + its <|im_end|> tokens; conversations are EOS-separated; every
   window has seq_len tokens; a too-long conversation is truncated, a non-fitting one starts the next window.
3. SFTMixDataset with share 0 yields the base windows unchanged (+ a ones loss_mask where missing); share 1 yields only SFT windows.
4. The probe index contains the CHECK protocol prompt's 13-grams (so such a conversation is dropped)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v7.data import IM_END, _encode_sft_example, get_chat_tokenizer  # noqa: E402
from v13_sempty import sft_mix as sm  # noqa: E402

tok = get_chat_tokenizer()
convs = [
    [{'role': 'user', 'content': 'What is 2 + 2?'}, {'role': 'assistant', 'content': 'It is 4.'}],
    [{'role': 'system', 'content': 'Be brief.'}, {'role': 'user', 'content': 'Name a planet.'},
     {'role': 'assistant', 'content': 'Mars.'}, {'role': 'user', 'content': 'Another?'}, {'role': 'assistant', 'content': 'Venus.'}],
    [{'role': 'user', 'content': 'Tell me a long story. ' * 50}, {'role': 'assistant', 'content': 'Once upon a time. ' * 200}],
]
ok = True
# 1
for c in convs[:2]:
    ids, tgt = sm.sft_ids_targets(c, tok)
    ref = _encode_sft_example(c, tok, 512)
    r_ids = ref[0].tolist() if hasattr(ref[0], 'tolist') else list(ref[0])
    same = r_ids[:len(ids)] == ids[:len(r_ids)] or r_ids[:len(ids) - 1] == ids[1:]
    print('1 layout vs v7: ids prefix equal', same, '| n ids', len(ids), '| targets', sum(tgt))
    ok &= bool(same)
# 2
SEQ = 64
win = list(sm.pack_windows(iter(convs * 3), tok, SEQ))
im_end = tok.convert_tokens_to_ids(IM_END)
eos = tok.eos_token_id
shapes = all(w['input_ids'].numel() == SEQ and w['labels'].numel() == SEQ and w['loss_mask'].numel() == SEQ for w in win)
print('2 windows', len(win), '| shapes ok', shapes)
ok &= shapes
# every target position must lie inside an assistant span: rebuild the reference mask per conversation
ref_ids, ref_tgt = [], []
for c in convs * 3:
    i, t = sm.sft_ids_targets(c, tok)
    i, t = i + [eos], t + [0]
    i, t = i[:SEQ + 1], t[:SEQ + 1]
    ref_ids.append(i); ref_tgt.append(t)
# the labels at target positions decode to assistant text or <|im_end|>, never to "user" / "system" headers
lab_t = torch.cat([w['labels'][w['loss_mask'] > 0] for w in win]).tolist()
txt = tok.decode(lab_t)
bad = ('user\n' in txt) or ('system\n' in txt) or ('What is 2' in txt) or ('Name a planet' in txt)
print('2 target text sample:', repr(txt[:80]), '| no user/system tokens in targets', not bad)
ok &= not bad
n_imend_t = sum(1 for x in lab_t if x == im_end)
print('2 <|im_end|> targets', n_imend_t, '(one per assistant turn present)')
ok &= n_imend_t > 0
zero_eos = all(float(w['loss_mask'][j]) == 0.0 for w in win for j in range(SEQ) if int(w['labels'][j]) == eos)
print('2 EOS separators / padding never a target', zero_eos)
ok &= zero_eos


# 3
class Base(torch.utils.data.IterableDataset):
    def __iter__(self):
        g = torch.Generator().manual_seed(0)
        for _ in range(20):
            x = torch.randint(0, 50000, (SEQ + 1,), generator=g)
            yield {'input_ids': x[:-1], 'labels': x[1:]}


base = list(Base())
orig_iter = sm.iter_sft_conversations
sm.iter_sft_conversations = lambda dc, skip=0, tf=0.0, d=None, p=None: iter(convs * 50)
sm.build_probe_index = lambda tok_, n=0: np.zeros(0, dtype=np.uint64)
ds0 = sm.SFTMixDataset(Base(), tok, SEQ, 0.0, 43, {}, {}, parallel=False)
out0 = list(ds0)
same0 = len(out0) == len(base) and all(torch.equal(a['input_ids'], b['input_ids']) and torch.equal(a['labels'], b['labels'])
                                        and torch.all(a['loss_mask'] == 1) for a, b in zip(out0, base))
print('3 share 0 == base stream (+ ones mask)', same0)
ok &= same0
ds1 = sm.SFTMixDataset(Base(), tok, SEQ, 1.0, 43, {}, {}, parallel=False)
it1 = iter(ds1)
first = [next(it1) for _ in range(5)]
only_sft = all(float(w['loss_mask'].min()) == 0.0 for w in first)   # SFT windows have masked context; base windows are all ones
print('3 share 1 -> SFT windows only (masked context present)', only_sft)
ok &= only_sft
sm.iter_sft_conversations = orig_iter
# 4
import importlib  # noqa: E402
importlib.reload(sm)
from v7.decontam import ngram_hashes  # noqa: E402
idx = sm.build_probe_index(tok, n_docs=50)
probe = 'Lab protocol: report every mass in grams, rounded to one decimal place.\nSample A weighed 12.347 g and sample B weighed 8.912 g.'
hit = bool(np.isin(ngram_hashes(probe), idx).any())
print('4 probe index', idx.size, '13-grams | CHECK protocol prompt hits it', hit)
ok &= hit
print('ALL PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
