"""sft1 parallel producers on REAL smoltalk2-SFT (network / HF cache): order is deterministic, and a stream resumed from a
mid-stream cursor (draws + per-split raw rows) continues exactly. python v13_sempty/tmp/test_sft_parallel.py"""
import copy, hashlib, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v7.data import get_chat_tokenizer  # noqa: E402
from v13_sempty import sft_mix as sm  # noqa: E402

h = lambda m: hashlib.md5(json.dumps(m)[:2000].encode()).hexdigest()[:10]  # noqa: E731
tok = get_chat_tokenizer()
pidx = sm.build_probe_index(tok)
dec = os.environ.get('DECONTAM_INDEX', '.cache/decontam/v2.npy')
dc = {}
t = time.time()
st = sm.ParallelSFTStream(dc, 0.0, dec, pidx)
it = iter(st)
a = [h(next(it)) for _ in range(400)]
rate = 400 / (time.time() - t)
snap = copy.deepcopy(dc)
b_rest = [h(next(it)) for _ in range(200)]
for p in st.procs: p.terminate()
st2 = sm.ParallelSFTStream({}, 0.0, dec, pidx)
it2 = iter(st2)
a2 = [h(next(it2)) for _ in range(400)]
for p in st2.procs: p.terminate()
st3 = sm.ParallelSFTStream(copy.deepcopy(snap), 0.0, dec, pidx)
it3 = iter(st3)
c = [h(next(it3)) for _ in range(200)]
for p in st3.procs: p.terminate()
print(f'first 400 kept at {rate:.0f} convs/s (incl. startup) | drops {({k: v for k, v in dc.items() if "#" in k})}')
print('deterministic order', a == a2, '| resume from the cursor at 400 continues exactly', c == b_rest)
ok = a == a2 and c == b_rest
print('ALL PASS' if ok else 'FAIL'); os._exit(0 if ok else 1)
