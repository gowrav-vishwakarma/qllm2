# proto_far2 hook smoke: (1) proto_far stream via _open_source_iter identical between HEAD and the hook (first 300 docs, skip 0 and
# skip 1000), (2) proto_far2 yields v2 docs with its own cursor key, deterministic resume (skip k == drop first k), realized modes by meta,
# (3) token lengths fit one 4096 window (whole-window packing precondition)
import importlib.util, subprocess, sys, tempfile, re
from pathlib import Path
ROOT = Path.cwd(); sys.path[:0] = [str(ROOT)]
from v7 import data as NEW
src = subprocess.run(['git', 'show', 'HEAD:v7/data.py'], cwd=ROOT, capture_output=True, text=True, check=True).stdout
tmp = Path(tempfile.mkdtemp()) / 'old_v7_data.py'; tmp.write_text(src)
sp = importlib.util.spec_from_file_location('old_v7_data', tmp); OLD = importlib.util.module_from_spec(sp); sp.loader.exec_module(OLD)
specs_name = [k for k, v in vars(NEW).items() if isinstance(v, dict) and 'proto_far2' in v][0]
SN, SO = getattr(NEW, specs_name), getattr(OLD, specs_name)
kw = dict(edu_score_min=3, fineweb_name='x', exclude_holdout=True, holdout_pct=5, mix_seed=43)
def take(mod, specs, name, n, skip=0, dc=None):
    it = mod._open_source_iter(name, specs[name], skip=skip, doc_counters=dc, **kw); return [next(it) for _ in range(n)]
for skip in (0, 1000):
    a, b = take(OLD, SO, 'proto_far', 300, skip), take(NEW, SN, 'proto_far', 300, skip)
    assert a == b, f'proto_far differs at skip {skip}'
print('1. proto_far identical HEAD vs hook (300 docs at skip 0 and 1000): OK')
dc = {}; d = take(NEW, SN, 'proto_far2', 400, 0, dc)
assert set(dc) == {'proto_far2'} and dc['proto_far2'] == 400, dc
assert take(NEW, SN, 'proto_far2', 100, 300) == d[300:400], 'resume mismatch'
assert d[:50] != take(NEW, SN, 'proto_far', 50), 'v2 == v1 stream?'
print('2. proto_far2: own cursor key', dc, '| skip 300 == docs 300-399: OK | differs from proto_far: OK')
from v14 import protocol_docs as P
import random
rng = random.Random(1); modes = {}
it = P.protocol_text_iter(seed=43, far=True, far_version=2)
from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained('gpt2')
L = [len(tok(x)['input_ids']) for x in d[:200]]
print('3. token length of 200 proto_far2 docs: min', min(L), 'max', max(L), '(<= 4096:', max(L) <= 4096, ')')
