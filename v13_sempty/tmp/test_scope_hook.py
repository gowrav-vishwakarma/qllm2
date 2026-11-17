"""Segment 3f hook checks (2026-11-17): proto_far3 + proto4k2 registry entries in v7/data.py.
  1. proto, proto4k, proto_far, proto_far2 streams via _open_source_iter identical HEAD vs this tree (skip 0 and 1000)
  2. proto_far3 / proto4k2: own cursor keys, exact resume (skip k == drop k), differ from their v1 sources
  3. held-out unit direction (asked volume, mass protocol) = 0 docs in 20,000 of each (meta), and the mode shares
  4. token length: proto_far3 vs proto_far, proto4k2 vs proto4k (mean / sd / max over 300 docs; whole-window: proto_far3 <= 4096)
    python v13_sempty/tmp/test_scope_hook.py"""
import importlib.util, random, statistics as st, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT)]
from v7 import data as NEW
from v14 import protocol_docs as P
src = subprocess.run(['git', 'show', 'HEAD:v7/data.py'], cwd=ROOT, capture_output=True, text=True, check=True).stdout
tmp = Path(tempfile.mkdtemp()) / 'old_v7_data.py'; tmp.write_text(src)
sp = importlib.util.spec_from_file_location('old_v7_data', tmp); OLD = importlib.util.module_from_spec(sp); sp.loader.exec_module(OLD)
kw = dict(edu_score_min=3, fineweb_name='x', exclude_holdout=True, holdout_pct=5, mix_seed=43)
def take(mod, name, n, skip=0, dc=None):
    it = mod._open_source_iter(name, mod.SOURCE_REGISTRY[name], skip=skip, doc_counters=dc, **kw); return [next(it) for _ in range(n)]
for name in ('proto', 'proto4k', 'proto_far', 'proto_far2'):
    for skip in (0, 1000):
        assert take(OLD, name, 200, skip) == take(NEW, name, 200, skip), (name, skip)
print('1. proto / proto4k / proto_far / proto_far2 identical HEAD vs hook (200 docs at skip 0 and 1000): OK')
for name, ref in (('proto_far3', 'proto_far'), ('proto4k2', 'proto4k')):
    dc = {}; d = take(NEW, name, 300, 0, dc)
    assert set(dc) == {name} and dc[name] == 300, dc
    assert take(NEW, name, 100, 200) == d[200:300], f'{name} resume'
    assert d[:50] != take(NEW, ref, 50), f'{name} == {ref}'
print('2. proto_far3 / proto4k2: own cursor keys, exact resume, differ from proto_far / proto4k: OK')
for label, fn in (('proto_far3', lambda r: P.build_far_doc_v3(r, with_meta=True)), ('proto4k2', lambda r: P.build_doc_v2(r, with_meta=True, max_gap=300))):
    rng = random.Random(5); modes = {}; held = 0; N = 20000
    for _ in range(N):
        _, m = fn(rng); modes[m.get('mode', '-')] = modes.get(m.get('mode', '-'), 0) + 1
        if m.get('mode') == 'other' and m.get('family') == 'unit' and m.get('quantity') == P.SCOPE_HELD_ASKED: held += 1
    print(f'3. {label}: held-out unit direction {held} / {N}; modes', {k: round(v / N, 3) for k, v in sorted(modes.items())})
    assert held == 0
from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained('gpt2')
for a, b in (('proto_far3', 'proto_far'), ('proto4k2', 'proto4k')):
    for name in (a, b):
        L = [len(tok(x)['input_ids']) for x in take(NEW, name, 300)]
        print(f'4. {name}: mean {st.mean(L):.0f} sd {st.pstdev(L):.0f} max {max(L)}')
        if name == 'proto_far3': assert max(L) <= 4096
print('ALL PASS')
