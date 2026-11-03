"""Run one of the real probe scripts unchanged, recording every scored item (prompt + candidates + pick).
  python rec.py <worktree> <module> <out.jsonl> <script args...>
The script's source is executed in its own namespace with score_candidate_logits (and, for the skill pack,
score()) wrapped to log each call; nothing else changes."""
import json, sys, types
from pathlib import Path
W, MOD, OUT = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
sys.path.insert(0, str(W)); sys.path.insert(0, str(W / 'scripts'))
fh = open(OUT, 'w'); st = dict(tok=None, mute=False, arm=None)

def short(ids):
    t = st['tok'].decode(ids)
    if len(t) <= 1600: return t
    return t[:700] + f'\n\n[... {len(ids)} tokens in total; the long filler text in the middle is skipped here ...]\n\n' + t[-700:]

def log(ex, ids, cand, ok):
    tok = st['tok']; ci = list(ex.candidate_token_ids); pick = max(range(len(cand)), key=lambda c: cand[c])
    d = ex.to_dict() if hasattr(ex, 'to_dict') else {}
    meta = {k: v for k, v in d.items() if not isinstance(v, (list, dict)) and k not in ('target_token_id',)}
    fh.write(json.dumps(dict(task=meta.pop('task', None), arm=st['arm'], prompt=short(list(ids)), n_tokens=len(ids),
                             target=tok.decode([ex.target_token_id]), pick=tok.decode([ci[pick]]),
                             candidates=[tok.decode([c]) for c in ci], ok=bool(ok), meta=meta)) + '\n')

src = (W / 'scripts' / f'{MOD}.py').read_text()
top, _, main_block = src.partition("\nif __name__ == '__main__':")
ns = dict(__name__=MOD, __file__=str(W / 'scripts' / f'{MOD}.py'))
exec(compile(top, ns['__file__'], 'exec'), ns)

orig = ns['score_candidate_logits']
def patched(ex, cand):
    r = orig(ex, cand)
    if not st['mute']: log(ex, ex.prompt_ids, cand, r.get('correct'))
    return r
ns['score_candidate_logits'] = patched

o_load = ns['_load_v13_sempty']
def load(*a, **k):
    m, t, c = o_load(*a, **k); st['tok'] = t; return m, t, c
ns['_load_v13_sempty'] = load

if MOD == 'run_skillpack_probe':      # log only the main arms through score(); tag pack-in-state vs re-read
    st['mute'] = True; o_score = ns['score']; seen = {}
    def sc(body, ex, ids):
        r = o_score(body, ex, ids)
        k = id(ex); n = seen.get(k, 0); seen[k] = n + 1
        st['arm'] = 'pack read once at start (held in state)' if n == 0 else 'pack placed right before the question (re-read)'
        lg = body.last_logits(ids)[ex.candidate_token_ids].float().cpu().tolist()
        log(ex, ids, lg, r[0]); st['arm'] = None
        return r
    ns['score'] = sc

sys.argv = [MOD] + sys.argv[4:]
if 'main' in ns and MOD != 'run_skillpack_probe':
    ns['main']()
else:
    exec(compile('if True:' + main_block, ns['__file__'], 'exec'), ns)
fh.close()
