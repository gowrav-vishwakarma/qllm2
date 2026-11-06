"""Packer hold for whole-window sources (v7/data.py StreamingTokenChunkDataset, 2026-11-07).
  1. a stream WITHOUT b2s_far / proto_far gives bit-identical chunks (input_ids + loss_mask) to the committed (HEAD) class
  2. yield: far docs (1,800-3,200 tokens) mixed into web docs at ~1.5 % of tokens, ANSWER_IN_WINDOW=1, answer weight 100:
     share of far docs whose answer keeps its weight (= the doc started in the window, so it sits whole in it), HEAD vs new
    python v13_sempty/tmp/test_packer_hold.py
"""
import importlib.util, os, random, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]
os.environ['ANSWER_IN_WINDOW'] = '1'
from v7 import data as NEW  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

src = subprocess.run(['git', 'show', 'HEAD:v7/data.py'], cwd=ROOT, capture_output=True, text=True, check=True).stdout
tmp = Path(tempfile.mkdtemp()) / 'old_v7_data.py'; tmp.write_text(src)
spec = importlib.util.spec_from_file_location('old_v7_data', tmp); OLD = importlib.util.module_from_spec(spec); spec.loader.exec_module(OLD)
tok = AutoTokenizer.from_pretrained('gpt2')
WORDS = 'the of and a to in is was for on that with as by at from his it an were are which this be or'.split()


def web(rng):
    return ' '.join(rng.choice(WORDS) for _ in range(rng.randint(40, 2400)))


def synth(rng):   # a short answer-marked doc (like b2s / recall) so the weighted path is exercised
    return 'Rule. ' + ' '.join(rng.choice(WORDS) for _ in range(rng.randint(10, 60))) + NEW._ans(' answer') + '\n'


def far(rng):     # a stub far doc of ~1,800-3,200 tokens ending in a marked answer
    return 'Map rule.' + ' '.join(rng.choice(WORDS) for _ in range(rng.randint(1750, 3100))) + NEW._ans(' 42') + '\n'


def stream(seed, with_far):
    rng = random.Random(seed)
    while True:
        r = rng.random()
        if with_far and r < 0.006:
            yield 'b2s_far', far(rng)
        elif r < 0.05:
            yield 'b2s', synth(rng)
        else:
            yield 'dclm', web(rng)


def chunks(mod, with_far, n):
    ds = mod.StreamingTokenChunkDataset(stream(7, with_far), tok, 4096, shuffle_buffer=0, token_counters={}, answer_weight=100.0)
    it = iter(ds)
    return [next(it) for _ in range(n)]


def test_identical():
    a = chunks(OLD, False, 400); b = chunks(NEW, False, 400)
    for x, y in zip(a, b):
        assert x['input_ids'].equal(y['input_ids']) and x['loss_mask'].equal(y['loss_mask']), 'stream without far docs changed'
    print('1. no far sources: 400 chunks (input_ids + loss_mask) identical to HEAD: OK')


def far_yield(mod, n=3000):
    cs = chunks(mod, True, n)
    head = tok.encode('Map rule.', add_special_tokens=False); ans = tok.encode(' 42', add_special_tokens=False)[0]
    starts = kept = 0
    for c in cs:
        ids = c['input_ids'].tolist(); lab = c['labels'].tolist(); lm = c['loss_mask'].tolist()
        starts += sum(1 for i in range(len(ids) - len(head)) if ids[i:i + len(head)] == head)
        kept += sum(1 for t, w in zip(lab, lm) if w > 50 and t == ans)   # the far doc's own answer token, weighted
    return starts, kept


def test_yield():
    for name, mod in (('HEAD (no hold)', OLD), ('new (hold)', NEW)):
        s, k = far_yield(mod)
        print(f'2. {name}: far docs started {s}, answers kept with weight {k}: yield {k / max(s, 1):.1%}')


if __name__ == '__main__':
    test_identical(); test_yield(); print('DONE')
