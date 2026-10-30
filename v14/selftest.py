"""v14 self-tests (CPU, minutes). Run: .venv/bin/python -m v14.selftest

1. off == v13      V14LM with every v14 switch off is bit-identical to v13 LM.
2. N7 lookup     parallel n-gram lookup == a brute-force bounded table (with
                   forced slot collisions) == token-by-token decode; the decode
                   state after a prefix matches; "X is Y" templates do not collide.
3. N7 start       N7 on (zero-init readers) is bit-identical to N7 off.
4. N7 decode      full-sequence logits == token-by-token decode with live readers.
5. N7 learns      every reader parameter gets a gradient.
6. B2 splits       held-out rule parameters appear only in the ``_new`` tasks,
                   trained ones only in the base tasks; answers stay out of
                   the context; the nearest-parameter decoy is a candidate.
(Tests of removed units: N6 operator memory at 78e4354; tied echo at 5100fd7;
N1 at 9e0a3fd.)
"""

from __future__ import annotations

import torch

from v13_sempty.model import LM
from v14.config import get_config
from v14.units import V14LM

ATOL = 2e-4


def _cfg(**kw):
    base = dict(chunk_size=32, delta=True, chrono=True, out_gate=True, echo=True,
                vocab_size=512, max_seq_len=256)
    base.update(kw)
    return get_config('tiny_real', **base)


def _logits(model, ids):
    with torch.no_grad():
        return model(ids)[0]


def test_off_is_v13():
    cfg = _cfg()
    torch.manual_seed(0)
    ref = LM(cfg).eval()
    torch.manual_seed(0)
    v14 = V14LM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 40))
    diff = (_logits(ref, ids) - _logits(v14, ids)).abs().max().item()
    assert diff == 0.0, f"v14 (switches off) != v13: {diff:.3e}"
    print(f"  off == v13: max |diff| = {diff:.1e}")


def _decode_diff(model, seq_len, vocab=512):
    ids = torch.randint(0, vocab, (2, seq_len))
    full = _logits(model, ids)
    states, steps = None, []
    with torch.no_grad():
        for t in range(seq_len):
            lt, states, _ = model(ids[:, t:t + 1], states=states, step_offset=t)
            steps.append(lt)
    return (full - torch.cat(steps, dim=1)).abs().max().item()


def _ref_ngram(row, n, S):
    """Brute-force bounded table: dict slot -> (fp, value), last write wins."""
    from v14.ngram import _FP_MULT, _SLOT_MULT, _hash
    def key(g):
        g = torch.tensor(g, dtype=torch.int64)
        return int(torch.remainder(_hash(g, _SLOT_MULT), S)), int(_hash(g, _FP_MULT))
    table, vals, founds = {}, [], []
    for t in range(len(row)):
        if t >= n:
            s, f = key(row[t - n:t])
            table[s] = (f, row[t])
        hit = False
        if t >= n - 1:
            s, f = key(row[t - n + 1:t + 1])
            e = table.get(s)
            hit = e is not None and e[0] == f
        vals.append(table[s][1] if hit else 0)
        founds.append(hit)
    return vals, founds


def test_n7_lookup():
    """Parallel == brute force == step decode, with and without slot collisions."""
    from v14.ngram import empty_state, lookup_parallel, lookup_step, state_from_prefix
    g = torch.Generator().manual_seed(3)
    for n, S, vocab in ((2, 7, 6), (2, 1 << 12, 6), (3, 5, 4), (3, 1 << 12, 20), (4, 64, 3)):
        ids = torch.randint(0, vocab, (3, 90), generator=g)
        v, f, _ = lookup_parallel(ids, n, S)
        st = empty_state(3, n, S, ids.device)
        for b in range(3):
            rv, rf = _ref_ngram(ids[b].tolist(), n, S)
            assert v[b].tolist() == rv and f[b].tolist() == rf, f"N7 parallel != reference (n={n}, S={S})"
        for t in range(90):
            vt, ft, st = lookup_step(st, ids[:, t], n, S)
            assert torch.equal(vt, v[:, t]) and torch.equal(ft, f[:, t]), f"N7 step != parallel at t={t}"
        pre = state_from_prefix(ids, n, S)
        for k in ('fp', 'val', 'used', 'tail'):
            same = torch.equal(pre[k], st[k]) if k != 'fp' and k != 'val' else \
                torch.equal(pre[k][pre['used']], st[k][st['used']])
            assert same, f"N7 prefix state != step state ({k}, n={n}, S={S})"
    # collision check (v13 diagnosis): "X is Y" items share the template "is";
    # every repeated item must return its own label, not the latest one
    xs, ys = list(range(100, 140)), list(range(200, 240))
    is_ = 7
    row = sum(([x, is_, y] for x, y in zip(xs, ys)), []) + sum(([x, is_] for x in xs[::-1]), [])
    v, f, _ = lookup_parallel(torch.tensor([row]), 2, 1 << 16)
    tail = len(xs) * 3
    got = [int(v[0, tail + 2 * i + 1]) for i in range(len(xs))]
    assert got == ys[::-1] and bool(f[0, tail + 1::2].all()), "N7: template keys collided"
    print(f"  N7: parallel == brute force == step decode (5 settings, forced collisions); "
          f"prefix state matches; 40 'X is Y' items, 0 template collisions")


def _n7(seed=0, **kw):
    torch.manual_seed(seed)
    return V14LM(_cfg(ngram_orders=(2, 3), ngram_layers=(0, -1), **kw)).eval()


def test_n7_off_at_init():
    torch.manual_seed(0)
    off = V14LM(_cfg()).eval()
    on = _n7(0)
    ids = torch.randint(0, 512, (2, 60))
    ids[:, 30:] = ids[:, :30]                                    # repeats, so lookups fire
    diff = (_logits(off, ids) - _logits(on, ids)).abs().max().item()
    assert diff == 0.0, f"N7 at init != N7 off: {diff:.3e}"
    print(f"  N7 on at init == off: max |diff| = {diff:.1e}")


def test_n7_decode_matches_prefill():
    model = _n7(1)
    with torch.no_grad():
        for r in model.ngram_read.values():
            r.proj.normal_(std=0.05)
            r.found_bias.normal_(std=0.05)
            r.gate_w.normal_(std=0.5)
    torch.manual_seed(2)
    diff = _decode_diff(model, 150, vocab=12)                    # small vocab: many repeats
    assert diff < ATOL, f"N7 prefill != decode: {diff:.3e}"
    print(f"  N7 decode == prefill (150 tokens, live readers): max |diff| = {diff:.2e}")


def test_n7_learns():
    model = _n7(4).train()
    with torch.no_grad():
        for r in model.ngram_read.values():
            r.proj.normal_(std=0.05)
    ids = torch.randint(0, 20, (2, 64))
    logits = model(ids)[0]
    torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)).backward()
    for name, p in model.ngram_read.named_parameters():
        assert p.grad is not None and p.grad.abs().max().item() > 0, f"N7: no gradient on {name}"
    print("  N7: every reader parameter gets a gradient")


def _b2_param(task, text):
    """The hidden parameter of a B2 prompt (k, d, or the affine b)."""
    lines = text.split('\n')[1:]
    if task.startswith('shift'):
        x, y = lines[0].split(' -> ')
        return ord(y) - ord(x)
    if task.startswith('step'):
        t = [int(v) for v in lines[0].rstrip(',').split(', ')]
        return t[1] - t[0]
    (x1, y1), (x2, y2) = [tuple(int(v) for v in ln.split(' -> ')) for ln in lines[:2]]
    a = (y2 - y1) // (x2 - x1)
    return y1 - a * x1


def test_b2_splits(seeds=40):
    """B2: no held-out parameter leaks into training tasks and vice versa."""
    from transformers import AutoTokenizer
    from v14 import rules_ood as O
    tok = AutoTokenizer.from_pretrained('gpt2')
    split = {'shift_wide': (O.SHIFT_TRAIN, O.SHIFT_NEW), 'step_wide': (O.STEP_TRAIN, O.STEP_NEW),
             'affine_map': (O.AFFINE_B_TRAIN, O.AFFINE_B_NEW)}
    n = 0
    for base, (train, held) in split.items():
        assert not set(train) & set(held), f"{base}: train and held-out overlap"
        for task, allowed in ((base, train), (base + '_new', held)):
            for shots in (1, 2, 4, 8):
                for seed in range(seeds):
                    ex = O.build_example(tok, task, shots, seed)
                    assert _b2_param(task, ex.prompt_text) in allowed, f"{task}: parameter leak"
                    c = ex.candidate_token_ids
                    assert len(set(c)) == 8 and ex.target_token_id in c, f"{task}: candidates"
                    shown = [ln.split(' -> ')[1] for ln in ex.prompt_text.split('\n')[1:]
                             if ' -> ' in ln and ln.split(' -> ')[1]]
                    assert ex.target_text.strip() not in shown, f"{task}: answer in context"
                    n += 1
    print(f"  B2: {n} episodes, splits disjoint, no parameter leak, answers out of context")


def main():
    torch.set_num_threads(8)
    for test in (test_off_is_v13, test_n7_lookup, test_n7_off_at_init,
                 test_n7_decode_matches_prefill, test_n7_learns, test_b2_splits):
        print(f"{test.__name__}:")
        test()
    print("ALL PASS")


if __name__ == '__main__':
    main()
