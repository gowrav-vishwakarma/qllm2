"""v14 self-tests (CPU, minutes). Run: .venv/bin/python -m v14.selftest

1. off == v13      V14LM with every v14 switch off is bit-identical to v13 LM.
2. B2 splits       held-out rule parameters appear only in the ``_new`` tasks,
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
    for test in (test_off_is_v13, test_b2_splits):
        print(f"{test.__name__}:")
        test()
    print("ALL PASS")


if __name__ == '__main__':
    main()
