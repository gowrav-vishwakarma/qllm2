"""B1 run 2: rule tasks whose hidden parameter never appears in training.

The shared generator (``memory_probes/rules.py``) draws letter_shift's shift
k from 1-3 and number_step's step d from 2-9, and B1 trains on exactly
those. B1 run 1 then scored 1.00 for every arm: the rules were learned in
the weights, and the context only had to pick k or d from a few known
values. These variants draw k from 4-5 and d from 10-13, so a model that
only knows the trained parameters cannot answer. It has to infer the rule
from the examples in the context.

Same prompt format and 8 candidates as the shared tasks. The tempting wrong
answers lead with what the training prior would say: the trained shifts for
letters, and last + 1 / last + 9 for numbers. A model that falls back on
its prior scores below chance, which shows that it fell back.
"""

from __future__ import annotations

import numpy as np

from memory_probes import rules as R

OOD_TASKS = ('letter_shift_ood', 'number_step_ood')
TRAIN_SHIFTS = (1, 2, 3)          # what rules._letter_shift draws
OOD_SHIFTS = (4, 5)
OOD_STEPS = (10, 11, 12, 13)      # rules._number_step draws 2-9


def _letter_shift_ood(tokenizer, rng, n_shots):
    k = int(rng.choice(OOD_SHIFTS))
    letters = R._pool(tokenizer, R.LETTERS)
    usable = [c for c in letters if chr(ord(c) + k) in letters]
    picks = list(rng.choice(usable, size=n_shots + 1, replace=False))
    query = picks[-1]
    target = chr(ord(query) + k)
    shown = set()
    for x in picks[:-1]:
        shown |= {x, chr(ord(x) + k)}
    if target in shown:                                    # keep the answer out of the context
        return None
    lines = ''.join(f'{x} -> {chr(ord(x) + k)}\n' for x in picks[:-1])
    prompt = f'Letter rule.\n{lines}{query} ->'
    prior = [chr(ord(query) + j) for j in TRAIN_SHIFTS + tuple(s for s in OOD_SHIFTS if s != k)]
    tempting = [c for c in prior if c in letters] + [query]
    return prompt, target, R._candidates(tokenizer, rng, target, tempting, letters, 8), 1 / 8


def _number_step_ood(tokenizer, rng, n_shots):
    d = int(rng.choice(OOD_STEPS))
    s = int(rng.integers(1, 40))
    terms = [s + i * d for i in range(max(n_shots, 2) + 1)]
    target = str(terms[-1])
    shown = terms[:-1]
    if any(R._single(tokenizer, ' ' + str(t)) is None for t in terms):
        return None
    last = shown[-1]
    tempting = [str(last + 1), str(last + 9), str(last), str(last + d + 1),
                str(last + d - 1), str(last + d + 2), str(last + d - 2)]
    pool = [str(n) for n in range(1, 300)]
    prompt = 'Number rule.\n' + ', '.join(str(t) for t in shown) + ','
    return prompt, target, R._candidates(tokenizer, rng, target, tempting, pool, 8), 1 / 8


_BUILDERS = {'letter_shift_ood': _letter_shift_ood, 'number_step_ood': _number_step_ood}


def build_example(tokenizer, task: str, n_shots: int, seed: int) -> R.RuleExample:
    """Any B1 task: the shared generator's, or one of OOD_TASKS. Deterministic
    in (task, n_shots, seed); retries a draw that leaks its answer."""
    if task not in _BUILDERS:
        return R.build_rule_example(tokenizer, task, n_shots, seed)
    for attempt in range(100):
        rng = np.random.default_rng([seed, n_shots, 100 + OOD_TASKS.index(task), attempt])
        built = _BUILDERS[task](tokenizer, rng, n_shots)
        if built is not None:
            break
    else:
        raise RuntimeError(f'could not build {task} n_shots={n_shots} seed={seed}')
    prompt, target, cands, chance = built
    return R.RuleExample(
        example_id=f'{task}_shots{n_shots}_seed{seed}', task=task,
        prompt_ids=tokenizer.encode(prompt, add_special_tokens=False), prompt_text=prompt,
        target_token_id=R._single(tokenizer, ' ' + target), target_text=' ' + target,
        candidate_token_ids=[R._single(tokenizer, ' ' + c) for c in cands],
        candidate_texts=[' ' + c for c in cands],
        n_shots=n_shots, seed=seed, in_context_chance=float(chance))
