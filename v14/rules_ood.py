"""B1 tasks beyond the shared generator.

B1 run 2: rule tasks whose hidden parameter never appears in training.

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

B1 run 3 (v13 review, format a): ``label_marker``. Each item is two words,
a category MARKER and a stem; items that share a marker share a label, and
the query is a new stem with a marker seen in the context, so its label
must be found by a partial match inside the context, never recalled:

    Group rule. moss tavern is alpha, moss quill is alpha, brine cord is beta, moss lute is

The marker comes first or second (per episode), so copying a fixed slot does
not work. Words are GPT-2 single-token lowercase words split by a fixed hash
into four disjoint pools: train markers, eval markers, train stems, eval
stems. ``label_marker`` draws from the train pools; ``label_marker_new``
from the eval pools, which are never seen in training (every item unseen).

N7 bench (exact copy in a format never trained): ``repeat_copy`` trains
generic copying in one format ("Copy rule.\n w1 .. wk\n w1 .. wp ->
w_{p+1}"); ``label_exact`` asks for it in another ("Group rule. moss is
alpha, quill is beta, ... quill is" -> beta, every item with its own label),
and ``label_exact_new`` does the same with words never seen in training.

B2 (bar-3 analysis, EXPERIMENTS_V14 "N4"): inference must beat selection.
B1's unseen parameters ask for extrapolation from 3 trained values, where
picking one of 3 memorised tables is the cheap solution. B2 trains on MANY
parameter values and holds a few out from inside the range, so a model that
selects a memorised table lands next to the answer and only a model that
infers the parameter from the examples gets it (interpolation):

    shift_wide    letter shift, k in 1-12 without {5, 9}
    step_wide     arithmetic sequence, step d in 2-20 without {7, 13, 17}
    affine_map    x -> a*x + b, a in 1-3, b in 0-20 without {7, 13, 17}

Each ``<task>_new`` draws only the held-out values. Tempting wrong answers
are what the nearest trained parameter would give (k +- 1, d +- 1,
b +- 1, a +- 1), so falling back to a memorised neighbour scores 0, not
chance.
"""

from __future__ import annotations

import re
import zlib

import numpy as np

from memory_probes import rules as R

OOD_TASKS = ('letter_shift_ood', 'number_step_ood')
MARKER_TASKS = ('label_marker', 'label_marker_new')
B2_TASKS = ('shift_wide', 'step_wide', 'affine_map')
B2_NEW_TASKS = tuple(t + '_new' for t in B2_TASKS)
COPY_TASKS = ('repeat_copy', 'label_exact', 'label_exact_new')   # N7 bench (format-free exact copy)
EXTRA_TASKS = OOD_TASKS + MARKER_TASKS + B2_TASKS + B2_NEW_TASKS + COPY_TASKS   # append only: index seeds the rng
TRAIN_SHIFTS = (1, 2, 3)          # what rules._letter_shift draws
OOD_SHIFTS = (4, 5)
OOD_STEPS = (10, 11, 12, 13)      # rules._number_step draws 2-9
SHIFT_NEW = (5, 9)                                            # B2 held-out values
SHIFT_TRAIN = tuple(k for k in range(1, 13) if k not in SHIFT_NEW)
STEP_NEW = (7, 13, 17)
STEP_TRAIN = tuple(d for d in range(2, 21) if d not in STEP_NEW)
AFFINE_A = (1, 2, 3)
AFFINE_B_NEW = (7, 13, 17)
AFFINE_B_TRAIN = tuple(b for b in range(0, 21) if b not in AFFINE_B_NEW)


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


def _shift_wide(tokenizer, rng, n_shots, new=False):
    k = int(rng.choice(SHIFT_NEW if new else SHIFT_TRAIN))
    letters = R._pool(tokenizer, R.LETTERS)
    usable = [c for c in letters if chr(ord(c) + k) in letters]
    if len(usable) < n_shots + 1:
        return None
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
    near = [chr(ord(query) + k + j) for j in (-1, 1, -2, 2)]
    tempting = [c for c in near if c in letters and c != query] + [query]
    return prompt, target, R._candidates(tokenizer, rng, target, tempting, letters, 8), 1 / 8


def _step_wide(tokenizer, rng, n_shots, new=False):
    d = int(rng.choice(STEP_NEW if new else STEP_TRAIN))
    s = int(rng.integers(1, 40))
    terms = [s + i * d for i in range(max(n_shots, 2) + 1)]
    target = str(terms[-1])
    shown = terms[:-1]
    last = shown[-1]
    tempting = [str(last + d + j) for j in (-1, 1, -2, 2)] + [str(last), str(last + 2 * d)]
    pool = [str(n) for n in range(1, 260)]
    prompt = 'Number rule.\n' + ', '.join(str(t) for t in shown) + ','
    return prompt, target, R._candidates(tokenizer, rng, target, tempting, pool, 8), 1 / 8


def _affine_map(tokenizer, rng, n_shots, new=False):
    a = int(rng.choice(AFFINE_A))
    b = int(rng.choice(AFFINE_B_NEW if new else AFFINE_B_TRAIN))
    xs = [int(x) for x in rng.choice(np.arange(1, 31), size=max(n_shots, 2) + 1, replace=False)]
    xq = xs[-1]
    target = str(a * xq + b)
    lines = ''.join(f'{x} -> {a * x + b}\n' for x in xs[:-1])
    prompt = f'Map rule.\n{lines}{xq} ->'
    tempting = [str(a * xq + b + j) for j in (-1, 1)] + \
               [str(a2 * xq + b) for a2 in (a - 1, a + 1) if a2 >= 1] + [str(xq + b), str(a * xq)]
    pool = [str(n) for n in range(0, 260)]
    return prompt, target, R._candidates(tokenizer, rng, target, tempting, pool, 8), 1 / 8


def _repeat_copy(tokenizer, rng, n_shots):
    pools = _marker_pools(tokenizer)
    k = n_shots + 3
    words = list(rng.choice(pools['train_stems'], size=k, replace=False))
    p = int(rng.integers(1, k))                                  # shown prefix of the repeat
    prompt = 'Copy rule.\n' + ' '.join(words) + '\n' + ' '.join(words[:p])
    target = str(words[p])
    tempting = [w for w in words if w != target]
    return prompt, target, R._candidates(tokenizer, rng, target, tempting, pools['train_stems'], 8), 1 / 8


def _label_exact(tokenizer, rng, n_shots, new=False):
    pools = _marker_pools(tokenizer)
    n_items = max(2, n_shots)
    items = list(rng.choice(pools['eval_stems' if new else 'train_stems'], size=n_items, replace=False))
    labels = list(rng.choice(R._pool(tokenizer, R.LABELS), size=n_items, replace=False))
    qi = int(rng.integers(n_items))
    prompt = 'Group rule. ' + ', '.join(f'{w} is {lab}' for w, lab in zip(items, labels)) \
        + f', {items[qi]} is'
    target = str(labels[qi])
    tempting = [str(lab) for lab in labels if lab != target]
    return prompt, target, R._candidates(tokenizer, rng, target, tempting, R.LABELS, 8), 1 / n_items


_MARKER_POOLS: dict = {}


def _marker_pools(tokenizer):
    """Four disjoint pools of single-token words, split by crc32(word) % 4."""
    base = getattr(tokenizer, 'tok', tokenizer)          # CachedTokenizer or HF tokenizer
    key = id(base)
    if key not in _MARKER_POOLS:
        words = sorted(t[1:] for t in base.get_vocab() if re.fullmatch(r'Ġ[a-z]{4,7}', t))
        split = {i: [w for w in words if zlib.crc32(w.encode()) % 4 == i] for i in range(4)}
        _MARKER_POOLS[key] = {'train_markers': split[0], 'eval_markers': split[1],
                              'train_stems': split[2], 'eval_stems': split[3]}
    return _MARKER_POOLS[key]


def _label_marker(tokenizer, rng, n_shots, new=False):
    pools = _marker_pools(tokenizer)
    side = 'eval' if new else 'train'
    n_cat = max(2, min(4, n_shots // 2))
    markers = list(rng.choice(pools[f'{side}_markers'], size=n_cat, replace=False))
    labels = list(rng.choice(R._pool(tokenizer, R.LABELS), size=n_cat, replace=False))
    n_items = max(n_shots, n_cat)                         # every marker shown at least once
    stems = list(rng.choice(pools[f'{side}_stems'], size=n_items + 1, replace=False))
    marker_first = bool(rng.integers(2))
    item = (lambda m, s: f'{m} {s}') if marker_first else (lambda m, s: f'{s} {m}')
    pairs = [(item(markers[i % n_cat], stems[i]), labels[i % n_cat]) for i in range(n_items)]
    pairs = [pairs[i] for i in rng.permutation(len(pairs))]
    qi = int(rng.integers(n_cat))
    query = item(markers[qi], stems[-1])                  # a new stem: never labelled
    target = str(labels[qi])
    prompt = 'Group rule. ' + ', '.join(f'{w} is {lab}' for w, lab in pairs) + f', {query} is'
    tempting = [str(lab) for lab in labels if lab != target]
    return prompt, target, R._candidates(tokenizer, rng, target, tempting, R.LABELS, 8), 1 / n_cat


_BUILDERS = {'letter_shift_ood': _letter_shift_ood, 'number_step_ood': _number_step_ood,
             'label_marker': _label_marker,
             'label_marker_new': lambda tok, rng, n: _label_marker(tok, rng, n, new=True),
             'shift_wide': _shift_wide, 'step_wide': _step_wide, 'affine_map': _affine_map,
             'shift_wide_new': lambda tok, rng, n: _shift_wide(tok, rng, n, new=True),
             'step_wide_new': lambda tok, rng, n: _step_wide(tok, rng, n, new=True),
             'affine_map_new': lambda tok, rng, n: _affine_map(tok, rng, n, new=True),
             'repeat_copy': _repeat_copy, 'label_exact': _label_exact,
             'label_exact_new': lambda tok, rng, n: _label_exact(tok, rng, n, new=True)}


def build_example(tokenizer, task: str, n_shots: int, seed: int) -> R.RuleExample:
    """Any B1 task: the shared generator's, or one of EXTRA_TASKS. Deterministic
    in (task, n_shots, seed); retries a draw that leaks its answer."""
    if task not in _BUILDERS:
        return R.build_rule_example(tokenizer, task, n_shots, seed)
    for attempt in range(100):
        rng = np.random.default_rng([seed, n_shots, 100 + EXTRA_TASKS.index(task), attempt])
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
