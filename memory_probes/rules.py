"""In-context rule-learning tasks: does a model LEARN a rule, or only recall?

The recall probe (``behavioral.py``) asks for a value that is written in the
context. These tasks show a few examples of a hidden rule and then ask about
a NEW input, so the answer must come from applying the rule:

==============  =========================================  ==================
task            what the context shows                     answer
==============  =========================================  ==================
letter_shift    ``c -> e``, ``k -> m`` (hidden shift k)    query + k; never
                                                           in the context
number_step     ``7, 12, 17, 22,`` (hidden step d)         last + d; never
                                                           in the context
reverse3        ``cat dog sun -> sun dog cat``             first query word,
                                                           placed by the rule
label_category  ``apple is beta, dog is gold, pear is      the label of the
                beta`` (invented label per category)       query's category;
                                                           the query item is
                                                           never labelled
two_hop         ``Ana owns the dog. The dog is called      owner -> pet ->
                Rex.`` (facts shuffled)                    name, chained
==============  =========================================  ==================

Scoring is contrastive like the recall probe: the target must get the
highest next-token logit among ``candidate_count`` single-token candidates
(chance = 1 / candidate_count). Candidates include the tempting wrong
answers (copy the query, wrong step, other in-context labels), and
``in_context_chance`` reports the chance level among the answers that DO
appear in the context (relevant for label_category / two_hop / reverse3).

``build_rule_example(tokenizer, task, n_shots, seed)`` is the importable
generator (shared with the v14 micro-bench, which trains on these tasks);
``scripts/run_learning_probe.py`` is the CLI over trained checkpoints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from memory_probes.behavioral import score_candidate_logits  # noqa: F401  (re-export)

TASKS = ('letter_shift', 'number_step', 'reverse3', 'label_category', 'two_hop')

LETTERS = tuple('abcdefghijklmnopqrstuvwxyz')
WORDS = (
    'cat', 'dog', 'sun', 'car', 'book', 'tree', 'door', 'fish', 'bird', 'milk',
    'rain', 'road', 'ship', 'hat', 'cup', 'bed', 'box', 'key', 'pen', 'map',
    'star', 'moon', 'ring', 'coin', 'lamp', 'rock', 'leaf', 'bell', 'nest', 'drum',
)
CATEGORIES = {
    'fruit': ('apple', 'banana', 'grape', 'pear', 'lemon', 'mango', 'cherry', 'peach', 'plum'),
    'animal': ('horse', 'cow', 'lion', 'tiger', 'bear', 'wolf', 'sheep', 'goat', 'rabbit'),
    'color': ('red', 'blue', 'green', 'yellow', 'purple', 'pink', 'brown', 'black', 'white'),
    'country': ('France', 'Germany', 'Japan', 'China', 'India', 'Brazil', 'Spain', 'Italy', 'Canada'),
    'body': ('hand', 'foot', 'arm', 'leg', 'head', 'nose', 'ear', 'knee', 'neck'),
    'vehicle': ('car', 'bus', 'train', 'truck', 'plane', 'boat', 'bike', 'ship', 'van'),
}
LABELS = ('alpha', 'beta', 'gamma', 'delta', 'omega', 'lambda', 'gold', 'silver', 'prime', 'orbit', 'pixel', 'vector')
OWNERS = ('Ana', 'Ben', 'Cara', 'Dev', 'Eli', 'Fay', 'Gus', 'Hana', 'Ivo', 'Jo', 'Kit', 'Lena')
PETS = ('dog', 'cat', 'horse', 'goat', 'fox', 'rabbit', 'turtle', 'mouse', 'duck', 'pig')
PET_NAMES = ('Max', 'Rex', 'Sam', 'Leo', 'Tom', 'Bob', 'Jack', 'Lucy', 'Rosa', 'Milo', 'Duke', 'Ruby')


@dataclass(frozen=True)
class RuleExample:
    example_id: str
    task: str
    prompt_ids: List[int]
    prompt_text: str
    target_token_id: int
    target_text: str
    candidate_token_ids: List[int]
    candidate_texts: List[str]
    n_shots: int
    seed: int
    in_context_chance: float

    def to_dict(self, include_prompt_ids: bool = False) -> Dict[str, Any]:
        row = asdict(self)
        if not include_prompt_ids:
            row.pop('prompt_ids')
        return row


def _single(tokenizer, text: str) -> int | None:
    """Token id of ``text`` if it is exactly one token, else None."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def _pool(tokenizer, items: Iterable[str]) -> List[str]:
    """Items whose space-prefixed form is one token (the answer slot)."""
    return [w for w in items if _single(tokenizer, ' ' + w) is not None]


def _candidates(tokenizer, rng, target: str, tempting: Sequence[str],
                pool: Sequence[str], count: int) -> List[str]:
    """target + tempting wrong answers + random fill from pool, shuffled, distinct."""
    out = [target]
    for w in list(tempting) + list(rng.permutation(np.array(pool, dtype=object))):
        if len(out) == count:
            break
        if w not in out and _single(tokenizer, ' ' + w) is not None:
            out.append(w)
    if len(out) < count:
        raise ValueError(f'only {len(out)} distinct single-token candidates for {target!r}')
    return [out[i] for i in rng.permutation(len(out))]


def _letter_shift(tokenizer, rng, n_shots):
    k = int(rng.integers(1, 4))
    letters = _pool(tokenizer, LETTERS)
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
    tempting = [query] + [chr(ord(query) + j) for j in (1, 2, 3, 4) if j != k
                          and chr(ord(query) + j) in letters]
    return prompt, target, _candidates(tokenizer, rng, target, tempting, letters, 8), 1 / 8


def _number_step(tokenizer, rng, n_shots):
    d = int(rng.integers(2, 10))
    s = int(rng.integers(1, 40))
    terms = [s + i * d for i in range(max(n_shots, 2) + 1)]
    target = str(terms[-1])
    shown = terms[:-1]
    if any(_single(tokenizer, ' ' + str(t)) is None for t in terms):
        return None
    last = shown[-1]
    tempting = [str(last), str(last + 2 * d), str(last + d + 1), str(last + d - 1),
                str(last + 1), str(last + d + 2), str(last + d - 2)]
    pool = [str(n) for n in range(1, 200)]
    prompt = 'Number rule.\n' + ', '.join(str(t) for t in shown) + ','
    return prompt, target, _candidates(tokenizer, rng, target, tempting, pool, 8), 1 / 8


def _reverse3(tokenizer, rng, n_shots):
    words = _pool(tokenizer, WORDS)
    lines = []
    for _ in range(n_shots):
        a, b, c = rng.choice(words, size=3, replace=False)
        lines.append(f'{a} {b} {c} -> {c} {b} {a}\n')
    a, b, c = rng.choice(words, size=3, replace=False)
    prompt = f'Word rule.\n{"".join(lines)}{a} {b} {c} -> {c} {b}'
    target = str(a)
    tempting = [str(b), str(c)]
    return prompt, target, _candidates(tokenizer, rng, target, tempting, words, 8), 1 / 3


def _label_category(tokenizer, rng, n_shots):
    n_cat = max(2, min(4, n_shots // 2))
    cats = list(rng.choice(sorted(CATEGORIES), size=n_cat, replace=False))
    labels = list(rng.choice(_pool(tokenizer, LABELS), size=n_cat, replace=False))
    members = {c: list(rng.permutation(_pool(tokenizer, CATEGORIES[c]))) for c in cats}
    shots = [(cats[i % n_cat], i) for i in range(max(n_shots, n_cat))]   # every category shown
    used = {c: 0 for c in cats}
    pairs = []
    for c, _ in shots:
        pairs.append((members[c][used[c]], labels[cats.index(c)]))
        used[c] += 1
    pairs = [pairs[i] for i in rng.permutation(len(pairs))]
    qc = cats[int(rng.integers(n_cat))]
    query = members[qc][used[qc]]                          # never labelled in the context
    target = labels[cats.index(qc)]
    prompt = 'Label rule. ' + ', '.join(f'{w} is {lab}' for w, lab in pairs) + f', {query} is'
    tempting = [lab for lab in labels if lab != target]
    return prompt, target, _candidates(tokenizer, rng, target, tempting, LABELS, 8), 1 / n_cat


def _two_hop(tokenizer, rng, n_shots):
    n = max(2, n_shots)
    owners = list(rng.choice(OWNERS, size=n, replace=False))
    pets = list(rng.choice(PETS, size=n, replace=False))
    names = list(rng.choice(_pool(tokenizer, PET_NAMES), size=n, replace=False))
    facts = [f'{o} owns the {p}.' for o, p in zip(owners, pets)]
    facts += [f'The {p} is called {m}.' for p, m in zip(pets, names)]
    facts = [facts[i] for i in rng.permutation(len(facts))]
    qi = int(rng.integers(n))
    prompt = 'Story. ' + ' '.join(facts) + f" {owners[qi]}'s pet is called"
    target = str(names[qi])
    tempting = [str(m) for m in names if m != target]
    return prompt, target, _candidates(tokenizer, rng, target, tempting, PET_NAMES, 8), 1 / n


_BUILDERS = {
    'letter_shift': _letter_shift,
    'number_step': _number_step,
    'reverse3': _reverse3,
    'label_category': _label_category,
    'two_hop': _two_hop,
}


def build_rule_example(tokenizer, task: str, n_shots: int, seed: int,
                       candidate_count: int = 8) -> RuleExample:
    """One rule-learning prompt. Deterministic in (task, n_shots, seed).

    Returns prompt ids, the single-token target and ``candidate_count``
    single-token candidates (target included). Retries with a derived
    seed when a draw is invalid (answer leaked into the context, a number
    that is not one token), so every (task, n_shots, seed) yields a prompt.
    """
    if task not in _BUILDERS:
        raise ValueError(f'unknown task {task!r}; choose from {TASKS}')
    if candidate_count != 8:
        raise ValueError('candidate_count is fixed at 8 for these tasks')
    if n_shots < 1:
        raise ValueError('n_shots must be >= 1')
    for attempt in range(100):
        rng = np.random.default_rng([seed, n_shots, TASKS.index(task), attempt])
        built = _BUILDERS[task](tokenizer, rng, n_shots)
        if built is not None:
            break
    else:
        raise RuntimeError(f'could not build {task} n_shots={n_shots} seed={seed}')
    prompt, target, cands, chance = built
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    cand_ids = [_single(tokenizer, ' ' + c) for c in cands]
    return RuleExample(
        example_id=f'{task}_shots{n_shots}_seed{seed}',
        task=task,
        prompt_ids=prompt_ids,
        prompt_text=prompt,
        target_token_id=_single(tokenizer, ' ' + target),
        target_text=' ' + target,
        candidate_token_ids=cand_ids,
        candidate_texts=[' ' + c for c in cands],
        n_shots=n_shots,
        seed=seed,
        in_context_chance=float(chance),
    )


def build_rule_suite(tokenizer, tasks: Sequence[str] = TASKS,
                     shots: Sequence[int] = (2, 4, 8),
                     seeds: Sequence[int] = tuple(range(100))) -> List[RuleExample]:
    return [build_rule_example(tokenizer, t, n, s) for t in tasks for n in shots for s in seeds]


def aggregate(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Accuracy per (task, n_shots) over scored rows."""
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        if r.get('skipped'):
            continue
        groups.setdefault((r['task'], r['n_shots']), []).append(r)
    out = []
    for (task, n), rs in sorted(groups.items()):
        out.append({
            'task': task, 'n_shots': n, 'n': len(rs),
            'accuracy': float(np.mean([r['correct'] for r in rs])),
            'mean_target_margin': float(np.mean([r['target_margin'] for r in rs])),
            'chance': 1 / 8,
            'in_context_chance': float(np.mean([r['in_context_chance'] for r in rs])),
        })
    return out


__all__ = ['TASKS', 'RuleExample', 'build_rule_example', 'build_rule_suite',
           'aggregate', 'score_candidate_logits']
