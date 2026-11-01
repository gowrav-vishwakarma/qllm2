"""Long-context in-context learning: a rule prompt with its examples spread
across a long document.

The examples of a rule prompt (memory_probes/rules.py or v14/rules_ood.py)
are pulled apart by blocks of held-out web text, so the prompt reaches a
target length L; the header stays first and the query stays last:

    <header> <ex 1> \\n\\n <filler> \\n\\n <ex 2> ... <ex k> \\n\\n <filler> \\n\\n <query>

The model must still infer the rule from examples that now sit thousands of
tokens apart (and far from the query). Target, candidates and scoring are
unchanged. L = 0 means the same examples with no filler (the baseline).
Filler is drawn deterministically from the pretraining holdout cache (web
text no model trained on), the same draw for every model.

Used by ``scripts/run_learning_probe.py --spread 0,1024,...``.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

import numpy as np

from memory_probes.rules import RuleExample

# task -> (header text, regex that splits the body into example/query units)
_FORMATS: Dict[str, Tuple[str, str]] = {
    'label_category': ('Label rule.', r'(?<=,)'),   # " red is alpha," ... " car is"
    'label_exact': ('Label rule.', r'(?<=,)'),      # label_category with an in-context query item
    'two_hop': ('Story.', r'(?<=\.)'),               # " The duck is called Sam." ... " Eli's pet is called"
    'affine_map': ('Map rule.', r'(?=\n)'),          # "\n21 -> 43" ... "\n29 ->"
    'shift_wide': ('Letter rule.', r'(?=\n)'),       # "\no -> u" ... "\nl ->"
    'letter_shift': ('Letter rule.', r'(?=\n)'),     # "\nc -> d" ... "\nj ->"
    'reverse3': ('Word rule.', r'(?=\n)'),           # "\nrock book nest -> nest book rock" ... "\nmilk bird rock -> rock bird"
    # Sequences have no query line: the last term is the query unit, so the
    # step must be read from terms a filler block (or more) away.
    'number_step': ('Number rule.', r'(?<=,)'),      # "\n36," " 45," ... " 99,"
    'step_wide': ('Number rule.', r'(?<=,)'),
}
SPREAD_TASKS = tuple(_FORMATS)
_SEP = '\n\n'


def _format(task: str) -> Tuple[str, str]:
    base = task[:-4] if task.endswith('_new') else task
    if base not in _FORMATS:
        raise ValueError(f'no spread format for task {task!r}; known: {SPREAD_TASKS}')
    return _FORMATS[base]


def split_units(example: RuleExample) -> Tuple[str, List[str]]:
    """(header, units) with ''.join([header] + units) == prompt_text; the
    last unit is the query."""
    header, pattern = _format(example.task)
    text = example.prompt_text
    if not text.startswith(header):
        raise ValueError(f'{example.example_id}: prompt does not start with {header!r}')
    units = [u for u in re.split(pattern, text[len(header):]) if u]
    if header + ''.join(units) != text or len(units) < 2:
        raise ValueError(f'{example.example_id}: could not split into units')
    return header, units


def filler_tokens(path: str, max_id: int = 50256) -> np.ndarray:
    """Plain-text token ids of the holdout cache (doc separators, chat ids and
    bare newline tokens dropped, so the filler reads as continuous prose)."""
    import torch
    blob = torch.load(path, weights_only=False)
    toks = blob['tokens'] if isinstance(blob, dict) else blob
    toks = np.asarray(toks.cpu().numpy() if hasattr(toks, 'cpu') else toks, dtype=np.int64).reshape(-1)
    toks = toks[toks < max_id]
    return toks[(toks != 198) & (toks != 628)]   # drop bare newline tokens (blank-line runs)


def spread_example(example: RuleExample, tokenizer, length: int,
                   filler: np.ndarray, tail: int = 0) -> RuleExample | None:
    """The example spread to ``length`` tokens (0 = no filler). The task is
    renamed ``<task>@<length>``. None if ``length`` is too short to fit.
    ``tail`` > 0 adds that many more filler tokens between the last example and
    the query (task ``<task>@<length>+t<tail>``), so the rule must be carried
    past any bounded attention window; tail = 0 gives the old prompts exactly."""
    header, units = split_units(example)
    head_ids = tokenizer.encode(header, add_special_tokens=False)
    unit_ids = [tokenizer.encode(u, add_special_tokens=False) for u in units]
    base = len(head_ids) + sum(len(u) for u in unit_ids)
    gaps = len(unit_ids) - 1
    sep = tokenizer.encode(_SEP, add_special_tokens=False)
    if length == 0:
        chunks = [[] for _ in range(gaps)]
    else:
        per_gap = (length - base) // gaps - 2 * len(sep)
        if per_gap < 16:
            return None
        rng = np.random.default_rng([example.seed, length, sum(map(ord, example.task))])
        starts = rng.integers(0, len(filler) - per_gap, size=gaps)
        chunks = [sep + filler[s:s + per_gap].tolist() + sep for s in starts]
    if tail:
        t0 = int(np.random.default_rng([example.seed, length, tail, 7]).integers(0, len(filler) - tail))
        tail_ids = filler[t0:t0 + tail].tolist()
        chunks[-1] = (chunks[-1][:-len(sep)] if chunks[-1] else sep) + tail_ids + sep
    ids = list(head_ids) + list(unit_ids[0])
    for chunk, unit in zip(chunks, unit_ids[1:]):
        ids += chunk + unit
    text_note = f'{header}{units[0]} ...[{gaps} filler blocks]... {units[-1]}' if length else header + ''.join(units)
    tag = f'{length}+t{tail}' if tail else f'{length}'
    return replace(example, example_id=f'{example.example_id}_L{tag}',
                   task=f'{example.task}@{tag}', prompt_ids=ids, prompt_text=text_note)


def spread_suite(examples: Sequence[RuleExample], tokenizer, lengths: Sequence[int],
                 filler: np.ndarray, tail: int = 0) -> List[RuleExample]:
    out = []
    for ex in examples:
        for L in lengths:
            s = spread_example(ex, tokenizer, L, filler, tail)
            if s is not None:
                out.append(s)
    return out


__all__ = ['SPREAD_TASKS', 'split_units', 'filler_tokens', 'spread_example', 'spread_suite']
