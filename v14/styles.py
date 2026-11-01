"""Natural test variants (user via Lead, 2026-11-01 10:00 IST: "test like normal, not toy: recall must also be asked
with its synonyms and in different styles"). Importable pools for v16/bench.py's --qstyle varied and any runner.

    STATEMENTS        how a fact is stated in the stream, {k} = key, {v} = value (one is sampled per fact)
    QUESTIONS         how a key-value fact is asked; each ends exactly where the answer starts
    FIELD_QUESTIONS   how one field of a tool / record row is asked ({k} = the record's key, {f} = the field)
    SYN_PAIRS         82 synonym phrase pairs sharing no word (v14/syn_cell.py): store one side, ask the other
    sample(rng, pool) -> (name, template)

The first entry of each pool is bench's template, kept as the secondary / control style.
"""
from __future__ import annotations

from v14.syn_cell import PAIRS as SYN_PAIRS  # noqa: F401  (re-exported)

STATEMENTS = (
    ('template', " Definition: {k} refers to {v}."),
    ('aside', " By the way, {k} is just another name for {v}."),
    ('local', " People around here call {v} by the name {k}."),
    ('turns_out', " It turns out that {k} is what everyone means when they say {v}."),
    ('records', " In the old records, {k} stands for {v}."),
    ('note', " Note to self: {k} means {v}."),
    ('dialogue', " \"What is {k}?\" she asked. \"It is {v},\" he said."),
    ('glossary', " Glossary entry - {k}: {v}."),
)

QUESTIONS = (
    ('template', " Query: {k} refers to"),
    ('direct', " Question: what does {k} refer to? Answer:"),
    ('forgot', " I forgot what {k} means. It means"),
    ('chat', " hey, {k} was what again? It was"),
    ('remind', " Remind me: {k} is the one that refers to"),
    ('qa', " Q: What is {k}? A:"),
    ('stand_for', " Question: what does {k} stand for? Answer:"),
    ('recall', " Earlier we said that {k} meant"),
)

FIELD_QUESTIONS = (
    ('template', " Query: job: {k}; {f}:"),
    ('direct', " Question: what is the {f} of job {k}? Answer:"),
    ('which', " Which {f} does job {k} have? It has"),
    ('chat', " quick one - job {k}, what was its {f} again? It was"),
    ('report', " Report: for job {k}, the {f} is"),
    ('qa', " Q: {k}'s {f}? A:"),
)


def sample(rng, pool):
    """(name, template) drawn uniformly from a pool (bench's rng keeps docs reproducible)."""
    return pool[rng.randrange(len(pool))]


def _check():
    """every template formats; questions end before the answer (no {v} in them); statements hold {k} and {v}."""
    for name, t in STATEMENTS:
        s = t.format(k='K1', v='V1')
        assert 'K1' in s and 'V1' in s, name
    for name, t in QUESTIONS:
        s = t.format(k='K1')
        assert 'K1' in s and '{' not in s and not s.rstrip().endswith('.'), name
    for name, t in FIELD_QUESTIONS:
        s = t.format(k='K1', f='F1')
        assert 'K1' in s and 'F1' in s and not s.rstrip().endswith('.'), name
    return len(STATEMENTS), len(QUESTIONS), len(FIELD_QUESTIONS), len(SYN_PAIRS)


if __name__ == '__main__':          # run as a module from the repo root: python -m v14.styles
    print('statements, questions, field questions, synonym pairs:', _check())
