"""Real reasoning / aptitude Q&A documents for a "reason first" curriculum.

User idea (2026-10-29): before the bulk web data, teach the model a little
grammar, then real aptitude-test style questions with worked answers, then
chat with complex relations, so it learns to reason first. Real datasets
only (no synthetic generators), TRAIN splits only, so every test split stays
clean for later evaluation.

Each item becomes one plain-text document: question, options, the worked
solution when the dataset has one, then the answer. Plain LM loss over the
whole document (no answer weighting): the model learns the reasoning text,
not only the final letter.

Sources (all hub data files; no dataset scripts are executed):
  deepmind/aqua_rat     GMAT/GRE algebra word problems + rationale   97,467
  openai/gsm8k          grade-school math, step-by-step solution      7,473
  datatune/LogiQA2.0    civil-service exam logical reasoning         12,567
  metaeval/reclor       LSAT/GMAT logical reasoning                   4,638
  ehovy/race            English-exam reading comprehension (grouped
                        per passage)                                 87,866 q
  allenai/ai2_arc       grade-school science (Easy + Challenge)       3,370
  tau/commonsense_qa    commonsense multiple choice                   9,741
  allenai/openbookqa    science facts + reasoning                     4,957
  allenai/sciq          science with support passage                 11,679
  allenai/qasc          two-fact composition (facts shown)            8,134
  google/boolq          yes/no over a passage                         9,427
  ChilleD/StrategyQA    implicit multi-step yes/no (facts shown)      1,603
  allenai/winogrande    commonsense coreference (xl)                 40,398
  Rowan/hellaswag       commonsense continuation                     39,905
"""

from __future__ import annotations

import ast
import json
import random
import re
from typing import Callable, Dict, Iterator, List, Optional

LETTERS = 'ABCDEFGHIJ'


def _hub(repo: str, path: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, path, repo_type='dataset')


def _parquet(repo: str, path: str):
    import pandas as pd
    return pd.read_parquet(_hub(repo, path)).to_dict('records')


def _as_list(x) -> List[str]:
    """Options arrive as numpy arrays, python lists, or their string repr."""
    if isinstance(x, str):
        s = x.strip()
        try:
            v = ast.literal_eval(s)
            return [str(o) for o in v]
        except (ValueError, SyntaxError):
            return re.findall(r"'((?:[^'\\]|\\.)*)'", s) or [s]
    return [str(o) for o in list(x)]


def _mc(question: str, options: List[str], answer_idx: int,
        passage: str = '', solution: str = '') -> str:
    opts = '\n'.join(f'{LETTERS[i]}) {o.strip()}' for i, o in enumerate(options))
    head = f'Passage: {passage.strip()}\n\n' if passage else ''
    sol = f'Solution: {solution.strip()}\n' if solution else ''
    return (f'{head}Question: {question.strip()}\n{opts}\n{sol}'
            f'Answer: {LETTERS[answer_idx]}) {options[answer_idx].strip()}')


def _aqua() -> List[str]:
    out = []
    for r in _parquet('deepmind/aqua_rat', 'raw/train-00000-of-00001.parquet'):
        opts = [re.sub(r'^[A-E]\)\s*', '', o) for o in _as_list(r['options'])]
        idx = LETTERS.index(r['correct'].strip())
        if idx < len(opts):
            out.append(_mc(r['question'], opts, idx, solution=r['rationale']))
    return out


def _gsm8k() -> List[str]:
    out = []
    for r in _parquet('openai/gsm8k', 'main/train-00000-of-00001.parquet'):
        sol, _, ans = r['answer'].rpartition('####')
        sol = re.sub(r'<<[^>]*>>', '', sol).strip()             # drop calculator annotations
        out.append(f"Question: {r['question'].strip()}\nSolution: {sol}\nAnswer: {ans.strip()}")
    return out


def _logiqa() -> List[str]:
    out = []
    for line in open(_hub('datatune/LogiQA2.0', 'MRC/train.txt')):
        if not line.strip():
            continue
        r = json.loads(line)
        out.append(_mc(r['question'], _as_list(r['options']), int(r['answer']), passage=r['text']))
    return out


def _reclor() -> List[str]:
    rows = json.load(open(_hub('metaeval/reclor', 'train.json')))
    return [_mc(r['question'], _as_list(r['answers']), int(r['label']), passage=r['context'])
            for r in rows]


def _race() -> List[str]:
    groups: Dict[str, List[dict]] = {}
    order: List[str] = []
    for r in _parquet('ehovy/race', 'all/train-00000-of-00001.parquet'):
        if r['article'] not in groups:
            groups[r['article']] = []
            order.append(r['article'])
        groups[r['article']].append(r)
    out = []
    for art in order:
        qs = []
        for r in groups[art]:
            opts = _as_list(r['options'])
            i = LETTERS.index(r['answer'].strip())
            body = '\n'.join(f'{LETTERS[j]}) {o.strip()}' for j, o in enumerate(opts))
            qs.append(f"Question: {r['question'].strip()}\n{body}\nAnswer: {LETTERS[i]}) {opts[i].strip()}")
        out.append(f'Passage: {art.strip()}\n\n' + '\n\n'.join(qs))
    return out


def _choices_mc(rows, qkey='question') -> List[str]:
    out = []
    for r in rows:
        ch = r['choices']
        texts, labels = _as_list(ch['text']), _as_list(ch['label'])
        if r['answerKey'] not in labels:
            continue
        out.append(_mc(r[qkey], texts, labels.index(r['answerKey'])))
    return out


def _arc() -> List[str]:
    rows = (_parquet('allenai/ai2_arc', 'ARC-Easy/train-00000-of-00001.parquet')
            + _parquet('allenai/ai2_arc', 'ARC-Challenge/train-00000-of-00001.parquet'))
    return _choices_mc(rows)


def _csqa() -> List[str]:
    return _choices_mc(_parquet('tau/commonsense_qa', 'data/train-00000-of-00001.parquet'))


def _obqa() -> List[str]:
    return _choices_mc(_parquet('allenai/openbookqa', 'main/train-00000-of-00001.parquet'),
                       qkey='question_stem')


def _sciq() -> List[str]:
    out = []
    rng = random.Random(1234)
    for r in _parquet('allenai/sciq', 'data/train-00000-of-00001.parquet'):
        opts = [r['correct_answer'], r['distractor1'], r['distractor2'], r['distractor3']]
        perm = list(range(4))
        rng.shuffle(perm)
        out.append(_mc(r['question'], [opts[i] for i in perm], perm.index(0), passage=r['support'] or ''))
    return out


def _qasc() -> List[str]:
    out = []
    for r in _parquet('allenai/qasc', 'data/train-00000-of-00001.parquet'):
        texts, labels = _as_list(r['choices']['text']), _as_list(r['choices']['label'])
        if r['answerKey'] not in labels:
            continue
        facts = f"Fact 1: {r['fact1']}\nFact 2: {r['fact2']}\nSo: {r['combinedfact']}"
        out.append(_mc(r['question'], texts, labels.index(r['answerKey']), passage=facts))
    return out


def _boolq() -> List[str]:
    return [f"Passage: {r['passage'].strip()}\n\nQuestion: {r['question'].strip()}?\n"
            f"Answer: {'yes' if str(r['answer']) == 'True' else 'no'}"
            for r in _parquet('google/boolq', 'data/train-00000-of-00001.parquet')]


def _strategyqa() -> List[str]:
    rows = _parquet('ChilleD/StrategyQA', 'data/train-00000-of-00001-506370352f622815.parquet')
    return [f"Question: {r['question'].strip()}\nFacts: {str(r['facts']).strip()}\n"
            f"Answer: {'yes' if str(r['answer']) == 'True' else 'no'}" for r in rows]


def _winogrande() -> List[str]:
    out = []
    for r in _parquet('allenai/winogrande', 'winogrande_xl/train-00000-of-00001.parquet'):
        if str(r['answer']) not in ('1', '2'):
            continue
        out.append(_mc(f"Fill the blank: {r['sentence']}", [r['option1'], r['option2']],
                       int(r['answer']) - 1))
    return out


def _hellaswag() -> List[str]:
    out = []
    for r in _parquet('Rowan/hellaswag', 'data/train-00000-of-00001.parquet'):
        if str(r['label']).strip() == '':
            continue
        out.append(_mc(f"What happens next? {r['ctx'].strip()}", _as_list(r['endings']), int(r['label'])))
    return out


APTITUDE_SOURCES: Dict[str, Callable[[], List[str]]] = {
    'aqua_rat': _aqua, 'gsm8k': _gsm8k, 'logiqa2': _logiqa, 'reclor': _reclor,
    'race': _race, 'arc': _arc, 'commonsense_qa': _csqa, 'openbookqa': _obqa,
    'sciq': _sciq, 'qasc': _qasc, 'boolq': _boolq, 'strategyqa': _strategyqa,
    'winogrande': _winogrande, 'hellaswag': _hellaswag,
}

_DOCS_CACHE: Optional[List[str]] = None


def aptitude_docs() -> List[str]:
    """All formatted train-split documents, in a fixed source order."""
    global _DOCS_CACHE
    if _DOCS_CACHE is None:
        docs: List[str] = []
        for name, build in APTITUDE_SOURCES.items():
            got = build()
            print(f'  [aptitude] {name}: {len(got):,} docs', flush=True)
            docs.extend(got)
        _DOCS_CACHE = docs
    return _DOCS_CACHE


def aptitude_text_iter(*, seed: int = 0, skip_docs: int = 0,
                       doc_counters: Optional[Dict[str, int]] = None,
                       counter_key: str = 'aptitude',
                       bump: Optional[Callable] = None) -> Iterator[str]:
    """Infinite, deterministic stream: shuffled epoch e uses Random(seed, e).

    ``skip_docs`` resumes exactly (documents are counted across epochs).
    """
    docs = aptitude_docs()
    n = len(docs)
    epoch, skipped = skip_docs // n, skip_docs % n
    while True:
        order = list(range(n))
        random.Random(f'aptitude-{seed}-{epoch}').shuffle(order)
        for i in order[skipped:]:
            if bump is not None:
                bump(doc_counters, counter_key)
            yield docs[i]
        skipped = 0
        epoch += 1


__all__ = ['APTITUDE_SOURCES', 'aptitude_docs', 'aptitude_text_iter']
