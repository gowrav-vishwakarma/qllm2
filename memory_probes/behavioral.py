"""Tokenizer-aware behavioral memory examples and scoring utilities.

The benchmark uses contrastive next-token scoring instead of free generation:
after reading a context containing invented key/value records, a model must
assign the highest next-token logit to the queried value. This reduces
instruction-following and decoding confounds while preserving a behavioral
test of whether the model retained the association.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


KEYS = (
    'dax', 'wug', 'blicket', 'fep', 'toma', 'koba', 'zorp', 'lurn',
    'mip', 'nust', 'pavo', 'rusk', 'sarn', 'tovin', 'vask', 'yoma',
)
VALUES = (
    'apple', 'banana', 'copper', 'dolphin', 'emerald', 'forest',
    'garden', 'harbor', 'island', 'jacket', 'kitten', 'lemon',
    'mirror', 'ocean', 'piano', 'rabbit',
)
FILLER = (
    'The archive clerk reviewed an ordinary report and placed it on the shelf. '
    'The weather remained mild while routine work continued through the day. '
    'Several notes described roads, buildings, meals, and uneventful meetings. '
)


@dataclass(frozen=True)
class BehavioralExample:
    example_id: str
    prompt_ids: List[int]
    prompt_text: str
    target_token_id: int
    target_text: str
    candidate_token_ids: List[int]
    candidate_texts: List[str]
    context_tokens: int
    target_position: float
    associations: int
    seed: int

    def to_dict(self, include_prompt_ids: bool = False) -> Dict[str, Any]:
        row = asdict(self)
        if not include_prompt_ids:
            row.pop('prompt_ids')
        return row


def _encode(tokenizer, text: str) -> List[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def single_token_values(tokenizer, values: Iterable[str] = VALUES) -> List[tuple[str, int]]:
    """Return candidate words represented by one token after ordinary prose."""
    out = []
    for value in values:
        ids = _encode(tokenizer, f' {value}')
        if len(ids) == 1:
            out.append((value, int(ids[0])))
    return out


def _filler_tokens(tokenizer, count: int) -> List[int]:
    if count <= 0:
        return []
    unit = _encode(tokenizer, FILLER)
    if not unit:
        raise ValueError('Tokenizer produced no filler tokens')
    repeats = (count + len(unit) - 1) // len(unit)
    return (unit * repeats)[:count]


def build_example(
    tokenizer,
    *,
    context_tokens: int,
    target_position: float,
    associations: int,
    seed: int,
    candidate_count: int = 8,
) -> BehavioralExample:
    """Build one exact-length invented-association prompt."""
    if context_tokens < 16:
        raise ValueError('context_tokens must be at least 16')
    if not 0.0 <= target_position <= 1.0:
        raise ValueError('target_position must be in [0, 1]')
    if associations < 1 or associations > len(KEYS):
        raise ValueError(f'associations must be in [1, {len(KEYS)}]')

    candidates = single_token_values(tokenizer)
    if len(candidates) < candidate_count:
        raise ValueError(
            f'Need {candidate_count} single-token candidate values; found {len(candidates)}'
        )

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(candidates), size=candidate_count, replace=False)
    candidate_pairs = [candidates[int(i)] for i in chosen]
    target_index = int(rng.integers(min(associations, candidate_count)))
    records = []
    for index in range(associations):
        key = KEYS[index]
        value = candidate_pairs[index % candidate_count][0]
        records.append(f'Memory record {index + 1}: {key} means {value}.\n')
    record_ids = _encode(tokenizer, ''.join(records))
    target_key = KEYS[target_index]
    query_ids = _encode(tokenizer, f'\nMemory query: {target_key} means')

    filler_count = context_tokens - len(record_ids) - len(query_ids)
    if filler_count < 0:
        raise ValueError(
            f'context_tokens={context_tokens} is too short for '
            f'{associations} associations ({-filler_count} tokens short)'
        )
    before_count = int(round(filler_count * target_position))
    after_count = filler_count - before_count
    prompt_ids = (
        _filler_tokens(tokenizer, before_count)
        + record_ids
        + _filler_tokens(tokenizer, after_count)
        + query_ids
    )
    if len(prompt_ids) != context_tokens:
        raise AssertionError('Prompt construction did not preserve exact token length')

    target_text, target_token_id = candidate_pairs[target_index]
    return BehavioralExample(
        example_id=(
            f'ctx{context_tokens}_pos{target_position:g}_'
            f'n{associations}_seed{seed}'
        ),
        prompt_ids=prompt_ids,
        prompt_text=tokenizer.decode(prompt_ids),
        target_token_id=target_token_id,
        target_text=target_text,
        candidate_token_ids=[token_id for _, token_id in candidate_pairs],
        candidate_texts=[text for text, _ in candidate_pairs],
        context_tokens=context_tokens,
        target_position=target_position,
        associations=associations,
        seed=seed,
    )


def build_suite(
    tokenizer,
    *,
    context_lengths: Sequence[int],
    positions: Sequence[float],
    association_counts: Sequence[int],
    seeds: Sequence[int],
    candidate_count: int = 8,
) -> List[BehavioralExample]:
    examples = []
    for context_tokens in context_lengths:
        for position in positions:
            for associations in association_counts:
                for seed in seeds:
                    examples.append(build_example(
                        tokenizer,
                        context_tokens=context_tokens,
                        target_position=position,
                        associations=associations,
                        seed=seed,
                        candidate_count=candidate_count,
                    ))
    return examples


def score_candidate_logits(
    example: BehavioralExample,
    candidate_logits: Sequence[float],
) -> Dict[str, Any]:
    """Score a candidate-only logit vector for one example."""
    logits = np.asarray(candidate_logits, dtype=np.float64)
    if logits.shape != (len(example.candidate_token_ids),):
        raise ValueError('candidate_logits shape does not match candidate set')
    target_index = example.candidate_token_ids.index(example.target_token_id)
    order = np.argsort(-logits)
    rank = int(np.where(order == target_index)[0][0]) + 1
    distractors = np.delete(logits, target_index)
    margin = float(logits[target_index] - distractors.max()) if distractors.size else 0.0
    return {
        'correct': rank == 1,
        'target_rank': rank,
        'target_margin': margin,
        'predicted_text': example.candidate_texts[int(order[0])],
        'target_text': example.target_text,
        'candidate_logits': logits.tolist(),
    }
