"""Interleaved duplex blocks for Stage 0 (text proxies for audio streams)."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import torch

from v11.duplex.thinking import (
    ASSISTANT_PHRASES,
    BACKCHANNEL_PHRASES,
    USER_PHRASES,
    VOCAB,
)


class ScenarioKind(Enum):
    USER_TURN = auto()
    ASSISTANT_TURN = auto()
    BARGE_IN = auto()
    BACKCHANNEL = auto()


@dataclass
class DuplexBlock:
    """One time block: env proxy, optional ast echo, thinking label, optional reply text."""

    env_tokens: List[int]
    ast_tokens: List[int]
    thinking: int
    reply_tokens: List[int]
    kind: ScenarioKind


def _phrase_to_ids(word_ids: List[int]) -> List[int]:
    return [VOCAB.text_token(w) for w in word_ids]


def _sample_phrase(pool: List[List[int]], rng) -> List[int]:
    return _phrase_to_ids(pool[int(rng.integers(0, len(pool)))])


def build_block(
    kind: ScenarioKind,
    rng,
    env_len: int = 4,
    ast_len: int = 2,
    reply_len: int = 6,
    truncate_reply: Optional[int] = None,
) -> DuplexBlock:
    if kind == ScenarioKind.USER_TURN:
        env = _sample_phrase(USER_PHRASES, rng)[:env_len]
        if len(env) < env_len:
            env = env + [VOCAB.text_token(i) for i in range(env_len - len(env))]
        return DuplexBlock(
            env_tokens=env,
            ast_tokens=[],
            thinking=VOCAB.listen,
            reply_tokens=[],
            kind=kind,
        )

    if kind == ScenarioKind.ASSISTANT_TURN:
        reply = _sample_phrase(ASSISTANT_PHRASES, rng)
        if len(reply) < reply_len:
            reply = reply + [VOCAB.text_token(100 + i) for i in range(reply_len - len(reply))]
        else:
            reply = reply[:reply_len]
        return DuplexBlock(
            env_tokens=[],
            ast_tokens=_sample_phrase(ASSISTANT_PHRASES, rng)[:ast_len],
            thinking=VOCAB.speak,
            reply_tokens=reply,
            kind=kind,
        )

    if kind == ScenarioKind.BARGE_IN:
        env = _sample_phrase(USER_PHRASES, rng)[:env_len]
        reply = _sample_phrase(ASSISTANT_PHRASES, rng)
        if truncate_reply is not None:
            reply = reply[:truncate_reply]
        return DuplexBlock(
            env_tokens=env,
            ast_tokens=reply[:ast_len],
            thinking=VOCAB.listen,
            reply_tokens=[],
            kind=kind,
        )

    # BACKCHANNEL
    env = _sample_phrase(BACKCHANNEL_PHRASES, rng)
    reply = _sample_phrase(ASSISTANT_PHRASES, rng)[:reply_len]
    return DuplexBlock(
        env_tokens=env,
        ast_tokens=reply[:ast_len],
        thinking=VOCAB.backchannel,
        reply_tokens=reply,
        kind=kind,
    )


def block_to_token_lists(block: DuplexBlock) -> Tuple[List[int], List[int]]:
    """Return (input_ids, label_ids). label_ids[i]=input_ids[i] where we train CE, else -100."""
    inp: List[int] = []
    lab: List[int] = []

    def append(tok: int, train: bool = False):
        inp.append(tok)
        lab.append(tok if train else -100)

    if block.env_tokens:
        append(VOCAB.env_mark)
        for t in block.env_tokens:
            append(t)

    if block.ast_tokens:
        append(VOCAB.ast_mark)
        for t in block.ast_tokens:
            append(t)

    append(block.thinking, train=True)

    for t in block.reply_tokens:
        append(t, train=True)

    return inp, lab


def generate_conversation(
    n_blocks: int,
    rng,
    barge_in_prob: float = 0.35,
    backchannel_prob: float = 0.15,
) -> Tuple[List[int], List[int], List[ScenarioKind]]:
    """Build a multi-block synthetic duplex transcript."""
    kinds: List[ScenarioKind] = []
    all_inp: List[int] = []
    all_lab: List[int] = []

    state = 'idle'
    for _ in range(n_blocks):
        r = rng.random()
        if state == 'assistant_speaking' and r < barge_in_prob:
            kind = ScenarioKind.BARGE_IN
            truncate = int(rng.integers(1, 4))
            block = build_block(kind, rng, truncate_reply=truncate)
            state = 'user'
        elif r < backchannel_prob:
            kind = ScenarioKind.BACKCHANNEL
            block = build_block(kind, rng)
            state = 'assistant_speaking'
        elif state in ('idle', 'user'):
            if rng.random() < 0.4:
                kind = ScenarioKind.USER_TURN
                block = build_block(kind, rng)
                state = 'user'
            else:
                kind = ScenarioKind.ASSISTANT_TURN
                block = build_block(kind, rng)
                state = 'assistant_speaking'
        else:
            kind = ScenarioKind.ASSISTANT_TURN
            block = build_block(kind, rng)
            state = 'assistant_speaking'

        kinds.append(kind)
        inp, lab = block_to_token_lists(block)
        all_inp.extend(inp)
        all_lab.extend(lab)
        all_inp.append(VOCAB.block_sep)
        all_lab.append(-100)

    return all_inp, all_lab, kinds


def pad_batch(
    sequences: List[Tuple[List[int], List[int]]],
    pad_id: int = VOCAB.pad,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(s[0]) for s in sequences)
    b = len(sequences)
    input_ids = torch.full((b, max_len), pad_id, dtype=torch.long)
    labels = torch.full((b, max_len), -100, dtype=torch.long)
    attn = torch.zeros(b, max_len, dtype=torch.bool)
    for i, (inp, lab) in enumerate(sequences):
        n = len(inp)
        input_ids[i, :n] = torch.tensor(inp, dtype=torch.long)
        labels[i, :n] = torch.tensor(lab, dtype=torch.long)
        attn[i, :n] = True
    return input_ids, labels, attn
