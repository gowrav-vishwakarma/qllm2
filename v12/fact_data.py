"""Purpose-built fact-supervision data for the V12 fact module.

Web/edu pretrain never *demands* that the model store an invented binding and
reproduce it later, so the PAM state never learns durable key->value recall
(4090: ~0.2 vs Mamba ~1.0). This loader builds store-then-query documents and
supervises ONLY the answer/value token (a per-token ``loss_mask``), so the loss
is a direct recall objective: predict the value from memory, not from local
context. Hard-negative distractors (other keys, reused values) force genuine
key->value binding rather than "a value appeared recently".

Contract matches the SFT path (``MaskedTextDataset``): each sample is
``{input_ids, labels, loss_mask}`` of length ``seq_len``; ``loss_mask`` is 1 only
on value tokens. The trainer already applies it in both the fused
(``ce_from_lm(..., loss_mask=)``) and standard (``_masked_ce``) paths.

Vocabulary is deliberately DISJOINT from ``memory_probes/behavioral.py``
(KEYS/VALUES) so the behavioral single_assoc@2048 eval stays a held-out test.

The dataset is map-style but generates each example deterministically on
``__getitem__`` (seeded by index), so it supports shuffle + ``__len__`` without
materializing millions of tokens in RAM.
"""

import random
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from v7.data import get_chat_tokenizer
from memory_probes.behavioral import KEYS as _EVAL_KEYS, VALUES as _EVAL_VALUES

# Nonce key building blocks (invented tokens; not real words).
_CONSONANTS = 'bdfgklmnprstvz'
_VOWELS = 'aeiou'

# Candidate value words; filtered at load time to those that are a SINGLE GPT-2
# token with a leading space AND not in the eval VALUES set (held-out).
_VALUE_CANDIDATES = (
    'gold', 'silver', 'iron', 'bronze', 'stone', 'river', 'mountain', 'valley',
    'meadow', 'tiger', 'eagle', 'falcon', 'otter', 'coffee', 'pepper', 'ginger',
    'cotton', 'velvet', 'marble', 'crystal', 'diamond', 'ruby', 'willow',
    'cedar', 'maple', 'birch', 'coral', 'pearl', 'jade', 'opal', 'slate',
    'granite', 'canyon', 'desert', 'glacier', 'prairie', 'thunder', 'breeze',
    'comet', 'meteor', 'cobalt', 'nickel', 'brass', 'amber', 'ivory', 'walnut',
    'almond', 'cherry', 'plum', 'peach', 'melon', 'lime', 'olive', 'wheat',
    'maize', 'clover', 'cactus', 'bamboo', 'orchid', 'tulip', 'daisy', 'raven',
    'hawk', 'bison', 'moose', 'lynx', 'seal', 'whale', 'shark', 'trout',
)

_FILLER_BANK = (
    "The committee reviewed the quarterly notes without any changes.",
    "A light rain fell over the harbor as the ferries came and went.",
    "Workers repainted the fence and swept the yard before noon.",
    "The lecture covered ordinary topics and ended a few minutes early.",
    "Shelves in the storeroom were dusted and the boxes were relabeled.",
    "Traffic moved slowly along the avenue during the afternoon.",
    "The librarian filed the returned volumes and updated the ledger.",
    "Clouds drifted past while the market stalls slowly packed up.",
    "A short memo described the schedule for the following week.",
    "The garden path was cleared and the benches were wiped down.",
)


def _build_value_pool(tokenizer) -> List[Tuple[str, int]]:
    """Value words that are one token (with leading space) and not eval VALUES."""
    eval_values = set(_EVAL_VALUES)
    pool = []
    seen = set()
    for word in _VALUE_CANDIDATES:
        if word in eval_values or word in seen:
            continue
        ids = tokenizer.encode(f' {word}', add_special_tokens=False)
        if len(ids) == 1:
            pool.append((word, int(ids[0])))
            seen.add(word)
    if len(pool) < 24:
        raise RuntimeError(
            f"fact value pool too small ({len(pool)}); need >=24 single-token values"
        )
    return pool


def _nonce_key(rng: random.Random) -> str:
    """Two-syllable invented key (6 chars), disjoint from eval KEYS by length."""
    def syl():
        return rng.choice(_CONSONANTS) + rng.choice(_VOWELS) + rng.choice(_CONSONANTS)
    key = syl() + syl()
    return key if key not in _EVAL_KEYS else key + 'a'


class FactRecallDataset(Dataset):
    """Store-then-query fact examples with answer-only ``loss_mask``.

    Each example: several ``Record: <key> means <value>.`` bindings (plus
    hard-negative distractors), a long filler gap, then one or more
    ``Query: <key> means <value>`` probes where only the value token is
    supervised. Exact length ``seq_len`` (value tokens land near the end for a
    long-range recall signal).
    """

    def __init__(
        self,
        n_samples: int,
        seq_len: int,
        tokenizer=None,
        *,
        seed: int = 0,
        max_facts: int = 8,
        max_distractors: int = 4,
        max_queries: int = 3,
    ):
        self.n_samples = int(n_samples)
        self.seq_len = int(seq_len)
        self.tokenizer = tokenizer or get_chat_tokenizer()
        self.seed = int(seed)
        self.max_facts = max_facts
        self.max_distractors = max_distractors
        self.max_queries = max_queries
        self.value_pool = _build_value_pool(self.tokenizer)
        self._eos = self.tokenizer.eos_token_id
        # Pre-tokenized filler unit for exact-length gap filling.
        self._filler_unit = self.tokenizer.encode(
            ' '.join(_FILLER_BANK), add_special_tokens=False
        )

    def __len__(self):
        return self.n_samples

    def _fill_tokens(self, n: int) -> List[int]:
        if n <= 0:
            return []
        unit = self._filler_unit
        reps = (n + len(unit) - 1) // len(unit)
        return (unit * reps)[:n]

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _build(self, rng: random.Random):
        L = self.seq_len + 1  # build full sequence; slice into input/labels
        n_facts = rng.randint(2, self.max_facts)
        # Distinct keys + values for the true facts.
        keys, seen_keys = [], set()
        while len(keys) < n_facts:
            k = _nonce_key(rng)
            if k not in seen_keys:
                seen_keys.add(k)
                keys.append(k)
        val_idx = rng.sample(range(len(self.value_pool)), n_facts)
        vals = [self.value_pool[i][0] for i in val_idx]

        # Which facts get queried (place them LAST in the record block so they
        # survive any front-truncation).
        n_q = min(rng.choice([1, 1, 1, 2, 3]), n_facts)
        q_facts = rng.sample(range(n_facts), n_q)
        q_set = set(q_facts)
        record_order = [i for i in range(n_facts) if i not in q_set] + list(q_facts)

        # Hard-negative distractor records: extra keys, values REUSED from the
        # true set (so the queried key cannot be answered by value frequency).
        n_dist = rng.randint(0, self.max_distractors)
        distractors = []
        for _ in range(n_dist):
            dk = _nonce_key(rng)
            if dk in seen_keys:
                continue
            seen_keys.add(dk)
            dv = vals[rng.randrange(n_facts)]
            distractors.append((dk, dv))

        # Interleave true records (non-query first) with distractors.
        record_items = [(keys[i], vals[i]) for i in record_order]
        insert_at = sorted(rng.randrange(len(record_items) + 1) for _ in distractors)
        merged, di = [], 0
        for pos in range(len(record_items) + 1):
            while di < len(distractors) and insert_at[di] == pos:
                merged.append(distractors[di]); di += 1
            if pos < len(record_items):
                merged.append(record_items[pos])

        rec_tokens: List[int] = []
        for k, v in merged:
            rec_tokens += self._encode(f"Record: {k} means {v}. ")

        # Query block (goes at the very end); supervise only the value token.
        q_tokens: List[int] = []
        q_sup: List[int] = []
        for j in q_facts:
            head = self._encode(f"Query: {keys[j]} means")
            value_ids = self._encode(f" {vals[j]}")
            tail = self._encode(". ")
            q_tokens += head + value_ids + tail
            q_sup += [0] * len(head) + [1] * len(value_ids) + [0] * len(tail)

        # Size the filler so the whole sequence is exactly L, value near the end.
        budget = L - len(q_tokens)
        if len(rec_tokens) >= budget:
            rec_tokens = rec_tokens[len(rec_tokens) - budget:]  # keep queried tail
            filler = []
        else:
            filler = self._fill_tokens(budget - len(rec_tokens))

        tokens = rec_tokens + filler + q_tokens
        sup = [0] * (len(rec_tokens) + len(filler)) + q_sup
        # Exact-length guard (pad/truncate defensively).
        if len(tokens) < L:
            pad = L - len(tokens)
            tokens += [self._eos] * pad
            sup += [0] * pad
        elif len(tokens) > L:
            tokens = tokens[:L]
            sup = sup[:L]
        return tokens, sup

    def __getitem__(self, idx):
        rng = random.Random((self.seed * 2654435761 + idx) & 0xFFFFFFFF)
        tokens, sup = self._build(rng)
        tok = torch.tensor(tokens, dtype=torch.long)
        supervise = torch.tensor(sup, dtype=torch.long)
        return {
            'input_ids': tok[:-1],
            'labels': tok[1:],
            'loss_mask': supervise[1:],  # aligns to predicted (label) tokens
        }


def load_fact_recall(
    seq_len: int = 1024,
    *,
    token_budget: Optional[int] = None,
    n_train: Optional[int] = None,
    n_val: int = 512,
    seed: int = 0,
    max_facts: int = 8,
    max_distractors: int = 4,
    max_queries: int = 3,
    **_unused,
):
    """Build (train_ds, val_ds, tokenizer) for the fact-recall stage.

    ``n_train`` defaults from ``token_budget`` (budget // seq_len) so a single
    epoch consumes ~token_budget tokens; falls back to 20000 examples.
    """
    tokenizer = get_chat_tokenizer()
    if n_train is None:
        n_train = (token_budget // seq_len) if token_budget else 20000
        n_train = max(int(n_train), 1000)
    common = dict(seq_len=seq_len, tokenizer=tokenizer, max_facts=max_facts,
                  max_distractors=max_distractors, max_queries=max_queries)
    train_ds = FactRecallDataset(n_train, seed=seed, **common)
    val_ds = FactRecallDataset(n_val, seed=seed + 10_000_019, **common)
    print(
        f"[fact_data] train={n_train:,} val={n_val:,} examples, seq_len={seq_len}, "
        f"value_pool={len(train_ds.value_pool)}, vocab={len(tokenizer)}"
    )
    return train_ds, val_ds, tokenizer
