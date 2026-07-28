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

DIVERSITY (2026-07-27). The first production run used ONE phrasing and ~50 value
words, and the resulting module scored 0.925 on that exact distribution but
chance the moment either the template or the vocabulary changed -- it learned a
surface pattern, not a binding operation. Three things are now randomized per
document, each with a deterministic held-out split so transfer is measurable:

  * template  -- 12 record/query phrasings (``TEMPLATES_TRAIN``); 4 more are held
                 out (``TEMPLATES_HELDOUT``), including the exact phrasing that
                 ``memory_probes/behavioral.py`` uses, so ``v12.eval_recall``
                 stays an honest transfer test.
  * vocabulary -- ~17k single-token words swept from the tokenizer, hash-split
                 80/20 into train/held-out. Behavioral ``VALUES`` are forced into
                 the held-out side.
  * filler    -- sampled from a 40-sentence bank rather than a fixed cycle, and
                 deliberately INDEPENDENT of the template, so a template shift at
                 eval time does not also shift the filler (that confound would
                 make the transfer grid uninterpretable).

Keys are invented nonces regenerated per document in one of four surface shapes,
so key form is not a constant either.

The dataset is map-style but generates each example deterministically on
``__getitem__`` (seeded by index), so it supports shuffle + ``__len__`` without
materializing millions of tokens in RAM.
"""

import hashlib
import random
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from v7.data import get_chat_tokenizer
from memory_probes.behavioral import KEYS as _EVAL_KEYS, VALUES as _EVAL_VALUES

# Nonce key building blocks (invented tokens; not real words).
_CONSONANTS = 'bdfgklmnprstvz'
_VOWELS = 'aeiou'

# Record/query phrasing pairs. ``{k}``=key, ``{v}``=value, ``{n}``=1-based index.
# A query stem must end immediately before the value with NO trailing space --
# the value is appended separately as ' <word>' so it stays exactly one token.
_TEMPLATE_BANK = (
    # -- train split ---------------------------------------------------------
    ('Record: {k} means {v}. ',                  'Query: {k} means'),
    ('{k} -> {v}; ',                             '{k} ->'),
    ('The code {k} stands for {v}. ',            'The code {k} stands for'),
    ('entry {n}: {k} = {v}\n',                   'entry: {k} ='),
    ('Note that {k} maps to {v}. ',              'Note that {k} maps to'),
    ('We define {k} as {v}. ',                   'We define {k} as'),
    ('{k} is assigned {v}. ',                    '{k} is assigned'),
    ('Set {k} to {v}. ',                         'Set {k} to'),
    ('Pairing {n}: {k} with {v}. ',              'Pairing: {k} with'),
    ('In the ledger, {k} refers to {v}. ',       'In the ledger, {k} refers to'),
    ('{k} corresponds to {v}. ',                 '{k} corresponds to'),
    ('Label {k} carries the tag {v}. ',          'Label {k} carries the tag'),
    # -- held-out split (index >= _N_TRAIN_TEMPLATES) ------------------------
    # First entry mirrors memory_probes/behavioral.py so eval_recall measures
    # exactly the "unseen template" cell of the transfer grid.
    ('Memory record {n}: {k} means {v}.\n',      '\nMemory query: {k} means'),
    ('{k} translates to {v}. ',                  '{k} translates to'),
    ('{k} yields {v}\n',                         '{k} yields'),
    ('Table row {n}: {k} | {v}\n',               'Table row: {k} |'),
)
_N_TRAIN_TEMPLATES = 12

TEMPLATES_TRAIN = _TEMPLATE_BANK[:_N_TRAIN_TEMPLATES]
TEMPLATES_HELDOUT = _TEMPLATE_BANK[_N_TRAIN_TEMPLATES:]

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
    "Someone left the side window open and the curtains kept moving.",
    "The bus arrived late and the passengers waited under the awning.",
    "A technician replaced the bulbs along the corridor before opening.",
    "The kettle was refilled twice during the long morning session.",
    "Papers on the desk were sorted into three uneven piles.",
    "The cat crossed the courtyard and settled beside the low wall.",
    "Rehearsal ran past the hour because the second act needed work.",
    "A delivery van idled at the corner while the driver checked a list.",
    "The river was low enough that the stones showed near the bank.",
    "Two students compared notes about the reading before the seminar.",
    "The clock in the hallway was five minutes fast all week.",
    "Someone had watered the plants and pulled the blinds halfway.",
    "The printer jammed again and the queue backed up until lunch.",
    "Fog settled over the fields and lifted only after ten.",
    "The old gate needed oil and complained whenever it swung.",
    "A radio played quietly in the back room throughout the shift.",
    "Boxes of samples were stacked against the far wall of the lab.",
    "The path behind the school was muddy after the weekend rain.",
    "Someone repaired the loose tile near the entrance on Tuesday.",
    "The meeting adjourned without resolving the second agenda item.",
    "A crow sat on the chimney for most of the afternoon.",
    "The chairs were rearranged into a circle and then back again.",
    "Deliveries were rescheduled because the loading dock was blocked.",
    "The lamp flickered whenever the heater switched itself on.",
    "A notice about the closure was pinned beside the main door.",
    "The bread came out slightly overdone but nobody complained.",
    "Snow melted off the roof and dripped steadily past the window.",
    "The inventory count matched the ledger for the first time in months.",
    "A visitor asked for directions and left without signing in.",
    "The last train was delayed and the platform emptied slowly.",
)

# Module-level caches: sweeping the tokenizer vocabulary is not free and the
# dataset is constructed several times per run (train + val).
_VALUE_POOL_CACHE = {}


def _sweep_single_token_words(tokenizer, *, min_id=2000, min_len=4):
    """Lowercase alphabetic words that are ONE token with a leading space.

    ``min_id`` skips the low BPE ids, which are the high-frequency function
    words and subword fragments -- a value of ' the' would collide with the
    filler text and make the recall objective meaningless.
    """
    out = []
    for tid in range(min_id, len(tokenizer)):
        piece = tokenizer.convert_ids_to_tokens(tid)
        if not piece or not piece.startswith('\u0120'):
            continue
        word = piece[1:]
        if len(word) < min_len or not word.isalpha() or not word.islower():
            continue
        # Guard the contract: the value must round-trip to this single id.
        ids = tokenizer.encode(f' {word}', add_special_tokens=False)
        if len(ids) == 1 and ids[0] == tid:
            out.append((word, tid))
    return out


def build_value_pool(tokenizer, split: str = 'train',
                     limit: Optional[int] = None) -> List[Tuple[str, int]]:
    """Value words for a split. ``split`` in {'train', 'heldout', 'all'}.

    The split is a stable md5 hash of the word (80/20), not Python's salted
    ``hash``, so train and held-out pools are identical across processes and
    runs. Behavioral ``VALUES`` are forced held-out to keep ``v12.eval_recall``
    an honest transfer test.

    ``limit`` truncates the pool (deterministically, in tokenizer-id order).
    The full ~14k sweep makes the readout a pure copy operation, which is the
    right target but needs a large token budget to learn; a few hundred to a
    few thousand values is still one to two orders of magnitude more diverse
    than the 50-word list that produced the memorized 2026-07 module.
    """
    if split not in ('train', 'heldout', 'all'):
        raise ValueError(f"split must be train|heldout|all, got {split!r}")
    key = (len(tokenizer), split, limit)
    if key in _VALUE_POOL_CACHE:
        return _VALUE_POOL_CACHE[key]

    all_key = (len(tokenizer), 'all', None)
    if all_key in _VALUE_POOL_CACHE:
        every = _VALUE_POOL_CACHE[all_key]
    else:
        # A value that also occurs in the filler would pollute the in-context
        # candidate set the transfer grid scores against.
        filler_ids = set()
        for sentence in _FILLER_BANK:
            filler_ids.update(tokenizer.encode(sentence + ' ', add_special_tokens=False))
        every = [(w, t) for w, t in _sweep_single_token_words(tokenizer)
                 if t not in filler_ids]
        _VALUE_POOL_CACHE[all_key] = every

    if split == 'all':
        pool = every
    else:
        eval_values = set(_EVAL_VALUES)
        pool = []
        for word, tid in every:
            digest = hashlib.md5(word.encode()).digest()[0]
            heldout = word in eval_values or digest >= 205  # ~20% of 0..255
            if heldout == (split == 'heldout'):
                pool.append((word, tid))

    if limit:
        pool = pool[:int(limit)]
    if len(pool) < 24:
        raise RuntimeError(
            f"fact value pool for split={split} too small ({len(pool)}); "
            f"need >=24 single-token values"
        )
    _VALUE_POOL_CACHE[key] = pool
    return pool


def _build_value_pool(tokenizer) -> List[Tuple[str, int]]:
    """Back-compat alias for the training-split pool."""
    return build_value_pool(tokenizer, 'train')


def templates_for(split: str = 'train'):
    if split == 'train':
        return TEMPLATES_TRAIN
    if split == 'heldout':
        return TEMPLATES_HELDOUT
    if split == 'all':
        return _TEMPLATE_BANK
    raise ValueError(f"split must be train|heldout|all, got {split!r}")


def nonce_key(rng: random.Random) -> str:
    """Invented key in one of four surface shapes, disjoint from eval KEYS."""
    def cvc():
        return rng.choice(_CONSONANTS) + rng.choice(_VOWELS) + rng.choice(_CONSONANTS)

    shape = rng.randrange(4)
    if shape == 0:
        key = cvc() + cvc()                                   # 6 chars
    elif shape == 1:
        key = cvc() + cvc() + cvc()                           # 9 chars
    elif shape == 2:
        key = cvc() + rng.choice(_VOWELS) + cvc()             # 7 chars
    else:
        key = cvc() + str(rng.randrange(10, 100))             # alnum
    return key if key not in _EVAL_KEYS else key + 'a'


def make_keys(rng: random.Random, n: int, split: str = 'train') -> List[str]:
    """``n`` distinct keys. 'heldout' draws the behavioral probe's pseudo-words,
    whose surface form the training generator never produces."""
    if split == 'heldout':
        if n > len(_EVAL_KEYS):
            raise ValueError(f"only {len(_EVAL_KEYS)} held-out keys, asked for {n}")
        return rng.sample(list(_EVAL_KEYS), n)
    out, seen = [], set()
    while len(out) < n:
        k = nonce_key(rng)
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _nonce_key(rng: random.Random) -> str:
    """Back-compat alias."""
    return nonce_key(rng)


def render(template, idx: int, key: str, value: Optional[str] = None) -> str:
    """Record line (``value`` given) or query stem (``value`` None)."""
    record_fmt, query_fmt = template
    if value is None:
        return query_fmt.format(k=key, n=idx + 1)
    return record_fmt.format(k=key, v=value, n=idx + 1)


class FactRecallDataset(Dataset):
    """Store-then-query fact examples with answer-only ``loss_mask``.

    Each example draws a template and a value set, emits several bindings (plus
    hard-negative distractors), a long filler gap, then one or more query stems
    where only the value token is supervised. Exact length ``seq_len`` (value
    tokens land near the end for a long-range recall signal).
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
        max_queries: int = 8,
        template_split: str = 'train',
        value_split: str = 'train',
        key_split: str = 'train',
        value_pool_limit: Optional[int] = None,
        mixed_templates: bool = True,
    ):
        self.n_samples = int(n_samples)
        self.seq_len = int(seq_len)
        self.tokenizer = tokenizer or get_chat_tokenizer()
        self.seed = int(seed)
        self.max_facts = max_facts
        self.max_distractors = max_distractors
        self.max_queries = max_queries
        self.template_split = template_split
        self.value_split = value_split
        self.key_split = key_split
        # Mix a second phrasing into the record block of some documents, so a
        # document is not internally uniform either.
        self.mixed_templates = bool(mixed_templates)
        self.templates = templates_for(template_split)
        self.value_pool_limit = value_pool_limit
        self.value_pool = build_value_pool(self.tokenizer, value_split,
                                           value_pool_limit)
        self._eos = self.tokenizer.eos_token_id
        self._filler_units = [self.tokenizer.encode(s + ' ', add_special_tokens=False)
                              for s in _FILLER_BANK]

    def __len__(self):
        return self.n_samples

    def _fill_tokens(self, rng: random.Random, n: int) -> List[int]:
        """``n`` tokens of filler sampled from the bank (template-independent)."""
        if n <= 0:
            return []
        out: List[int] = []
        while len(out) < n:
            out += self._filler_units[rng.randrange(len(self._filler_units))]
        return out[:n]

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _build(self, rng: random.Random):
        L = self.seq_len + 1  # build full sequence; slice into input/labels
        template = self.templates[rng.randrange(len(self.templates))]
        # Distractor records may use a different phrasing than the queried ones.
        alt = (self.templates[rng.randrange(len(self.templates))]
               if self.mixed_templates else template)

        n_facts = rng.randint(2, self.max_facts)
        n_dist = rng.randint(0, self.max_distractors)
        # All keys up front: the held-out split draws without replacement from a
        # fixed 16-word pool, so facts and distractors must not collide.
        all_keys = make_keys(rng, n_facts + n_dist, self.key_split)
        keys, dist_keys = all_keys[:n_facts], all_keys[n_facts:]
        val_idx = rng.sample(range(len(self.value_pool)), n_facts)
        vals = [self.value_pool[i][0] for i in val_idx]

        # Which facts get queried (place them LAST in the record block so they
        # survive any front-truncation). Query as many as the budget allows:
        # only the value tokens carry gradient, so one query per 1024-token
        # document wastes ~99.9% of the forward pass. ``rng.sample`` also
        # randomizes query order, so answer order never mirrors record order.
        n_q = min(n_facts, self.max_queries)
        q_facts = rng.sample(range(n_facts), n_q)
        q_set = set(q_facts)
        record_order = [i for i in range(n_facts) if i not in q_set] + list(q_facts)

        # Hard-negative distractor records: extra keys, values REUSED from the
        # true set (so the queried key cannot be answered by value frequency).
        distractors = [(dk, vals[rng.randrange(n_facts)]) for dk in dist_keys]

        # Interleave true records (non-query first) with distractors; queried
        # facts keep the document's main template, distractors may use ``alt``.
        record_items = [(keys[i], vals[i], template) for i in record_order]
        insert_at = sorted(rng.randrange(len(record_items) + 1) for _ in distractors)
        merged, di = [], 0
        for pos in range(len(record_items) + 1):
            while di < len(distractors) and insert_at[di] == pos:
                merged.append((*distractors[di], alt)); di += 1
            if pos < len(record_items):
                merged.append(record_items[pos])

        rec_tokens: List[int] = []
        for i, (k, v, tmpl) in enumerate(merged):
            rec_tokens += self._encode(render(tmpl, i, k, v))

        # Query block (goes at the very end); supervise only the value token.
        q_tokens: List[int] = []
        q_sup: List[int] = []
        for j in q_facts:
            head = self._encode(render(template, 0, keys[j]))
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
            filler = self._fill_tokens(rng, budget - len(rec_tokens))

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
    max_queries: int = 8,
    template_split: str = 'train',
    value_split: str = 'train',
    key_split: str = 'train',
    value_pool_limit: Optional[int] = None,
    **_unused,
):
    """Build (train_ds, val_ds, tokenizer) for the fact-recall stage.

    ``n_train`` defaults from ``token_budget`` (budget // seq_len) so a single
    epoch consumes ~token_budget tokens; falls back to 20000 examples.
    ``template_split`` / ``value_split`` select the diversity slice; the defaults
    train on the train slices and leave the held-out ones for the transfer grid.
    """
    tokenizer = get_chat_tokenizer()
    if n_train is None:
        n_train = (token_budget // seq_len) if token_budget else 20000
        n_train = max(int(n_train), 1000)
    common = dict(seq_len=seq_len, tokenizer=tokenizer, max_facts=max_facts,
                  max_distractors=max_distractors, max_queries=max_queries,
                  template_split=template_split, value_split=value_split,
                  key_split=key_split, value_pool_limit=value_pool_limit)
    train_ds = FactRecallDataset(n_train, seed=seed, **common)
    val_ds = FactRecallDataset(n_val, seed=seed + 10_000_019, **common)
    print(
        f"[fact_data] train={n_train:,} val={n_val:,} examples, seq_len={seq_len}, "
        f"templates={len(train_ds.templates)} ({template_split}), "
        f"value_pool={len(train_ds.value_pool)} ({value_split}), "
        f"vocab={len(tokenizer)}"
    )
    return train_ds, val_ds, tokenizer
