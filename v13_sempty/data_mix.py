"""Recall-curriculum mixing for v13_sempty training.

Wraps a map-style base dataset (e.g. v7.data.TextDataset over WikiText-103)
and, with probability ``frac``, replaces a sample with a packed sequence of
synthetic recall documents. The recall generator is v7's ``_build_recall_doc``
(passkey / kv / multi / dense) -- imported, never edited. Values there are
disjoint from the behavioral probe's, so the eval stays honest.

This is the training-time analogue of the FineWeb+3%-recall curriculum the
matched transformer used (see EXPERIMENTS_SEMPY "Recall"). Validation is never
mixed (val PPL stays a clean WikiText number).

Determinism: sample i's recall/no-recall decision and its recall content are a
pure function of ``(seed, i)`` -- so a resumed run sees the same mix, and two
arms trained with the same seed see identical recall batches.
"""
from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from v7.data import _build_recall_doc


class RecallMixDataset(Dataset):
    def __init__(self, base: Dataset, *, frac: float, seq_len: int,
                 tokenizer, seed: int = 42):
        self.base = base
        self.frac = float(frac)
        self.seq_len = int(seq_len)
        self.tokenizer = tokenizer
        self.seed = int(seed)
        self.eos = (tokenizer.eos_token_id
                    if tokenizer.eos_token_id is not None else 0)

    def __len__(self):
        return len(self.base)

    def _recall_chunk(self, idx: int):
        """A (seq_len+1)-token tensor packed from synthetic recall docs."""
        rng = random.Random((self.seed, idx, 0x5EED))
        need = self.seq_len + 1
        ids: list[int] = []
        while len(ids) < need:
            doc = _build_recall_doc(rng)
            ids.extend(self.tokenizer.encode(doc))
            ids.append(self.eos)
        chunk = torch.tensor(ids[:need], dtype=torch.long)
        return {'input_ids': chunk[:-1], 'labels': chunk[1:]}

    def __getitem__(self, idx):
        if self.frac > 0.0:
            r = random.Random((self.seed, idx)).random()
            if r < self.frac:
                return self._recall_chunk(idx)
        return self.base[idx]
