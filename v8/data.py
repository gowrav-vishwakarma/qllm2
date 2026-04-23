"""V8 data loading.

Re-exports the v7 loaders for WikiText-103 and TinyStories so we don't fork
the (well-tested) tokenization+caching pipeline. Adds a small synthetic
entity-attribute cloze generator used by Stage B's InfoNCE auxiliary on the
EffectAlgebraBank.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

# Re-export v7 loaders so callers can `from v8.data import load_wikitext103`.
from v7.data import (  # noqa: F401
    TeeLogger,
    TextDataset,
    load_wikitext103,
    load_tinystories,
    compute_text_quality,
    resolve_amp_dtype,
    build_lr_scheduler,
    build_param_groups,
    repair_text,
)


# ── Synthetic Entity-Attribute Cloze (Stage B InfoNCE) ───────────────────────


@dataclass
class EntityClozeExample:
    """One synthetic entity-attribute QA example.

    The "context" describes an entity's attributes (e.g. ``"alpha lives in
    boston"``); the "query" is a masked sentence asking for one attribute
    (``"alpha lives in <MASK>"``); the gold answer is the masked attribute.

    InfoNCE in the bank uses (psi_query, psi_distractors, gold_effect_idx)
    where the gold_effect_idx is a hash of the entity name modulo bank_size
    -- we want each entity to consistently route to a small subset of
    effects.
    """

    context_ids: torch.Tensor
    query_ids: torch.Tensor
    answer_ids: torch.Tensor
    entity_idx: int


class EntityClozeDataset(Dataset):
    r"""Tiny synthetic dataset of ``"<entity> <relation> <value>"`` triples.

    The vocabulary used here is a small fixed set of made-up tokens so that
    the model has to *learn* the routing rather than rely on tokenizer
    priors. Entities are integers mapped to a span of words drawn from a
    fixed pool; relations and values likewise.
    """

    def __init__(
        self,
        n_entities: int = 64,
        n_relations: int = 4,
        n_values_per_relation: int = 8,
        n_examples: int = 4096,
        seq_len: int = 32,
        tokenizer=None,
        seed: int = 0,
    ):
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.n_values_per_relation = n_values_per_relation
        self.seq_len = seq_len
        rng = random.Random(seed)

        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.mask_id = tokenizer.eos_token_id

        ent_pool = [f" entity_{i}" for i in range(n_entities)]
        rel_pool = [" lives_in", " works_at", " owns", " likes"][:n_relations]
        val_pool = [
            [f" val_{r}_{v}" for v in range(n_values_per_relation)]
            for r in range(n_relations)
        ]

        # Build a deterministic table of (entity, relation, value) facts.
        self.facts: Dict[Tuple[int, int], int] = {}
        for e in range(n_entities):
            for r in range(n_relations):
                self.facts[(e, r)] = rng.randrange(n_values_per_relation)

        self.examples: List[EntityClozeExample] = []
        for _ in range(n_examples):
            e = rng.randrange(n_entities)
            r = rng.randrange(n_relations)
            v = self.facts[(e, r)]
            ent_w, rel_w, val_w = ent_pool[e], rel_pool[r], val_pool[r][v]

            ctx = ent_w + rel_w + val_w + " ."
            qry = ent_w + rel_w + " "
            ans = val_w

            ctx_ids = torch.tensor(
                tokenizer.encode(ctx, add_special_tokens=False), dtype=torch.long
            )
            qry_ids = torch.tensor(
                tokenizer.encode(qry, add_special_tokens=False), dtype=torch.long
            )
            ans_ids = torch.tensor(
                tokenizer.encode(ans, add_special_tokens=False), dtype=torch.long
            )

            self.examples.append(EntityClozeExample(
                context_ids=ctx_ids,
                query_ids=qry_ids,
                answer_ids=ans_ids,
                entity_idx=e,
            ))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        # Pad / truncate context+query to seq_len.
        full = torch.cat([ex.context_ids, ex.query_ids, ex.answer_ids], dim=0)
        full = full[: self.seq_len]
        if len(full) < self.seq_len:
            pad = torch.full(
                (self.seq_len - len(full),), self.mask_id, dtype=torch.long,
            )
            full = torch.cat([full, pad], dim=0)
        # input_ids = full[:-1], labels = full[1:] -- standard LM next-token shift.
        return {
            "input_ids": full[:-1],
            "labels": full[1:],
            "entity_idx": torch.tensor(ex.entity_idx, dtype=torch.long),
            # Token position of the answer: where the model needs to retrieve
            # the value via the bank.
            "answer_pos": torch.tensor(
                len(ex.context_ids) + len(ex.query_ids) - 1, dtype=torch.long,
            ).clamp_max(self.seq_len - 2),
        }


def make_entity_cloze_loaders(
    tokenizer,
    n_entities: int = 64,
    n_examples_train: int = 4096,
    n_examples_val: int = 512,
    seq_len: int = 32,
    seed: int = 0,
):
    """Convenience constructor returning (train_ds, val_ds) for entity cloze."""
    train_ds = EntityClozeDataset(
        n_entities=n_entities, n_examples=n_examples_train,
        seq_len=seq_len, tokenizer=tokenizer, seed=seed,
    )
    val_ds = EntityClozeDataset(
        n_entities=n_entities, n_examples=n_examples_val,
        seq_len=seq_len, tokenizer=tokenizer, seed=seed + 1,
    )
    return train_ds, val_ds
