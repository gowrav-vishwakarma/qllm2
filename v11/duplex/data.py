"""Synthetic Stage-0 duplex dataset."""

from typing import Iterator, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from v11.duplex.interleave import generate_conversation, pad_batch
from v11.duplex.thinking import VOCAB


class SyntheticDuplexDataset(Dataset):
    """Fixed-size dataset of random synthetic conversations."""

    def __init__(
        self,
        n_samples: int = 4096,
        n_blocks: int = 8,
        seed: int = 42,
        barge_in_prob: float = 0.35,
    ):
        self.samples: List[Tuple[List[int], List[int]]] = []
        rng = np.random.default_rng(seed)
        for _ in range(n_samples):
            inp, lab, _ = generate_conversation(
                n_blocks=n_blocks,
                rng=rng,
                barge_in_prob=barge_in_prob,
            )
            if len(inp) < 4:
                continue
            self.samples.append((inp, lab))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inp, lab = self.samples[idx]
        return (
            torch.tensor(inp, dtype=torch.long),
            torch.tensor(lab, dtype=torch.long),
        )


def collate_duplex(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    seqs = [(inp.tolist(), lab.tolist()) for inp, lab in batch]
    input_ids, labels, attn = pad_batch(seqs, pad_id=VOCAB.pad)
    return input_ids, labels, attn


class StreamingDuplexDataset(IterableDataset):
    """Infinite stream of synthetic duplex batches (for long training)."""

    def __init__(
        self,
        batch_size: int = 16,
        n_blocks: int = 8,
        seed: int = 0,
        barge_in_prob: float = 0.35,
    ):
        self.batch_size = batch_size
        self.n_blocks = n_blocks
        self.seed = seed
        self.barge_in_prob = barge_in_prob

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        rng = np.random.default_rng(self.seed)
        while True:
            batch = []
            for _ in range(self.batch_size):
                inp, lab, _ = generate_conversation(
                    n_blocks=self.n_blocks,
                    rng=rng,
                    barge_in_prob=self.barge_in_prob,
                )
                batch.append((inp, lab))
            yield pad_batch(batch, pad_id=VOCAB.pad)
