"""
Memory-aligned training objectives for V6.

SpanCorruptionDataset: T5-style span masking that trains cross-gap reasoning.
DelayedRecallDataset: fact-then-cue tasks that train write/retrieve directly.

Both operate on the same token streams as standard TextDataset and produce
(input_ids, labels, loss_mask) where loss_mask selects which positions
contribute to the loss.
"""

import random
import torch
from torch.utils.data import Dataset
from typing import List


class SpanCorruptionDataset(Dataset):
    """Corrupt contiguous spans and train the model to infill them.

    For each chunk of tokens, randomly masks ~corruption_rate of positions
    in contiguous spans of mean_length. The model sees sentinel tokens in
    place of masked spans and must predict the original tokens.

    Loss is computed only on masked positions, forcing the model to build
    relational context across the gap.
    """

    SENTINEL_BASE = 50254

    def __init__(
        self,
        tokens: torch.Tensor,
        seq_len: int = 512,
        corruption_rate: float = 0.15,
        mean_length: int = 3,
        seed: int = 42,
    ):
        self.seq_len = seq_len
        self.corruption_rate = corruption_rate
        self.mean_length = mean_length
        self.rng = random.Random(seed)

        n_chunks = len(tokens) // (seq_len + 1)
        self.data = tokens[:n_chunks * (seq_len + 1)].view(n_chunks, seq_len + 1)

    def __len__(self):
        return self.data.shape[0]

    def _sample_spans(self, length: int) -> List[tuple]:
        """Sample non-overlapping corruption spans."""
        target_masked = max(1, int(length * self.corruption_rate))
        spans = []
        total_masked = 0
        attempts = 0
        occupied = set()

        while total_masked < target_masked and attempts < 100:
            span_len = max(1, int(self.rng.expovariate(1.0 / self.mean_length)))
            span_len = min(span_len, length - 1, target_masked - total_masked)
            start = self.rng.randint(0, length - span_len - 1)

            overlap = any(pos in occupied for pos in range(start, start + span_len))
            if not overlap:
                spans.append((start, start + span_len))
                for pos in range(start, start + span_len):
                    occupied.add(pos)
                total_masked += span_len
            attempts += 1

        return sorted(spans)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        input_ids = chunk[:-1].clone()
        labels = chunk[1:].clone()
        seq_len = len(input_ids)

        spans = self._sample_spans(seq_len)

        loss_mask = torch.zeros(seq_len, dtype=torch.bool)

        for span_idx, (start, end) in enumerate(spans):
            sentinel = self.SENTINEL_BASE - span_idx
            if sentinel < 0:
                sentinel = 0
            input_ids[start] = sentinel
            for pos in range(start + 1, end):
                input_ids[pos] = sentinel
            for pos in range(start, end):
                loss_mask[pos] = True

        return {
            'input_ids': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
        }


class DelayedRecallDataset(Dataset):
    """Train the model to recall facts from earlier in the sequence.

    Constructs training pairs where a fact region early in the chunk must
    be recalled when a cue appears later, with a configurable gap between
    them. Loss is weighted more heavily on recall positions.

    The gap ensures the model must store the fact across many tokens,
    directly training the memory write/retrieve path.
    """

    def __init__(
        self,
        tokens: torch.Tensor,
        seq_len: int = 512,
        gap: int = 64,
        fact_len: int = 8,
        seed: int = 42,
    ):
        self.seq_len = seq_len
        self.gap = gap
        self.fact_len = fact_len
        self.rng = random.Random(seed)

        n_chunks = len(tokens) // (seq_len + 1)
        self.data = tokens[:n_chunks * (seq_len + 1)].view(n_chunks, seq_len + 1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        chunk = self.data[idx]
        input_ids = chunk[:-1].clone()
        labels = chunk[1:].clone()
        seq_len = len(input_ids)

        loss_mask = torch.ones(seq_len, dtype=torch.float32)

        min_fact_start = 4
        max_fact_start = seq_len - self.gap - self.fact_len - 4
        if max_fact_start > min_fact_start:
            fact_start = self.rng.randint(min_fact_start, max_fact_start)
            fact_end = fact_start + self.fact_len
            recall_start = fact_end + self.gap
            recall_end = min(recall_start + self.fact_len, seq_len)

            if recall_end <= seq_len:
                for pos in range(recall_start, recall_end):
                    loss_mask[pos] = 3.0
                for pos in range(fact_start, fact_end):
                    loss_mask[pos] = 2.0

        return {
            'input_ids': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
        }
