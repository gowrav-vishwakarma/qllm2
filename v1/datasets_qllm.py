# ==========================
# file: datasets_qllm.py
# ==========================
from typing import Optional, Tuple
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, Subset
import torch
from qllm_utils import bytes_encode
import random
import itertools


class ByteLMChunked(Dataset):
    """
    Concatenate text, convert to bytes [0..255], make fixed-length chunks (stride == seq_length).
    Supports optional streaming-ish behavior for large corpora (by sampling slices).
    """
    def __init__(self, dataset_name: str, split: str, seq_length: int, max_samples: Optional[int] = None, streaming: bool = False):
        self.seq_length = seq_length
        texts = []

        if dataset_name == "wikitext2":
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            texts = [ex["text"] for ex in ds]
        elif dataset_name == "tinystories":
            ds = load_dataset("roneneldan/TinyStories", split=split)
            texts = [ex["text"] for ex in ds]
        elif dataset_name == "c4_en_small":
            # Try different approaches for C4 dataset compatibility
            try:
                # First try: use the newer C4 dataset format
                ds = load_dataset("c4", "en", split=split if split in ["validation"] else "train[:1%]", trust_remote_code=True)
                texts = [ex["text"] for ex in ds]
            except Exception as e1:
                try:
                    # Second try: use a different C4 variant
                    ds = load_dataset("allenai/c4", "en", split=split if split in ["validation"] else "train[:1%]")
                    texts = [ex["text"] for ex in ds]
                except Exception as e2:
                    try:
                        # Third try: use a smaller C4 sample
                        ds = load_dataset("c4", "en", split=split if split in ["validation"] else "train[:0.1%]")
                        texts = [ex["text"] for ex in ds]
                    except Exception as e3:
                        print(f"Warning: C4 dataset not available, falling back to wikitext2")
                        print(f"Errors: {e1}, {e2}, {e3}")
                        # Fallback to wikitext2
                        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
                        texts = [ex["text"] for ex in ds]
        elif dataset_name == "fineweb_sample":
            try:
                ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train[:1%]")
                texts = [ex.get("text", "") for ex in ds]
            except:
                print("Warning: FineWeb dataset not available, falling back to wikitext2")
                ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
                texts = [ex["text"] for ex in ds]
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        if 'toks' not in locals():
            joined = "\n\n".join(texts)
            toks = bytes_encode(joined)

        if max_samples is not None:
            cap = max_samples * seq_length
            toks = toks[: max(cap, seq_length)]

        n = len(toks)
        self.x, self.y = [], []
        # stride can be seq_length (non-overlapping) or smaller to add overlap
        stride = seq_length
        for i in range(0, n - seq_length - 1, stride):
            chunk = toks[i : i + seq_length]
            target = toks[i + 1 : i + 1 + seq_length]
            if len(chunk) == seq_length and len(target) == seq_length:
                self.x.append(torch.tensor(chunk, dtype=torch.long))
                self.y.append(torch.tensor(target, dtype=torch.long))

        # If dataset is enormous but user requested small max_samples, we may sample subset
        if max_samples is not None and len(self.x) > max_samples:
            indices = random.sample(range(len(self.x)), max_samples)
            self.x = [self.x[i] for i in indices]
            self.y = [self.y[i] for i in indices]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def build_loaders(dataset: str, seq_length: int, batch_size: int, max_samples: Optional[int],
                  streaming: bool = False, num_workers: int = 4, drop_last: bool = True):
    """
    Builds DataLoader objects. For large data use streaming=True to sample slices.
    """
    if dataset == "wikitext2":
        train_ds = ByteLMChunked("wikitext2", "train", seq_length, max_samples, streaming=streaming)
        val_ds = ByteLMChunked("wikitext2", "validation", seq_length, None, streaming=False)
    else:
        full = ByteLMChunked(dataset, "train", seq_length, max_samples, streaming=streaming)
        n = len(full)
        val_n = max(64, n // 100)
        train_n = max(0, n - val_n)
        train_ds = Subset(full, list(range(0, train_n)))
        val_ds = Subset(full, list(range(train_n, n)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, drop_last=drop_last)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=max(1, num_workers//2), drop_last=False)
    return train_loader, val_loader
