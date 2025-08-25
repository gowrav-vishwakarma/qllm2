# ==========================
# file: datasets_qllm.py
# ==========================
from typing import Optional, Tuple
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, Subset
import torch
from qllm_utils import bytes_encode


class ByteLMChunked(Dataset):
    """
    Concatenate text, convert to bytes [0..255], make fixed-length chunks (stride == seq_length).
    """
    def __init__(self, dataset_name: str, split: str, seq_length: int, max_samples: Optional[int] = None):
        self.seq_length = seq_length
        texts = []

        if dataset_name == "wikitext2":
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            texts = [ex["text"] for ex in ds]
        elif dataset_name == "tinystories":
            ds = load_dataset("roneneldan/TinyStories", split=split)
            texts = [ex["text"] for ex in ds]
        elif dataset_name == "c4_en_small":
            # Reasonable alternative to deprecated OpenWebText script; cap via max_samples
            # Using a small slice to keep it feasible on single GPU
            ds = load_dataset("c4", "en", split=split if split in ["validation"] else "train[:1%]")
            texts = [ex["text"] for ex in ds]
        elif dataset_name == "fineweb_sample":
            ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train[:1%]")
            texts = [ex.get("text", "") for ex in ds]
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        joined = "\n\n".join(texts)
        toks = bytes_encode(joined)

        if max_samples is not None:
            cap = max_samples * seq_length
            toks = toks[: max(cap, seq_length)]

        n = len(toks)
        self.x, self.y = [], []
        for i in range(0, n - seq_length - 1, seq_length):
            chunk = toks[i : i + seq_length]
            target = toks[i + 1 : i + 1 + seq_length]
            if len(chunk) == seq_length and len(target) == seq_length:
                self.x.append(torch.tensor(chunk, dtype=torch.long))
                self.y.append(torch.tensor(target, dtype=torch.long))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def build_loaders(dataset: str, seq_length: int, batch_size: int, max_samples: Optional[int]):
    if dataset == "wikitext2":
        train_ds = ByteLMChunked("wikitext2", "train", seq_length, max_samples)
        val_ds = ByteLMChunked("wikitext2", "validation", seq_length, None)
    else:
        full = ByteLMChunked(dataset, "train", seq_length, max_samples)
        n = len(full)
        val_n = max(64, n // 100)
        train_n = max(0, n - val_n)
        train_ds = Subset(full, list(range(0, train_n)))
        val_ds = Subset(full, list(range(train_n, n)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2, drop_last=False)
    return train_loader, val_loader
