# ==========================
# file: qllm_utils.py
# ==========================
import os, json, math, torch
from typing import List, Dict, Any


def device_str() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def bytes_encode(text: str) -> List[int]:
    return list(text.encode("utf-8", errors="ignore"))


def bytes_decode(ids: List[int]) -> str:
    return bytes([int(x) % 256 for x in ids]).decode("utf-8", errors="ignore")


def causal_mask(seq_len: int, device: str):
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


def save_args_json(path: str, args: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(args, f, indent=2)


def load_args_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def reconcile_model_args(cli: Dict[str, Any], saved: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prefer CLI overrides *only* if explicitly set (not None). Otherwise, use saved.
    Ensures seq_length and global_tokens match to avoid pos_embed mismatch.
    """
    out = saved.copy()
    for k, v in cli.items():
        if v is not None:
            out[k] = v
    return out