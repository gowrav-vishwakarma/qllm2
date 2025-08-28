# ==========================
# file: qllm_utils.py
# ==========================
import os
import json
import math
import torch
from typing import List, Dict, Any, Optional


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
    # returns shape (1,1,seq_len,seq_len) with -inf above diag
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
    Prefer CLI overrides only if explicitly set (not None). Otherwise use saved.
    Ensures seq_length/global tokens match to avoid pos_embed mismatch.
    """
    out = saved.copy()
    for k, v in cli.items():
        if v is not None:
            out[k] = v
    return out


# --- distributed helpers (optional; for torchrun) ---
def is_dist() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank() -> int:
    if not is_dist():
        return 0
    return torch.distributed.get_rank()


def get_world_size() -> int:
    if not is_dist():
        return 1
    return torch.distributed.get_world_size()


def ddp_setup(backend: str = "nccl"):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "29500")
        init_method = f"tcp://{addr}:{port}"
        torch.distributed.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(local_rank)
        return True
    # If not launched with torchrun, try environment LOCAL_RANK
    if "LOCAL_RANK" in os.environ:
        try:
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            return True
        except:
            pass
    return False


def save_checkpoint(state: Dict[str, Any], path: str, rank: Optional[int] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if rank is None or rank == 0:
        torch.save(state, path)
