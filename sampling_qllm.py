# ==========================
# file: sampling_qllm.py
# ==========================
import torch
import torch.nn.functional as F
from typing import List, Optional
import random
import math

def top_k_logits(logits, k):
    if k == 0:
        return logits
    v, _ = torch.topk(logits, k)
    minv = v[:, -1].unsqueeze(-1)
    return torch.where(logits < minv, torch.full_like(logits, -float("Inf")), logits)

def top_p_logits(logits, p):
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # remove tokens with cumulative probability above p
    sorted_indices_to_remove = cumulative_probs > p
    # keep at least one token
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits_flat = logits.flatten()
    logits_flat[indices_to_remove] = -float("Inf")
    return logits_flat.view_as(logits)

def apply_repetition_penalty(logits, recent_ids: List[int], penalty: float):
    if penalty == 1.0 or len(recent_ids) == 0:
        return logits
    # logits: [B, V]
    for rid in set(recent_ids):
        logits[..., rid] /= penalty
    return logits

def sample_next_token(logits: torch.Tensor,
                      temperature: float = 1.0,
                      top_k: int = 0,
                      top_p: float = 1.0,
                      repetition_penalty: float = 1.0,
                      recent_ids: Optional[List[List[int]]] = None,
                      min_p: float = 0.0) -> torch.Tensor:
    """
    logits: [B, V] tensor of logits for next token
    recent_ids: list of lists, recent ids per batch (each an int list). We'll use only last row when B==1 commonly.
    returns: tensor of sampled ids shape [B]
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    # basic temperature scaling
    if temperature != 1.0:
        logits = logits / max(1e-8, temperature)

    # repetition penalty
    if recent_ids is not None and repetition_penalty != 1.0:
        # apply on batch 0 only if recent_ids is nested: recent_ids[0] list
        rids = recent_ids[0] if isinstance(recent_ids, list) and len(recent_ids) > 0 else []
        logits = apply_repetition_penalty(logits, rids, repetition_penalty)

    # top-k
    if top_k > 0:
        logits = top_k_logits(logits, top_k)

    # top-p / nucleus
    if top_p < 1.0:
        logits = top_p_logits(logits, top_p)

    # min_p: simple floor on probs
    probs = F.softmax(logits, dim=-1)
    if min_p > 0.0:
        probs = torch.clamp(probs, min=min_p)
        probs = probs / probs.sum(dim=-1, keepdim=True)

    # sample
    next_id = torch.multinomial(probs, num_samples=1)  # [B,1]
    return next_id.squeeze(1)
