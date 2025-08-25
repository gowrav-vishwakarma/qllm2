# ==========================
# file: sampling_qllm.py
# ==========================
import torch
from typing import List

@torch.no_grad()
def sample_next_token(logits, temperature=1.0, top_k=0, top_p=0.0, repetition_penalty=1.0, recent_ids=None, min_p=0.0):
    if repetition_penalty != 1.0 and recent_ids is not None:
        for b in range(logits.size(0)):
            for tid in recent_ids[b]:
                logits[b, tid] /= repetition_penalty
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    logits = logits / max(1e-8, temperature)
    if top_k > 0:
        v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        kth = v[:, -1].unsqueeze(-1)
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cums = torch.cumsum(probs, dim=-1)
        mask = cums > top_p
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = torch.empty_like(logits).scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
    if min_p > 0.0:
        probs = torch.softmax(logits, dim=-1)
        logits = torch.where(probs < min_p, torch.full_like(logits, float("-inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

