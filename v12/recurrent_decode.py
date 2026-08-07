"""Batched recurrent decode step for V12 (eager, compile-safe).

Today ``V12LM.generate`` loops Python-over-layers each token. This helper fuses
the *structure* of a one-token decode into a single function suitable for a
future Triton/CUDA kernel. Training still uses the chunked parallel path.

Usage:
    from v12.recurrent_decode import decode_one_token
    logits, states = decode_one_token(model, token_id, states, step_offset)
"""

from __future__ import annotations

import torch

from v12.complex_ops import imag_part, real_part, stack_complex


@torch.inference_mode()
def decode_one_token(model, token_id: torch.Tensor, states, step_offset: int):
    """One greedy-ready logits row [B,V] and updated PAM states."""
    if token_id.dim() == 1:
        token_id = token_id.unsqueeze(-1)
    logits, new_states, _aux = model.forward(token_id, states=states, step_offset=step_offset)
    return logits[:, -1, :], new_states


def pam_state_nbytes(states) -> int:
    """Approximate recurrent state size (bytes) for logging."""
    if states is None:
        return 0
    total = 0
    for s in states:
        if s is None:
            continue
        if isinstance(s, list):
            for t in s:
                if t is not None:
                    total += t.numel() * t.element_size()
        else:
            total += s.numel() * s.element_size()
    return total
