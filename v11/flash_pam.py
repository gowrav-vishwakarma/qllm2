"""Flash-PAM: faster training-time evaluation of the PAM chunked dual form.

Background / constraint (see EXPERIMENTS_V11.md "Flash-PAM design constraint"):
custom-autograd Triton kernels historically fight `torch.compile` on this split-real
complex stack (they hit the `is_compiling()` guard or cause graph breaks), and removing
Triton in V7 gave +66% tok/s. So the *primary* Flash-PAM is a pure-PyTorch
**chunk-parallel** reformulation that inductor can fuse end-to-end — no custom autograd.

Baseline (`V11PAMLayer._forward_chunked_head`) loops over chunks in Python and runs the
expensive intra-chunk GEMMs (score [C,C] and AV [C,d]) *inside* that loop. Flash-PAM
instead:
  * folds the chunk axis into the batch and runs **all** intra-chunk GEMMs as one big
    batched matmul (better GPU utilisation, fewer launches), then
  * runs only the cheap d x d **state carry** sequentially over the n chunks.

It is numerically identical to `_forward_chunked_head` (same math, just regrouped), so
gradients match and it is a drop-in. Complex tensors are split-real `[..., 2]` throughout.

Public:
    flash_pam_chunked_head(queries, keys, protected_values, decay_gamma, head_dim, chunk_size)
        queries, keys, protected_values: [B,H,T,d,2] (queries NOT pre-scaled; scaled by
        head_dim**-0.5 inside, matching the baseline).
        decay_gamma: [B,H,T] head-scalar decay in (0,1].
        Returns output [B,H,T,d,2] and final memory_state [B,H,d,d,2].
"""

import math
import torch
from torch import Tensor


def _chunk_decay_matrices(decay_gamma_batch: Tensor, chunk_len: int):
    """decay_gamma_batch: [N, C] -> (decay_matrix, cumulative_decay, total_decay, decay_last).

    decay_matrix[t,s]      = prod_{s<u<=t} decay_gamma   (lower-tri)   [N,C,C]
    cumulative_decay[t]    = prod_{0<=u<=t} decay_gamma  (inclusive)    [N,C]
    total_decay            = cumulative_decay[:, -1]                    [N]
    decay_last[s]          = prod_{s<u<=C-1} decay_gamma = total/cum_excl [N,C]
    """
    compute_dtype = (
        torch.float32 if decay_gamma_batch.dtype in (torch.bfloat16, torch.float16)
        else decay_gamma_batch.dtype
    )
    log_decay = torch.log(decay_gamma_batch.to(compute_dtype) + 1e-6)
    cum_neg_log = torch.cumsum(-log_decay, dim=-1)
    log_decay_matrix = (cum_neg_log.unsqueeze(-1) - cum_neg_log.unsqueeze(-2)).transpose(-1, -2)
    causal = torch.tril(torch.ones(chunk_len, chunk_len, device=decay_gamma_batch.device))
    log_decay_matrix = log_decay_matrix * causal + (1 - causal) * (-1e4)
    decay_matrix = torch.exp(log_decay_matrix.clamp(max=0.0))
    cumulative_decay = torch.exp(torch.cumsum(log_decay, dim=-1))
    total_decay = cumulative_decay[:, -1]
    decay_last = decay_matrix[:, -1, :]
    return decay_matrix, cumulative_decay, total_decay, decay_last


def flash_pam_chunked_head(
    queries: Tensor,
    keys: Tensor,
    protected_values: Tensor,
    decay_gamma: Tensor,
    head_dim: int,
    chunk_size: int,
):
    """Chunk-parallel equivalent of V11PAMLayer._forward_chunked_head (head decay)."""
    batch_size, num_heads, seq_len = queries.shape[:3]
    chunk_len = chunk_size
    query_scale = head_dim ** -0.5
    num_chunks = (seq_len + chunk_len - 1) // chunk_len
    padded_seq_len = num_chunks * chunk_len
    pad_len = padded_seq_len - seq_len

    scaled_queries = queries * query_scale
    if pad_len:
        # pad keys/values with 0 (contribute nothing), decay_gamma with 1 (log 0).
        scaled_queries = torch.nn.functional.pad(scaled_queries, (0, 0, 0, 0, 0, pad_len))
        keys = torch.nn.functional.pad(keys, (0, 0, 0, 0, 0, pad_len))
        protected_values = torch.nn.functional.pad(protected_values, (0, 0, 0, 0, 0, pad_len))
        decay_gamma = torch.nn.functional.pad(decay_gamma, (0, pad_len), value=1.0)

    # [B,H,n,C,...] -> fold (B,H,n) into one batch axis num_batch_chunks.
    num_batch_chunks = batch_size * num_heads * num_chunks
    queries_chunk = scaled_queries.view(batch_size, num_heads, num_chunks, chunk_len, head_dim, 2).reshape(
        num_batch_chunks, chunk_len, head_dim, 2
    )
    keys_chunk = keys.view(batch_size, num_heads, num_chunks, chunk_len, head_dim, 2).reshape(
        num_batch_chunks, chunk_len, head_dim, 2
    )
    values_chunk = protected_values.view(batch_size, num_heads, num_chunks, chunk_len, head_dim, 2).reshape(
        num_batch_chunks, chunk_len, head_dim, 2
    )
    decay_gamma_chunk = decay_gamma.view(batch_size, num_heads, num_chunks, chunk_len).reshape(
        num_batch_chunks, chunk_len
    )

    decay_matrix, cumulative_decay, total_decay, decay_last = _chunk_decay_matrices(
        decay_gamma_chunk, chunk_len
    )

    query_real, query_imag = queries_chunk[..., 0], queries_chunk[..., 1]
    key_real, key_imag = keys_chunk[..., 0], keys_chunk[..., 1]
    value_real, value_imag = values_chunk[..., 0], values_chunk[..., 1]

    score_real = torch.bmm(query_real, key_real.transpose(-1, -2)) + torch.bmm(query_imag, key_imag.transpose(-1, -2))
    score_imag = torch.bmm(query_imag, key_real.transpose(-1, -2)) - torch.bmm(query_real, key_imag.transpose(-1, -2))
    decay_matrix_typed = decay_matrix.to(score_real.dtype)
    weighted_real, weighted_imag = score_real * decay_matrix_typed, score_imag * decay_matrix_typed
    output_real = torch.bmm(weighted_real, value_real) - torch.bmm(weighted_imag, value_imag)
    output_imag = torch.bmm(weighted_real, value_imag) + torch.bmm(weighted_imag, value_real)
    intra_output = torch.stack([output_real, output_imag], dim=-1)

    decay_last_expanded = decay_last.unsqueeze(-1).to(value_real.dtype)
    write_value_real, write_value_imag = value_real * decay_last_expanded, value_imag * decay_last_expanded
    state_real = torch.bmm(write_value_real.transpose(-1, -2), key_real) + torch.bmm(write_value_imag.transpose(-1, -2), key_imag)
    state_imag = torch.bmm(write_value_imag.transpose(-1, -2), key_real) - torch.bmm(write_value_real.transpose(-1, -2), key_imag)
    state_chunk = torch.stack([state_real, state_imag], dim=-1)

    intra_output = intra_output.view(batch_size, num_heads, num_chunks, chunk_len, head_dim, 2)
    state_chunk = state_chunk.view(batch_size, num_heads, num_chunks, head_dim, head_dim, 2)
    cumulative_decay = cumulative_decay.view(batch_size, num_heads, num_chunks, chunk_len).to(queries.dtype)
    total_decay = total_decay.view(batch_size, num_heads, num_chunks).to(queries.dtype)
    scaled_queries_chunked = scaled_queries.view(batch_size, num_heads, num_chunks, chunk_len, head_dim, 2)

    outputs = []
    memory_state = queries.new_zeros(batch_size, num_heads, head_dim, head_dim, 2)
    for chunk_idx in range(num_chunks):
        output_chunk = intra_output[:, :, chunk_idx]
        if chunk_idx > 0:
            state_real, state_imag = memory_state[..., 0], memory_state[..., 1]
            query_real_chunk, query_imag_chunk = (
                scaled_queries_chunked[:, :, chunk_idx, ..., 0],
                scaled_queries_chunked[:, :, chunk_idx, ..., 1],
            )
            carried_real = (
                state_real @ query_real_chunk.transpose(-1, -2)
                - state_imag @ query_imag_chunk.transpose(-1, -2)
            ).transpose(-1, -2)
            carried_imag = (
                state_real @ query_imag_chunk.transpose(-1, -2)
                + state_imag @ query_real_chunk.transpose(-1, -2)
            ).transpose(-1, -2)
            cumulative_decay_expanded = cumulative_decay[:, :, chunk_idx].unsqueeze(-1)
            output_chunk = output_chunk + torch.stack(
                [carried_real * cumulative_decay_expanded, carried_imag * cumulative_decay_expanded],
                dim=-1,
            )
        outputs.append(output_chunk)
        memory_state = memory_state * total_decay[:, :, chunk_idx][..., None, None, None] + state_chunk[:, :, chunk_idx]

    output = torch.cat(outputs, dim=2)
    if pad_len:
        output = output[:, :, :seq_len]
    return output, memory_state
