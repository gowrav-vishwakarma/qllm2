"""Named PAM primitives — the E3 delta chunk kernel written in sempyt.

Everything here takes and returns ``NamedTensor``. The only torch that appears
is inside ``v13_sempty.complex_ops.fused_decay_matrix`` (a [time, time] lag
table) and inside ``sempyt.solve_triangular``.

Axis vocabulary used throughout:
  memory_states   E3 notebooks (K)
  batch, heads    minibatch item, PAM head
  chunk_time      token positions inside one chunk
  write_row       chunk position of the write / query  (chunk_time alias)
  source_col      chunk position of the key / source   (chunk_time alias)
  head_row        notebook value axis   (d_i)
  head_col        notebook key axis     (d_j)
  head_feature    per-head channel of Q / K / V
  complex_pair    real, imag
"""

from __future__ import annotations

from sempyt.dim import Dim
from sempyt.ops import as_complex, contract, outer, solve_triangular
from sempyt.structural import (
    cos,
    cumsum,
    exp,
    log,
    select,
    sin,
    tril_mask,
)
from sempyt.tensor import NamedTensor

from v13_sempty.complex_ops import named_decay_matrix


def conjugate_scores(left: NamedTensor, right: NamedTensor, *, over: Dim) -> NamedTensor:
    """``Σ_f left[f] · conj(right[f])`` — the Q·K* / K·K* score of one chunk."""
    return contract(left, right.conj(), over=over)


def cumulative_decay(decay_gamma: NamedTensor, *, over: Dim) -> NamedTensor:
    """Inclusive decay product ``exp(cumsum(log γ))`` along ``over``."""
    return exp(cumsum(log(decay_gamma + 1e-6), over=over))


def decay_matrix(decay_gamma: NamedTensor, *, over: Dim, rows: Dim, cols: Dim) -> NamedTensor:
    """Lag table ``D[t, s] = Π_{s<i≤t} γ_i``, causal-masked (v13 lag kernel).

    ``over`` is the time axis of ``decay_gamma``; it becomes ``rows`` (target
    position) against ``cols`` (source position).
    """
    return named_decay_matrix(decay_gamma.alias(over, rows), rows, cols)


def phase_rotation(
    routing_weights: NamedTensor,
    retrieval_phase: NamedTensor,
    *,
    complex_pair: Dim,
) -> NamedTensor:
    """Retrieval gain ``α · e^{iφ}`` as one complex tensor."""
    return as_complex(
        routing_weights * cos(retrieval_phase),
        routing_weights * sin(retrieval_phase),
        complex_pair,
    )


def delta_chunk(
    *,
    key_gram: NamedTensor,
    query_key: NamedTensor,
    write: NamedTensor,
    decay_gamma: NamedTensor,
    cumulative: NamedTensor,
    erase_beta: NamedTensor,
    chunk_time: Dim,
    write_row: Dim,
    source_col: Dim,
    head_row: Dim,
    query_scale: float,
    factored: bool,
) -> tuple[NamedTensor, NamedTensor]:
    """Solve one delta-rule chunk. Returns ``(read on write_row, write on source_col)``.

    The intra-chunk delta rule is the unit-lower-triangular system
    ``(I + M) update = write`` with ``M[t,s] = β_e(t) · D[t,s] · (K K*)[t,s]``
    below the diagonal. ``solve_triangular`` broadcasts the state axis of
    ``write`` against the state-free system, so all K notebooks share one
    factorisation without any packing.

    ``factored=True`` substitutes ``update = diag(α) · y``, which cancels the
    decay table out of the system entirely (``D[t,s] = α_t / α_s``) — cheaper,
    but only safe while ``min α`` stays clear of zero.
    """
    erase_rows = erase_beta.alias(chunk_time, write_row)
    alpha_rows = cumulative.alias(chunk_time, write_row)
    alpha_cols = cumulative.alias(chunk_time, source_col)
    alpha_last = select(cumulative, over=chunk_time, index=-1)
    strictly_past = tril_mask(
        write_row, source_col, diagonal=-1,
        device=key_gram.device, dtype=key_gram.dtype,
    )
    causal = tril_mask(
        write_row, source_col,
        device=key_gram.device, dtype=key_gram.dtype,
    )

    if factored:
        mass = key_gram * (erase_rows * strictly_past)
        solution = solve_triangular(
            mass, write / alpha_rows,
            over=(write_row, source_col), feature=head_row, unitriangular=True,
        )
        read = contract(query_key * causal, solution, over=source_col)
        return read * alpha_rows * query_scale, solution * alpha_last

    decay = decay_matrix(decay_gamma, over=chunk_time, rows=write_row, cols=source_col)
    mass = key_gram * (erase_rows * decay * strictly_past)
    solution = solve_triangular(
        mass, write,
        over=(write_row, source_col), feature=head_row, unitriangular=True,
    )
    read = contract(query_key * (decay * causal), solution, over=source_col)
    return read * query_scale, solution * (alpha_last / (alpha_cols + 1e-12))


def recur_step_delta(
    memory: NamedTensor,
    decay_gamma: NamedTensor,
    value: NamedTensor,
    key: NamedTensor,
    query: NamedTensor,
    write_beta: NamedTensor,
    erase_beta: NamedTensor,
    *,
    head_row: Dim,
    head_col: Dim,
) -> tuple[NamedTensor, NamedTensor]:
    """One gated delta step of the decode recurrence.

    Forget, predict what the notebook already says about this key, write the
    correction, then read with the query. ``memory`` is ``(..., head_row,
    head_col, pair)``; ``key``/``query`` live on ``head_col`` and ``value`` on
    ``head_row``.
    """
    memory = memory * decay_gamma
    predicted = contract(memory, key, over=head_col)
    update = write_beta * value - erase_beta * predicted
    memory = memory + outer(update, key.conj(), (head_row, head_col))
    return contract(memory, query, over=head_col), memory
