"""
v13_sempty — a simple, pure Phase-Associative Memory (PAM) language model
on the sempyt named-axis framework.

What this model is
------------------

Every layer carries one fixed-size *notebook* per head — a complex d×d
matrix — that remembers everything seen so far, one association at a time:

    notebook_t  =  decay_t * notebook_{t-1}  +  value_t (x) conjugate(key_t)
    read_t      =  scale * ( notebook_t . query_t )

In words, each token:

  1. fades what the notebook holds, by a learned per-head number decay_t in
     (0, 1);
  2. writes one new association — the token's value, hung on the phase of its
     key (the write is conjugate, so it can later be recalled exactly);
  3. reads what the notebook now says about its query.

That is the whole memory. No gates, no erasures, no extra states: the decay
is the only learned knob on the notebook, and it is a single number per head.

Position comes from RoPE, applied to the query and the key. The recurrence
alone only knows recency (a decaying number); word order needs a position
carrier, and RoPE is the free one — a buffer and one complex multiply, no
parameters. Every lean variant in v7/v11 kept it; learned input embeddings
were measured neutral there, so they stay out (see EXPERIMENTS_SEMPY.md).

Training and decoding run the same algebra two ways:

  * train / prefill  — time is processed in windows of ``chunk_size``. Inside
    a window the recurrence has a closed form: if a_s is the product of the
    decays up to position s, then

        notebook_s = a_s * ( notebook_in + sum_{j <= s} write_j / a_j )

    so the window is one cumulative sum, and the notebook at the window's end
    is carried into the next. This is O(T) work, never O(T^2).
  * decode           — the same recurrence, one step per token, on the
    carried notebook. O(1) per token, independent of context length.

The two agree to round-off (selftest ``test_parallel_vs_recurrent``).

Style
-----

The code speaks in the model's own words: ``decay``, ``write``, ``notebook``,
``read``, ``rotation``. All layout is named — ``.to()`` / ``.alias()`` /
``contract`` / ``outer`` / ``cumprod`` — and ``check_torch_layout.py`` fails
the build if anything reaches for ``view`` / ``permute`` / ``[..., 0]``. Raw
torch appears only at the declared boundaries: the RoPE table (built once,
outside the graph), the tied-logit hand-off, the chunked-CE Function, and
the sampling loop.

Named axes
----------
  batch             sequence items in the minibatch
  time              token positions
  model_dim         residual / embedding width
  heads             PAM heads
  head_feature      per-head channel width (d)
  complex_pair      last axis of size 2: real then imag
  qkv_slot / qkv_fused   fused Q / K / V packing
  head_row / head_col    the two axes of the d x d notebook
  chunk_time        token positions inside one window
  real_imag_feature  concat(real, imag) along model_dim
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from sempyt.dim import Dim, ProductDim
from sempyt.nn import Linear as NamedLinear
from sempyt.ops import contract, imag, outer, real
from sempyt.policies import SplitComplex
from sempyt.structural import (
    cat, cumprod, exp, flatten, select, softplus, stack, take, zeros,
)
from sempyt.tensor import NamedTensor, named

from v13_sempty.config import PAMConfig
from v13_sempty.complex_ops import (
    ComplexEmbed,
    ComplexGatedUnit,
    ComplexLinear,
    ComplexNorm,
    as_complex_dropout_mask,
    build_rope_cache,
    to_real_concat,
)


# ── Phase-Associative Memory ─────────────────────────────────────────────────


class PAMLayer(nn.Module):
    r"""One head-fan of phase-associative memories.

    Per head, a complex d×d notebook ``S`` carries the whole past:

        S_t = decay_t * S_{t-1} + value_t (x) conj(key_t)
        y_t = d^{-1/2} * (S_t . query_t)

    ``decay_t = exp(-softplus(linear(real+imag of token) + bias))`` is the
    only learned memory quantity — a single number per head per token. The
    write conjugates the key (so a later read with the same key recalls the
    value); the read uses the raw query. Reads are *after* the write, so a
    token can recall what it just wrote.

    ``forward`` returns the PAM output ``[batch, time, inner]`` (inner =
    heads × head_feature, folded back to model_dim by the output projection)
    and the notebook to carry on (raw ``[batch, heads, d, d, 2]`` per
    ``state``; ``None`` when nothing was carried in and nothing should be
    returned).
    """

    def __init__(self, cfg: PAMConfig, layer_idx: int = 0,
                 model_dim: Dim | None = None, complex_pair: Dim | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.model_dim = model_dim or Dim("model_dim", cfg.dim)
        self.complex_pair = complex_pair or Dim("complex_pair", 2)
        self.policy = SplitComplex(self.complex_pair)
        self.heads = Dim("heads", cfg.n_heads)
        self.head_feature = Dim("head_feature", cfg.head_dim)
        self.head_row = Dim("head_row", cfg.head_dim)
        self.head_col = Dim("head_col", cfg.head_dim)
        self.inner = ProductDim(self.heads, self.head_feature)
        self.real_imag_feature = Dim("real_imag_feature", cfg.dim * 2)

        # Fused QKV: one projection, then split. The three share the same
        # width, so one matrix is the whole fan.
        self.qkv_slot = Dim("qkv_slot", 3)
        self.qkv_fused = Dim("qkv_fused", 3 * cfg.n_heads * cfg.head_dim)
        self.qkv_proj = ComplexLinear(self.model_dim, self.qkv_fused,
                                      bias=False, pair=self.complex_pair)
        self.o_proj = ComplexLinear(self.inner, self.model_dim,
                                    bias=False, pair=self.complex_pair)

        # The decay: one number per head, read from the token's real+imag.
        self.decay_out = Dim("decay_out", cfg.n_heads)
        self.dt_proj = NamedLinear(self.real_imag_feature, self.decay_out)
        self.dt_bias = nn.Parameter(torch.zeros(cfg.n_heads) + cfg.base_dt_bias)

        # RoPE table, built once outside the graph (declared boundary).
        if cfg.use_rope:
            self.register_buffer(
                'rope_cache',
                build_rope_cache(cfg.max_seq_len, cfg.head_dim),
                persistent=False,
            )
        self.use_rope = cfg.use_rope

        self.chunk_size = cfg.chunk_size
        self.head_dim = cfg.head_dim
        self.out_dropout = nn.Dropout(cfg.dropout)

    # ── small named helpers ──────────────────────────────────────────────────

    def _as_token(self, x: NamedTensor) -> NamedTensor:
        """Re-state the token axes under this layer's own Dim identities.

        Each forward wraps its batch/time in fresh Dims, and a carried
        notebook keeps the Dims of the call that created it; restating is
        an identity change only — no data moves.
        """
        batch, time = x.layout[0], x.layout[1]
        return named(x.data, (batch, time, self.model_dim, self.complex_pair),  # named-exit: restate
                     self.policy)

    def _as_notebook(self, state, batch: Dim) -> NamedTensor:
        """Re-state a carried notebook onto this call's axes (no data moves)."""
        layout = (batch, self.heads, self.head_row, self.head_col, self.complex_pair)
        data = state.data if isinstance(state, NamedTensor) else state  # named-exit: restate
        return named(data, layout, self.policy)

    def _project(self, tokens: NamedTensor, step_offset: int):
        """Query, key, value on (batch, heads, time, head_feature, complex_pair).

        The fused QKV axis splits with its factors at the end of the layout —
        the one order sempty's named split accepts — then heads move before
        time for the notebook. RoPE rotates query and key by their position,
        so the notebook's associations carry word order.
        """
        batch, time = tokens.layout[0], tokens.layout[1]
        qkv = self.qkv_proj(tokens)
        qkv = qkv.to(batch, time, self.qkv_slot, self.heads, self.head_feature,
                     self.complex_pair)
        queries, keys, values = (qkv.select(self.qkv_slot, slot) for slot in (0, 1, 2))

        if self.use_rope:
            position_end = step_offset + tokens.size(time)
            if position_end > self.rope_cache.shape[0]:
                self.register_buffer(
                    'rope_cache',
                    build_rope_cache(position_end * 2, self.head_dim).to(tokens.device),
                    persistent=False,
                )
            rotation = named(
                self.rope_cache[step_offset:position_end].to(dtype=tokens.dtype),
                (time, self.head_feature, self.complex_pair), self.policy,
            )  # named-exit: the RoPE table is a plain buffer, sliced here
            queries = queries * rotation
            keys = keys * rotation

        layout = (batch, self.heads, time, self.head_feature, self.complex_pair)
        return (queries.to(*layout), keys.to(*layout), values.to(*layout))

    def _decay(self, tokens: NamedTensor) -> NamedTensor:
        """Per-head retention in (0, 1): exp(-softplus(linear(token) + bias))."""
        batch, time = tokens.layout[0], tokens.layout[1]
        real_imag = to_real_concat(tokens, into=self.real_imag_feature)
        logit = self.dt_proj(real_imag).alias(self.decay_out, self.heads)
        retention = exp(-softplus(logit + named(self.dt_bias, (self.heads,))))
        return retention.to(batch, self.heads, time)

    # ── train / prefill: one closed form per window, notebook carried ────────

    def _chunked(self, tokens: NamedTensor, queries, keys, values,
                 decay) -> tuple[NamedTensor, NamedTensor]:
        seq_len = tokens.size(tokens.layout[1])
        time = tokens.layout[1]
        carried = None  # the notebook coming in from earlier windows
        reads = []
        for start in range(0, seq_len, self.chunk_size):
            length = min(self.chunk_size, seq_len - start)
            window = Dim("chunk_time", length)

            write = outer(
                take(values, over=time, start=start, length=length, new=window)
                    .alias(self.head_feature, self.head_row),
                take(keys, over=time, start=start, length=length, new=window)
                    .alias(self.head_feature, self.head_col).conj(),
                (self.head_row, self.head_col),
            )
            window_decay = take(decay, over=time, start=start, length=length,
                                new=window)
            window_query = take(queries, over=time, start=start, length=length,
                                new=window).alias(self.head_feature, self.head_col)

            # Closed form of S_t = g_t S_{t-1} + u_t inside this window: with
            # a_s = prod_{j <= s} g_j,  S_s = a_s * (S_in + sum_{j<=s} u_j / a_j).
            retention = cumprod(window_decay, over=window)
            accumulated = (write / retention).cumsum(over=window)
            window_notebook = (retention * (accumulated + carried)
                               if carried is not None else retention * accumulated)

            reads.append(
                contract(window_notebook, window_query, over=self.head_col)
                * (self.head_dim ** -0.5)
            )
            carried = select(window_notebook, over=window, index=length - 1)
        output = cat(reads, over="chunk_time", into=time)
        return output, carried

    # ── decode: one step per token on the carried notebook ───────────────────

    def _stepwise(self, tokens: NamedTensor, queries, keys, values, decay,
                  state) -> tuple[NamedTensor, NamedTensor]:
        batch, time = tokens.layout[0], tokens.layout[1]
        seq_len = tokens.size(time)
        notebook = self._as_notebook(state, batch) if state is not None else zeros(
            batch, self.heads, self.head_row, self.head_col, self.complex_pair,
            policy=self.policy, device=tokens.device, dtype=tokens.dtype,
        )
        reads = []
        for t in range(seq_len):
            # The recurrence, one step: fade, write, then read.
            notebook = (select(decay, over=time, index=t) * notebook) + outer(
                select(values, over=time, index=t).alias(self.head_feature, self.head_row),
                select(keys, over=time, index=t).alias(self.head_feature, self.head_col).conj(),
                (self.head_row, self.head_col),
            )
            reads.append(
                contract(notebook,
                         select(queries, over=time, index=t)
                         .alias(self.head_feature, self.head_col),
                         over=self.head_col)
                * (self.head_dim ** -0.5)
            )
        output = stack(reads, into=time, at=self.head_row)
        return output, notebook

    def forward(self, x: NamedTensor, state=None, step_offset: int = 0):
        tokens = self._as_token(x)
        seq_len = tokens.size(tokens.layout[1])
        queries, keys, values = self._project(tokens, step_offset)
        decay = self._decay(tokens)

        if state is None and seq_len > 1:
            output, new_state = self._chunked(tokens, queries, keys, values, decay)
        else:
            output, new_state = self._stepwise(tokens, queries, keys, values, decay, state)

        out = self.o_proj(output.alias(self.head_row, self.head_feature)
                          .to(tokens.layout[0], tokens.layout[1], self.inner,
                              self.complex_pair))
        if self.training:
            out = out * as_complex_dropout_mask(self.out_dropout, out)
        return out, (new_state.data if new_state is not None else None)  # named-exit: state hand-off


# ── Block ────────────────────────────────────────────────────────────────────


class Block(nn.Module):
    """Pre-norm residual: gated channel mix, then PAM sequence mix."""

    def __init__(self, cfg: PAMConfig, layer_idx: int = 0,
                 model_dim: Dim | None = None, complex_pair: Dim | None = None):
        super().__init__()
        self.model_dim = model_dim or Dim("model_dim", cfg.dim)
        self.complex_pair = complex_pair or Dim("complex_pair", 2)
        self.dropout = cfg.dropout
        self.norm1 = ComplexNorm(self.model_dim, pair=self.complex_pair)
        self.cgu = ComplexGatedUnit(self.model_dim, cfg.expand,
                                    activation=cfg.activation,
                                    pair=self.complex_pair)
        self.cgu_dropout = nn.Dropout(cfg.dropout)
        self.cgu_scale = nn.Parameter(torch.tensor(1.0))
        self.norm2 = ComplexNorm(self.model_dim, pair=self.complex_pair)
        self.pam = PAMLayer(cfg, layer_idx=layer_idx,
                            model_dim=self.model_dim, complex_pair=self.complex_pair)
        # PAM starts soft: the memory path learns while the residual carries.
        self.pam_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: NamedTensor, pam_state=None, step_offset: int = 0):
        cgu_out = self.cgu(self.norm1(x))
        if self.training:
            cgu_out = cgu_out * as_complex_dropout_mask(self.cgu_dropout, cgu_out)
        x = x + cgu_out * self.cgu_scale
        pam_out, new_state = self.pam(self.norm2(x), state=pam_state,
                                      step_offset=step_offset)
        return x + pam_out * self.pam_scale, new_state


# ── Language model ───────────────────────────────────────────────────────────


class LM(nn.Module):
    """ComplexEmbed -> [Block] x N -> tied complex head.

    The head is tied to the embedding: the score of a candidate token is the
    real dot product of its embedding with the (complex) hidden state,
    ``real . embed_real + imag . embed_imag``.
    """

    def __init__(self, cfg: PAMConfig):
        super().__init__()
        self.config = cfg
        self.model_dim = Dim("model_dim", cfg.dim)
        self.complex_pair = Dim("complex_pair", 2)
        self.real_imag_feature = Dim("real_imag_feature", cfg.dim * 2)
        self.policy = SplitComplex(self.complex_pair)
        self.embed = ComplexEmbed(cfg.vocab_size, self.model_dim, self.complex_pair)
        self.embed_norm = ComplexNorm(self.model_dim, pair=self.complex_pair)
        self.blocks = nn.ModuleList([
            Block(cfg, layer_idx=i,
                  model_dim=self.model_dim, complex_pair=self.complex_pair)
            for i in range(cfg.n_layers)
        ])
        self.output_norm = ComplexNorm(self.model_dim, pair=self.complex_pair)
        # A distinct out-axis: contract cannot tell two copies of one Dim apart.
        self.lm_head_out = Dim("lm_head_out", cfg.dim)
        self.lm_head_proj = ComplexLinear(self.model_dim, self.lm_head_out,
                                          pair=self.complex_pair)
        self.lm_head_norm = ComplexNorm(self.model_dim, pair=self.complex_pair)
        self._init_weights()

    def _init_weights(self):
        embed_weights = {self.embed.embed_real, self.embed.embed_imag}
        for module in self.modules():
            if isinstance(module, (nn.Linear, NamedLinear)):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding) and module not in embed_weights:
                nn.init.normal_(module.weight, std=0.02)

    def _stem(self, input_ids, step_offset: int = 0) -> tuple:
        batch, time = Dim("batch", input_ids.shape[0]), Dim("time", input_ids.shape[1])
        return self.embed_norm(self.embed(input_ids, batch, time)), batch, time

    def _tied_logits(self, lm: NamedTensor, batch: Dim, time: Dim) -> torch.Tensor:
        embed_real = named(self.embed.embed_real.weight, (self.embed.vocab, self.model_dim))
        embed_imag = named(self.embed.embed_imag.weight, (self.embed.vocab, self.model_dim))
        logits = (contract(real(lm), embed_real, over=self.model_dim)
                  + contract(imag(lm), embed_imag, over=self.model_dim))
        return logits.raw(batch, time, self.embed.vocab)  # named-exit: public logits are raw

    def _run_blocks(self, z, batch, time, states, step_offset):
        use_ckpt = self.config.gradient_checkpointing and self.training and states is None
        new_states = []
        for i, block in enumerate(self.blocks):
            state = states[i] if states is not None else None
            if use_ckpt:
                z, new_state = self._ckpt_block(block, z, step_offset)
            else:
                z, new_state = block(z, pam_state=state, step_offset=step_offset)
            new_states.append(new_state)
        return z, new_states

    def forward(self, input_ids, states=None, step_offset: int = 0, labels=None):
        z, batch, time = self._stem(input_ids, step_offset)
        z, new_states = self._run_blocks(z, batch, time, states, step_offset)
        lm = self.lm_head_norm(self.lm_head_proj(self.output_norm(z)))
        logits = self._tied_logits(lm, batch, time)
        aux_loss = torch.zeros((), device=input_ids.device)
        return logits, new_states, aux_loss

    def _hidden_to_lm(self, input_ids, step_offset: int = 0):
        """Training path: the stack, stopped before full [B, T, V] logits."""
        z, batch, time = self._stem(input_ids, step_offset)
        z, _ = self._run_blocks(z, batch, time, states=None, step_offset=step_offset)
        lm = self.lm_head_norm(self.lm_head_proj(self.output_norm(z)))
        aux_loss = torch.zeros((), device=input_ids.device)
        return lm, aux_loss

    def ce_from_lm(self, lm: NamedTensor, labels, loss_mask=None,
                   ignore_index=-100, chunk: int = 4096, return_nll: bool = False):
        """Chunked cross-entropy from the pre-logit complex hidden ``lm``.

        The tied head folds into one real matmul: ``H @ W^T`` with
        ``H = concat(lm_real, lm_imag)`` and ``W = concat(embed_real,
        embed_imag)``. The chunked-CE Function never holds the full
        ``[N, vocab]`` softmax. Named up to the Function hand-off.
        """
        from v13_sempty.fused_ce import fused_linear_cross_entropy
        batch, time = lm.layout[0], lm.layout[1]
        flat = Dim("flat", batch.size * time.size)

        hidden = flatten(to_real_concat(lm, into=self.real_imag_feature),
                         (batch, time), flat)
        embed_real = named(self.embed.embed_real.weight, (self.embed.vocab, self.model_dim))
        embed_imag = named(self.embed.embed_imag.weight, (self.embed.vocab, self.model_dim))
        weight = cat([embed_real, embed_imag], over=self.model_dim,
                     into=self.real_imag_feature)

        out = fused_linear_cross_entropy(
            hidden.raw(flat, self.real_imag_feature),  # named-exit: the CE Function takes raw torch
            weight.raw(self.embed.vocab, self.real_imag_feature),  # named-exit: ditto
            labels.reshape(-1),  # named-exit: raw int tensor from the dataloader
            mask=(loss_mask.reshape(-1) if loss_mask is not None else None),  # named-exit: raw mask
            chunk=chunk, ignore_index=ignore_index, return_nll=return_nll,
        )
        if return_nll:
            return out, getattr(out, '_nll', None).reshape(batch.size, time.size)  # named-exit: raw nll
        return out

    def fused_ce_loss(self, input_ids, labels, loss_mask=None, ignore_index=-100,
                      chunk: int = 4096):
        """Eager convenience: the hidden stack plus chunked CE in one call."""
        lm, aux_loss = self._hidden_to_lm(input_ids)
        main = self.ce_from_lm(lm, labels, loss_mask=loss_mask,
                               ignore_index=ignore_index, chunk=chunk)
        return main, aux_loss

    @staticmethod
    def _ckpt_block(block, z, step_offset):
        # Non-reentrant checkpoint: backward recomputes the block with grad.
        # Do NOT detach the input — the checkpoint needs the input edge to
        # push gradient back through the preceding block.
        compiled = getattr(block, '_compiled_fwd', None)

        def run(z_in):
            if compiled is not None:
                return compiled(z_in, pam_state=None, step_offset=step_offset)
            return block(z_in, pam_state=None, step_offset=step_offset)

        return grad_checkpoint(run, z, use_reentrant=False)

    def compile_blocks(self, mode: str = 'default'):
        """Compile each Block.forward so inductor can see the PAM chunk."""
        for block in self.blocks:
            block._compiled_fwd = torch.compile(block.forward, mode=mode, dynamic=False)
        return self

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0,
                 top_k=50, top_p=0.0, repetition_penalty=1.0, eos_token_id=None):
        """Autoregressive decode: prefill builds the notebooks, then O(1)/token."""
        self.eval()
        generated = input_ids.clone()
        logits, states, _ = self.forward(generated)
        step = generated.shape[1]
        finished = torch.zeros(generated.shape[0], dtype=torch.bool,
                               device=generated.device)
        for _ in range(max_new_tokens):
            if temperature <= 0.0:
                # Greedy: the argmax is the whole distribution.
                next_logits = logits[:, -1]
                if repetition_penalty != 1.0:
                    score = torch.gather(next_logits, 1, generated)
                    score = torch.where(score > 0, score / repetition_penalty,
                                        score * repetition_penalty)
                    next_logits.scatter_(1, generated, score)
                nxt = next_logits.argmax(-1, keepdim=True)
            else:
                next_logits = logits[:, -1] / temperature
                if repetition_penalty != 1.0:
                    score = torch.gather(next_logits, 1, generated)
                    score = torch.where(score > 0, score / repetition_penalty,
                                        score * repetition_penalty)
                    next_logits.scatter_(1, generated, score)
                if top_k > 0:
                    v, _ = next_logits.topk(min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, -1:]] = float('-inf')
                if top_p > 0:
                    sl, si = next_logits.sort(descending=True)
                    cum = sl.softmax(dim=-1).cumsum(dim=-1)
                    sl[cum - sl.softmax(dim=-1) >= top_p] = float('-inf')
                    next_logits = sl.scatter(1, si, sl)
                nxt = torch.multinomial(next_logits.softmax(dim=-1), 1)
            generated = torch.cat([generated, nxt], dim=1)
            if eos_token_id is not None:
                finished |= nxt.squeeze(1) == eos_token_id
                if bool(finished.all()):
                    break
            logits, states, _ = self.forward(nxt, states=states, step_offset=step)
            step += 1
        return generated

    def count_parameters(self) -> Dict[str, int]:
        embed_p = sum(p.numel() for p in self.embed.parameters())
        block_p = sum(p.numel() for b in self.blocks for p in b.parameters())
        head_p = (sum(p.numel() for p in self.lm_head_proj.parameters())
                  + sum(p.numel() for p in self.lm_head_norm.parameters()))
        norm_p = (sum(p.numel() for p in self.embed_norm.parameters())
                  + sum(p.numel() for p in self.output_norm.parameters()))
        total = embed_p + block_p + head_p + norm_p
        return {'embedding (tied)': embed_p, 'blocks': block_p,
                'norms': norm_p, 'lm_head': head_p, 'total': total}


__all__ = ["PAMLayer", "Block", "LM"]
