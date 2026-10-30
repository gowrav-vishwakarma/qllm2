"""Selftest for the simple PAM model (CPU, float32, self-contained).

The model is intentionally NOT compared against v13 — it is a different,
leaner architecture. Instead the tests pin its own contract:

  * test_param_count_and_state   — parameter accounting and state_dict sanity
  * test_parallel_vs_recurrent   — the equivalence contract: the chunked
                                   window path (train / prefill) and the
                                   stepwise path (decode) agree on logits and
                                   on the carried notebook, to round-off
  * test_tied_logits             — the tied head equals the manual
                                   real @ E_r.T + imag @ E_i.T score,
                                   built with the same named axes the model uses
  * test_fused_ce                — chunked fused CE equals plain
                                   F.cross_entropy in loss and gradients
  * test_smoke_loss_decreases    — the tiny preset trains on synthetic data
  * test_generate_smoke          — the decode loop runs and stays in-vocab
  * test_real_*                  — the same six for the fully-real arm
  * test_real_fused_kernel_parity— (CUDA only) the Triton fused PAM scan and
                                   its torch form agree on logits, carried
                                   notebooks and every parameter gradient

This module is a declared raw-torch boundary (it compares against plain
``F.cross_entropy``), so raw exits are legal here — but the named tensors
stay named up to each hand-off, exactly as the model does.

Run:
    .venv/bin/python -m v13_sempty.selftest
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sempyt.dim import Dim
from sempyt.structural import cat
from sempyt.tensor import named

from v13_sempty.config import get_config
from v13_sempty.complex_ops import to_real_concat
from v13_sempty.model import LM
from v13_sempty.train import Trainer, synthetic_loader

RECUR_ATOL = 1e-4  # fp32 tolerance: closed form vs one-step-per-token


def _unwrap(t):
    return t.data if hasattr(t, "layout") else t


def _max_diff(a, b) -> float:
    return (_unwrap(a).float() - _unwrap(b).float()).abs().max().item()


def test_param_count_and_state():
    cfg = get_config('tiny')
    model = LM(cfg)
    params = model.count_parameters()
    assert params['total'] > 0
    assert params['total'] == sum(p.numel() for p in model.parameters()), \
        "count_parameters disagrees with the actual parameter total"
    sd = model.state_dict()
    assert sd['embed.embed_real.weight'].shape == (cfg.vocab_size, cfg.dim)
    n = cfg.n_layers
    assert f'blocks.{n - 1}.pam.o_proj.weight_real' in sd
    assert f'blocks.{n - 1}.pam.dt_bias' in sd
    # The RoPE table is a buffer, not a parameter, and is not checkpointed.
    assert 'blocks.0.pam.rope_cache' not in sd
    print(f"  params total={params['total']:,}")
    return True


def test_parallel_vs_recurrent(batch_size=2, seq_len=17, seed=0):
    """Chunked (windows of chunk_size) vs one-token-at-a-time, same algebra.

    seq_len is deliberately longer than chunk_size so the window loop carries
    the notebook across more than one window: 17 = 7 + 7 + 3.
    """
    torch.manual_seed(seed)
    cfg = get_config('tiny')
    cfg.chunk_size = 7  # force multiple windows
    model = LM(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits_par, states_par, _ = model.forward(ids)
        # Stepwise: one token at a time, carrying the notebooks.
        logits_list, states = [], None
        for t in range(seq_len):
            lt, states, _ = model.forward(
                ids[:, t:t + 1], states=states, step_offset=t,
            )
            logits_list.append(lt)
        logits_seq = torch.cat(logits_list, dim=1)

    diff_logits = (logits_par - logits_seq).abs().max().item()
    diff_state = max(
        _max_diff(sp, ss) for sp, ss in zip(states_par, states)
    )
    assert diff_logits < RECUR_ATOL, f"logits disagree: {diff_logits:.3e}"
    assert diff_state < RECUR_ATOL, f"carried notebook disagrees: {diff_state:.3e}"
    print(f"  max |logit diff| = {diff_logits:.3e}   "
          f"max |state diff| = {diff_state:.3e}")
    return True


def test_tied_logits(batch_size=2, seq_len=8, seed=1):
    """The tied head equals real @ E_r.T + imag @ E_i.T, named end to end.

    The manual score is built with the same named axes the model's own
    ``ce_from_lm`` uses: the two embedding matrices are ``named`` onto the
    vocab x model_dim layout and joined along ``model_dim`` into
    ``real_imag_feature``; the hidden state is folded to the same axis.
    """
    torch.manual_seed(seed)
    cfg = get_config('tiny')
    model = LM(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits, _, _ = model.forward(ids)
        # Re-run the stack with our own batch/time axes (the model wraps ids
        # in fresh Dims; the data is identical in eval).
        batch, time = Dim("batch", batch_size), Dim("time", seq_len)
        z = model.embed_norm(model.embed(ids, batch, time))
        for block in model.blocks:
            z, _ = block(z, pam_state=None, step_offset=0)
        lm = model.lm_head_norm(model.lm_head_proj(model.output_norm(z)))

        embed_real = named(model.embed.embed_real.weight,
                           (model.embed.vocab, model.model_dim))
        embed_imag = named(model.embed.embed_imag.weight,
                           (model.embed.vocab, model.model_dim))
        weight = cat([embed_real, embed_imag], over=model.model_dim,
                     into=model.real_imag_feature)

        flat = to_real_concat(lm, into=model.real_imag_feature).raw(
            batch, time, model.real_imag_feature)
        manual = flat @ weight.raw(model.embed.vocab, model.real_imag_feature).T

    diff = (logits - manual).abs().max().item()
    assert diff < 1e-4, f"tied logits disagree: {diff:.3e}"
    print(f"  max |tied logit diff| = {diff:.3e}")
    return True


def test_fused_ce(batch_size=2, seq_len=16, seed=0):
    torch.manual_seed(seed)
    cfg = get_config('tiny')
    model = LM(cfg)
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    # Plain CE through the public forward (raw boundary: F.cross_entropy).
    model.zero_grad()
    logits, _, _ = model.forward(ids, labels=labels)
    plain = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    plain.backward()
    plain_grads = {n: p.grad.clone() for n, p in model.named_parameters()
                   if p.grad is not None}

    # Chunked fused CE through the training path (small chunk forces many).
    model.zero_grad()
    lm, _aux = model._hidden_to_lm(ids)
    fused = model.ce_from_lm(lm, labels, chunk=5)
    fused.backward()
    fused_grads = {n: p.grad.clone() for n, p in model.named_parameters()
                   if p.grad is not None}

    loss_diff = (plain - fused).abs().item()
    max_grad_diff = max(
        (pg - fg).abs().max().item()
        for pg, fg in zip(plain_grads.values(), fused_grads.values())
    )
    assert set(plain_grads) == set(fused_grads), "parameter gradient sets differ"
    assert max_grad_diff < 1e-3, f"gradients disagree: {max_grad_diff:.3e}"
    print(f"  loss diff = {loss_diff:.3e}   max grad diff = {max_grad_diff:.3e}")
    return True


def test_smoke_loss_decreases(steps=12, seed=0):
    torch.manual_seed(seed)
    cfg = get_config('tiny')
    cfg.vocab_size = 256
    model = LM(cfg)
    loader = synthetic_loader(256, batch_size=4, seq_len=32,
                              n_batches=steps, seed=seed)
    trainer = Trainer(model, loader, learning_rate=3e-4, warmup_steps=2,
                      total_steps=steps, device=torch.device('cpu'))
    losses = trainer.train(max_steps=steps)
    delta = losses[-1] - losses[0]
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    assert delta < 0, f"loss did not decrease: {losses}"
    print(f"  loss {losses[0]:.4f} -> {losses[-1]:.4f}  (delta {delta:+.4f})")
    return True


def test_generate_smoke(seed=0):
    torch.manual_seed(seed)
    cfg = get_config('tiny')
    cfg.vocab_size = 256
    model = LM(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    out = model.generate(ids, max_new_tokens=4, temperature=0.0, top_k=1)
    assert out.shape == (1, 12), f"bad shape {out.shape}"
    assert (out >= 0).all() and (out < cfg.vocab_size).all()
    print(f"  generated {out.shape[1] - 8} tokens, shape {tuple(out.shape)}")
    return True


# ── Fully-real twins: same contracts, the tiny_real preset ─────────────────


def test_real_param_count_and_state():
    cfg = get_config('tiny_real')
    model = LM(cfg)
    params = model.count_parameters()
    assert params['total'] > 0
    assert params['total'] == sum(p.numel() for p in model.parameters()), \
        "count_parameters disagrees with the actual parameter total"
    sd = model.state_dict()
    # The real model has one embedding, not a real/imag pair.
    assert sd['embed.embed.weight'].shape == (cfg.vocab_size, cfg.dim)
    assert 'embed.embed_real.weight' not in sd
    n = cfg.n_layers
    assert f'blocks.{n - 1}.pam.o_proj.weight' in sd
    assert f'blocks.{n - 1}.pam.dt_bias' in sd
    # The RoPE table is a buffer, not a parameter, and is not checkpointed.
    assert 'blocks.0.pam.rope_cache' not in sd
    print(f"  params total={params['total']:,}")
    return True


def test_real_parallel_vs_recurrent(batch_size=2, seq_len=17, seed=0):
    """Chunked window path vs one-token-at-a-time, real arithmetic.

    seq_len is deliberately longer than chunk_size so the window loop carries
    the notebook across more than one window: 17 = 7 + 7 + 3.
    """
    torch.manual_seed(seed)
    cfg = get_config('tiny_real')
    cfg.chunk_size = 7  # force multiple windows
    model = LM(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits_par, states_par, _ = model.forward(ids)
        # Stepwise: one token at a time, carrying the notebooks.
        logits_list, states = [], None
        for t in range(seq_len):
            lt, states, _ = model.forward(
                ids[:, t:t + 1], states=states, step_offset=t,
            )
            logits_list.append(lt)
        logits_seq = torch.cat(logits_list, dim=1)

    diff_logits = (logits_par - logits_seq).abs().max().item()
    diff_state = max(
        _max_diff(sp, ss) for sp, ss in zip(states_par, states)
    )
    assert diff_logits < RECUR_ATOL, f"logits disagree: {diff_logits:.3e}"
    assert diff_state < RECUR_ATOL, f"carried notebook disagrees: {diff_state:.3e}"
    print(f"  max |logit diff| = {diff_logits:.3e}   "
          f"max |state diff| = {diff_state:.3e}")
    return True


def test_real_tied_logits(batch_size=2, seq_len=8, seed=1):
    """The real tied head equals ``hidden @ embed.T``, named end to end."""
    torch.manual_seed(seed)
    cfg = get_config('tiny_real')
    model = LM(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits, _, _ = model.forward(ids)
        # Re-run the stack with our own batch/time axes (the model wraps ids
        # in fresh Dims; the data is identical in eval).
        batch, time = Dim("batch", batch_size), Dim("time", seq_len)
        z = model.embed_norm(model.embed(ids, batch, time))
        for block in model.blocks:
            z, _ = block(z, pam_state=None, step_offset=0)
        lm = model.lm_head_norm(model.lm_head_proj(model.output_norm(z)))

        flat = lm.raw(batch, time, model.model_dim)
        manual = flat @ model.embed.embed.weight.T

    diff = (logits - manual).abs().max().item()
    assert diff < 1e-4, f"tied logits disagree: {diff:.3e}"
    print(f"  max |tied logit diff| = {diff:.3e}")
    return True


def test_real_fused_ce(batch_size=2, seq_len=16, seed=0):
    torch.manual_seed(seed)
    cfg = get_config('tiny_real')
    model = LM(cfg)
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    # Plain CE through the public forward (raw boundary: F.cross_entropy).
    model.zero_grad()
    logits, _, _ = model.forward(ids, labels=labels)
    plain = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    plain.backward()
    plain_grads = {n: p.grad.clone() for n, p in model.named_parameters()
                   if p.grad is not None}

    # Chunked fused CE through the training path (small chunk forces many).
    model.zero_grad()
    lm, _aux = model._hidden_to_lm(ids)
    fused = model.ce_from_lm(lm, labels, chunk=5)
    fused.backward()
    fused_grads = {n: p.grad.clone() for n, p in model.named_parameters()
                   if p.grad is not None}

    loss_diff = (plain - fused).abs().item()
    max_grad_diff = max(
        (pg - fg).abs().max().item()
        for pg, fg in zip(plain_grads.values(), fused_grads.values())
    )
    assert set(plain_grads) == set(fused_grads), "parameter gradient sets differ"
    assert max_grad_diff < 1e-3, f"gradients disagree: {max_grad_diff:.3e}"
    print(f"  loss diff = {loss_diff:.3e}   max grad diff = {max_grad_diff:.3e}")
    return True


def test_real_smoke_loss_decreases(steps=12, seed=0):
    torch.manual_seed(seed)
    cfg = get_config('tiny_real')
    cfg.vocab_size = 256
    model = LM(cfg)
    loader = synthetic_loader(256, batch_size=4, seq_len=32,
                              n_batches=steps, seed=seed)
    trainer = Trainer(model, loader, learning_rate=3e-4, warmup_steps=2,
                      total_steps=steps, device=torch.device('cpu'))
    losses = trainer.train(max_steps=steps)
    delta = losses[-1] - losses[0]
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    assert delta < 0, f"loss did not decrease: {losses}"
    print(f"  loss {losses[0]:.4f} -> {losses[-1]:.4f}  (delta {delta:+.4f})")
    return True


def test_real_generate_smoke(seed=0):
    torch.manual_seed(seed)
    cfg = get_config('tiny_real')
    cfg.vocab_size = 256
    model = LM(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    out = model.generate(ids, max_new_tokens=4, temperature=0.0, top_k=1)
    assert out.shape == (1, 12), f"bad shape {out.shape}"
    assert (out >= 0).all() and (out < cfg.vocab_size).all()
    print(f"  generated {out.shape[1] - 8} tokens, shape {tuple(out.shape)}")
    return True


def test_real_fused_kernel_parity(batch_size=2, seq_len=130, seed=0):
    """Triton fused PAM scan vs the torch form, through the whole real model.

    CUDA only (skipped otherwise). Same model, same batch; kernel on vs off
    must agree on logits, carried notebooks and every parameter gradient.
    seq_len 130 = two 64-tiles plus a partial one; chunk 7 for the torch form.
    """
    from v13_sempty.triton_kernels import kernel_enabled, set_kernel_enabled
    if not (torch.cuda.is_available() and kernel_enabled()):
        print("  skipped (no CUDA / Triton)")
        return True
    torch.manual_seed(seed)
    cfg = get_config('tiny_real')
    cfg.chunk_size = 7
    model = LM(cfg).cuda().train()
    ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device='cuda')
    labels = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device='cuda')

    def run(on: bool):
        set_kernel_enabled(on)
        model.zero_grad(set_to_none=True)
        logits, states, _ = model.forward(ids)
        F.cross_entropy(logits.reshape(-1, cfg.vocab_size), labels.reshape(-1)).backward()
        grads = [p.grad.detach().clone() for p in model.parameters()]
        return logits.detach(), [s.detach() for s in states], grads

    try:
        lg_k, st_k, gr_k = run(True)
        lg_t, st_t, gr_t = run(False)
    finally:
        set_kernel_enabled(True)
    d_logits = _max_diff(lg_k, lg_t)
    d_state = max(_max_diff(a, b) for a, b in zip(st_k, st_t))
    d_grad = max((a - b).abs().max().item() / (b.abs().max().item() + 1e-12)
                 for a, b in zip(gr_k, gr_t))
    assert d_logits < RECUR_ATOL, f"logits disagree: {d_logits:.3e}"
    assert d_state < RECUR_ATOL, f"carried notebook disagrees: {d_state:.3e}"
    assert d_grad < 1e-3, f"parameter grads disagree: rel {d_grad:.3e}"
    print(f"  max |logit diff| = {d_logits:.3e}   max |state diff| = {d_state:.3e}   "
          f"max rel grad diff = {d_grad:.3e}")
    return True


def test_complex_fused_kernel_parity(batch_size=2, seq_len=130, seed=0):
    """Complex fused PAM scan vs the _stable_notebook form, whole complex model.

    CUDA only. Kernel on (fused_complex_pam_read) vs off (per-window
    _stable_notebook) must agree on logits, carried notebooks and every
    parameter gradient. Same shape convention as the real parity test.
    """
    from v13_sempty.triton_kernels import kernel_enabled, set_kernel_enabled
    if not (torch.cuda.is_available() and kernel_enabled()):
        print("  skipped (no CUDA / Triton)")
        return True
    torch.manual_seed(seed)
    cfg = get_config('tiny')
    cfg.chunk_size = 7
    model = LM(cfg).cuda().train()
    ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device='cuda')
    labels = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device='cuda')

    def run(on: bool):
        set_kernel_enabled(on)
        model.zero_grad(set_to_none=True)
        logits, states, _ = model.forward(ids)
        F.cross_entropy(logits.reshape(-1, cfg.vocab_size), labels.reshape(-1)).backward()
        grads = [p.grad.detach().clone() for p in model.parameters()]
        return logits.detach(), [s.detach() for s in states], grads

    try:
        lg_k, st_k, gr_k = run(True)
        lg_t, st_t, gr_t = run(False)
    finally:
        set_kernel_enabled(True)
    d_logits = _max_diff(lg_k, lg_t)
    d_state = max(_max_diff(a, b) for a, b in zip(st_k, st_t))
    d_grad = max((a - b).abs().max().item() / (b.abs().max().item() + 1e-12)
                 for a, b in zip(gr_k, gr_t))
    assert d_logits < 1e-4, f"logits disagree: {d_logits:.3e}"
    assert d_state < 1e-4, f"carried notebook disagrees: {d_state:.3e}"
    assert d_grad < 2e-3, f"parameter grads disagree: rel {d_grad:.3e}"
    print(f"  max |logit diff| = {d_logits:.3e}   max |state diff| = {d_state:.3e}   "
          f"max rel grad diff = {d_grad:.3e}")
    return True


def test_chrono_rotary_parity(batch_size=2, seq_len=24, seed=0):
    """N1 Chrono-PAM at init == baseline RoPE (bit-parity) + warp grads flow.

    The learned time-warp W is zero-init, so g=1 and the cumulative phase is
    exactly pos*inv_freq -- identical to fixed RoPE. Runs on CPU (no CUDA
    needed): chrono-on logits must equal the same model with chrono flipped
    off, and one backward must reach every warp_proj.
    """
    torch.manual_seed(seed)
    cfg = get_config('tiny_real')
    cfg.chrono = True
    model = LM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    logits_on, _, _ = model.forward(ids)
    for blk in model.blocks:
        blk.pam.chrono = False
    logits_off, _, _ = model.forward(ids)
    d = (logits_on - logits_off).abs().max().item()
    assert d < 1e-4, f"chrono@init != baseline RoPE: {d:.3e}"
    for blk in model.blocks:
        blk.pam.chrono = True
    model.train()
    model.zero_grad(set_to_none=True)
    logits, _, _ = model.forward(ids)
    F.cross_entropy(logits.reshape(-1, cfg.vocab_size), ids.reshape(-1)).backward()
    n_none = sum(1 for blk in model.blocks if blk.pam.warp_proj.weight.grad is None)
    assert n_none == 0, f"{n_none} warp_proj got no grad"
    print(f"  max |logit diff| = {d:.3e}   warp grads: all {len(model.blocks)} present")
    return True


def test_chrono_parallel_vs_recurrent(batch_size=2, seq_len=17, seed=0):
    """Chrono decode: chunked prefill == prefill(prefix) + one-token steps.

    The learned warp is set to random non-zero weights so the clock is
    genuinely content-dependent (not plain RoPE). The carried state is
    (notebook, clock); logits from the full chunked pass must equal those
    from a 5-token prefill followed by 12 stepwise tokens, and a pure
    token-by-token run from an empty state.
    """
    torch.manual_seed(seed)
    cfg = get_config('tiny_real')
    cfg.chunk_size = 7
    cfg.chrono = True
    model = LM(cfg).eval()
    for blk in model.blocks:
        nn.init.normal_(blk.pam.warp_proj.weight, std=0.5)
        nn.init.normal_(blk.pam.warp_proj.bias, std=0.5)
    ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    prefix = 5

    with torch.no_grad():
        logits_par, states_par, _ = model.forward(ids)
        # prefill + stepwise
        lp, states, _ = model.forward(ids[:, :prefix])
        parts = [lp]
        for t in range(prefix, seq_len):
            lt, states, _ = model.forward(ids[:, t:t + 1], states=states, step_offset=t)
            parts.append(lt)
        logits_mix = torch.cat(parts, dim=1)
        # pure stepwise from empty
        parts, states0 = [], None
        for t in range(seq_len):
            lt, states0, _ = model.forward(ids[:, t:t + 1], states=states0, step_offset=t)
            parts.append(lt)
        logits_seq = torch.cat(parts, dim=1)

    d_mix = (logits_par - logits_mix).abs().max().item()
    d_seq = (logits_par - logits_seq).abs().max().item()
    d_nb = max(_max_diff(sp[0], ss[0]) for sp, ss in zip(states_par, states))
    d_clk = max(_max_diff(sp[1], ss[1]) for sp, ss in zip(states_par, states))
    assert d_mix < RECUR_ATOL, f"prefill+step logits disagree: {d_mix:.3e}"
    assert d_seq < RECUR_ATOL, f"stepwise logits disagree: {d_seq:.3e}"
    assert d_nb < RECUR_ATOL, f"carried notebook disagrees: {d_nb:.3e}"
    assert d_clk < 1e-3, f"carried clock disagrees: {d_clk:.3e}"
    print(f"  max |logit diff| prefill+step = {d_mix:.3e}  stepwise = {d_seq:.3e}   "
          f"state {d_nb:.3e}  clock {d_clk:.3e}")
    return True


def test_out_gate_parity_and_decode(batch_size=2, seq_len=17, seed=0):
    """N4 read-out gate: identity at init, grads flow, decode == chunked.

    (1) out_gate on (zero W_g, b_g = silu^-1(1)) must reproduce the ungated
    chrono model's logits exactly; (2) one backward reaches every gate_proj;
    (3) with random gate weights, chunked prefill == token-by-token decode
    (the gate is per-token elementwise, so the (notebook, clock) carry is
    unchanged).
    """
    torch.manual_seed(seed)
    cfg = get_config('tiny_real')
    cfg.chunk_size = 7
    cfg.chrono = True
    cfg.out_gate = True
    model = LM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    with torch.no_grad():
        logits_on, _, _ = model.forward(ids)
        for blk in model.blocks:
            blk.pam.out_gate = False
        logits_off, _, _ = model.forward(ids)
    d_init = (logits_on - logits_off).abs().max().item()
    assert d_init < 1e-5, f"out_gate@init != ungated: {d_init:.3e}"
    for blk in model.blocks:
        blk.pam.out_gate = True

    model.train()
    model.zero_grad(set_to_none=True)
    logits, _, _ = model.forward(ids)
    F.cross_entropy(logits.reshape(-1, cfg.vocab_size), ids.reshape(-1)).backward()
    n_none = sum(1 for blk in model.blocks if blk.pam.gate_proj.weight.grad is None)
    assert n_none == 0, f"{n_none} gate_proj got no grad"

    model.eval()
    for blk in model.blocks:
        nn.init.normal_(blk.pam.gate_proj.weight, std=0.5)
        nn.init.normal_(blk.pam.gate_proj.bias, std=0.5)
    with torch.no_grad():
        logits_par, _, _ = model.forward(ids)
        parts, states = [], None
        for t in range(seq_len):
            lt, states, _ = model.forward(ids[:, t:t + 1], states=states, step_offset=t)
            parts.append(lt)
        logits_seq = torch.cat(parts, dim=1)
    d_dec = (logits_par - logits_seq).abs().max().item()
    assert d_dec < RECUR_ATOL, f"gated decode disagrees: {d_dec:.3e}"
    print(f"  init parity {d_init:.3e}   gate grads: all {len(model.blocks)} present   "
          f"decode vs chunked {d_dec:.3e}")
    return True


def test_short_conv(batch_size=2, seq_len=40, seed=0):
    """G1 short conv on the Phase 3b arm (chrono+gate+delta).

    (1) short_conv at init (zero conv) == the model without it; (2) with
    random conv weights the model stays causal (changing token t moves no
    logit before t); (3) one backward reaches every qkv_conv.
    """
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    cfg = get_config('tiny_real')
    cfg.chunk_size = 16
    cfg.chrono = cfg.out_gate = cfg.delta = True
    cfg.short_conv = True
    model = LM(cfg).to(dev).eval()
    ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=dev)
    with torch.no_grad():
        on, _, _ = model.forward(ids)
        for blk in model.blocks:
            blk.pam.short_conv = False
        off, _, _ = model.forward(ids)
        for blk in model.blocks:
            blk.pam.short_conv = True
    d_init = (on - off).abs().max().item()
    assert d_init < 1e-5, f"short_conv@init != off: {d_init:.3e}"

    for blk in model.blocks:
        nn.init.normal_(blk.pam.qkv_conv.weight, std=0.3)
    t0 = 25
    ids2 = ids.clone()
    ids2[:, t0] = (ids2[:, t0] + 1) % cfg.vocab_size
    with torch.no_grad():
        a, _, _ = model.forward(ids)
        b, _, _ = model.forward(ids2)
    d_past = (a[:, :t0] - b[:, :t0]).abs().max().item()
    d_now = (a[:, t0:] - b[:, t0:]).abs().max().item()
    assert d_past < 1e-4 and d_now > 1e-3, f"conv not causal: past {d_past:.3e} now {d_now:.3e}"

    cfg2 = get_config('tiny_real')
    cfg2.chunk_size = 16
    cfg2.chrono = cfg2.out_gate = cfg2.delta = True
    cfg2.short_conv = True
    m2 = LM(cfg2).to(dev).train()
    lg, _, _ = m2.forward(ids)
    F.cross_entropy(lg.reshape(-1, cfg2.vocab_size).float(), ids.reshape(-1)).backward()
    n_none = sum(1 for blk in m2.blocks
                 if blk.pam.qkv_conv.weight.grad is None)
    assert n_none == 0, f"{n_none} blocks missing G1 grads"

    print(f"  conv init parity {d_init:.3e}   causal: past {d_past:.1e} now {d_now:.1e}   "
          f"grads ok ({dev})")
    return True


def test_echo_memory(batch_size=2, seq_len=48, seed=0):
    """E1 echo memory: identity at init, causal, grads flow, and it can copy.

    (1) echo on with zero echo_out == the model without it; (2) with a random
    echo_out, changing token t moves no logit before t; (3) one backward
    reaches echo_qk / echo_gates / echo_out; (4) mechanism: hand-wire the
    echo to a 'hidden = embedding of current token' model and check that the
    read after a repeated token points at the token that followed it before.
    """
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    cfg = get_config('tiny_real')
    cfg.chunk_size = 16
    cfg.chrono = cfg.out_gate = cfg.delta = True
    cfg.echo = True
    model = LM(cfg).to(dev).eval()
    ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=dev)
    with torch.no_grad():
        on, _, _ = model.forward(ids)
        model.echo = False
        off, _, _ = model.forward(ids)
        model.echo = True
    d_init = (on - off).abs().max().item()
    assert d_init < 1e-5, f"echo@init != off: {d_init:.3e}"

    nn.init.normal_(model.echo_out.weight, std=0.3)
    t0 = 30
    ids2 = ids.clone()
    ids2[:, t0] = (ids2[:, t0] + 1) % cfg.vocab_size
    with torch.no_grad():
        a, _, _ = model.forward(ids)
        b, _, _ = model.forward(ids2)
    d_past = (a[:, :t0] - b[:, :t0]).abs().max().item()
    d_now = (a[:, t0:] - b[:, t0:]).abs().max().item()
    assert d_past < 1e-4 and d_now > 1e-3, f"echo not causal: past {d_past:.3e} now {d_now:.3e}"

    model.train()
    model.zero_grad(set_to_none=True)
    lg, _, _ = model.forward(ids)
    F.cross_entropy(lg.reshape(-1, cfg.vocab_size).float(), ids.reshape(-1)).backward()
    for name in ('echo_qk', 'echo_gates', 'echo_out'):
        g = getattr(model, name).weight.grad
        assert g is not None and g.abs().sum() > 0, f"{name} got no grad"

    # (4) mechanism, on the raw read: hidden_t := unit embedding of x_t,
    # W_q = W_k = I, write/erase ~1, no fade. Sequence: A B ... A -> the read
    # at the second A must be closest to embedding(B).
    model.eval()
    D = cfg.dim
    with torch.no_grad():
        model.echo_qk.weight.copy_(torch.cat([torch.eye(D), torch.eye(D)]).to(dev))
        model.echo_gates.weight.zero_()
        Hh = cfg.n_heads
        model.echo_gates.bias[:Hh].fill_(-20.0)     # retention ~ 1
        model.echo_gates.bias[Hh:2 * Hh].fill_(20.0)  # write ~ 1
        model.echo_gates.bias[2 * Hh:].fill_(20.0)    # erase -> 0.95
        model.echo_out.weight.copy_(torch.eye(D).to(dev))
        seq = torch.randint(0, cfg.vocab_size, (1, 12), device=dev)
        seq[0, 11] = seq[0, 3]                        # repeat token 3 at the end
        from sempyt.tensor import named as _named
        from sempyt.dim import Dim as _Dim
        bt, tt = _Dim("batch", 1), _Dim("time", 12)
        E = model.embed.embed.weight
        hid = nn.functional.normalize(E[seq], dim=-1)
        r = model._echo_read(_named(hid, (bt, tt, model.model_dim)), seq, bt, tt)[0].raw(bt, tt, model.model_dim)
        sims = nn.functional.normalize(r[0, 11], dim=-1) @ nn.functional.normalize(E, dim=-1).T
        top = int(sims.argmax())
    assert top == int(seq[0, 4]), f"echo read at repeat -> {top}, want successor {int(seq[0, 4])}"

    # (5) decode: chunked prefill == token-by-token decode (random echo).
    torch.manual_seed(seed + 1)
    m5 = LM(cfg).to(dev).eval()
    nn.init.normal_(m5.echo_out.weight, std=0.3)
    nn.init.normal_(m5.echo_gates.weight, std=0.3)
    T5 = 20
    with torch.no_grad():
        par, _, _ = m5.forward(ids[:, :T5])
        half, st, _ = m5.forward(ids[:, :7])                 # prefill 7, then decode
        parts = [half]
        for t in range(7, T5):
            lt, st, _ = m5.forward(ids[:, t:t + 1], states=st, step_offset=t)
            parts.append(lt)
        seq_l = torch.cat(parts, dim=1)
    d_dec = (par - seq_l).abs().max().item()
    assert d_dec < 2e-3, f"echo decode disagrees: {d_dec:.3e}"
    print(f"  init parity {d_init:.3e}   causal: past {d_past:.1e} now {d_now:.1e}   "
          f"grads ok   copy: repeat of x3 recalls x4   decode vs chunked {d_dec:.2e} ({dev})")
    return True


def test_echo_qk_tied_init(seed=0):
    """Q1: with echo_qk_tied_init the echo query rows start equal to the key
    rows (so q_t = k-projection of h_t at init), the model is still identity
    at init (echo_out zero), and one optimiser step makes them differ (they
    train freely; nothing ties them)."""
    torch.manual_seed(seed)
    cfg = get_config('tiny_real')
    cfg.chunk_size = 16
    cfg.chrono = cfg.out_gate = cfg.delta = True
    cfg.echo = cfg.echo_qk_tied_init = True
    m = LM(cfg)
    D = cfg.dim
    W = m.echo_qk.weight
    assert torch.equal(W[:D], W[D:]), "q rows != k rows at init"
    ids = torch.randint(0, cfg.vocab_size, (2, 40))
    with torch.no_grad():
        on, _, _ = m.forward(ids)
        m.echo = False
        off, _, _ = m.forward(ids)
        m.echo = True
    d_init = (on - off).abs().max().item()
    assert d_init < 1e-5, f"echo@init != off: {d_init:.3e}"
    nn.init.normal_(m.echo_out.weight, std=0.3)          # give the read a path to the loss
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    lg, _, _ = m.forward(ids)
    F.cross_entropy(lg.reshape(-1, cfg.vocab_size).float(), ids.reshape(-1)).backward()
    opt.step()
    gap = (W[:D] - W[D:]).abs().max().item()
    assert gap > 0, "q and k rows stayed tied after a step"
    print(f"  q == k at init; init parity {d_init:.1e}; after one step |q - k| max {gap:.2e} (untied)")
    return True


def test_sparse_memory(batch_size=2, seq_len=30, seed=0):
    """M1 sparse memory: only cfg.pam_layers keep a notebook; the rest are
    gated-MLP blocks. Builds, trains one step, and decode == chunked (with
    echo on, so the carried states mix None, notebooks and the echo state)."""
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    cfg = get_config('tiny_real')
    cfg.n_layers = 4
    cfg.chunk_size = 16
    cfg.chrono = cfg.out_gate = cfg.delta = cfg.echo = True
    cfg.pam_layers = (1, 3)
    cfg.mlp_hidden = 320
    m = LM(cfg).to(dev)
    has = [b.pam is not None for b in m.blocks]
    assert has == [False, True, False, True], has
    assert m.blocks[0].cgu.hidden.size == 320
    ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=dev)
    m.train()
    lg, _, _ = m.forward(ids)
    F.cross_entropy(lg.reshape(-1, cfg.vocab_size).float(), ids.reshape(-1)).backward()
    assert m.blocks[1].pam.qkv_proj.weight.grad is not None
    m.eval()
    nn.init.normal_(m.echo_out.weight, std=0.3)
    with torch.no_grad():
        par, _, _ = m.forward(ids)
        out, st, _ = m.forward(ids[:, :5])
        parts = [out]
        for t in range(5, seq_len):
            lt, st, _ = m.forward(ids[:, t:t + 1], states=st, step_offset=t)
            parts.append(lt)
        seq_l = torch.cat(parts, dim=1)
    d = (par - seq_l).abs().max().item()
    assert d < 2e-3, f"sparse decode disagrees: {d:.3e}"
    print(f"  notebooks in {list(cfg.pam_layers)} of {cfg.n_layers}; mlp_hidden 320; "
          f"grads ok; decode vs chunked {d:.2e} ({dev})")
    return True


def test_shared_reads(batch_size=2, seq_len=40, seed=0):
    """M2 shared reads: (1) pam_shared_read with the writer's own query equals
    the writer's read (the write is reusable, exactly); (2) share_reads at
    init == off; (3) causal; (4) a reader's gradient reaches its q_proj AND
    the writer's qkv (the notebook learns what the readers need)."""
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    from v13_sempty.triton_kernels import pam_delta_batched, pam_shared_read
    torch.manual_seed(seed)
    BH, T, K = 4, 70, 32
    q = torch.randn(BH, T, K, device=dev)
    k = F.normalize(torch.randn(BH, T, K, device=dev), dim=-1)
    v = torch.randn(BH, T, K, device=dev)
    ret = torch.rand(BH, T, device=dev) * 0.2 + 0.8
    bw, be = torch.rand(BH, T, device=dev), torch.rand(BH, T, device=dev) * 0.5
    r0, _, write = pam_delta_batched(q, k, v, ret, bw, be, None, return_write=True)
    r1 = pam_shared_read(q, write)
    d_k = (r0 - r1).abs().max().item()
    assert d_k < 1e-4, f"shared read != writer read: {d_k:.3e}"

    cfg = get_config('tiny_real')
    cfg.n_layers = 4
    cfg.chunk_size = 16
    cfg.chrono = cfg.out_gate = cfg.delta = True
    cfg.pam_layers = (1,)
    cfg.share_reads = True
    m = LM(cfg).to(dev).eval()
    assert [b.reader is not None for b in m.blocks] == [False, False, True, True]
    ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=dev)
    with torch.no_grad():
        on, _, _ = m.forward(ids)
        m.config.share_reads = False
        readers = [b.reader for b in m.blocks]
        for b in m.blocks:
            b.reader = None
        off, _, _ = m.forward(ids)
        for b, r in zip(m.blocks, readers):
            b.reader = r
        m.config.share_reads = True
    d_init = (on - off).abs().max().item()
    assert d_init < 1e-5, f"share_reads@init != off: {d_init:.3e}"

    for b in m.blocks:
        if b.reader is not None:
            nn.init.normal_(b.reader.o_proj.weight, std=0.3)
    t0 = 25
    ids2 = ids.clone()
    ids2[:, t0] = (ids2[:, t0] + 1) % cfg.vocab_size
    with torch.no_grad():
        a, _, _ = m.forward(ids)
        b2, _, _ = m.forward(ids2)
    d_past = (a[:, :t0] - b2[:, :t0]).abs().max().item()
    assert d_past < 1e-4, f"shared reads not causal: {d_past:.3e}"

    # gradient of the READER path alone reaches the writer's qkv.
    m.train()
    m.zero_grad(set_to_none=True)
    lg, _, _ = m.forward(ids)
    lg_off = None
    F.cross_entropy(lg.reshape(-1, cfg.vocab_size).float(), ids.reshape(-1)).backward()
    g_reader = m.blocks[2].reader.q_proj.weight.grad
    assert g_reader is not None and g_reader.abs().sum() > 0, "reader q_proj got no grad"
    w_with = m.blocks[1].pam.qkv_proj.weight.grad.clone()
    m.zero_grad(set_to_none=True)
    for b in m.blocks:
        if b.reader is not None:
            b.reader.o_proj.weight.data.zero_()
    lg, _, _ = m.forward(ids)
    F.cross_entropy(lg.reshape(-1, cfg.vocab_size).float(), ids.reshape(-1)).backward()
    w_without = m.blocks[1].pam.qkv_proj.weight.grad
    d_w = (w_with - w_without).abs().max().item()
    assert d_w > 1e-8, "readers send no gradient into the writer"
    print(f"  shared read == writer read {d_k:.1e}   init parity {d_init:.1e}   "
          f"causal {d_past:.1e}   reader grad ok, writer grad shift {d_w:.1e} ({dev})")
    return True


def test_transformer_baseline(seed=0):
    """Transformer++ baseline (equal-data quality reference).

    (1) the default config is param-matched to PAM (101.9M vs 102.1M);
    (2) causal: changing later tokens leaves earlier logits unchanged;
    (3) the training loss (fused chunked CE) equals plain CE on the logits,
    and its gradients match; (4) runs past any fixed window (RoPE grows);
    (5) config + state_dict round-trip as the probe loader does it.
    """
    from dataclasses import asdict
    from v13_sempty.baseline_transformer import TransformerConfig, TransformerLM
    torch.manual_seed(seed)
    n = TransformerLM(TransformerConfig()).count_parameters()['total']
    assert 101.5e6 < n < 102.5e6, f"default size {n/1e6:.2f}M not ~102M"

    cfg = TransformerConfig(vocab_size=300, d_model=64, n_layer=2, n_head=4, d_ff=96)
    model = TransformerLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 24))
    with torch.no_grad():
        a, _, _ = model(ids)
        ids2 = ids.clone()
        ids2[:, 12:] = torch.randint(0, cfg.vocab_size, (2, 12))
        b, _, _ = model(ids2)
    d_causal = (a[:, :12] - b[:, :12]).abs().max().item()
    assert d_causal < 1e-5, f"future tokens leak into the past: {d_causal:.3e}"

    labels = torch.randint(0, cfg.vocab_size, ids.shape)
    model.zero_grad()
    logits, _, _ = model(ids)
    plain = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), labels.reshape(-1))
    plain.backward()
    g_plain = {k: p.grad.clone() for k, p in model.named_parameters()}
    model.zero_grad()
    lm, _ = model._hidden_to_lm(ids)
    fused = model.ce_from_lm(lm, labels, chunk=5)
    fused.backward()
    d_loss = (plain - fused).abs().item()
    d_grad = max((g_plain[k] - p.grad).abs().max().item() for k, p in model.named_parameters())
    assert d_loss < 1e-5 and d_grad < 1e-4, f"fused CE != plain CE: {d_loss:.3e} {d_grad:.3e}"
    assert all(p.grad is not None for p in model.parameters()), "a parameter got no grad"

    with torch.no_grad():
        long_logits, _, _ = model(torch.randint(0, cfg.vocab_size, (1, 700)))
    assert long_logits.shape == (1, 700, cfg.vocab_size)

    clone = TransformerLM(TransformerConfig(**asdict(model.config)))
    clone.load_state_dict(model.state_dict())
    with torch.no_grad():
        d_rt = (clone(ids)[0] - model(ids)[0]).abs().max().item()
    assert d_rt == 0.0, f"round-trip changed logits: {d_rt:.3e}"
    print(f"  {n/1e6:.2f}M   causal {d_causal:.1e}   fused CE loss {d_loss:.1e} grad {d_grad:.1e}   "
          f"T=700 ok   round-trip exact")
    return True


def test_prefetch_order_and_cursor():
    """train._Prefetch: the batches built on the background thread are exactly
    the inline ones, in order, and each carries the data cursor as it stood
    right after that batch was built (what an inline checkpoint would save).
    The toy stream mimics StreamingTokenChunkDataset: a shuffle buffer popped
    with the global CPU RNG and live cursor dicts mutated while reading."""
    from torch.utils.data import DataLoader, IterableDataset
    from v13_sempty.train import _Prefetch

    cursor = {'doc_counters': {}, 'token_counters': {}}

    class _Stream(IterableDataset):
        def __iter__(self):
            pending = []
            for i in range(600):
                cursor['doc_counters']['s'] = cursor['doc_counters'].get('s', 0) + 1
                pending.append(torch.full((4,), i))
                if len(pending) >= 16:
                    j = torch.randint(len(pending), (1,)).item()
                    cursor['token_counters']['s'] = cursor['token_counters'].get('s', 0) + 4
                    yield {'input_ids': pending.pop(j)}

    def run(prefetch):
        for d in cursor.values():
            d.clear()
        torch.manual_seed(0)
        dl = DataLoader(_Stream(), batch_size=3, num_workers=0, drop_last=True)
        out = []
        if prefetch:
            for batch, snap in _Prefetch(dl, 4, cursor):
                out.append((batch['input_ids'].clone(), snap))
        else:
            for batch in dl:
                out.append((batch['input_ids'].clone(),
                            {k: dict(v) for k, v in cursor.items()}))
        return out

    inline, pre = run(False), run(True)
    assert len(inline) == len(pre) > 100, (len(inline), len(pre))
    for k, ((x1, c1), (x2, c2)) in enumerate(zip(inline, pre)):
        assert torch.equal(x1, x2), f"batch {k}: prefetch changed the order"
        assert c1 == c2, f"batch {k}: cursor snapshot {c2} != inline {c1}"

    def broken():
        yield {'input_ids': torch.zeros(1)}
        raise RuntimeError('producer boom')

    got = []
    try:
        for batch, _ in _Prefetch(broken(), 2, None):
            got.append(batch)
        raise AssertionError('producer exception was swallowed')
    except RuntimeError as e:
        assert 'producer boom' in str(e) and len(got) == 1
    print(f"  {len(inline)} batches identical in order; cursor snapshots match; "
          f"producer errors re-raised")
    return True


def test_worker_stream_order_and_cursor():
    """train._WorkerStream (batches built in a forked worker): identical
    batches/order to the inline loader and the worker-side cursor snapshot
    per batch, with torch, python and numpy RNGs all used by the stream (the
    worker gets the states the inline path would have had)."""
    import random as _random
    import numpy as _np
    from torch.utils.data import DataLoader, IterableDataset
    from v13_sempty.train import _WorkerStream

    cursor = {'doc_counters': {}, 'token_counters': {}}

    class _Stream(IterableDataset):
        def __iter__(self):
            pending = []
            for i in range(500):
                cursor['doc_counters']['s'] = cursor['doc_counters'].get('s', 0) + 1
                noise = _random.random() + float(_np.random.rand())
                pending.append(torch.full((4,), float(i) + noise))
                if len(pending) >= 16:
                    j = torch.randint(len(pending), (1,)).item()
                    cursor['token_counters']['s'] = cursor['token_counters'].get('s', 0) + 4
                    yield {'input_ids': pending.pop(j)}

    def run(worker):
        for d in cursor.values():
            d.clear()
        torch.manual_seed(0); _random.seed(1); _np.random.seed(2)
        out = []
        if worker:
            for batch in _WorkerStream(_Stream(), cursor, 3, 4):
                out.append((batch['input_ids'].clone(), batch['_cursor']))
        else:
            for batch in DataLoader(_Stream(), batch_size=3, num_workers=0, drop_last=True):
                out.append((batch['input_ids'].clone(),
                            {k: dict(v) for k, v in cursor.items()}))
        return out

    inline, work = run(False), run(True)
    assert len(inline) == len(work) > 100, (len(inline), len(work))
    for k, ((x1, c1), (x2, c2)) in enumerate(zip(inline, work)):
        assert torch.equal(x1, x2), f"batch {k}: worker changed the batches/order"
        assert c1 == c2, f"batch {k}: worker cursor {c2} != inline {c1}"
    print(f"  {len(inline)} batches identical (torch/python/numpy RNG handed over); "
          f"worker cursor snapshots match")
    return True


def main():
    torch.set_num_threads(2)
    tests = [
        test_param_count_and_state,
        test_parallel_vs_recurrent,
        test_tied_logits,
        test_fused_ce,
        test_smoke_loss_decreases,
        test_generate_smoke,
        test_real_param_count_and_state,
        test_real_parallel_vs_recurrent,
        test_real_tied_logits,
        test_real_fused_ce,
        test_real_smoke_loss_decreases,
        test_real_generate_smoke,
        test_real_fused_kernel_parity,
        test_complex_fused_kernel_parity,
        test_chrono_rotary_parity,
        test_chrono_parallel_vs_recurrent,
        test_out_gate_parity_and_decode,
        test_short_conv,
        test_echo_memory,
        test_echo_qk_tied_init,
        test_sparse_memory,
        test_shared_reads,
        test_transformer_baseline,
        test_prefetch_order_and_cursor,
        test_worker_stream_order_and_cursor,
    ]
    failures = 0
    for t in tests:
        name = t.__name__
        print(f"{name} ...", flush=True)
        try:
            t()
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"  FAIL: {type(e).__name__}: {e}")
        print(flush=True)
    if failures:
        print(f"{failures}/{len(tests)} failed")
        raise SystemExit(1)
    print(f"all {len(tests)} passed")


if __name__ == '__main__':
    main()
