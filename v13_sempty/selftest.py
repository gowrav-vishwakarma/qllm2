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


def test_couple_gates_and_decode(batch_size=2, seq_len=17, seed=0):
    """C1 coupled gates: one step dt drives write, erase and fade.

    (1) write == erase == 1 - e^{-dt} and fade == e^{-A dt} with
    dt = softplus(dt_proj(x) + dt_bias); with A = 1 (init) the fade equals
    the uncoupled A3 fade for the same weights; (2) one backward reaches
    couple_a_log and dt_proj, and the A3 erase/write projections do not
    exist; (3) with live random gates, chunked prefill == prefill + steps ==
    token-by-token decode, notebook and clock included.
    """
    import v13_sempty.triton_kernels as tk
    torch.manual_seed(seed)
    cfg = get_config('tiny_real')
    cfg.chunk_size = 7
    cfg.chrono = True
    cfg.delta = True
    cfg.couple = True
    model = LM(cfg).eval()
    pam = model.blocks[0].pam
    assert not hasattr(pam, 'erase_proj') and not hasattr(pam, 'write_proj')
    base = get_config('tiny_real')
    base.chunk_size, base.chrono, base.delta = 7, True, True
    ref = LM(base).eval()
    ref.load_state_dict(model.state_dict(), strict=False)

    x = torch.randn(batch_size, seq_len, cfg.dim)
    batch, time = Dim("batch", batch_size), Dim("time", seq_len)
    tokens = named(x, (batch, time, pam.model_dim))
    with torch.no_grad():
        nn.init.normal_(pam.dt_proj.weight, std=0.5)
        ref.blocks[0].pam.dt_proj.weight.copy_(pam.dt_proj.weight)
        dt = F.softplus(x @ pam.dt_proj.weight.T + pam.dt_proj.bias + pam.dt_bias)  # [B,T,H]
        fade_init = pam._decay(tokens).raw(batch, pam.heads, time)
        rpam = ref.blocks[0].pam
        fade_ref = rpam._decay(named(x, (batch, time, rpam.model_dim))).raw(
            batch, rpam.heads, time)
        pam.couple_a_log.normal_(std=0.7)
        a = pam.couple_a_log.exp()
        bw, be = pam._delta_gates(tokens)
        fade = pam._decay(tokens).raw(batch, pam.heads, time)
    want_b = (1 - torch.exp(-dt)).permute(0, 2, 1)
    want_fade = torch.exp(-dt * a).permute(0, 2, 1)
    d_b = max((bw - want_b).abs().max().item(), (be - want_b).abs().max().item())
    d_fade = (fade - want_fade).abs().max().item()
    d_init = (fade_init - fade_ref).abs().max().item()
    assert d_b < 1e-6, f"write/erase gate != 1-e^-dt: {d_b:.3e}"
    assert d_fade < 1e-6, f"fade != e^-A dt: {d_fade:.3e}"
    assert d_init < 1e-6, f"A=1 fade != uncoupled fade: {d_init:.3e}"

    old = tk._DELTA_COMPILE
    tk._DELTA_COMPILE = False          # CPU selftest: eager delta prep
    try:
        ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
        model.train()
        model.zero_grad(set_to_none=True)
        logits, _, _ = model.forward(ids)
        F.cross_entropy(logits.reshape(-1, cfg.vocab_size), ids.reshape(-1)).backward()
        for blk in model.blocks:
            assert blk.pam.couple_a_log.grad is not None, "couple_a_log got no grad"
            assert blk.pam.dt_proj.weight.grad is not None, "dt_proj got no grad"

        model.eval()
        with torch.no_grad():
            for blk in model.blocks:
                nn.init.normal_(blk.pam.dt_proj.weight, std=0.5)
                nn.init.normal_(blk.pam.dt_proj.bias, std=1.0)
                blk.pam.couple_a_log.normal_(std=0.7)
                nn.init.normal_(blk.pam.warp_proj.weight, std=0.5)
            logits_par, states_par, _ = model.forward(ids)
            prefix = 5
            lp, states, _ = model.forward(ids[:, :prefix])
            parts = [lp]
            for t in range(prefix, seq_len):
                lt, states, _ = model.forward(ids[:, t:t + 1], states=states, step_offset=t)
                parts.append(lt)
            logits_mix = torch.cat(parts, dim=1)
            parts, states0 = [], None
            for t in range(seq_len):
                lt, states0, _ = model.forward(ids[:, t:t + 1], states=states0, step_offset=t)
                parts.append(lt)
            logits_seq = torch.cat(parts, dim=1)
    finally:
        tk._DELTA_COMPILE = old
    d_mix = (logits_par - logits_mix).abs().max().item()
    d_seq = (logits_par - logits_seq).abs().max().item()
    d_nb = max(_max_diff(sp[0], ss[0]) for sp, ss in zip(states_par, states))
    assert d_mix < RECUR_ATOL, f"prefill+step logits disagree: {d_mix:.3e}"
    assert d_seq < RECUR_ATOL, f"stepwise logits disagree: {d_seq:.3e}"
    assert d_nb < RECUR_ATOL, f"carried notebook disagrees: {d_nb:.3e}"
    print(f"  gates {d_b:.1e}  fade {d_fade:.1e}  A=1 vs A3 fade {d_init:.1e}   "
          f"decode: prefill+step {d_mix:.3e}  stepwise {d_seq:.3e}  state {d_nb:.3e}")
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
        test_couple_gates_and_decode,
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
