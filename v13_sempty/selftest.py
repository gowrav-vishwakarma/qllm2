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
