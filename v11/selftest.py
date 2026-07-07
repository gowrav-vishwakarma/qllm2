"""Correctness self-tests for V11 PAM memory dynamics.

For each mode we verify that the parallel TRAINING form (chunked / dual / UT)
produces the same output as the sequential O(1) RECURRENT inference form on the
same random input. If these agree, the chunked math is correct.

Run:
    .venv/bin/python -m v11.selftest
"""

import torch

from v11.model import V11Config, V11PAMLayer


def _run_mode(name, cfg, B=2, T=80, atol=2e-3, seed=0):
    torch.manual_seed(seed)
    layer = V11PAMLayer(cfg, layer_idx=0).eval()
    x = torch.randn(B, T, cfg.dim, 2) * 0.5

    with torch.no_grad():
        # Parallel training form.
        y_par, _ = layer(x, state=None, step_offset=0)

        # Sequential recurrent form, one token at a time.
        y_steps = []
        state = None
        for t in range(T):
            y_t, state = layer(x[:, t:t+1], state=state, step_offset=t)
            y_steps.append(y_t)
        y_rec = torch.cat(y_steps, dim=1)

    diff = (y_par - y_rec).abs()
    rel = diff.max().item() / (y_par.abs().max().item() + 1e-8)
    ok = diff.max().item() < atol
    print(f"[{name:14s}] max|Δ|={diff.max().item():.2e}  rel={rel:.2e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_warmstart_chatml():
    """ChatML rows 50257/50258 should become the mean of base-vocab rows."""
    from v11.train import _warmstart_chatml_embeddings, _CHATML_BASE_VOCAB, _CHATML_TOKEN_IDS

    torch.manual_seed(0)
    vocab, dim = 50261, 8
    state = {
        'embed.embed_real.weight': torch.randn(vocab, dim),
        'embed.embed_imag.weight': torch.randn(vocab, dim),
    }
    # Mark ChatML rows distinctly so we can detect change.
    for idx in _CHATML_TOKEN_IDS:
        state['embed.embed_real.weight'][idx] = 999.0
        state['embed.embed_imag.weight'][idx] = -999.0

    _warmstart_chatml_embeddings(state)
    for key in ('embed.embed_real.weight', 'embed.embed_imag.weight'):
        w = state[key]
        expected = w[:_CHATML_BASE_VOCAB].mean(dim=0)
        for idx in _CHATML_TOKEN_IDS:
            assert torch.allclose(w[idx], expected, atol=1e-6), f"{key}[{idx}] not mean"
    print("[warmstart_chatml] PASS")


def test_fused_e3_equiv(B=2, T=80, seed=0):
    """Fused E3 path must equal the reference K-loop exactly (fwd, grad, state)."""
    common = dict(
        vocab_size=512, dim=48, n_heads=3, head_dim=16, n_layers=1, expand=2,
        dropout=0.0, max_seq_len=256, chunk_size=24, gradient_checkpointing=False,
        use_rope=True, use_gsp=True, n_states=3, gate_content_aware=True,
    )
    torch.manual_seed(seed)
    loop = V11PAMLayer(V11Config(**{**common, 'fused_e3': False}))
    fused = V11PAMLayer(V11Config(**{**common, 'fused_e3': True}))
    rc = V11PAMLayer(V11Config(**{**common, 'fused_e3': True, 'recompute_pam_chunks': True}))
    fused.load_state_dict(loop.state_dict())
    rc.load_state_dict(loop.state_dict())
    rc.train()

    x = torch.randn(B, T, common['dim'], 2) * 0.5
    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)
    x3 = x.clone().requires_grad_(True)
    y1, S1 = loop(x1)
    y2, S2 = fused(x2)
    y3, S3 = rc(x3)
    (y1 ** 2).sum().backward()
    (y2 ** 2).sum().backward()
    (y3 ** 2).sum().backward()

    dy = (y1 - y2).abs().max().item()
    dS = (S1 - S2).abs().max().item()
    dg = (x1.grad - x2.grad).abs().max().item()
    dyr = (y1 - y3).abs().max().item()
    dgr = (x1.grad - x3.grad).abs().max().item()
    ok = max(dy, dS, dg, dyr, dgr) < 1e-10
    print(f"[fused_e3 equiv ] y={dy:.2e} S={dS:.2e} grad={dg:.2e}  "
          f"recompute[y={dyr:.2e} grad={dgr:.2e}]  {'PASS' if ok else 'FAIL'}")
    return ok


def test_fused_ce_equiv(B=2, T=40, seed=0):
    """Model-level fused CE must equal standard forward + F.cross_entropy."""
    import torch.nn.functional as F
    from v11.model import V11LM
    torch.manual_seed(seed)
    cfg = V11Config(
        vocab_size=256, dim=48, n_heads=3, head_dim=16, n_layers=2, expand=2,
        dropout=0.0, max_seq_len=128, chunk_size=24, gradient_checkpointing=False,
        n_states=3, gate_content_aware=True,
    )
    m = V11LM(cfg).train()
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    lbl = torch.randint(0, cfg.vocab_size, (B, T))
    m.zero_grad(); logits, _, aux1 = m(ids)
    ref = F.cross_entropy(logits.view(-1, logits.size(-1)), lbl.view(-1))
    ref.backward()
    gref = {n: p.grad.clone() for n, p in m.named_parameters()}
    m.zero_grad(); main, aux2 = m.fused_ce_loss(ids, lbl, chunk=16)
    main.backward()
    dloss = (main - ref).abs().item()
    dg = max((gref[n] - p.grad).abs().max().item() for n, p in m.named_parameters())
    ok = max(dloss, dg) < 1e-5 and aux1.item() == aux2.item()
    print(f"[fused_ce equiv ] loss={dloss:.2e} grad={dg:.2e} aux={aux1.item():.2e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_competitive_retrieval_equiv(B=2, T=80, seed=0):
    """Competitive E3: fused == loop == recurrent with routing+compete flags."""
    common = dict(
        vocab_size=512, dim=48, n_heads=3, head_dim=16, n_layers=1, expand=2,
        dropout=0.0, max_seq_len=256, chunk_size=24, gradient_checkpointing=False,
        use_rope=True, use_gsp=True, n_states=3, gate_content_aware=True,
        routing_content_aware=True, state_compete=True, phase_init='spread',
        route_balance_lambda=0.01,
    )
    torch.manual_seed(seed)
    loop = V11PAMLayer(V11Config(**{**common, 'fused_e3': False}))
    fused = V11PAMLayer(V11Config(**{**common, 'fused_e3': True}))
    fused.load_state_dict(loop.state_dict())
    loop.train()
    fused.train()

    x = torch.randn(B, T, common['dim'], 2) * 0.5
    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)
    y1, S1 = loop(x1)
    y2, S2 = fused(x2)
    (y1 ** 2).sum().backward()
    (y2 ** 2).sum().backward()

    with torch.no_grad():
        ok_rec = _run_mode(
            "compete_recur",
            V11Config(**{**common, 'fused_e3': False}),
            B=B, T=T, atol=2e-3, seed=seed + 1,
        )

    dy = (y1 - y2).abs().max().item()
    dS = (S1 - S2).abs().max().item()
    dg = (x1.grad - x2.grad).abs().max().item()
    ok = max(dy, dS, dg) < 1e-10 and ok_rec
    print(f"[compete fused  ] y={dy:.2e} S={dS:.2e} grad={dg:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def test_drop_shape_mismatches():
    """Resume-safe: growing phase_proj (dim -> 2*dim) reinits cleanly."""
    from v11.train import _drop_shape_mismatches
    cfg_old = V11Config(
        vocab_size=128, dim=32, n_heads=2, head_dim=16, n_layers=1, expand=2,
        n_states=3, routing_content_aware=False,
    )
    cfg_new = V11Config(
        vocab_size=128, dim=32, n_heads=2, head_dim=16, n_layers=1, expand=2,
        n_states=3, routing_content_aware=True,
    )
    old = V11PAMLayer(cfg_old)
    new = V11PAMLayer(cfg_new)
    state = old.state_dict()
    dropped = _drop_shape_mismatches(new, state)
    new.load_state_dict(state, strict=False)
    keys = [k for k, _, _ in dropped]
    assert any('phase_proj' in k for k in keys), dropped
    print(f"[resume_shape   ] dropped {keys}  PASS")
    return True


def main():
    torch.set_default_dtype(torch.float64)  # high precision for the math check
    test_warmstart_chatml()
    common = dict(
        vocab_size=512, dim=32, n_heads=2, head_dim=16, n_layers=1,
        expand=2, dropout=0.0, max_seq_len=512, chunk_size=24,
        gradient_checkpointing=False, use_rope=True, use_gsp=True,
    )
    results = []
    results.append(_run_mode("baseline", V11Config(**common)))
    results.append(_run_mode("E1 perchannel", V11Config(**{**common, 'decay_mode': 'per_channel'})))
    results.append(_run_mode("E2 delta", V11Config(**{**common, 'write_mode': 'delta', 'delta_chunk': 20})))
    results.append(_run_mode("E3 multistate", V11Config(**{**common, 'n_states': 2})))
    results.append(_run_mode("E1+E3 combo", V11Config(**{**common, 'decay_mode': 'per_channel', 'n_states': 2})))
    results.append(test_fused_e3_equiv())
    results.append(test_fused_ce_equiv())
    results.append(test_competitive_retrieval_equiv())
    results.append(test_drop_shape_mismatches())
    print()
    if all(results):
        print("ALL MODES PASS: parallel train form == O(1) recurrent form.")
    else:
        print("SOME MODES FAILED — see above.")
        raise SystemExit(1)


if __name__ == '__main__':
    main()
