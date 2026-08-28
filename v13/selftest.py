"""Correctness self-tests for V13 selective PAM memory dynamics.

For each mode we verify that the parallel TRAINING form (chunked / dual / UT)
produces the same output as the sequential O(1) RECURRENT inference form on the
same random input. If these agree, the chunked math is correct.

Run:
    .venv/bin/python -m v13.selftest
"""

import torch

from v13.model import V13Config, V13PAMLayer


def _run_mode(name, cfg, batch_size=2, seq_len=80, atol=2e-3, seed=0):
    torch.manual_seed(seed)
    layer = V13PAMLayer(cfg, layer_idx=0).eval()
    x = torch.randn(batch_size, seq_len, cfg.dim, 2) * 0.5

    with torch.no_grad():
        # Parallel training form.
        parallel_output, _ = layer(x, state=None, step_offset=0)

        # Sequential recurrent form, one token at a time.
        recurrent_steps = []
        state = None
        for time_idx in range(seq_len):
            step_output, state = layer(x[:, time_idx:time_idx + 1], state=state, step_offset=time_idx)
            recurrent_steps.append(step_output)
        recurrent_output = torch.cat(recurrent_steps, dim=1)

    diff = (parallel_output - recurrent_output).abs()
    rel = diff.max().item() / (parallel_output.abs().max().item() + 1e-8)
    ok = diff.max().item() < atol
    print(f"[{name:14s}] max|Δ|={diff.max().item():.2e}  rel={rel:.2e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_warmstart_chatml():
    """ChatML rows 50257/50258 should become the mean of base-vocab rows."""
    from v13.train import _warmstart_chatml_embeddings, _CHATML_BASE_VOCAB, _CHATML_TOKEN_IDS

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
    return True


def test_fused_e3_equiv(batch_size=2, seq_len=80, seed=0):
    """Fused E3 path must equal the reference K-loop exactly (fwd, grad, state)."""
    common = dict(
        vocab_size=512, dim=48, n_heads=3, head_dim=16, n_layers=1, expand=2,
        dropout=0.0, max_seq_len=256, chunk_size=24, gradient_checkpointing=False,
        use_rope=True, use_gsp=True, n_states=3, gate_content_aware=True,
    )
    torch.manual_seed(seed)
    loop = V13PAMLayer(V13Config(**{**common, 'fused_e3': False}))
    fused = V13PAMLayer(V13Config(**{**common, 'fused_e3': True}))
    rc = V13PAMLayer(V13Config(**{**common, 'fused_e3': True, 'recompute_pam_chunks': True}))
    fused.load_state_dict(loop.state_dict())
    rc.load_state_dict(loop.state_dict())
    rc.train()

    x = torch.randn(batch_size, seq_len, common['dim'], 2) * 0.5
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


def test_fused_ce_equiv(batch_size=2, seq_len=40, seed=0):
    """Model-level fused CE must equal standard forward + F.cross_entropy."""
    import torch.nn.functional as F
    from v13.model import V13LM
    torch.manual_seed(seed)
    cfg = V13Config(
        vocab_size=256, dim=48, n_heads=3, head_dim=16, n_layers=2, expand=2,
        dropout=0.0, max_seq_len=128, chunk_size=24, gradient_checkpointing=False,
        n_states=3, gate_content_aware=True,
    )
    m = V13LM(cfg).train()
    ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    lbl = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
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


def test_competitive_retrieval_equiv(batch_size=2, seq_len=80, seed=0):
    """Competitive E3: fused == loop == recurrent with routing+compete flags."""
    common = dict(
        vocab_size=512, dim=48, n_heads=3, head_dim=16, n_layers=1, expand=2,
        dropout=0.0, max_seq_len=256, chunk_size=24, gradient_checkpointing=False,
        use_rope=True, use_gsp=True, n_states=3, gate_content_aware=True,
        routing_content_aware=True, state_compete=True, phase_init='spread',
        route_balance_lambda=0.01,
    )
    torch.manual_seed(seed)
    loop = V13PAMLayer(V13Config(**{**common, 'fused_e3': False}))
    fused = V13PAMLayer(V13Config(**{**common, 'fused_e3': True}))
    fused.load_state_dict(loop.state_dict())
    loop.train()
    fused.train()

    x = torch.randn(batch_size, seq_len, common['dim'], 2) * 0.5
    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)
    y1, S1 = loop(x1)
    y2, S2 = fused(x2)
    (y1 ** 2).sum().backward()
    (y2 ** 2).sum().backward()

    with torch.no_grad():
        ok_rec = _run_mode(
            "compete_recur",
            V13Config(**{**common, 'fused_e3': False}),
            batch_size=batch_size, seq_len=seq_len, atol=2e-3, seed=seed + 1,
        )

    dy = (y1 - y2).abs().max().item()
    dS = (S1 - S2).abs().max().item()
    dg = (x1.grad - x2.grad).abs().max().item()
    ok = max(dy, dS, dg) < 1e-10 and ok_rec
    print(f"[compete fused  ] y={dy:.2e} S={dS:.2e} grad={dg:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def test_delta_keynorm_equiv(batch_size=2, seq_len=80, seed=0):
    """Production V13 path: delta + unit-norm keys (the 500M NaN fix).

    Verifies fused == K-loop == recurrent with delta_key_norm=True, and that
    keys leaving _project are truly per-vector unit-norm (||k||_2 == 1 across
    head_dim, NOT per-element). The delta k-direction eigenvalue is
    gamma*(1 - beta_e*||k||^2); unit keys keep it in [1-beta_e, 1).
    """
    common = dict(
        vocab_size=512, dim=48, n_heads=3, head_dim=16, n_layers=1, expand=2,
        dropout=0.0, max_seq_len=256, chunk_size=24, gradient_checkpointing=False,
        use_rope=True, use_gsp=True, n_states=3, gate_content_aware=True,
        write_mode='delta', delta_chunk=20, delta_erase_gate=True,
        vault_state=True, vault_state_idx=0, write_phase_address=True,
        delta_key_norm=True,
    )
    torch.manual_seed(seed)
    loop = V13PAMLayer(V13Config(**{**common, 'fused_e3': False}))
    fused = V13PAMLayer(V13Config(**{**common, 'fused_e3': True}))
    fused.load_state_dict(loop.state_dict())
    loop.train()
    fused.train()

    x = torch.randn(batch_size, seq_len, common['dim'], 2) * 0.5
    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)
    y1, S1 = loop(x1)
    y2, S2 = fused(x2)
    (y1 ** 2).sum().backward()
    (y2 ** 2).sum().backward()

    # Keys leaving _project must be per-vector unit-norm.
    with torch.no_grad():
        _, kproj, _, _ = fused._project(x, 0)
        knorm = (kproj[..., 0].square() + kproj[..., 1].square()).sum(-1).sqrt()
        d_k = (knorm - 1.0).abs().max().item()

    with torch.no_grad():
        ok_rec = _run_mode(
            "delta_keynorm",
            V13Config(**{**common, 'fused_e3': False}),
            batch_size=batch_size, seq_len=seq_len, atol=2e-3, seed=seed + 1,
        )

    dy = (y1 - y2).abs().max().item()
    dS = (S1 - S2).abs().max().item()
    dg = (x1.grad - x2.grad).abs().max().item()
    # Equivalence must be near-exact (fused == loop == recurrent, incl. grads).
    # The unit-norm check only needs to reject a MISSING norm (|d| ~ 0.2) or a
    # per-element norm (|d| ~ 3.9): the 1e-8 floor in the denominator leaves
    # |d| ~ 1e-8/(2*min|k|^2) ~ 1e-7 for near-zero keys, which is harmless.
    ok = max(dy, dS, dg) < 1e-10 and d_k < 1e-5 and ok_rec
    print(f"[delta_keynorm  ] y={dy:.2e} S={dS:.2e} grad={dg:.2e} |d||k|||={d_k:.2e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_delta_raw_readout_equiv(batch_size=2, seq_len=80, seed=0):
    """delta_raw_key_readout: fused == K-loop == recurrent with the two-state
    (S_erase unit-key, S_read raw-key) readout split, including grads.

    Also verifies the split is real: S_erase (stacked dim-0 index 0) must equal
    the flag-OFF state exactly (the erase/mass system is untouched) while S_read
    (index 1) and the outputs differ from flag-OFF (magnitude restored).
    """
    common = dict(
        vocab_size=512, dim=48, n_heads=3, head_dim=16, n_layers=1, expand=2,
        dropout=0.0, max_seq_len=256, chunk_size=24, gradient_checkpointing=False,
        use_rope=True, use_gsp=True, n_states=3, gate_content_aware=True,
        write_mode='delta', delta_chunk=20, delta_erase_gate=True,
        vault_state=True, vault_state_idx=0, write_phase_address=True,
        delta_key_norm=True,
    )
    torch.manual_seed(seed)
    loop = V13PAMLayer(V13Config(**{**common, 'fused_e3': False,
                                    'delta_raw_key_readout': True}))
    fused = V13PAMLayer(V13Config(**{**common, 'fused_e3': True,
                                     'delta_raw_key_readout': True}))
    fused.load_state_dict(loop.state_dict())
    loop.train()
    fused.train()

    x = torch.randn(batch_size, seq_len, common['dim'], 2) * 0.5
    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)
    y1, S1 = loop(x1)
    y2, S2 = fused(x2)
    (y1 ** 2).sum().backward()
    (y2 ** 2).sum().backward()

    # Stacked state must be [2, K, B, H, d, d, 2] with dim 0 = {S_erase, S_read}.
    ok_shape = S1.shape[0] == 2 and S2.shape == S1.shape
    dy = (y1 - y2).abs().max().item()
    dS = (S1 - S2).abs().max().item()
    dg = (x1.grad - x2.grad).abs().max().item()

    # Flag-OFF reference: same seed/params, no raw-key readout.
    torch.manual_seed(seed)
    off = V13PAMLayer(V13Config(**{**common, 'fused_e3': True})).train()
    y_off, S_off = off(x.clone())

    # S_erase (index 0) must be EXACTLY the flag-OFF state; S_read (index 1) and
    # the outputs must differ (magnitude axis restored).
    d_erase = (S1[0] - S_off).abs().max().item()
    d_read = (S1[1] - S_off).abs().max().item()
    d_y = (y1 - y_off).abs().max().item()

    ok_rec = _run_mode(
        "rawreadout_recur",
        V13Config(**{**common, 'fused_e3': True, 'delta_raw_key_readout': True}),
        batch_size=batch_size, seq_len=seq_len, atol=2e-3, seed=seed + 1,
    )
    ok = (max(dy, dS, dg) < 1e-10 and ok_shape and ok_rec
          and d_erase < 1e-12 and d_read > 1e-3 and d_y > 1e-3)
    print(f"[raw_readout    ] y={dy:.2e} S={dS:.2e} grad={dg:.2e} "
          f"eraseΔ={d_erase:.2e} readΔ={d_read:.2e} yΔ={d_y:.2e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_delta_erase_cap():
    """Erase-gain cap: beta_e <= delta_erase_beta_cap (stability), write beta free.

    The vault delta k-direction eigenvalue is 1 - beta_e (key-norm ||k||^2=1);
    capping beta_e <= 0.95 keeps it in [0.05, 1) for ANY learned init. This
    test forces erase_beta_proj to saturate (weight x10, bias +5 -> sigmoid
    ~1.0) and checks the clamp holds while the write path is untouched.
    """
    common = dict(
        vocab_size=512, dim=48, n_heads=3, head_dim=16, n_layers=1, expand=2,
        dropout=0.0, max_seq_len=256, chunk_size=24, gradient_checkpointing=False,
        use_rope=True, use_gsp=True, n_states=3, gate_content_aware=True,
        write_mode='delta', delta_chunk=20, delta_erase_gate=True,
        vault_state=True, vault_state_idx=0, write_phase_address=True,
        delta_key_norm=True, delta_erase_beta_cap=0.95,
    )
    torch.manual_seed(0)
    p = V13PAMLayer(V13Config(**common))
    p.train()
    # Saturate the erase projection: unclamped sigmoid output would be ~1.0.
    with torch.no_grad():
        p.erase_beta_proj.weight.mul_(10.0)
        p.erase_beta_proj.bias.fill_(5.0)
    x = torch.randn(2, 64, common['dim'], 2) * 0.5
    wb, eb = p._gate_betas(x)
    eb_max = eb.max().item()
    # Sanity: recompute the pre-clamp sigmoid (must actually exceed the cap,
    # otherwise the test is vacuous). erase_beta_proj takes cabs(x).
    from v13.complex_ops import cabs
    raw_sigmoid = torch.sigmoid(p.erase_beta_proj(cabs(x))).transpose(1, 2)
    raw_max = raw_sigmoid.max().item()
    ok = eb_max <= 0.95 + 1e-12 and raw_max > 0.95 and wb.max().item() > 0.5
    print(f"[delta_erasecap ] beta_e_max={eb_max:.4f} raw={raw_max:.4f} "
          f"beta_w_max={wb.max().item():.4f}  {'PASS' if ok else 'FAIL'}")
    return ok


def test_grad_ckpt_equiv(batch_size=2, seq_len=64, seed=0):
    """Non-reentrant checkpoint must match no-ckpt grads AND reach every block.

    Every other selftest sets gradient_checkpointing=False, which is how both
    the 2026-08-22 detach freeze (only last block learned) and the 2026-08-23
    TorchScript cnormalize_vec CheckpointError slipped through. This test
    exercises the production selective stack (delta + vault + phase +
    key-norm + erase-cap) under checkpointing.

    2026-08-23: cnormalize_vec is eager (not @jit.script) so the checkpoint
    forward and recompute build the same saved-tensor order.
    """
    from v13.model import V13LM
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    common = dict(
        vocab_size=256, dim=32, n_heads=2, head_dim=16, n_layers=4,
        expand=2, dropout=0.0, max_seq_len=128, chunk_size=32,
        use_rope=True, use_gsp=True, n_states=3,
        write_mode='delta', delta_chunk=16, delta_erase_gate=True,
        gate_content_aware=True, vault_state=True, vault_state_idx=0,
        write_phase_address=True, fused_e3=True,
        delta_key_norm=True, delta_erase_beta_cap=0.95,
    )
    torch.manual_seed(seed)
    m = V13LM(V13Config(**{**common, 'gradient_checkpointing': True})).to(device).train()
    ids = torch.randint(0, 256, (batch_size, seq_len), device=device)
    lab = torch.randint(0, 256, (batch_size, seq_len), device=device)

    def _run(use_ckpt):
        m.config.gradient_checkpointing = use_ckpt
        m.zero_grad(set_to_none=True)
        lm, _, _ = m._hidden_to_lm(ids)
        loss = m.ce_from_lm(lm, lab, chunk=64)
        loss.backward()
        grads = {n: p.grad.detach().clone()
                 for n, p in m.named_parameters() if p.grad is not None}
        return float(loss.detach()), grads

    try:
        l_ck, g_ck = _run(True)
        l_no, g_no = _run(False)
    except Exception as exc:
        torch.set_default_dtype(prev_dtype)
        print(f"[grad_ckpt equiv] RAISED {type(exc).__name__}: {exc}  FAIL")
        return False

    dloss = abs(l_ck - l_no)
    worst = 0.0
    missing = [n for n in g_no if n not in g_ck]
    for name in g_no:
        if name not in g_ck:
            continue
        ref = g_no[name]
        rel = (g_ck[name] - ref).norm().item() / (ref.norm().item() + 1e-12)
        if rel > worst:
            worst = rel
    block_norms = []
    for i in range(common['n_layers']):
        tot = 0.0
        for n, p in m.named_parameters():
            if n.startswith(f'blocks.{i}.') and p.grad is not None:
                tot += float(p.grad.detach().pow(2).sum())
        block_norms.append(tot ** 0.5)
    mx, mn = max(block_norms), min(block_norms)
    ok = (
        dloss < 1e-5
        and worst < 1e-5
        and not missing
        and mn > 0.0
        and (mx / mn) < 10.0
    )
    print(
        f"[grad_ckpt equiv] dloss={dloss:.2e} worst_rel={worst:.2e} "
        f"blocks=[{', '.join(f'{v:.2e}' for v in block_norms)}] "
        f"{'PASS' if ok else 'FAIL'}"
    )
    torch.set_default_dtype(prev_dtype)
    return ok


def test_delta_decay_factored_equiv(batch_size=2, seq_len=80, seed=0):
    """Factored decay (K-independent triangular system) == materialized D path."""
    common = dict(
        vocab_size=512, dim=48, n_heads=3, head_dim=16, n_layers=1, expand=2,
        dropout=0.0, max_seq_len=256, chunk_size=24, gradient_checkpointing=False,
        use_rope=True, use_gsp=True, n_states=3, gate_content_aware=True,
        write_mode='delta', delta_chunk=20, delta_erase_gate=True,
        vault_state=True, vault_state_idx=0, write_phase_address=True,
        delta_key_norm=True, fused_e3=True,
    )
    torch.manual_seed(seed)
    ref = V13PAMLayer(V13Config(**{**common, 'delta_decay_factored': False}))
    fac = V13PAMLayer(V13Config(**{**common, 'delta_decay_factored': True}))
    fac.load_state_dict(ref.state_dict())
    ref.train()
    fac.train()
    x = torch.randn(batch_size, seq_len, common['dim'], 2) * 0.5
    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)
    y1, S1 = ref(x1)
    y2, S2 = fac(x2)
    (y1 ** 2).sum().backward()
    (y2 ** 2).sum().backward()
    dy = (y1 - y2).abs().max().item()
    dS = (S1 - S2).abs().max().item()
    dg = (x1.grad - x2.grad).abs().max().item()
    ok = max(dy, dS, dg) < 1e-6
    print(f"[delta_factored ] y={dy:.2e} S={dS:.2e} grad={dg:.2e}  "
          f"{'PASS' if ok else 'FAIL'}")
    # Fast-decay stress: large softplus(dt) → tiny a[t]. The factored path
    # must either match the materialized path or fall back (min_a guard).
    torch.manual_seed(seed + 7)
    ref2 = V13PAMLayer(V13Config(**{**common, 'delta_decay_factored': False,
                                    'base_dt_bias': 4.0}))
    fac2 = V13PAMLayer(V13Config(**{**common, 'delta_decay_factored': True,
                                    'base_dt_bias': 4.0}))
    fac2.load_state_dict(ref2.state_dict())
    x3 = torch.randn(batch_size, seq_len, common['dim'], 2) * 0.5
    with torch.no_grad():
        y3, S3 = ref2(x3)
        y4, S4 = fac2(x3)
    dy2 = (y3 - y4).abs().max().item()
    dS2 = (S3 - S4).abs().max().item()
    ok2 = max(dy2, dS2) < 1e-5
    print(f"[delta_fact_fast] y={dy2:.2e} S={dS2:.2e}  {'PASS' if ok2 else 'FAIL'}")
    return ok and ok2


def test_ngram_read(batch_size=2, seq_len=40, seed=0):
    """Zero-parameter n-gram read: OFF is bit-identical; ON adds a content
    fingerprint, stays deterministic, keeps parallel==recurrent (decode buffer),
    matches the fused-CE path, and adds no parameters / state keys."""
    import torch.nn.functional as F
    from v13.model import V13LM

    def _cfg(**kw):
        base = dict(
            vocab_size=256, dim=48, n_heads=3, head_dim=16, n_layers=2, expand=2,
            dropout=0.0, max_seq_len=128, chunk_size=24,
            gradient_checkpointing=False, use_rope=True, use_gsp=True,
            n_states=3, gate_content_aware=True,
            write_mode='delta', delta_chunk=20,
            vault_state=True, vault_state_idx=0, write_phase_address=True,
            fused_e3=True,
        )
        base.update(kw)
        return V13Config(**base)

    torch.manual_seed(seed)
    off = V13LM(_cfg()).eval()
    torch.manual_seed(seed)
    on = V13LM(_cfg(ngram_read=True, ngram_size=3, ngram_scale=0.5)).eval()

    # (f) Zero-parameter: identical state_dict keys, identical param count.
    keys_off = set(off.state_dict().keys())
    keys_on = set(on.state_dict().keys())
    n_off = sum(p.numel() for p in off.parameters())
    n_on = sum(p.numel() for p in on.parameters())
    ok_zero = keys_off == keys_on and n_off == n_on

    ids = torch.randint(0, 256, (batch_size, seq_len))
    lbl = torch.randint(0, 256, (batch_size, seq_len))
    with torch.no_grad():
        # (a) OFF bit-identical across two fresh inits (no state leakage).
        off2 = V13LM(_cfg()).eval()
        off2.load_state_dict(off.state_dict())
        lg_off, _, _ = off(ids)
        lg_off2, _, _ = off2(ids)
        d_off = (lg_off - lg_off2).abs().max().item()
        ok_off = d_off == 0.0
        # (b) ON != OFF (fingerprint actually changes the logits).
        lg_on, _, _ = on(ids)
        d_on = (lg_on - lg_off).abs().max().item()
        ok_diff = d_on > 1e-6
        # (c) ON deterministic.
        lg_on2, _, _ = on(ids)
        d_det = (lg_on - lg_on2).abs().max().item()
        ok_det = d_det == 0.0
        # (d) Parallel == one-token recurrent (exercises the decode ctx buffer,
        #     including the boundary zero-fill at the sequence start).
        on._ngram_ctx = None
        lg_par, states, _ = on(ids)
        on._ngram_ctx = None
        st = None
        rec = []
        for t in range(seq_len):
            o, st, _ = on(ids[:, t:t + 1], states=st, step_offset=t)
            rec.append(o)
        lg_rec = torch.cat(rec, dim=1)
        d_par = (lg_par - lg_rec).abs().max().item()
        ok_par = d_par < 1e-8  # fp64 accumulation of the per-token PAM loop
        # (e) Fused-CE path matches the plain forward + CE (ngram injected in
        #     _hidden_to_lm too).
        ref = F.cross_entropy(lg_on.view(-1, 256), lbl.view(-1))
        main, _ = on.fused_ce_loss(ids, lbl, chunk=16)
        d_ce = (main - ref).abs().item()
        ok_ce = d_ce < 1e-5
        # (g) generate() decode path: top_k=1 makes multinomial degenerate
        #     (single-candidate support) → deterministic, so the buffer set on
        #     prefill and consumed on every decode step is bit-reproducible.
        gen1 = on.generate(ids, max_new_tokens=4, top_k=1)
        gen2 = on.generate(ids, max_new_tokens=4, top_k=1)
        ok_gen = torch.equal(gen1, gen2)

    ok = all([ok_zero, ok_off, ok_diff, ok_det, ok_par, ok_ce, ok_gen])
    print(f"[ngram_read     ] off={d_off:.1e} on-off={d_on:.2e} det={d_det:.1e} "
          f"par-rec={d_par:.2e} ce={d_ce:.1e} zero={int(ok_zero)} gen={int(ok_gen)}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_ngram_fusion(batch_size=2, seq_len=40, seed=0):
    """Learned ngram fusion block (Qwen PLE port):
      (a) OFF == ON-at-step-0 bit-identical (key_proj zero-init),
      (b) state_dict gains exactly the fusion keys,
      (c) deterministic, parallel == one-token recurrent (row buffer),
      (d) fused-CE path matches forward + CE,
      (e) generate() reproducible,
      (f) step-0 grads: key_proj ALIVE, conv1d exactly zero (expected),
      (g) row buffer populated after parallel forward.
    """
    import torch.nn.functional as F
    from v13.model import V13LM

    def _cfg(**kw):
        base = dict(
            vocab_size=256, dim=48, n_heads=3, head_dim=16, n_layers=2, expand=2,
            dropout=0.0, max_seq_len=128, chunk_size=24,
            gradient_checkpointing=False, use_rope=True, use_gsp=True,
            n_states=3, gate_content_aware=True,
            write_mode='delta', delta_chunk=20,
            vault_state=True, vault_state_idx=0, write_phase_address=True,
            fused_e3=True,
        )
        base.update(kw)
        return V13Config(**base)

    torch.manual_seed(seed)
    off = V13LM(_cfg()).eval()
    torch.manual_seed(seed)
    on = V13LM(_cfg(ngram_read=True, ngram_size=3, ngram_scale=0.5,
                    ngram_fusion=True)).eval()
    # The fusion module's conv1d init consumes RNG before _init_weights(),
    # so the two builds draw different base weights; align them so the
    # step-0 comparison isolates the fusion block's contribution.
    on.load_state_dict(off.state_dict(), strict=False)

    # (b) state_dict: same keys PLUS exactly the fusion params.
    keys_off = set(off.state_dict().keys())
    keys_on = set(on.state_dict().keys())
    new_keys = sorted(keys_on - keys_off)
    expected = sorted(
        f'ngram_fusion.{n}' for n in
        ['conv1d.weight', 'conv1d.bias',
         'key_proj.weight_real', 'key_proj.weight_imag',
         'key_proj.bias_real', 'key_proj.bias_imag', 'norm.scale']
    )
    ok_keys = (new_keys == expected and keys_off == keys_on - set(new_keys))
    ok_zero_init = all(
        torch.equal(on.state_dict()[k],
                    torch.zeros_like(on.state_dict()[k]))
        for k in new_keys
        if 'key_proj' in k
    )

    ids = torch.randint(0, 256, (batch_size, seq_len))
    lbl = torch.randint(0, 256, (batch_size, seq_len))
    with torch.no_grad():
        # (a) step-0 bit-identity: key_proj zero-init => injection exactly 0.
        lg_off, _, _ = off(ids)
        on._ngram_ctx = None
        on._ngram_row_ctx = None
        lg_on, _, _ = on(ids)
        d_init = (lg_on - lg_off).abs().max().item()
        ok_init = d_init == 0.0
        # (g) row buffer: last n-1 looked-up rows after parallel forward.
        ok_buf = (on._ngram_row_ctx is not None
                  and on._ngram_row_ctx.shape == (batch_size, 2, 48, 2)
                  and not on._ngram_row_ctx.requires_grad)
        # (c) deterministic.
        on._ngram_ctx = None
        on._ngram_row_ctx = None
        lg_on2, states, _ = on(ids)
        d_det = (lg_on - lg_on2).abs().max().item()
        ok_det = d_det == 0.0
        # (c2) parallel == one-token recurrent (exercises the decode row
        #      buffer, including the boundary zero-fill).
        on._ngram_ctx = None
        on._ngram_row_ctx = None
        lg_par, _, _ = on(ids)
        st = None
        rec = []
        for t in range(seq_len):
            o, st, _ = on(ids[:, t:t + 1], states=st, step_offset=t)
            rec.append(o)
        lg_rec = torch.cat(rec, dim=1)
        d_par = (lg_par - lg_rec).abs().max().item()
        ok_par = d_par < 1e-8
        # (d) fused-CE path matches the plain forward + CE.
        ref = F.cross_entropy(lg_on.view(-1, 256), lbl.view(-1))
        main, _ = on.fused_ce_loss(ids, lbl, chunk=16)
        d_ce = (main - ref).abs().item()
        ok_ce = d_ce < 1e-5
        # (e) generate() decode path reproducible (top_k=1 => degenerate
        #     multinomial), buffer set on prefill + every decode step.
        on._ngram_ctx = None
        on._ngram_row_ctx = None
        gen1 = on.generate(ids, max_new_tokens=4, top_k=1)
        on._ngram_ctx = None
        on._ngram_row_ctx = None
        gen2 = on.generate(ids, max_new_tokens=4, top_k=1)
        ok_gen = torch.equal(gen1, gen2)

    # (f) step-0 gradient contract: key_proj alive, conv1d exactly zero.
    on.train()
    lg, _, _ = on(ids)
    loss = F.cross_entropy(lg.view(-1, 256), lbl.view(-1))
    loss.backward()
    g_conv = float(on.ngram_fusion.conv1d.weight.grad.detach().abs().max())
    g_key = float(on.ngram_fusion.key_proj.weight_real.grad.detach().abs().max())
    ok_grad = (g_conv == 0.0 and g_key > 0.0)
    on.eval()

    ok = all([ok_keys, ok_zero_init, ok_init, ok_buf, ok_det,
              ok_par, ok_ce, ok_gen, ok_grad])
    print(f"[ngram_fusion   ] init-off={d_init:.1e} det={d_det:.1e} "
          f"par-rec={d_par:.2e} ce={d_ce:.1e} "
          f"keys={int(ok_keys)} zinit={int(ok_zero_init)} buf={int(ok_buf)} "
          f"gen={int(ok_gen)} conv_g={g_conv:.1e} key_g={g_key:.2e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_drop_shape_mismatches():
    """Resume-safe: growing phase_proj (dim -> 2*dim) reinits cleanly."""
    from v13.train import _drop_shape_mismatches
    cfg_old = V13Config(
        vocab_size=128, dim=32, n_heads=2, head_dim=16, n_layers=1, expand=2,
        n_states=3, routing_content_aware=False,
    )
    cfg_new = V13Config(
        vocab_size=128, dim=32, n_heads=2, head_dim=16, n_layers=1, expand=2,
        n_states=3, routing_content_aware=True,
    )
    old = V13PAMLayer(cfg_old)
    new = V13PAMLayer(cfg_new)
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
        n_states=1, write_mode='additive', vault_state=False,
        write_phase_address=False, gate_content_aware=False,
        gate_surprisal_lambda=0.0, fused_e3=True,
    )
    results = []
    results.append(_run_mode("baseline", V13Config(**common)))
    results.append(_run_mode("E1 perchannel", V13Config(**{**common, 'decay_mode': 'per_channel'})))
    results.append(_run_mode("E2 delta", V13Config(**{**common, 'write_mode': 'delta', 'delta_chunk': 20})))
    results.append(_run_mode("E3 multistate", V13Config(**{**common, 'n_states': 2})))
    results.append(_run_mode("E1+E3 combo", V13Config(**{**common, 'decay_mode': 'per_channel', 'n_states': 2})))
    # Stage-6 levers (defaults OFF elsewhere; flag-gated identity when disabled)
    results.append(_run_mode(
        "vault state",
        V13Config(**{**common, 'n_states': 3, 'vault_state': True, 'vault_state_idx': 0}),
    ))
    results.append(_run_mode(
        "phase address",
        V13Config(**{**common, 'n_states': 3, 'write_phase_address': True}),
    ))
    results.append(_run_mode(
        "vault+phase",
        V13Config(**{
            **common, 'n_states': 3, 'vault_state': True, 'vault_state_idx': 0,
            'write_phase_address': True,
        }),
    ))
    results.append(_run_mode(
        "gamma_floor",
        V13Config(**{**common, 'n_states': 3, 'gamma_floor': 0.9}),
    ))
    results.append(_run_mode(
        "v13 selective",
        V13Config(**{
            **common, 'n_states': 3, 'write_mode': 'delta', 'delta_chunk': 20,
            'gate_content_aware': True, 'vault_state': True, 'vault_state_idx': 0,
            'write_phase_address': True, 'fused_e3': False,
        }),
    ))
    results.append(test_fused_e3_equiv())
    results.append(test_fused_ce_equiv())
    results.append(test_competitive_retrieval_equiv())
    results.append(test_drop_shape_mismatches())
    results.append(test_delta_keynorm_equiv())
    results.append(test_delta_raw_readout_equiv())
    results.append(test_delta_erase_cap())
    results.append(test_grad_ckpt_equiv())
    results.append(test_delta_decay_factored_equiv())
    results.append(test_ngram_read())
    results.append(test_ngram_fusion())
    print()
    if all(results):
        print("ALL MODES PASS: parallel train form == O(1) recurrent form.")
    else:
        print("SOME MODES FAILED — see above.")
        raise SystemExit(1)


if __name__ == '__main__':
    main()
