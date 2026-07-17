"""Correctness self-tests for V12 PAM memory dynamics.

For each mode we verify that the parallel TRAINING form (chunked / dual / UT)
produces the same output as the sequential O(1) RECURRENT inference form on the
same random input. If these agree, the chunked math is correct.

Run:
    .venv/bin/python -m v12.selftest
"""

import torch

from v12.model import V12Config, V12PAMLayer


def _run_mode(name, cfg, batch_size=2, seq_len=80, atol=2e-3, seed=0):
    torch.manual_seed(seed)
    layer = V12PAMLayer(cfg, layer_idx=0).eval()
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
    from v12.train import _warmstart_chatml_embeddings, _CHATML_BASE_VOCAB, _CHATML_TOKEN_IDS

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


def test_fused_e3_equiv(batch_size=2, seq_len=80, seed=0):
    """Fused E3 path must equal the reference K-loop exactly (fwd, grad, state)."""
    common = dict(
        vocab_size=512, dim=48, n_heads=3, head_dim=16, n_layers=1, expand=2,
        dropout=0.0, max_seq_len=256, chunk_size=24, gradient_checkpointing=False,
        use_rope=True, use_gsp=True, n_states=3, gate_content_aware=True,
    )
    torch.manual_seed(seed)
    loop = V12PAMLayer(V12Config(**{**common, 'fused_e3': False}))
    fused = V12PAMLayer(V12Config(**{**common, 'fused_e3': True}))
    rc = V12PAMLayer(V12Config(**{**common, 'fused_e3': True, 'recompute_pam_chunks': True}))
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
    from v12.model import V12LM
    torch.manual_seed(seed)
    cfg = V12Config(
        vocab_size=256, dim=48, n_heads=3, head_dim=16, n_layers=2, expand=2,
        dropout=0.0, max_seq_len=128, chunk_size=24, gradient_checkpointing=False,
        n_states=3, gate_content_aware=True,
    )
    m = V12LM(cfg).train()
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


def test_head_gate(seed=0):
    """M1: learnable-head-count gate. Parallel==recurrent still holds; a closed
    gate zeroes that head's contribution; L0 penalty flows grad to log_alpha."""
    from v12.model import V12PAMLayer
    common = dict(
        vocab_size=512, dim=48, n_heads=3, head_dim=16, n_layers=1, expand=2,
        dropout=0.0, max_seq_len=256, chunk_size=24, gradient_checkpointing=False,
        use_rope=True, use_gsp=True, n_states=3, gate_content_aware=True,
        head_gate=True, head_gate_l0_lambda=0.01,
    )
    ok_equiv = _run_mode("head_gate E3", V12Config(**common), seed=seed)

    # A fully-closed gate (log_alpha -> -20) must zero that head's output block.
    torch.manual_seed(seed)
    layer = V12PAMLayer(V12Config(**common)).eval()
    with torch.no_grad():
        layer.head_gate.log_alpha[1] = -20.0
    z = layer.head_gate.gate()
    ok_zero = float(z[1].abs()) < 1e-6 and float(z[0]) > 0.5
    active = int(layer.head_gate.active_mask().sum())
    ok_count = active == 2

    # L0 penalty is finite, positive, and differentiable w.r.t. log_alpha.
    layer.train()
    x = torch.randn(2, 40, common['dim'], 2) * 0.5
    layer.head_gate.log_alpha.grad = None
    _out, _s = layer(x)
    pen = layer._route_aux
    ok_pen = pen is not None and torch.isfinite(pen).all() and float(pen) > 0
    pen.backward()
    ok_grad = layer.head_gate.log_alpha.grad is not None and \
        float(layer.head_gate.log_alpha.grad.abs().sum()) > 0

    ok = ok_equiv and ok_zero and ok_count and ok_pen and ok_grad
    print(f"[head_gate M1  ] closed_zero={ok_zero} active={active}/3 "
          f"l0={float(pen):.3e} grad={ok_grad}  {'PASS' if ok else 'FAIL'}")
    return ok


def test_head_freeze(seed=0):
    """M2: freezing head slots zeros grad on their fused-projection slices while
    active heads still receive gradient. Parallel==recurrent is unaffected."""
    from v12.model import V12LM
    torch.manual_seed(seed)
    cfg = V12Config(
        vocab_size=256, dim=48, n_heads=3, head_dim=16, n_layers=1, expand=2,
        dropout=0.0, max_seq_len=128, chunk_size=24, gradient_checkpointing=False,
        n_states=3, gate_content_aware=True, write_phase_address=True,
    )
    m = V12LM(cfg).train()
    # Train only head slot 2; freeze slots 0 and 1.
    m.set_stage_active_heads([2])
    ids = torch.randint(0, cfg.vocab_size, (2, 40))
    m.zero_grad()
    loss, _ = m.fused_ce_loss(ids, ids, chunk=16)
    loss.backward()

    d = cfg.head_dim
    pam = m.blocks[0].pam
    gr = pam.qkv_proj.weight_real.grad         # [3*inner, dim]
    inner = pam.inner_dim
    # Frozen head 0 rows (across q,k,v blocks) must be exactly zero.
    frozen_rows = []
    for blk in range(3):
        frozen_rows += list(range(blk * inner + 0 * d, blk * inner + 1 * d))
    active_rows = []
    for blk in range(3):
        active_rows += list(range(blk * inner + 2 * d, blk * inner + 3 * d))
    frozen_zero = float(gr[frozen_rows].abs().sum()) == 0.0
    active_nonzero = float(gr[active_rows].abs().sum()) > 0.0
    # o_proj frozen columns zero.
    go = pam.o_proj.weight_real.grad           # [dim, inner]
    o_frozen_zero = float(go[:, 0 * d:1 * d].abs().sum()) == 0.0
    o_active_nonzero = float(go[:, 2 * d:3 * d].abs().sum()) > 0.0
    # M3 per-head phase-band params: frozen head 0 row zero, active head 2 nonzero.
    wp = pam.write_phase_w.grad                # [H, head_dim]
    wp_frozen_zero = float(wp[0].abs().sum()) == 0.0
    wp_active_nonzero = float(wp[2].abs().sum()) > 0.0

    ok = (frozen_zero and active_nonzero and o_frozen_zero and o_active_nonzero
          and wp_frozen_zero and wp_active_nonzero)
    print(f"[head_freeze M2 ] qkv_frozen0={frozen_zero} qkv_active={active_nonzero} "
          f"o_frozen0={o_frozen_zero} o_active={o_active_nonzero} "
          f"phase_frozen0={wp_frozen_zero} phase_active={wp_active_nonzero}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def _model_parallel_recurrent(cfg, name, seq_len=48, seed=0, atol=2e-3):
    """Model-level parallel(full forward) == O(1) recurrent(token-by-token)."""
    from v12.model import V12LM
    torch.manual_seed(seed)
    m = V12LM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, seq_len))
    with torch.no_grad():
        par, _, _ = m(ids)
        steps, states = [], None
        for t in range(seq_len):
            lg, states, _ = m(ids[:, t:t + 1], states=states, step_offset=t)
            steps.append(lg)
        rec = torch.cat(steps, dim=1)
    diff = (par - rec).abs().max().item()
    ok = diff < atol
    print(f"[{name:14s}] max|Δlogits|={diff:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def test_grown_stack_equiv(seed=0):
    """M4.1: a non-uniform (spec-built) stack — grammar base + facts group with
    different n_heads / vault / phase-address — still has parallel==recurrent."""
    from v12.model import V12Config
    cfg = V12Config(
        vocab_size=256, dim=48, n_heads=3, head_dim=16, n_layers=2, expand=2,
        dropout=0.0, max_seq_len=128, chunk_size=24, gradient_checkpointing=False,
        n_states=3, gate_content_aware=True,
        layer_specs=[
            {'skill': 'grammar', 'group_id': 'base'},
            {'skill': 'grammar', 'group_id': 'base'},
            {'skill': 'facts', 'group_id': 'facts', 'n_heads': 4,
             'vault_state': True, 'write_phase_address': True},
            {'skill': 'facts', 'group_id': 'facts', 'n_heads': 4,
             'vault_state': True, 'write_phase_address': True},
        ],
    )
    return _model_parallel_recurrent(cfg, "grown_stack M4", seed=seed)


def test_uniform_fallback_identity(seed=0):
    """M4.1: layer_specs of N uniform 'base' entries builds a bit-identical model
    to the plain n_layers path (same RNG draw order => same weights)."""
    from v12.model import V12Config, V12LM
    base = dict(
        vocab_size=256, dim=48, n_heads=3, head_dim=16, n_layers=3, expand=2,
        dropout=0.0, max_seq_len=128, chunk_size=24, gradient_checkpointing=False,
        n_states=3, gate_content_aware=True,
    )
    torch.manual_seed(seed)
    m_none = V12LM(V12Config(**base))
    uniform = [{'group_id': 'base', 'stage': 0} for _ in range(base['n_layers'])]
    torch.manual_seed(seed)
    m_spec = V12LM(V12Config(**base, layer_specs=uniform))
    sa, sb = m_none.state_dict(), m_spec.state_dict()
    same_keys = set(sa) == set(sb)
    maxd = max((sa[k] - sb[k]).abs().max().item() for k in sa) if same_keys else float('inf')
    ok = same_keys and maxd == 0.0
    print(f"[uniform_ident  ] keys_match={same_keys} max|Δw|={maxd:.2e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_grow_and_freeze(seed=0):
    """M4.2: grow_layers preserves base indices/weights, freeze_layers zeroes grad
    on frozen blocks while grown blocks train, and the manifest round-trips."""
    from v12.model import V12Config, V12LM
    torch.manual_seed(seed)
    base = V12Config(
        vocab_size=256, dim=48, n_heads=3, head_dim=16, n_layers=2, expand=2,
        dropout=0.0, max_seq_len=128, chunk_size=24, gradient_checkpointing=False,
        n_states=3, gate_content_aware=True,
    )
    m = V12LM(base)
    base_state = {f'blocks.0.{k}': v.clone()
                  for k, v in m.blocks[0].state_dict().items()}
    # Grow two 'facts' layers on top of the frozen base.
    new_idx = m.grow_layers([
        {'skill': 'facts', 'group_id': 'facts', 'vault_state': True},
        {'skill': 'facts', 'group_id': 'facts', 'vault_state': True},
    ])
    ok_idx = new_idx == [2, 3] and len(m.blocks) == 4
    # Base block-0 weights unchanged after growth.
    ok_preserve = all(
        torch.equal(base_state[f'blocks.0.{k}'], v)
        for k, v in m.blocks[0].state_dict().items()
    )
    # Manifest: substrate_hash stamped on grown layers, provenance present.
    man = m.layer_manifest()
    ok_manifest = (
        len(man) == 4
        and man[2]['skill'] == 'facts' and man[2].get('substrate_hash')
        and man[0]['group_id'] == 'base'
    )

    # Freeze base (0,1), train grown (2,3); frozen blocks must get zero/no grad.
    m.freeze_layers(range(0, 2))
    m.train()
    ids = torch.randint(0, base.vocab_size, (2, 32))
    m.zero_grad()
    loss, _ = m.fused_ce_loss(ids, ids, chunk=16)
    loss.backward()
    frozen_grad = sum(
        float(p.grad.abs().sum()) for p in m.blocks[0].parameters() if p.grad is not None
    )
    grown_has_grad = any(
        p.grad is not None and float(p.grad.abs().sum()) > 0 for p in m.blocks[2].parameters()
    )
    frozen_reqgrad = any(p.requires_grad for p in m.blocks[0].parameters())
    ok_freeze = frozen_grad == 0.0 and grown_has_grad and not frozen_reqgrad

    # Config round-trips: rebuild from asdict(config) and load state strict.
    from dataclasses import asdict
    cfg2 = V12Config(**asdict(m.config))
    m2 = V12LM(cfg2)
    missing, unexpected = m2.load_state_dict(m.state_dict(), strict=False)
    ok_roundtrip = (len(m2.blocks) == 4 and not missing and not unexpected)

    ok = ok_idx and ok_preserve and ok_manifest and ok_freeze and ok_roundtrip
    print(f"[grow_freeze M4 ] idx={ok_idx} preserve={ok_preserve} manifest={bool(ok_manifest)} "
          f"freeze={ok_freeze} roundtrip={ok_roundtrip}  {'PASS' if ok else 'FAIL'}")
    return ok


def test_attach_mode_moe_fallback(seed=0):
    """M4.2: an attach_mode='moe' group runs (sequential fallback) and keeps
    parallel==recurrent; attach_mode survives config round-trip."""
    from v12.model import V12Config, V12LM
    from dataclasses import asdict
    cfg = V12Config(
        vocab_size=256, dim=48, n_heads=3, head_dim=16, n_layers=1, expand=2,
        dropout=0.0, max_seq_len=128, chunk_size=24, gradient_checkpointing=False,
        n_states=3, gate_content_aware=True,
        layer_specs=[
            {'skill': 'grammar', 'group_id': 'base'},
            {'skill': 'code_c', 'group_id': 'code', 'attach_mode': 'moe'},
            {'skill': 'code_java', 'group_id': 'code', 'attach_mode': 'moe'},
        ],
    )
    ok_equiv = _model_parallel_recurrent(cfg, "moe_fallback M4", seed=seed)
    torch.manual_seed(seed)
    m = V12LM(cfg)
    modes = m._attach_modes
    ok_modes = modes == ['sequential', 'moe', 'moe']
    cfg2 = V12Config(**asdict(m.config))
    ok_persist = [s.get('attach_mode') for s in cfg2.layer_specs] == ['sequential', 'moe', 'moe']
    ok = ok_equiv and ok_modes and ok_persist
    print(f"[moe_schema M4  ] modes={modes} persist={ok_persist}  {'PASS' if ok else 'FAIL'}")
    return ok


def test_compact_heads(seed=0):
    """M4.4: compacting a head-gated checkpoint drops closed slots (per-layer
    n_heads) and preserves inference logits (dropped heads had z≈0)."""
    import os
    import tempfile
    from dataclasses import asdict
    from v12.model import V12Config, V12LM
    from v12.compact import compact_checkpoint
    torch.manual_seed(seed)
    cfg = V12Config(
        vocab_size=64, dim=48, n_heads=6, head_dim=16, n_layers=2, expand=2,
        dropout=0.0, max_seq_len=64, chunk_size=16, gradient_checkpointing=False,
        n_states=3, gate_content_aware=True, head_gate=True, head_gate_l0_lambda=0.001,
        write_phase_address=True,
    )
    m = V12LM(cfg)
    with torch.no_grad():
        m.blocks[0].pam.head_gate.log_alpha[4] = -20.0
        m.blocks[0].pam.head_gate.log_alpha[5] = -20.0
        m.blocks[1].pam.head_gate.log_alpha[3] = -20.0
    d = tempfile.mkdtemp()
    src, out = os.path.join(d, 'src.pt'), os.path.join(d, 'slim.pt')
    torch.save({'config': asdict(m.config), 'model_state_dict': m.state_dict()}, src)
    slim, maxd = compact_checkpoint(src, out, threshold=1e-3, verify_seq_len=24)
    heads = [b.pam.num_heads for b in slim.blocks]
    ok = heads == [4, 5] and maxd < 1e-4
    print(f"[compact M4.4  ] kept_heads={heads} max|Δlogits|={maxd:.2e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_drop_shape_mismatches():
    """Resume-safe: growing phase_proj (dim -> 2*dim) reinits cleanly."""
    from v12.train import _drop_shape_mismatches
    cfg_old = V12Config(
        vocab_size=128, dim=32, n_heads=2, head_dim=16, n_layers=1, expand=2,
        n_states=3, routing_content_aware=False,
    )
    cfg_new = V12Config(
        vocab_size=128, dim=32, n_heads=2, head_dim=16, n_layers=1, expand=2,
        n_states=3, routing_content_aware=True,
    )
    old = V12PAMLayer(cfg_old)
    new = V12PAMLayer(cfg_new)
    state = old.state_dict()
    dropped = _drop_shape_mismatches(new, state)
    new.load_state_dict(state, strict=False)
    keys = [k for k, _, _ in dropped]
    assert any('phase_proj' in k for k in keys), dropped
    print(f"[resume_shape   ] dropped {keys}  PASS")
    return True


# ── Playable module system (registry + resolver + publish + pack) ────────────

def _mini_geom():
    return dict(vocab_size=64, dim=32, n_heads=2, head_dim=16, n_states=3,
                gate_content_aware=True, gradient_checkpointing=False,
                chunk_size=16, max_seq_len=64)


def _register_synthetic(reg, module_id, version, *, role, n_layers, requires=(),
                        substrate_hash=None):
    """Register a module with real (random) blocks but a chosen substrate_hash."""
    from dataclasses import asdict
    from v12.model import V12Config, V12LM
    from v12.registry import ModuleCard, hash_block_states
    cfg = V12Config(n_layers=n_layers, **_mini_geom())
    m = V12LM(cfg)
    state = m.state_dict()
    self_hash = hash_block_states([(state, i) for i in range(n_layers)])
    card = ModuleCard(
        module_id=module_id, version=version, role=role, skill=module_id,
        group_id=module_id, dim=cfg.dim, head_dim=cfg.head_dim,
        vocab_size=cfg.vocab_size, n_layers=n_layers,
        layer_specs=m.config.layer_specs, self_hash=self_hash,
        substrate_hash=substrate_hash, requires=list(requires),
    )
    reg.add({'config': asdict(m.config), 'model_state_dict': state}, card, overwrite=True)
    return card


def test_card_roundtrip():
    """Card I/O: sidecar + embedded copy both read back identically."""
    import os
    import tempfile
    from dataclasses import asdict
    from v12.model import V12Config, V12LM
    from v12.registry import ModuleCard, Requirement, read_card, sidecar_path, write_card
    cfg = V12Config(n_layers=2, **_mini_geom())
    m = V12LM(cfg)
    d = tempfile.mkdtemp()
    ckpt_path = os.path.join(d, 'model.pt')
    torch.save({'config': asdict(m.config), 'model_state_dict': m.state_dict()}, ckpt_path)
    card = ModuleCard(module_id='grammar_by_x', version='1.0', role='base',
                      skill='grammar', dim=cfg.dim, head_dim=cfg.head_dim,
                      vocab_size=cfg.vocab_size, n_layers=2,
                      requires=[Requirement('foo', '>=1.0', 'prelayer')])
    write_card(ckpt_path, card, embed=True)
    from_sidecar = read_card(ckpt_path)
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    embedded = ModuleCard.from_dict(ck['config']['module_card'])
    os.remove(sidecar_path(ckpt_path))
    from_embedded = read_card(ckpt_path)  # sidecar gone -> falls back to embedded
    ok = (from_sidecar.module_id == 'grammar_by_x' and from_sidecar.role == 'base'
          and from_sidecar.requires[0].module_id == 'foo'
          and embedded.version == '1.0' and from_embedded.module_id == 'grammar_by_x')
    print(f"[card_roundtrip ] sidecar+embedded read-back  {'PASS' if ok else 'FAIL'}")
    return ok


def test_resolver_multisource():
    """Multi-source graph: bio requires grammar + reasoning + fact + bio_base;
    resolves to a single ordered stack with grammar first and bio last."""
    import os
    import tempfile
    from v12.registry import Registry, Requirement, resolve
    reg = Registry(os.path.join(tempfile.mkdtemp(), 'reg'))
    _register_synthetic(reg, 'grammar', '1.0', role='base', n_layers=2)
    for gid in ('fact', 'reasoning', 'bio_base'):
        _register_synthetic(reg, gid, '1.0', role='group', n_layers=2,
                             requires=[Requirement('grammar', '>=1.0', 'prelayer')])
    _register_synthetic(reg, 'bio', '1.0', role='group', n_layers=2, requires=[
        Requirement('grammar', '>=1.0', 'prelayer'),
        Requirement('reasoning', '>=1.0', 'prelayer'),
        Requirement('fact', '>=1.0', 'prelayer'),
        Requirement('bio_base', '>=1.0', 'prelayer'),
    ])
    plan = resolve('bio', '*', reg)
    order = [r.module_id for r in plan.stack]
    ok = (order[0] == 'grammar' and order[-1] == 'bio'
          and set(order) == {'grammar', 'fact', 'reasoning', 'bio_base', 'bio'}
          and all(order.index('grammar') < order.index(x)
                  for x in ('fact', 'reasoning', 'bio_base', 'bio'))
          and not plan.report['warnings'])
    print(f"[resolve_multi  ] order={order}  {'PASS' if ok else 'FAIL'}")
    return ok


def test_version_conflict():
    """Incompatible constraints on the same module raise a conflict."""
    import os
    import tempfile
    from v12.registry import Registry, Requirement, resolve
    reg = Registry(os.path.join(tempfile.mkdtemp(), 'reg'))
    _register_synthetic(reg, 'grammar', '1.0', role='base', n_layers=2)
    _register_synthetic(reg, 'bad', '1.0', role='group', n_layers=2,
                        requires=[Requirement('grammar', '<1.0', 'prelayer')])
    raised = False
    try:
        resolve('bad', '*', reg)
    except ValueError:
        raised = True
    print(f"[version_conflict] raised={raised}  {'PASS' if raised else 'FAIL'}")
    return raised


def test_prelayer_hash_mismatch():
    """A wrong substrate_hash is a conflict; --force downgrades to a warning."""
    import os
    import tempfile
    from v12.registry import Registry, Requirement, resolve
    reg = Registry(os.path.join(tempfile.mkdtemp(), 'reg'))
    _register_synthetic(reg, 'grammar', '1.0', role='base', n_layers=2)
    _register_synthetic(reg, 'fact', '1.0', role='group', n_layers=2,
                        requires=[Requirement('grammar', '>=1.0', 'prelayer')],
                        substrate_hash='deadbeef' * 8)
    raised = False
    try:
        resolve('fact', '*', reg)
    except ValueError:
        raised = True
    forced = resolve('fact', '*', reg, force=True)
    ok = raised and bool(forced.report['warnings'])
    print(f"[prelayer_hash  ] raised={raised} force_warns={bool(forced.report['warnings'])}"
          f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_finetuned_standalone():
    """A finetuned dependency is recorded as lineage, not stacked."""
    import os
    import tempfile
    from v12.registry import Registry, Requirement, resolve
    reg = Registry(os.path.join(tempfile.mkdtemp(), 'reg'))
    _register_synthetic(reg, 'bio_ft', '1.0', role='base', n_layers=2,
                        requires=[Requirement('bio_base', '*', 'finetuned')])
    plan = resolve('bio_ft', '*', reg)
    order = [r.module_id for r in plan.stack]
    ok = (order == ['bio_ft'] and len(plan.lineage) == 1
          and plan.lineage[0]['module_id'] == 'bio_base')
    print(f"[finetuned      ] stack={order} lineage={[l['module_id'] for l in plan.lineage]}"
          f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_module_end_to_end(seed=0):
    """Train a grown stack, publish base+fact+reasoning, resolve -> pack -> load
    strict -> forward; packed logits match the original AND parallel==recurrent."""
    import os
    import tempfile
    from dataclasses import asdict
    from v12.model import V12Config, V12LM
    from v12.pack import pack_from_registry
    from v12.publish import parse_requires, publish
    from v12.registry import Registry, resolve
    torch.manual_seed(seed)
    d = tempfile.mkdtemp()
    reg = Registry(os.path.join(d, 'reg'))
    cfg = V12Config(n_layers=2, **_mini_geom())
    m = V12LM(cfg)
    m.grow_layers([{'skill': 'fact', 'group_id': 'fact'}] * 2)
    m.grow_layers([{'skill': 'reasoning', 'group_id': 'reasoning'}] * 2)
    m.eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 24))
    with torch.no_grad():
        ref, _, _ = m(ids)
    full = os.path.join(d, 'full.pt')
    torch.save({'config': asdict(m.config), 'model_state_dict': m.state_dict()}, full)
    publish(full, module_id='grammar', version='1.0', role='base', registry=reg)
    publish(full, module_id='fact', version='1.0', role='group', group_id='fact',
            requires=parse_requires('grammar@>=1.0:prelayer'), registry=reg)
    publish(full, module_id='reasoning', version='1.0', role='group', group_id='reasoning',
            requires=parse_requires('grammar@>=1.0:prelayer,fact@>=1.0:prelayer'),
            registry=reg)

    plan = resolve('reasoning', '*', reg)
    assert [r.module_id for r in plan.stack] == ['grammar', 'fact', 'reasoning']
    assert not plan.report['warnings'], plan.report['warnings']

    out = os.path.join(d, 'packed.pt')
    pack_from_registry('reasoning', '*', reg, out)
    ck = torch.load(out, map_location='cpu', weights_only=False)
    c2 = V12Config(**{k: v for k, v in ck['config'].items()
                      if k in V12Config.__dataclass_fields__})
    m2 = V12LM(c2)
    missing, unexpected = m2.load_state_dict(ck['model_state_dict'], strict=True)
    m2.eval()
    with torch.no_grad():
        p2, _, _ = m2(ids)
    match = (ref - p2).abs().max().item()
    equiv = _model_parallel_recurrent(c2, "packed M5", seq_len=24, seed=seed)
    ok = match < 1e-6 and equiv
    print(f"[module_e2e     ] packed==orig Δ={match:.2e} strict_load_ok  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    torch.set_default_dtype(torch.float64)  # high precision for the math check
    test_warmstart_chatml()
    common = dict(
        vocab_size=512, dim=32, n_heads=2, head_dim=16, n_layers=1,
        expand=2, dropout=0.0, max_seq_len=512, chunk_size=24,
        gradient_checkpointing=False, use_rope=True, use_gsp=True,
    )
    results = []
    results.append(_run_mode("baseline", V12Config(**common)))
    results.append(_run_mode("E2 delta", V12Config(**{**common, 'write_mode': 'delta', 'delta_chunk': 20})))
    results.append(_run_mode("E3 multistate", V12Config(**{**common, 'n_states': 2})))
    # M3 write-mechanism levers (defaults OFF elsewhere; flag-gated identity when disabled)
    results.append(_run_mode(
        "vault state",
        V12Config(**{**common, 'n_states': 3, 'vault_state': True, 'vault_state_idx': 0}),
    ))
    results.append(_run_mode(
        "phase address",
        V12Config(**{**common, 'n_states': 3, 'write_phase_address': True}),
    ))
    results.append(_run_mode(
        "vault+phase",
        V12Config(**{
            **common, 'n_states': 3, 'vault_state': True, 'vault_state_idx': 0,
            'write_phase_address': True,
        }),
    ))
    results.append(_run_mode(
        "gamma_floor",
        V12Config(**{**common, 'n_states': 3, 'gamma_floor': 0.9}),
    ))
    results.append(test_fused_e3_equiv())
    results.append(test_fused_ce_equiv())
    results.append(test_head_gate())
    results.append(test_head_freeze())
    results.append(test_grown_stack_equiv())
    results.append(test_uniform_fallback_identity())
    results.append(test_grow_and_freeze())
    results.append(test_attach_mode_moe_fallback())
    results.append(test_compact_heads())
    results.append(test_drop_shape_mismatches())
    # Playable module system (M5): registry + resolver + publish + pack.
    results.append(test_card_roundtrip())
    results.append(test_resolver_multisource())
    results.append(test_version_conflict())
    results.append(test_prelayer_hash_mismatch())
    results.append(test_finetuned_standalone())
    results.append(test_module_end_to_end())
    print()
    if all(results):
        print("ALL MODES PASS: parallel train form == O(1) recurrent form.")
    else:
        print("SOME MODES FAILED — see above.")
        raise SystemExit(1)


if __name__ == '__main__':
    main()
