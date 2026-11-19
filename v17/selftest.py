"""v17 selftest: ``sempyt`` named-axis invariants, zero-spin phasor, causality, decode parity, gradients, param parity."""

from __future__ import annotations

import torch

from sempyt import Dim, SplitComplex, cos, named, sin, stack
from v17.model import (
    POWConfig,
    POWLM,
    build_arm,
    stateful_scan,
    unit_circle,
)


def test_stateful_scan():
    torch.manual_seed(0)
    for T in (1, 17, 64, 195):
        b, t, d = Dim("batch", 6), Dim("time", T), Dim("feat", 8)
        x_raw = torch.randn(6, T, 8)
        g_raw = torch.sigmoid(torch.randn(6, T)) * 0.95 + 0.03
        s0_raw = torch.randn(6, 8)

        y_nt, last_nt = stateful_scan(
            named(x_raw, (b, t, d)),
            named(g_raw, (b, t)),
            over=t,
            carry=named(s0_raw, (b, d)),
            chunk=64,
        )
        s = s0_raw.clone()
        ys = []
        for step in range(T):
            s = g_raw[:, step:step + 1] * s + x_raw[:, step, :]
            ys.append(s)
        y_seq = torch.stack(ys, dim=1)
        err = (y_nt.raw(b, t, d) - y_seq).abs().max().item()
        err_last = (last_nt.raw(b, d) - s).abs().max().item()
        assert err < 2e-5 and err_last < 2e-5, f"stateful_scan mismatch T={T}: {err}, {err_last}"
    print("[PASS] sempyt stateful_scan == sequential loop (all ragged T, with carry)")


def test_zero_spin_phasor():
    """Prove S^1 carrier phasor holds its angle without spinning when s_t -> 0 (fixes N9s bug)."""
    b, t, p = Dim("batch", 4), Dim("time", 200), Dim("rot_pair", 2)
    cplx = SplitComplex(p)
    theta = named(torch.tensor([0.7, -1.9, 2.5, -0.4]), (b,))
    u0 = stack([cos(theta), sin(theta)], into=p)                            # (batch, rot_pair)
    x_raw = torch.zeros(4, 200, 2)
    x_raw[:, 0, :] = u0.raw(b, p)
    g_raw = torch.full((4, 200), 0.98)
    g_raw[:, 0] = 0.0
    y_nt, _ = stateful_scan(named(x_raw, (b, t, p)), named(g_raw, (b, t)), over=t)
    z_nt = unit_circle(y_nt, p, cplx)
    drift = (z_nt.raw(b, t, p) - u0.raw(b, p).unsqueeze(1)).abs().max().item()
    assert drift < 1e-4, f"phasor spun during filler: drift={drift}"
    print(f"[PASS] sempyt S^1 carrier phasor zero-spin over 200 filler steps (max drift={drift:.2e})")


def test_causality_and_streaming_decode(device: str):
    torch.manual_seed(7)
    cfg = POWConfig(vocab_size=256, dim=128, n_layers=3, n_heads=4, head_dim=32,
                    mlp_hidden=256, op_rings=64, op_heads=4, n_bounces=3, short_conv_k=4)
    model = POWLM(cfg).to(device).eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if any(k in name for k in ("sync_alpha", "bounce_alpha", "echo_out", "qkv_conv", "warp_proj")):
                param.normal_(0.0, 0.15)

    B, T = 2, 145
    ids = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    # 1. Exact causality check
    ids_pert = ids.clone()
    ids_pert[:, 70:] = (ids_pert[:, 70:] + 13) % cfg.vocab_size
    with torch.no_grad():
        logits_a, _ = model(ids)
        logits_b, _ = model(ids_pert)
    causal_diff = (logits_a[:, :70] - logits_b[:, :70]).abs().max().item()
    assert causal_diff < 1e-4, f"causality violated: {causal_diff}"
    print(f"[PASS] strict causality across DualPhase + 3-Bounce + CausalOperator (diff={causal_diff:.2e})")

    # 2. Full parallel prefill == chunked prefill (80 + 65) == 1-token O(1) decode
    # (Triton _delta_prep uses log(g + 1e-6), giving ~3e-4 fp32 drift over 145 steps, matching v13's 1e-3 bar)
    with torch.no_grad():
        logits_full, _ = model(ids)
        logits_c1, st_c1 = model(ids[:, :80], states=None)
        logits_c2, _ = model(ids[:, 80:], states=st_c1)
        logits_chunk = torch.cat([logits_c1, logits_c2], dim=1)

        st_step = None
        step_outs = []
        for step in range(T):
            lt, st_step = model(ids[:, step:step + 1], states=st_step)
            step_outs.append(lt)
        logits_step = torch.cat(step_outs, dim=1)

    err_chunk = (logits_full - logits_chunk).abs().max().item()
    err_step = (logits_full - logits_step).abs().max().item()
    assert err_chunk < 1e-3 and err_step < 1e-3, f"decode parity failed: chunk={err_chunk}, step={err_step}"
    print(f"[PASS] prefill == chunked ({err_chunk:.2e}) == 1-token O(1) decode ({err_step:.2e})")


def test_gradients(device: str):
    torch.manual_seed(11)
    cfg = POWConfig(vocab_size=256, dim=128, n_layers=2, n_heads=4, head_dim=32,
                    mlp_hidden=256, op_rings=64, op_heads=4, n_bounces=3)
    model = POWLM(cfg).to(device).train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    x = torch.randint(0, cfg.vocab_size, (2, 64), device=device)
    y = torch.randint(0, cfg.vocab_size, (2, 64), device=device)

    h, _ = model._hidden_to_lm(x)
    loss = model.ce_from_lm(h, y)
    loss.backward()
    for key in ("sync_alpha", "bounce_alpha", "bounce_q", "in_coord", "out_coord", "q_coord", "op_gates", "echo_out"):
        matched = [p.grad.abs().max().item() for n, p in model.named_parameters() if key in n and p.grad is not None]
        assert matched and max(matched) > 0.0, f"zero step-0 gradient on {key}: {matched}"
    opt.step()
    opt.zero_grad()

    h2, _ = model._hidden_to_lm(x)
    loss2 = model.ce_from_lm(h2, y)
    loss2.backward()
    dead = [n for n, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().max().item() == 0.0)]
    assert not dead, f"dead parameters after step 1: {dead}"
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"[PASS] non-zero gradients verified on all {n_trainable} trainable parameter tensors")


def test_arm_param_parity():
    arms = ("pow_v17", "pow_sync_only", "pow_work_only", "pam_v2", "gdn_like", "tpp")
    counts = {}
    for a in arms:
        m = build_arm(a, vocab_size=50257, dim=384, n_layers=4, n_heads=6, head_dim=64, base_mlp=1024)
        counts[a] = sum(p.numel() for p in m.parameters() if p.requires_grad)
    ref = counts["pow_v17"]
    for a, c in counts.items():
        rel = abs(c - ref) / ref
        assert rel < 0.003, f"param mismatch on {a}: {c} vs {ref} ({rel:.4%})"
        print(f"  arm {a:15s}: {c:,} trainable params ({(c - ref) / ref:+.3%})")
    print("[PASS] exact parameter parity (< 0.3%) across all 6 arms")


def test_imb_generators():
    from v17.bench import (
        FastTok, IMBPools, collate_docs, gen_b2_rules, gen_facts_recall,
        gen_prose_lure, gen_rxk, gen_scope_doc, gen_two_hop, make_train_batch,
    )
    import random
    P = IMBPools(FastTok())
    assert set(P.train_entities).isdisjoint(P.eval_entities)
    assert set(P.train_hop).isdisjoint(P.eval_hop)
    assert set(P.train_keys).isdisjoint(P.eval_keys)

    rng = random.Random(123)
    for L in (512, 2048, 4096):
        for fn in (gen_prose_lure, gen_scope_doc, gen_two_hop, gen_b2_rules, gen_rxk, gen_facts_recall):
            d_tr = fn(P, rng, min(L, 1024), False)
            d_ev = fn(P, rng, L, True)
            assert len(d_ev.ids) == L + 1 and len(d_ev.answers) >= 1
            for idx, gold, cands in d_ev.answers:
                assert d_ev.ids[idx] == gold and gold in cands and len(set(cands)) == 8
    from v14.rules_ood import B2_NEW_TASKS
    for t in B2_NEW_TASKS:
        for sp in (False, True):
            d_sp = gen_b2_rules(P, rng, 2048 if sp else 512, True, task=t, spread=sp)
            assert len(d_sp.ids) == (2049 if sp else 513) and len(d_sp.answers) == 1
    batch = make_train_batch(P, 42, 8, 512)
    x, y, m = collate_docs(batch, torch.device("cpu"))
    assert x.shape == (8, 512) and y.shape == (8, 512) and (m > 0).all()
    print("[PASS] IMB-17 6-axis generators & disjoint train/eval splits verified")


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running v17 sempyt selftest on {dev}...")
    test_stateful_scan()
    test_zero_spin_phasor()
    test_causality_and_streaming_decode(dev)
    test_gradients(dev)
    test_arm_param_parity()
    test_imb_generators()
    print("ALL v17 SEMPYT SELFTESTS PASSED.")


if __name__ == "__main__":
    main()
