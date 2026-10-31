"""v14 self-tests (CPU, minutes). Run: .venv/bin/python -m v14.selftest

1. off == v13      V14LM with every v14 switch off is bit-identical to v13 LM.
2. N7 lookup     parallel n-gram lookup == a brute-force bounded table (with
                   forced slot collisions) == token-by-token decode; the decode
                   state after a prefix matches; "X is Y" templates do not collide.
3. N7 start       N7 on (zero-init readers) is bit-identical to N7 off.
4. N7 decode      full-sequence logits == token-by-token decode with live readers.
5. N7 learns      every reader parameter gets a gradient.
6. N9a           content-phase echo: off at init == echo; decode == prefill
                   with live phases; "X is" template keys separate by X
                   (collision check) and the same X stays aligned; learns.
6b. N9s          salient phase on top of N9a: starts == N9a; the parallel
                   running angle == a step-by-step loop; decode == prefill with
                   live gates; every new parameter learns.
7. N6 / N10      operator memory ('phase' = N6, 'coord' = N10 circles):
                   starts bit-identical to echo at the same seed (body and RNG
                   untouched); decode == prefill with a live read; every
                   parameter learns.
8. v13 hook      v14/hook.py (how v13_sempty/train.py and the probe loaders
                   build v14 units): every v14 switch off -> the plain v13 LM
                   class; a checkpoint config rebuilds the same V14LM (strict
                   load, equal logits); gradient checkpointing on == off for
                   the v13 LM, N9a, N7 and N12 (loss and grads).
8b. N12          surprise-selected writes: same init as echo (no new params);
                   surprise == -log_softmax of the body's tied logits; no
                   writes -> the memory adds nothing; decode == prefill with a
                   mix of writes; the memory learns, echo's gates go unused.
9. B2 splits       held-out rule parameters appear only in the ``_new`` tasks,
                   trained ones only in the base tasks; answers stay out of
                   the context; the nearest-parameter decoy is a candidate.
(Tests of removed units: N6 operator memory at 78e4354; tied echo at 5100fd7;
N1 at 9e0a3fd.)
"""

from __future__ import annotations

import torch

from v13_sempty.model import LM
from v14.config import get_config
from v14.units import V14LM

ATOL = 2e-4


def _cfg(**kw):
    base = dict(chunk_size=32, delta=True, chrono=True, out_gate=True, echo=True,
                vocab_size=512, max_seq_len=256)
    base.update(kw)
    return get_config('tiny_real', **base)


def _logits(model, ids):
    with torch.no_grad():
        return model(ids)[0]


def test_off_is_v13():
    cfg = _cfg()
    torch.manual_seed(0)
    ref = LM(cfg).eval()
    torch.manual_seed(0)
    v14 = V14LM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 40))
    diff = (_logits(ref, ids) - _logits(v14, ids)).abs().max().item()
    assert diff == 0.0, f"v14 (switches off) != v13: {diff:.3e}"
    print(f"  off == v13: max |diff| = {diff:.1e}")


def _decode_diff(model, seq_len, vocab=512):
    ids = torch.randint(0, vocab, (2, seq_len))
    full = _logits(model, ids)
    states, steps = None, []
    with torch.no_grad():
        for t in range(seq_len):
            lt, states, _ = model(ids[:, t:t + 1], states=states, step_offset=t)
            steps.append(lt)
    return (full - torch.cat(steps, dim=1)).abs().max().item()


def _ref_ngram(row, n, S):
    """Brute-force bounded table: dict slot -> (fp, value), last write wins."""
    from v14.ngram import _FP_MULT, _SLOT_MULT, _hash
    def key(g):
        g = torch.tensor(g, dtype=torch.int64)
        return int(torch.remainder(_hash(g, _SLOT_MULT), S)), int(_hash(g, _FP_MULT))
    table, vals, founds = {}, [], []
    for t in range(len(row)):
        if t >= n:
            s, f = key(row[t - n:t])
            table[s] = (f, row[t])
        hit = False
        if t >= n - 1:
            s, f = key(row[t - n + 1:t + 1])
            e = table.get(s)
            hit = e is not None and e[0] == f
        vals.append(table[s][1] if hit else 0)
        founds.append(hit)
    return vals, founds


def test_n7_lookup():
    """Parallel == brute force == step decode, with and without slot collisions."""
    from v14.ngram import empty_state, lookup_parallel, lookup_step, state_from_prefix
    g = torch.Generator().manual_seed(3)
    for n, S, vocab in ((2, 7, 6), (2, 1 << 12, 6), (3, 5, 4), (3, 1 << 12, 20), (4, 64, 3)):
        ids = torch.randint(0, vocab, (3, 90), generator=g)
        v, f, _ = lookup_parallel(ids, n, S)
        st = empty_state(3, n, S, ids.device)
        for b in range(3):
            rv, rf = _ref_ngram(ids[b].tolist(), n, S)
            assert v[b].tolist() == rv and f[b].tolist() == rf, f"N7 parallel != reference (n={n}, S={S})"
        for t in range(90):
            vt, ft, st = lookup_step(st, ids[:, t], n, S)
            assert torch.equal(vt, v[:, t]) and torch.equal(ft, f[:, t]), f"N7 step != parallel at t={t}"
        pre = state_from_prefix(ids, n, S)
        for k in ('fp', 'val', 'used', 'tail'):
            same = torch.equal(pre[k], st[k]) if k != 'fp' and k != 'val' else \
                torch.equal(pre[k][pre['used']], st[k][st['used']])
            assert same, f"N7 prefix state != step state ({k}, n={n}, S={S})"
    # collision check (v13 diagnosis): "X is Y" items share the template "is";
    # every repeated item must return its own label, not the latest one
    xs, ys = list(range(100, 140)), list(range(200, 240))
    is_ = 7
    row = sum(([x, is_, y] for x, y in zip(xs, ys)), []) + sum(([x, is_] for x in xs[::-1]), [])
    v, f, _ = lookup_parallel(torch.tensor([row]), 2, 1 << 16)
    tail = len(xs) * 3
    got = [int(v[0, tail + 2 * i + 1]) for i in range(len(xs))]
    assert got == ys[::-1] and bool(f[0, tail + 1::2].all()), "N7: template keys collided"
    print(f"  N7: parallel == brute force == step decode (5 settings, forced collisions); "
          f"prefix state matches; 40 'X is Y' items, 0 template collisions")


def _n7(seed=0, **kw):
    torch.manual_seed(seed)
    return V14LM(_cfg(ngram_orders=(2, 3), ngram_layers=(0, -1), **kw)).eval()


def test_n7_off_at_init():
    torch.manual_seed(0)
    off = V14LM(_cfg()).eval()
    on = _n7(0)
    ids = torch.randint(0, 512, (2, 60))
    ids[:, 30:] = ids[:, :30]                                    # repeats, so lookups fire
    diff = (_logits(off, ids) - _logits(on, ids)).abs().max().item()
    assert diff == 0.0, f"N7 at init != N7 off: {diff:.3e}"
    print(f"  N7 on at init == off: max |diff| = {diff:.1e}")


def test_n7_decode_matches_prefill():
    model = _n7(1)
    with torch.no_grad():
        for r in model.ngram_read.values():
            r.proj.normal_(std=0.05)
            r.found_bias.normal_(std=0.05)
            r.gate_w.normal_(std=0.5)
    torch.manual_seed(2)
    diff = _decode_diff(model, 150, vocab=12)                    # small vocab: many repeats
    assert diff < ATOL, f"N7 prefill != decode: {diff:.3e}"
    print(f"  N7 decode == prefill (150 tokens, live readers): max |diff| = {diff:.2e}")


def test_n7_learns():
    model = _n7(4).train()
    with torch.no_grad():
        for r in model.ngram_read.values():
            r.proj.normal_(std=0.05)
    ids = torch.randint(0, 20, (2, 64))
    logits = model(ids)[0]
    torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)).backward()
    for name, p in model.ngram_read.named_parameters():
        assert p.grad is not None and p.grad.abs().max().item() > 0, f"N7: no gradient on {name}"
    print("  N7: every reader parameter gets a gradient")


def _n9a(seed=0):
    torch.manual_seed(seed)
    return V14LM(_cfg(echo_phase_lags=3)).eval()


def test_n9a():
    """Content-phase echo keys: identity at init, decode == prefill, collisions separate."""
    torch.manual_seed(0)
    echo = V14LM(_cfg()).eval()
    model = _n9a(0)
    ids = torch.randint(0, 512, (2, 60))
    diff0 = (_logits(echo, ids) - _logits(model, ids)).abs().max().item()
    assert diff0 == 0.0, f"N9a at init != echo: {diff0:.3e}"
    with torch.no_grad():
        model.phase_proj.normal_(std=0.3)
        torch.nn.init.normal_(model.echo_out.weight, std=0.1)
    torch.manual_seed(5)
    diff = _decode_diff(model, 150)
    assert diff < ATOL, f"N9a prefill != decode: {diff:.3e}"
    # collision check: the same template key turned by the windows (X_i, is)
    # of different nouns X_i must spread apart; the same X must stay aligned
    is_ = 7
    xs = list(range(100, 140))
    win = torch.tensor([[x, is_] for x in xs])                       # [40, 2]: windows ending at "is"
    with torch.no_grad():
        alpha = model._content_phase(win)[:, -1]                     # [40, planes]
        key = torch.randn(1, 1, alpha.shape[-1] * 2).expand(len(xs), 1, -1)
        turned = model._turn(key, alpha[:, None])[:, 0]
        turned = torch.nn.functional.normalize(turned.reshape(len(xs), -1), dim=-1)
        cos = turned @ turned.t()
        off = cos[~torch.eye(len(xs), dtype=torch.bool)].abs().mean().item()
        again = model._turn(key[:1], model._content_phase(win[:1])[:, -1][:, None])[:, 0]
        same = torch.nn.functional.cosine_similarity(again.reshape(1, -1), turned[:1]).item()
    assert off < 0.3 and same > 0.999, f"N9a collision check: different-X |cos| {off:.2f}, same-X {same:.3f}"
    model.train()
    out = model(torch.randint(0, 512, (2, 40)))[0]
    out.float().pow(2).mean().backward()
    g = model.phase_proj.grad
    assert g is not None and g.abs().max().item() > 0, "N9a: no gradient on the phases"
    print(f"  N9a: init == echo; decode == prefill {diff:.2e}; template keys of 40 nouns |cos| {off:.2f} "
          f"(was 1.0 unturned), same noun {same:.3f}; phases learn")


def test_n9s():
    torch.manual_seed(0)
    n9a = V14LM(_cfg(echo_phase_lags=3)).eval()
    torch.manual_seed(0)
    model = V14LM(_cfg(echo_phase_lags=3, echo_phase_salient=True)).eval()
    ids = torch.randint(0, 512, (2, 60))
    d0 = (_logits(n9a, ids) - _logits(model, ids)).abs().max().item()
    assert d0 == 0.0, f"N9s at init != N9a: {d0:.3e}"
    with torch.no_grad():
        model.phase_proj.normal_(std=0.3)
        model.sal_proj.normal_(std=0.3)
        model.sal_gate_w.normal_(std=0.5)
        torch.nn.init.normal_(model.echo_out.weight, std=0.1)
    # the parallel scan against a plain loop
    h = torch.randn(2, 37, model.config.dim)
    x = torch.randint(0, 512, (2, 37))
    with torch.no_grad():
        psi, last = model._salient_phase(h, x)
        e = model.embed.embed.weight[x].float()
        e = e * torch.rsqrt(e.pow(2).mean(-1, keepdim=True) + 1e-6)
        hn = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        sg = torch.sigmoid(hn @ model.sal_gate_w.t() + model.sal_gate_b)
        per = model.sal_proj.shape[0] // sg.shape[-1]
        sp = sg.repeat_interleave(per, dim=-1)
        a = e @ model.sal_proj.t()
        ref, cur = [], torch.zeros_like(a[:, 0])
        for t in range(37):
            cur = (1 - sp[:, t]) * cur + sp[:, t] * a[:, t]
            ref.append(cur)
        ref = torch.stack(ref, 1)
    dscan = (psi - ref).abs().max().item()
    assert dscan < 1e-4, f"N9s scan != loop: {dscan:.3e}"
    torch.manual_seed(11)
    diff = _decode_diff(model, 150)
    assert diff < ATOL, f"N9s prefill != decode: {diff:.3e}"
    model.train()
    out = model(torch.randint(0, 512, (2, 40)))[0]
    out.float().pow(2).mean().backward()
    for name in ('sal_proj', 'sal_gate_w', 'sal_gate_b', 'phase_proj'):
        g = getattr(model, name).grad
        assert g is not None and g.abs().max().item() > 0, f"N9s: no gradient on {name}"
    print(f"  N9s: init == N9a; scan == loop {dscan:.1e}; decode == prefill {diff:.2e}; gates learn")


def test_n12():
    torch.manual_seed(0)
    echo = V14LM(_cfg()).eval()
    torch.manual_seed(0)
    model = V14LM(_cfg(echo_surprise_tau=3.0)).eval()
    a, b = echo.state_dict(), model.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), "N12 init != echo init"
    # surprise against a direct log-softmax over the tied head
    lmb = torch.randn(2, 33, model.config.dim)
    ids = torch.randint(0, 512, (2, 33))
    E = model.embed.embed.weight.float()
    ref = -torch.log_softmax(lmb[:, :-1] @ E.t(), -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    u = model._surprise(lmb, ids)
    du = (u[:, 1:] - ref).abs().max().item()
    assert torch.isinf(u[:, 0]).all() and du < 1e-4, f"N12 surprise off: {du:.3e}"
    u2 = model._surprise(lmb[:, 1:], ids[:, 1:], lm_prev=lmb[:, 0])
    assert (u2 - ref).abs().max().item() < 1e-4, "N12 surprise with a carried vector off"
    with torch.no_grad():
        torch.nn.init.normal_(model.echo_out.weight, std=0.1)
    # no writes: the store is empty and the read adds nothing
    x = torch.randint(0, 512, (2, 60))
    model.surp_tau = 1e9
    body = _logits(model, x)
    with torch.no_grad():
        model.echo_out.weight.mul_(3.0)
    d_empty = (_logits(model, x) - body).abs().max().item()
    assert d_empty == 0.0, f"N12 with no writes still reads: {d_empty:.3e}"
    # a threshold in the widest gap near the median: about half the tokens write
    torch.manual_seed(11)
    probe = torch.randint(0, 512, (2, 150))
    with torch.no_grad():
        z, batch, time = model._stem(probe, 0)
        z, _ = model._run_blocks(z, batch, time, None, 0, probe)
        lm = model.lm_head_norm(model.lm_head_proj(model.output_norm(z)))
        uu = model._surprise(lm.raw(batch, time, model.model_dim), probe)
    vals = uu[torch.isfinite(uu)].sort().values
    mid = len(vals) // 2
    lo, hi = mid - len(vals) // 8, mid + len(vals) // 8
    gaps = vals[lo + 1:hi] - vals[lo:hi - 1]
    j = lo + int(gaps.argmax())
    model.surp_tau = float((vals[j] + vals[j + 1]) / 2)
    torch.manual_seed(11)
    diff = _decode_diff(model, 150)
    rate = float(model.surp_write_rate)
    _logits(model, probe)
    rate = float(model.surp_write_rate)
    assert 0.2 < rate < 0.8, f"N12 test threshold should write about half: {rate:.2f}"
    assert diff < ATOL, f"N12 prefill != decode: {diff:.3e}"
    model.train()
    out = model(torch.randint(0, 512, (2, 40)))[0]
    out.float().pow(2).mean().backward()
    for name in ('echo_qk', 'echo_out'):
        g = getattr(model, name).weight.grad
        assert g is not None and g.abs().max().item() > 0, f"N12: no gradient on {name}"
    assert model.echo_gates.weight.grad is None, "N12 should not use echo's learned gates"
    print(f"  N12: init == echo; surprise == log-softmax {du:.1e}; empty store adds 0; "
          f"decode == prefill {diff:.2e} at write rate {rate:.2f}; memory learns, gates unused")


def test_l1():
    import os
    import tempfile
    from v14 import l1_key as L
    torch.manual_seed(0)
    T, D = 300, L.DIM
    nll = torch.rand(T) * 6
    for p in (50, 51, 52, 120, 200):                 # three bursts: 50-52 (gaps merge), 120, 200
        nll[p] = 12.0
    assert L.bursts(nll) == [(50, 53), (120, 121), (200, 201)], L.bursts(nll)
    fwd = L.row_spans(T, nll, 'fwd')
    evt = L.row_spans(T, nll, 'evt')
    assert fwd == [(50, 74), (120, 144), (200, 224)], fwd
    assert evt == [(0, 50), (50, 120), (120, 200)], evt
    head = L.SpanKey()
    vecs = torch.randn(T, D)
    # the harness hook (every position) == the training path (spans) for the fwd form
    with tempfile.TemporaryDirectory() as tmp:
        ck = os.path.join(tmp, 'h.pt')
        torch.save({'state': head.state_dict(), 'form': 'fwd'}, ck)
        os.environ['L1_CKPT'] = ck
        L._HEAD.clear()
        K = L.key_fn(vecs, None)
        assert K.shape == (T, L.KEY_DIM)
        ref = L.pool_spans(head, vecs, [(p, min(T, p + L.SPAN)) for p in (0, 77, T - 5, T - 1)], L.SPAN)
        d = (K[[0, 77, T - 5, T - 1]] - ref).abs().max().item()
        assert d < 1e-5, f"L1 key_fn != pool_spans: {d:.2e}"
        kb = L.key_fn(vecs, None, [(50, 53), (120, 121)])
        assert kb.shape == (2, L.KEY_DIM)
    # training signal: the loss falls when the query span carries the target's content
    doc = dict(vecs=torch.randn(T, D).half(), nll=nll, target=122, tlen=1, qpos=250)
    doc['vecs'][250:274] = (0.3 * doc['vecs'][120:144].float() + torch.randn(24, D)).half()   # a noisy repeat
    l0, nq, _ = L.batch_loss(head, [doc], 'fwd', torch.device('cpu'))
    l0 = l0.detach()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
    for _ in range(30):
        loss, _, _ = L.batch_loss(head, [doc], 'fwd', torch.device('cpu'))
        opt.zero_grad(); loss.backward(); opt.step()
    assert nq == 1 and float(l0) > 0.1 and float(loss) < 0.5 * float(l0), f"L1 loss did not fall: {float(l0):.3f} -> {float(loss):.3f}"
    print(f"  L1: bursts / fwd / evt rows as specified; key_fn == pool_spans {d:.1e}; loss {float(l0):.3f} -> {float(loss.detach()):.3f}")


def test_opmem():
    for mode in ('phase', 'coord'):
        torch.manual_seed(0)
        echo = V14LM(_cfg()).eval()
        torch.manual_seed(0)
        model = V14LM(_cfg(opmem=mode)).eval()
        ids = torch.randint(0, 512, (2, 50))
        d0 = (_logits(echo, ids) - _logits(model, ids)).abs().max().item()
        assert d0 == 0.0, f"{mode}: opmem at init != echo: {d0:.3e}"
        with torch.no_grad():
            torch.nn.init.normal_(model.opmem.out.weight, std=0.5)    # make the read matter
            torch.nn.init.normal_(model.opmem.fade.weight, std=0.5)
            torch.nn.init.normal_(model.opmem.write.weight, std=0.5)
            torch.nn.init.normal_(model.opmem.w_in.weight, std=0.5)
            torch.nn.init.normal_(model.opmem.w_role.weight, std=0.5)
        torch.manual_seed(7)
        diff = _decode_diff(model, 150)
        assert diff < ATOL, f"{mode}: opmem prefill != decode: {diff:.3e}"
        model.train()
        out = model(torch.randint(0, 512, (2, 40)))[0]
        out.float().pow(2).mean().backward()
        for name, prm in model.opmem.named_parameters():
            assert prm.grad is not None and prm.grad.abs().max().item() > 0, f"{mode}: no grad on {name}"
        print(f"  {mode}: init == echo; decode == prefill {diff:.2e}; every parameter learns")


def test_hook():
    from dataclasses import asdict
    from v13_sempty.config import get_config as v13_preset
    from v14.hook import build_lm, config_from_raw, with_v14_switches
    base = dict(chunk_size=32, delta=True, chrono=True, out_gate=True, echo=True,
                vocab_size=512, max_seq_len=256)

    def v13_get_config(preset, **kw):                 # v13's get_config takes no overrides
        c = v13_preset(preset)
        for k, v in kw.items():
            setattr(c, k, v)
        return c
    plain = v13_get_config('tiny_real', **base)
    assert with_v14_switches(plain, echo_phase_lags=0) is plain
    assert type(build_lm(plain)) is LM, "hook: switches off must build the v13 LM class"
    cfg = with_v14_switches(v13_get_config('tiny_real', **base), echo_phase_lags=3)
    torch.manual_seed(0)
    model = build_lm(cfg).eval()
    assert type(model) is V14LM
    with torch.no_grad():
        model.phase_proj.normal_(std=0.3)
    re = build_lm(config_from_raw(asdict(cfg))).eval()               # what a loader does
    re.load_state_dict(model.state_dict(), strict=True)
    ids = torch.randint(0, 512, (2, 40))
    d = (_logits(model, ids) - _logits(re, ids)).abs().max().item()
    assert d == 0.0, f"hook: reloaded V14LM differs: {d:.3e}"
    # gradient checkpointing parity (the 4090 suspect): loss and grads, on vs off
    worst = 0.0
    for lags, orders, tau in ((0, (), 0.0), (3, (), 0.0), (0, (2, 3), 0.0), (0, (), 3.0)):
        runs = []
        for ckpt in (False, True):
            c = with_v14_switches(v13_get_config('tiny_real', gradient_checkpointing=ckpt, **base),
                                  echo_phase_lags=lags, ngram_orders=orders, echo_surprise_tau=tau)
            torch.manual_seed(1)
            m = build_lm(c).train()
            with torch.no_grad():
                if lags:
                    m.phase_proj.normal_(std=0.3)
                if tau:
                    torch.nn.init.normal_(m.echo_out.weight, std=0.1)
                for r in getattr(m, 'ngram_read', {}).values():
                    r.proj.normal_(std=0.05)
            ids_rep = torch.cat([ids[:, :20], ids[:, :20]], dim=1) if orders else ids  # repeats: lookups fire
            logits = m(ids_rep)[0]
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]), ids_rep[:, 1:].reshape(-1))
            loss.backward()
            runs.append((loss.item(), {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}))
        (l0, g0), (l1, g1) = runs
        assert abs(l0 - l1) < 1e-6 and g0.keys() == g1.keys(), f"hook: ckpt changes the loss (lags={lags}, tau={tau})"
        worst = max(worst, max((g0[k] - g1[k]).abs().max().item() for k in g0))
    assert worst < 1e-5, f"hook: grad checkpointing changes grads: {worst:.3e}"
    print(f"  hook: off -> v13 LM class; reload strict, logits equal; grad ckpt on == off "
          f"(v13 LM, N9a, N7, N12): max grad diff {worst:.1e}")


def _b2_param(task, text):
    """The hidden parameter of a B2 prompt (k, d, or the affine b)."""
    lines = text.split('\n')[1:]
    if task.startswith('shift'):
        x, y = lines[0].split(' -> ')
        return ord(y) - ord(x)
    if task.startswith('step'):
        t = [int(v) for v in lines[0].rstrip(',').split(', ')]
        return t[1] - t[0]
    (x1, y1), (x2, y2) = [tuple(int(v) for v in ln.split(' -> ')) for ln in lines[:2]]
    a = (y2 - y1) // (x2 - x1)
    return y1 - a * x1


def test_b2_splits(seeds=40):
    """B2: no held-out parameter leaks into training tasks and vice versa."""
    from transformers import AutoTokenizer
    from v14 import rules_ood as O
    tok = AutoTokenizer.from_pretrained('gpt2')
    split = {'shift_wide': (O.SHIFT_TRAIN, O.SHIFT_NEW), 'step_wide': (O.STEP_TRAIN, O.STEP_NEW),
             'affine_map': (O.AFFINE_B_TRAIN, O.AFFINE_B_NEW)}
    n = 0
    for base, (train, held) in split.items():
        assert not set(train) & set(held), f"{base}: train and held-out overlap"
        for task, allowed in ((base, train), (base + '_new', held)):
            for shots in (1, 2, 4, 8):
                for seed in range(seeds):
                    ex = O.build_example(tok, task, shots, seed)
                    assert _b2_param(task, ex.prompt_text) in allowed, f"{task}: parameter leak"
                    c = ex.candidate_token_ids
                    assert len(set(c)) == 8 and ex.target_token_id in c, f"{task}: candidates"
                    shown = [ln.split(' -> ')[1] for ln in ex.prompt_text.split('\n')[1:]
                             if ' -> ' in ln and ln.split(' -> ')[1]]
                    assert ex.target_text.strip() not in shown, f"{task}: answer in context"
                    n += 1
    print(f"  B2: {n} episodes, splits disjoint, no parameter leak, answers out of context")


def main():
    torch.set_num_threads(8)
    for test in (test_off_is_v13, test_n7_lookup, test_n7_off_at_init,
                 test_n7_decode_matches_prefill, test_n7_learns, test_n9a, test_n9s, test_n12, test_l1, test_opmem, test_hook, test_b2_splits):
        print(f"{test.__name__}:")
        test()
    print("ALL PASS")


if __name__ == '__main__':
    main()
