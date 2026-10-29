"""v14 self-tests (CPU, minutes). Run: .venv/bin/python -m v14.selftest

1. off == v13      V14LM with every v14 switch off is bit-identical to v13 LM.
2. N1 start        sum / gate / phase give the same logits at init (shared
                   weights copied), so any later gap is learned, not built in.
3. N1 decode       full-sequence (chunked kernel) == token-by-token decode,
                   for each read rule, across the 64-token sub-chunk edge.
4. N1 learns       every new parameter gets a nonzero gradient at init.
5. N1 turns        with nonzero angles, phase and gate give different reads.
6. N6 (op memory)  starts as echo, decode == prefill, learns, applies an
                   inferred rotation; each for shared and own query projection.
7. tied echo       W_q = W_k gives q = k; decode == prefill; learns.
"""

from __future__ import annotations

import torch

from v13_sempty.model import LM
from v14.config import get_config
import sempyt
from sempyt import Dim, named

from v14.units import OPMEM_MODES, OperatorMemory, PHASE_READS, PhaseStatesPAM, TiedEchoQK, V14LM

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


def _n1(read, seed=0):
    torch.manual_seed(seed)
    return V14LM(_cfg(phase_states=2, phase_read=read)).eval()


def test_n1_reads_equal_at_start():
    models = {r: _n1(r) for r in PHASE_READS}
    shared = models['phase'].state_dict()
    for r in ('sum', 'gate'):
        missing = models[r].load_state_dict(shared, strict=False)
        assert all('angle_proj' in k for k in missing.unexpected_keys), missing
    ids = torch.randint(0, 512, (2, 40))
    ref = _logits(models['phase'], ids)
    for r in ('sum', 'gate'):
        diff = (_logits(models[r], ids) - ref).abs().max().item()
        assert diff < 1e-5, f"{r} != phase at start: {diff:.3e}"
        print(f"  start: {r} == phase  max |diff| = {diff:.1e}")


def test_n1_decode_matches_prefill(seq_len=70):
    for read in PHASE_READS:
        model = _n1(read, seed=1)
        if read != 'sum':     # give the angles something to do
            for m in model.modules():
                if isinstance(m, PhaseStatesPAM):
                    torch.nn.init.normal_(m.angle_proj.weight, std=0.5)
        ids = torch.randint(0, 512, (2, seq_len))
        full = _logits(model, ids)
        states, steps = None, []
        with torch.no_grad():
            for t in range(seq_len):
                lt, states, _ = model(ids[:, t:t + 1], states=states, step_offset=t)
                steps.append(lt)
        diff = (full - torch.cat(steps, dim=1)).abs().max().item()
        assert diff < ATOL, f"{read}: prefill != decode: {diff:.3e}"
        print(f"  decode == prefill ({read}): max |logit diff| = {diff:.2e}")


def test_n1_new_params_learn():
    for read in PHASE_READS:
        model = _n1(read, seed=2).train()
        ids = torch.randint(0, 512, (2, 40))
        logits = model(ids)[0]
        torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)).backward()
        for name, p in model.named_parameters():
            if any(s in name for s in ('notebook_dt_offset', 'notebook_mix', 'angle_proj')):
                g = 0.0 if p.grad is None else p.grad.abs().max().item()
                assert g > 0, f"{read}: {name} has no gradient"
        print(f"  {read}: every new parameter has a gradient")


def test_n1_phase_differs_from_gate():
    models = {r: _n1(r, seed=3) for r in ('gate', 'phase')}
    models['gate'].load_state_dict(models['phase'].state_dict())
    for model in models.values():
        torch.manual_seed(7)
        for m in model.modules():
            if isinstance(m, PhaseStatesPAM):
                torch.nn.init.normal_(m.angle_proj.weight, std=0.5)
    ids = torch.randint(0, 512, (2, 40))
    diff = (_logits(models['gate'], ids) - _logits(models['phase'], ids)).abs().max().item()
    assert diff > 1e-3, f"phase and gate agree with nonzero angles: {diff:.3e}"
    print(f"  nonzero angles: phase vs gate max |diff| = {diff:.2e}")


# (mode, query projection): rotation-only, plain diagonal, diagonal with its own W_q
N6_VARIANTS = (('phase', 'shared'), ('diag', 'shared'), ('diag', 'own'))


def _n6(mode, seed=0, query='shared', **kw):
    torch.manual_seed(seed)
    return V14LM(_cfg(opmem=mode, opmem_query=query, **kw)).eval()


def _decode_diff(model, seq_len):
    ids = torch.randint(0, 512, (2, seq_len))
    full = _logits(model, ids)
    states, steps = None, []
    with torch.no_grad():
        for t in range(seq_len):
            lt, states, _ = model(ids[:, t:t + 1], states=states, step_offset=t)
            steps.append(lt)
    return (full - torch.cat(steps, dim=1)).abs().max().item()


def test_n6_starts_as_echo():
    """Zero-init read-out: with the shared weights copied, N6 == the echo body."""
    torch.manual_seed(0)
    ref = LM(_cfg()).eval()
    ids = torch.randint(0, 512, (2, 40))
    for mode, query in N6_VARIANTS:
        model = _n6(mode, query=query)
        missing = model.load_state_dict(ref.state_dict(), strict=False)
        assert all(k.startswith('opmem.') for k in missing.missing_keys), missing
        diff = (_logits(model, ids) - _logits(ref, ids)).abs().max().item()
        assert diff == 0.0, f"{mode}/{query}: N6 at start != echo: {diff:.3e}"
        print(f"  start: N6 {mode}/{query} == echo body  max |diff| = {diff:.1e}")


def test_n6_decode_matches_prefill(seq_len=150):
    """Prefill (chunked scan, 64-step chunks) == one token at a time, both modes."""
    for mode, query in N6_VARIANTS:
        model = _n6(mode, seed=4, query=query)
        torch.nn.init.normal_(model.opmem.out.weight, std=0.5)       # make the read matter
        torch.nn.init.normal_(model.opmem.fade.weight, std=0.5)
        torch.nn.init.normal_(model.opmem.write.weight, std=0.5)
        diff = _decode_diff(model, seq_len)
        assert diff < ATOL, f"{mode}/{query}: prefill != decode: {diff:.3e}"
        print(f"  decode == prefill (N6 {mode}/{query}): max |logit diff| = {diff:.2e}")


def test_n6_learns():
    for mode, query in N6_VARIANTS:
        model = _n6(mode, seed=5, query=query).train()
        torch.nn.init.normal_(model.opmem.out.weight, std=0.02)       # past step 0
        ids = torch.randint(0, 512, (2, 40))
        logits = model(ids)[0]
        torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)).backward()
        for name, p in model.opmem.named_parameters():
            g = 0.0 if p.grad is None else p.grad.abs().max().item()
            assert g > 0, f"{mode}/{query}: opmem.{name} has no gradient"
        print(f"  N6 {mode}/{query}: every operator-memory parameter has a gradient")


def test_n6_applies_an_inferred_rotation(T=12, seed=6):
    """The mechanism, by hand. Every example output is its input turned by a
    hidden angle θ per channel; the read must turn a NEW input by θ.
    W_in = W_role = I, write gate ~1, fade ~1 (retention e^-softplus(-9))."""
    torch.manual_seed(seed)
    cfg = _cfg(n_heads=2, head_dim=8, dim=16, opmem='phase')
    model_dim = Dim("model_dim", 16)
    batch, time = Dim("batch", 1), Dim("time", T)
    results = {}
    for mode in OPMEM_MODES:
        cfg.opmem = mode
        op = OperatorMemory(cfg, model_dim).eval()
        with torch.no_grad():
            op.w_in.weight.copy_(torch.eye(16)); op.w_role.weight.copy_(torch.eye(16))
            op.write.weight.zero_(); op.write.bias.fill_(9.0)
            op.fade.weight.zero_(); op.fade.bias.fill_(-9.0)
        heads, plane, pair = op.heads, op.plane, op.pair
        theta = named(torch.rand(2, 4) * 6.28 - 3.14, (heads, plane))
        h = named(torch.randn(1, T, 2, 4, 2) * (0.5 + torch.rand(1, T, 2, 4, 1)),
                  (batch, time, heads, plane, pair))                  # uneven magnitudes
        # embedded_t = h_{t-1} turned by θ (the rule); embedded_0 = noise
        turned = sempyt.rotate(sempyt.shift(h, time, 1), theta, pair)
        hidden = h.to(batch, time, op.feat).alias(op.feat, model_dim)
        embedded = turned.to(batch, time, op.feat).alias(op.feat, model_dim)
        with torch.no_grad():
            p, _ = op.read(hidden, embedded)
        want = sempyt.rotate(h, theta, pair)                          # the new input, turned by θ
        got = p.with_policy(sempyt.Real()).to(batch, time, heads, plane, pair).data[0, -1]
        ref = want.data[0, -1]
        cos = torch.nn.functional.cosine_similarity(got.reshape(2, -1), ref.reshape(2, -1), dim=-1)
        results[mode] = cos.min().item()
    assert results['phase'] > 0.999, f"phase read did not apply the rotation: cos {results['phase']:.4f}"
    print(f"  rule applied to a new input: cos(read, input turned by θ) "
          f"phase {results['phase']:.4f}, diag {results['diag']:.4f}")


def test_echo_tied():
    """echo_tied_qk: q = k by construction; prefill == decode; the tied map learns."""
    torch.manual_seed(8)
    model = V14LM(_cfg(echo_tied_qk=True)).eval()
    assert isinstance(model.echo_qk, TiedEchoQK)
    h = torch.randn(2, 5, model.config.dim)
    q, k = model.echo_qk(h).chunk(2, dim=-1)
    assert torch.equal(q, k), "tied echo: q != k"
    # make the read matter. std 0.1, not 0.5: at 0.5 even untied v13 echo
    # misses ATOL (2.3e-4; tied 2.8e-4, the same ~4e-5 relative error of the
    # chunked WY form vs the fp32 step loop), so 0.5 tests v13's rounding, not the tie.
    torch.nn.init.normal_(model.echo_out.weight, std=0.1)
    diff = _decode_diff(model, 150)
    assert diff < ATOL, f"tied echo: prefill != decode: {diff:.3e}"
    model.train()
    ids = torch.randint(0, 512, (2, 40))
    logits = model(ids)[0]
    torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)).backward()
    g = model.echo_qk.proj.weight.grad
    assert g is not None and g.abs().max().item() > 0, "tied echo: no gradient"
    print(f"  tied echo: q == k, decode == prefill (max |diff| = {diff:.2e}), learns")


def main():
    torch.set_num_threads(8)
    for test in (test_off_is_v13, test_n1_reads_equal_at_start,
                 test_n1_decode_matches_prefill, test_n1_new_params_learn,
                 test_n1_phase_differs_from_gate, test_n6_starts_as_echo,
                 test_n6_decode_matches_prefill, test_n6_learns,
                 test_n6_applies_an_inferred_rotation, test_echo_tied):
        print(f"{test.__name__}:")
        test()
    print("ALL PASS")


if __name__ == '__main__':
    main()
