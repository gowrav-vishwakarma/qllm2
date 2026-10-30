"""v14 self-tests (CPU, minutes). Run: .venv/bin/python -m v14.selftest

1. off == v13      V14LM with every v14 switch off is bit-identical to v13 LM.
2. N1 start        sum / gate / phase give the same logits at init (shared
                   weights copied), so any later gap is learned, not built in.
3. N1 decode       full-sequence (chunked kernel) == token-by-token decode,
                   for each read rule, across the 64-token sub-chunk edge.
4. N1 learns       every new parameter gets a nonzero gradient at init.
5. N1 turns        with nonzero angles, phase and gate give different reads.
6. tied echo       W_q = W_k gives q = k; decode == prefill; learns.
7. B2 splits       held-out rule parameters appear only in the ``_new`` tasks,
                   trained ones only in the base tasks; answers stay out of
                   the context; the nearest-parameter decoy is a candidate.
(The N6 operator-memory tests left with N6, after B1 run 2; they are at 78e4354.)
"""

from __future__ import annotations

import torch

from v13_sempty.model import LM
from v14.config import get_config
from v14.units import PHASE_READS, PhaseStatesPAM, TiedEchoQK, V14LM

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


def _decode_diff(model, seq_len):
    ids = torch.randint(0, 512, (2, seq_len))
    full = _logits(model, ids)
    states, steps = None, []
    with torch.no_grad():
        for t in range(seq_len):
            lt, states, _ = model(ids[:, t:t + 1], states=states, step_offset=t)
            steps.append(lt)
    return (full - torch.cat(steps, dim=1)).abs().max().item()


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
    for test in (test_off_is_v13, test_n1_reads_equal_at_start,
                 test_n1_decode_matches_prefill, test_n1_new_params_learn,
                 test_n1_phase_differs_from_gate, test_echo_tied, test_b2_splits):
        print(f"{test.__name__}:")
        test()
    print("ALL PASS")


if __name__ == '__main__':
    main()
