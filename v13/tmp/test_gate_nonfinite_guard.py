"""Verify the gate-surprisal safety net (v7/train.py _gate_surprisal_loss).

The 500M NaN crash was a device-side assert in F.binary_cross_entropy
(Loss.cu:91 target in [0,1]) triggered when a per-token NLL went non-finite:
target_p = sigmoid(median - nll) -> NaN -> assert. The erase-gain cap
(V13Config.delta_erase_beta_cap) is the real fix for the blowup; this test
proves the *safety net* (mask non-finite tokens + nan_to_num the target) keeps
the aux loss finite even when an NLL is +inf/-inf/NaN, so a single pathological
token can never take down a long run.

Run: .venv/bin/python -m v13.tmp.test_gate_nonfinite_guard
"""
import sys
sys.path.insert(0, '.')
import torch

from v7.train import V7Trainer
from v13.model import V13PAMLayer, V13Config


def make_gate_probs(cfg, B, T):
    """Real [L,B,T] protect-prob tensor from a production PAM layer (in [0,1])."""
    layer = V13PAMLayer(cfg).eval()
    from v13.complex_ops import to_real_concat
    x = torch.randn(B, T, cfg.dim, 2) * 0.5
    with torch.no_grad():
        gi = to_real_concat(x)                          # [B,T,2*dim] content-aware input
        prob = torch.sigmoid(layer.protect_gate(gi)).mean(dim=-1)  # [B,T]
    # [L,B,T]: stack L identical layers (shape is what the trainer consumes).
    L = 3
    return prob.unsqueeze(0).expand(L, B, T).contiguous().float()


def run_case(name, nll, expect_event):
    B, T = nll.shape
    cfg = V13Config(
        vocab_size=256, dim=32, n_heads=2, head_dim=16, n_layers=1,
        expand=2, dropout=0.0, max_seq_len=256, chunk_size=24,
        gradient_checkpointing=False, use_rope=True, use_gsp=True,
        n_states=1, gate_content_aware=True, write_mode='delta',
        delta_chunk=16, delta_erase_gate=True,
    )
    m_cfg = cfg
    gp = make_gate_probs(cfg, B, T)
    labels = torch.zeros(B, T, dtype=torch.long)      # all valid (no -100)
    loss_mask = torch.ones(B, T)

    # Minimal trainer shim: _gate_surprisal_loss only touches these attrs.
    tr = type('T', (), {})()
    tr.global_step = 7
    tr._gate_nonfinite_events = 0
    tr.fused_ce_chunk = 4096

    loss = V7Trainer._gate_surprisal_loss(tr, gp, nll, None, labels, loss_mask, m_cfg)
    finite = bool(torch.isfinite(loss).all())
    events = getattr(tr, '_gate_nonfinite_events', 0)
    ok = finite and (events > 0) == expect_event
    print(f"[{name:16s}] loss={float(loss):.4f} finite={finite} "
          f"events={events} (expect_event={expect_event})  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    B, T = 4, 64
    base = torch.rand(B, T) * 3.0 + 1.0  # finite NLLs in [1,4]
    torch.manual_seed(0)
    results = []
    results.append(run_case("all_finite", base, expect_event=False))
    inf = base.clone(); inf[0, 3] = float('inf')
    results.append(run_case("one_plus_inf", inf, expect_event=True))
    ninf = base.clone(); ninf[1, 5] = float('-inf')
    results.append(run_case("one_minus_inf", ninf, expect_event=True))
    nan = base.clone(); nan[2, 7] = float('nan')
    results.append(run_case("one_nan", nan, expect_event=True))
    many = base.clone(); many[::2] = float('inf')  # many non-finite at once
    results.append(run_case("many_inf", many, expect_event=True))
    # All non-finite -> neutral target, still finite loss, no crash.
    allbad = torch.full((B, T), float('nan'))
    results.append(run_case("all_nan", allbad, expect_event=True))
    print()
    if all(results):
        print("ALL SAFETY-NET CASES PASS: non-finite NLL never asserts; aux stays finite.")
    else:
        print("SOME SAFETY-NET CASES FAILED.")
        raise SystemExit(1)


if __name__ == '__main__':
    main()
