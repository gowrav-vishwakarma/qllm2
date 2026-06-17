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


def main():
    torch.set_default_dtype(torch.float64)  # high precision for the math check
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
    print()
    if all(results):
        print("ALL MODES PASS: parallel train form == O(1) recurrent form.")
    else:
        print("SOME MODES FAILED — see above.")
        raise SystemExit(1)


if __name__ == '__main__':
    main()
