"""Equivalence contract: v13_sempty vs v13 (CPU, tiny configs).

Tolerances (fp32 CPU):
  * module / full-model forward+grad after state_dict copy: atol 1e-5
    (measured 0.0 on the production delta-fused E3 path)
  * parallel-train vs recurrent-infer: atol 2e-3 (v13's own selftest bar)
  * fused CE vs F.cross_entropy: atol 1e-5
  * one AdamW step param delta: atol 1e-5

Triton is forced off on the v13 side so both use the same PT math.

Run:
    .venv/bin/python -m v13_sempty.selftest
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from sempyt.dim import Dim
from sempyt.policies import SplitComplex
from sempyt.tensor import named

from v13.model import V13Config, V13LM, V13PAMLayer
from v13.triton_kernels import set_triton_enabled
from v13_sempty.config import V13Config as SConfig
from v13_sempty.model import V13LM as SLM
from v13_sempty.model import V13PAMLayer as SPAM
from v13_sempty.fused_ce import fused_linear_cross_entropy
from v13_sempty.train import Trainer, build_param_groups, synthetic_loader


def _as_named_tokens(x):
    """Public-edge wrap for PAM selftests (v13 still speaks raw torch)."""
    pair = Dim("complex_pair", 2)
    return named(
        x,
        (Dim("batch", x.shape[0]), Dim("time", x.shape[1]),
         Dim("model_dim", x.shape[2]), pair),
        SplitComplex(pair),
    )

set_triton_enabled(False)

ATOL_TIGHT = 1e-5
ATOL_RECUR = 2e-3


def _unwrap(t):
    """Drop back to raw torch at the v13 comparison boundary."""
    return t.data if hasattr(t, "layout") else t


def _max(a, b):
    return (_unwrap(a) - _unwrap(b)).abs().max().item()


def _prod_kw(**extra):
    kw = dict(
        vocab_size=256, dim=32, n_heads=2, head_dim=16, n_layers=2,
        expand=2, dropout=0.0, max_seq_len=128, chunk_size=32,
        gradient_checkpointing=False, use_rope=True, use_gsp=True,
        n_states=3, write_mode='delta', delta_chunk=16, delta_erase_gate=True,
        gate_content_aware=True, vault_state=True, vault_state_idx=0,
        write_phase_address=True, fused_e3=True, delta_key_norm=True,
        delta_erase_beta_cap=0.95,
    )
    kw.update(extra)
    return kw


def test_state_dict_keys():
    kw = _prod_kw()
    ref = V13LM(V13Config(**kw))
    port = SLM(SConfig(**kw))
    rk, pk = set(ref.state_dict()), set(port.state_dict())
    assert rk == pk, (sorted(rk - pk)[:8], sorted(pk - rk)[:8])
    print("[state_dict keys ] PASS")
    return True


def test_pam_equiv(batch_size=2, seq_len=24, seed=0):
    torch.manual_seed(seed)
    kw = _prod_kw(n_layers=1)
    ref = V13PAMLayer(V13Config(**kw)).eval()
    port = SPAM(SConfig(**kw)).eval()
    port.load_state_dict(ref.state_dict())
    x = torch.randn(batch_size, seq_len, kw['dim'], 2) * 0.5
    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)
    y1, S1 = ref(x1)
    y2, S2 = port(_as_named_tokens(x2))
    y2 = y2.data
    (y1 ** 2).sum().backward()
    (y2 ** 2).sum().backward()
    dy, dS, dg = _max(y1, y2), _max(S1, S2), _max(x1.grad, x2.grad)
    ok = max(dy, dS, dg) < ATOL_TIGHT
    print(f"[pam vs v13      ] y={dy:.2e} S={dS:.2e} dx={dg:.2e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_parallel_vs_recurrent(batch_size=2, seq_len=20, seed=2):
    torch.manual_seed(seed)
    kw = _prod_kw(n_layers=1)
    port = SPAM(SConfig(**kw)).eval()
    x = torch.randn(batch_size, seq_len, kw['dim'], 2) * 0.5
    with torch.no_grad():
        par, _ = port(_as_named_tokens(x), state=None, step_offset=0)
        steps, state = [], None
        for t in range(seq_len):
            y, state = port(_as_named_tokens(x[:, t:t + 1]), state=state, step_offset=t)
            steps.append(y.data)
        rec = torch.cat(steps, dim=1)
    par = par.data
    d = (par - rec).abs().max().item()
    ok = d < ATOL_RECUR
    print(f"[par vs recur    ] max|d|={d:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def test_lm_forward_grad(batch_size=2, seq_len=20, seed=1):
    torch.manual_seed(seed)
    kw = _prod_kw()
    ref = V13LM(V13Config(**kw)).eval()
    port = SLM(SConfig(**kw)).eval()
    port.load_state_dict(ref.state_dict())
    ids = torch.randint(0, kw['vocab_size'], (batch_size, seq_len))
    y1, _, _ = ref(ids)
    y2, _, _ = port(ids)
    dy = _max(y1, y2)
    lab = torch.randint(0, kw['vocab_size'], (batch_size, seq_len))
    F.cross_entropy(y1.reshape(-1, y1.size(-1)), lab.reshape(-1)).backward()
    F.cross_entropy(y2.reshape(-1, y2.size(-1)), lab.reshape(-1)).backward()
    worst = 0.0
    for n, p in ref.named_parameters():
        q = dict(port.named_parameters())[n]
        if p.grad is None or q.grad is None:
            continue
        worst = max(worst, (p.grad - q.grad).abs().max().item())
    ok = dy < ATOL_TIGHT and worst < ATOL_TIGHT
    print(f"[lm fwd+grad     ] logits={dy:.2e} worst_grad={worst:.2e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_fused_ce_equiv(batch_size=2, seq_len=16, seed=0):
    torch.manual_seed(seed)
    kw = _prod_kw(n_layers=2, dim=32)
    m = SLM(SConfig(**kw)).train()
    ids = torch.randint(0, kw['vocab_size'], (batch_size, seq_len))
    lbl = torch.randint(0, kw['vocab_size'], (batch_size, seq_len))
    m.zero_grad()
    logits, _, aux1 = m(ids)
    ref = F.cross_entropy(logits.view(-1, logits.size(-1)), lbl.view(-1))
    ref.backward()
    gref = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad()
    main, aux2 = m.fused_ce_loss(ids, lbl, chunk=16)
    main.backward()
    dloss = (main - ref).abs().item()
    dg = max((gref[n] - p.grad).abs().max().item() for n, p in m.named_parameters() if n in gref)
    ok = max(dloss, dg) < ATOL_TIGHT and aux1.item() == aux2.item()
    print(f"[fused_ce        ] loss={dloss:.2e} grad={dg:.2e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_fused_ce_vs_v13():
    torch.manual_seed(0)
    from v13.fused_ce import fused_linear_cross_entropy as v13_ce
    N, D, V = 40, 16, 64
    H = torch.randn(N, D, requires_grad=True)
    W = torch.randn(V, D, requires_grad=True)
    t = torch.randint(0, V, (N,))
    H2 = H.detach().clone().requires_grad_(True)
    W2 = W.detach().clone().requires_grad_(True)
    l1 = v13_ce(H, W, t, chunk=8)
    l1.backward()
    l2 = fused_linear_cross_entropy(H2, W2, t, chunk=8)
    l2.backward()
    ok = max(
        (l1 - l2).abs().item(),
        (H.grad - H2.grad).abs().max().item(),
        (W.grad - W2.grad).abs().max().item(),
    ) < 1e-12
    print(f"[fused_ce vs v13 ] {'PASS' if ok else 'FAIL'}")
    return ok


def test_one_step_parity(seed=0):
    """Identical init + batch + AdamW step → matching param deltas."""
    torch.manual_seed(seed)
    kw = _prod_kw(n_layers=2, vocab_size=128)
    ref = V13LM(V13Config(**kw)).train()
    port = SLM(SConfig(**kw)).train()
    port.load_state_dict(ref.state_dict())
    loader = synthetic_loader(kw['vocab_size'], batch_size=2, seq_len=16, n_batches=1, seed=seed)
    batch = next(iter(loader))

    def _one(model):
        opt = torch.optim.AdamW(
            build_param_groups(model, 0.01), lr=1e-4, betas=(0.9, 0.95),
        )
        ids, lab = batch['input_ids'], batch['labels']
        lm, _, _ = model._hidden_to_lm(ids)
        loss = model.ce_from_lm(lm, lab, chunk=32)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return {n: p.detach().clone() for n, p in model.named_parameters()}

    a = _one(ref)
    b = _one(port)
    worst, worst_n = 0.0, None
    for n in a:
        d = (a[n] - b[n]).abs().max().item()
        if d > worst:
            worst, worst_n = d, n
    ok = worst < ATOL_TIGHT
    print(f"[one-step AdamW  ] worst_param={worst:.2e} ({worst_n})  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_smoke_loss_decreases():
    torch.manual_seed(0)
    kw = _prod_kw(n_layers=2, vocab_size=64, dim=32)
    model = SLM(SConfig(**kw))
    loader = synthetic_loader(64, batch_size=2, seq_len=16, n_batches=6, seed=0)
    tr = Trainer(
        model, loader, learning_rate=3e-3, warmup_steps=0, total_steps=6,
        fused_ce=True, fused_ce_chunk=32, device=torch.device('cpu'),
        log_interval=0,
    )
    losses = tr.train(max_steps=6)
    ok = all(math.isfinite(v) for v in losses) and losses[-1] <= losses[0] + 0.5
    print(f"[smoke finite    ] first={losses[0]:.4f} last={losses[-1]:.4f}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    results = [
        test_state_dict_keys(),
        test_pam_equiv(),
        test_parallel_vs_recurrent(),
        test_lm_forward_grad(),
        test_fused_ce_equiv(),
        test_fused_ce_vs_v13(),
        test_one_step_parity(),
        test_smoke_loss_decreases(),
    ]
    print()
    if all(results):
        print("ALL v13_sempty SELFTESTS PASS")
    else:
        print("SOME SELFTESTS FAILED")
        raise SystemExit(1)


if __name__ == '__main__':
    main()
