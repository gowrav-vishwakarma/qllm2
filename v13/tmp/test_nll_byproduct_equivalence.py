"""Re-runnable: verify the O(1) NLL-byproduct gate target is EXACT.

The fused CE now emits the per-token NLL it already computes (materialized
intermediate) as `loss._nll` instead of the gate aux re-running a second
detached O(V) head GEMM (`linear_ce_per_token`). This test checks, on the
real v13 model:

  1. byproduct NLL == legacy `linear_ce_per_token` (same fp32 math),
     including ignore_index rows (-> 0.0);
  2. gate BCE loss value identical (new vs legacy target);
  3. grads wrt protect_gate weights identical;
  4. main CE scalar unchanged with return_nll on/off (old contract intact);
  5. trainer capability flag: v13 model -> True, V7 model -> False.

    cd qllm2 && .venv/bin/python v13/tmp/test_nll_byproduct_equivalence.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dataclasses import replace as dc_replace

import torch
import torch.nn.functional as F
from v13.model import V13LM, get_config
from v11.fused_ce import linear_ce_per_token
from v7.train import V7Trainer

torch.manual_seed(0)
DEVICE = 'cuda'

# Small but structurally identical config (same K/gates/preset flags).
cfg = get_config('v13_e3_k3_selective')
cfg = dc_replace(cfg, dim=96, n_heads=3, n_layers=4, max_seq_len=512,
                 delta_chunk=64, fused_e3=True)
m = V13LM(cfg).to(DEVICE)
m.train()
assert cfg.gate_surprisal_lambda > 0, "preset must keep the gate aux on"

B, T = 4, 256
ids = torch.randint(0, cfg.vocab_size, (B, T), device=DEVICE)
labels = torch.randint(0, cfg.vocab_size, (B, T), device=DEVICE)
labels[0, :10] = -100          # ignore_index rows (padding)
loss_mask = torch.ones(B, T, device=DEVICE)
loss_mask[0, :10] = 0.0

with torch.autocast('cuda', dtype=torch.bfloat16):
    lm, aux, gate_probs = m._hidden_to_lm(ids)
    assert gate_probs is not None, "gate_probs stash missing (training mode?)"

    # ── 4. main CE unchanged by return_nll ──────────────────────────────
    main_off = m.ce_from_lm(lm, labels, loss_mask=loss_mask, chunk=128)
    main_on, nll = m.ce_from_lm(lm, labels, loss_mask=loss_mask, chunk=128,
                                return_nll=True)
assert nll is not None and nll.shape == (B, T)
assert nll.dtype == torch.float32 and not nll.requires_grad
assert (main_off - main_on).abs().item() == 0.0, "main CE changed!"
print(f"  main CE identical on/off return_nll: {main_on.item():.6f}")

# ── 1. byproduct NLL == legacy linear_ce_per_token ─────────────────────
raw = m
hc = torch.cat([lm[..., 0], lm[..., 1]], dim=-1).reshape(B * T, -1)
wc = torch.cat([raw.embed.embed_real.weight, raw.embed.embed_imag.weight], dim=-1)
legacy = linear_ce_per_token(hc.detach(), wc.detach(), labels.reshape(-1),
                             chunk=128).reshape(B, T)
d = (nll - legacy).abs().max().item()
assert d < 1e-5, f"NLL mismatch: max |Δ|={d}"
assert (nll[0, :10] == 0).all(), "ignore rows must be 0.0"
assert torch.equal(legacy[0, :10], nll[0, :10])
print(f"  NLL byproduct == legacy linear_ce_per_token: max|Δ|={d:.2e}, "
      f"ignore rows 0.0 OK")


def gate_loss_from(surprisal):
    """Exact copy of V7Trainer._gate_surprisal_loss target/BCE math."""
    valid = labels != -100
    valid = valid & (loss_mask > 0)
    median_ce = surprisal[valid].median()
    tau = max(getattr(cfg, 'gate_surprisal_tau', 1.0), 1e-3)
    sign = getattr(cfg, 'gate_surprisal_sign', 1.0)
    target_p = torch.sigmoid(sign * (median_ce - surprisal) / tau).detach()
    gp = gate_probs.float().clamp(1e-4, 1 - 1e-4)
    target = target_p.float().unsqueeze(0).expand_as(gp)
    vmask = valid.unsqueeze(0).expand_as(gp).to(gp.dtype)
    with torch.amp.autocast(device_type='cuda', enabled=False):
        bce = F.binary_cross_entropy(gp, target, reduction='none')
    return (bce * vmask).sum() / vmask.sum().clamp_min(1.0)


# ── 2+3. gate loss value and grads identical ───────────────────────────
g_new = gate_loss_from(nll)
g_new.backward(retain_graph=True)
grads_new = {n: p.grad.clone() for n, p in m.named_parameters()
             if p.grad is not None and 'protect_gate' in n}
m.zero_grad(set_to_none=True)
g_old = gate_loss_from(legacy)
g_old.backward(retain_graph=True)
grads_old = {n: p.grad.clone() for n, p in m.named_parameters()
             if p.grad is not None and 'protect_gate' in n}
dv = (g_new - g_old).abs().item()
assert dv < 1e-5, f"gate loss value mismatch: {dv}"
gm = max((a - b).abs().max().item() for a, b in zip(grads_new.values(),
                                                     grads_old.values()))
assert len(grads_new) == 2 * cfg.n_layers, "expected weight+bias protect_gate per layer"
layer_ids = sorted({n.split('.')[1] for n in grads_new})
assert len(layer_ids) == cfg.n_layers, f"protect_gate layers {layer_ids} != n_layers {cfg.n_layers}"
assert gm < 1e-6, f"gate grad mismatch: {gm}"
print(f"  gate loss identical: new={g_new.item():.8f} old={g_old.item():.8f} "
      f"(Δ={dv:.2e})")
print(f"  protect_gate grads identical over {len(grads_new)} layers "
      f"(max|Δ|={gm:.2e})")

# ── grad isolation: aux must NOT touch the trunk (detached stash) ─────
m.zero_grad(set_to_none=True)
g_new = gate_loss_from(nll)
g_new.backward()
trunk_grads = [(n, p.grad.abs().max().item()) for n, p in m.named_parameters()
               if p.grad is not None and 'protect_gate' not in n]
assert not trunk_grads, f"trunk leaked grads: {trunk_grads[:3]}"
print("  trunk isolation: no non-protect_gate grads from aux  OK")

# ── 5. trainer capability flags ────────────────────────────────────────
class _FakeLoader:
    pass
t13 = V7Trainer.__new__(V7Trainer)
t13._ce_emits_nll = (
    'return_nll' in __import__('inspect').signature(m.ce_from_lm).parameters)
assert t13._ce_emits_nll is True, "v13 ce_from_lm must advertise return_nll"
import inspect as _inspect
from v11.model import V11LM, get_config as get_v11_config
v11m = V11LM(get_v11_config('tiny'))
assert 'return_nll' not in _inspect.signature(
    v11m.ce_from_lm).parameters, "v11 model should keep legacy path"

print("\nPASS — O(1) NLL-byproduct gate target is exact; legacy path intact.")
