"""Numerical A/B: is the v13 fork bit-equal to v11 on the additive path?

Build V11LM(v11_e3_k3_chat) and V13LM(v13_e3_k3_selective with v11 features
forced OFF: additive, no vault, no phase-address, no erase gate, lambda 0),
copy the v11 state_dict into v13, run both on identical int64 inputs on GPU,
and compare per-block hidden states + final logits + CE loss.

If all diffs ~1e-6 (fp32) => the v13 fork does NOT change the additive forward
path; any train-curve gap is recipe/data/shared-code-drift, not the v13 fork.
If a specific block diverges => that's the bug.

Run: .venv/bin/python v13/tmp/test_v11_v13_forward_ab.py
"""
import sys
import torch

sys.path.insert(0, '/home/gowrav/Development/qllm2')

from v11.model import V11LM, V11Config  # noqa: E402
from v13.model import V13LM, V13Config  # noqa: E402
from dataclasses import replace  # noqa: E402

torch.manual_seed(0)

dev = 'cuda'
T, B = 2048, 2
V = 50261

# ---- configs: force v13 to the v11 additive feature set ----
c11 = V11Config(
    vocab_size=V, dim=384, n_heads=6, head_dim=64, n_layers=16, expand=3,
    dropout=0.1, max_seq_len=T, use_rope=True, use_gsp=True, fused_qkv=True,
    tie_weights=True, gradient_checkpointing=False, chunk_size=256,
    decay_mode='head', write_mode='additive', n_states=3, state_dt_spread=2.0,
    base_dt_bias=-4.0, gate_content_aware=True, protect_gate_bias=-3.0,
    fused_e3=True,
)
c13 = V13Config(
    vocab_size=V, dim=384, n_heads=6, head_dim=64, n_layers=16, expand=3,
    dropout=0.1, max_seq_len=T, use_rope=True, use_gsp=True, fused_qkv=True,
    tie_weights=True, gradient_checkpointing=False, chunk_size=256,
    decay_mode='head', write_mode='additive', n_states=3, state_dt_spread=2.0,
    base_dt_bias=-4.0, gate_content_aware=True, protect_gate_bias=-3.0,
    fused_e3=True,
    # v13 features OFF (match v11):
    delta_erase_gate=False, vault_state=False, write_phase_address=False,
    gate_surprisal_lambda=0.0,
)

m11 = V11LM(c11).to(dev).eval()
m13 = V13LM(c13).to(dev).eval()

# ---- weight transfer ----
sd11 = m11.state_dict()
sd13 = m13.state_dict()
only11 = set(sd11) - set(sd13)
only13 = set(sd13) - set(sd11)
shape_mismatch = {k for k in set(sd11) & set(sd13) if sd11[k].shape != sd13[k].shape}
print(f"keys only in v11: {sorted(only11) or 'none'}")
print(f"keys only in v13: {sorted(only13) or 'none'}")
print(f"shape mismatches: {sorted(shape_mismatch) or 'none'}")
if shape_mismatch:
    sys.exit("shape mismatch — abort")
missing, unexpected = m13.load_state_dict(sd11, strict=False)
print(f"load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
# make v13 EXACTLY equal (incl. any keys only in v11, e.g. none expected)
for k in missing:
    if k in sd11:
        m13.state_dict()[k].copy_(sd11[k])  # noqa
# verify
for k in sd11:
    if k in m13.state_dict():
        d = (m11.state_dict()[k].float() - m13.state_dict()[k].float()).abs().max()
        if d > 0:
            print(f"  weight diff {k}: {d:.3e}")

# ---- inputs ----
input_ids = torch.randint(0, V, (B, T), device=dev)
labels = torch.randint(0, V, (B, T), device=dev)

def block_hiddens(m, x):
    z = m.embed(x)
    if m.pos_embed is not None:
        z = m.pos_embed(z, step_offset=0)
    z = m.embed_norm(z)
    out = []
    for blk in m.blocks:
        z, _ = blk(z, pam_state=None, step_offset=0)
        out.append(z)
    return out

with torch.no_grad():
    h11 = block_hiddens(m11, input_ids)
    h13 = block_hiddens(m13, input_ids)
    lg11, _, _ = m11(input_ids)
    lg13, _, _ = m13(input_ids)

print("\nper-block hidden max-abs diff (v13 - v11):")
maxd = 0.0
for i, (a, b) in enumerate(zip(h11, h13)):
    d = (a.float() - b.float()).abs().max().item()
    maxd = max(maxd, d)
    print(f"  block {i:2d}: {d:.3e}")
dlogits = (lg11.float() - lg13.float()).abs().max().item()
drel = ((lg11.float() - lg13.float()).abs() / (lg11.float().abs() + 1e-6)).max().item()
print(f"\nfinal logits: max-abs diff {dlogits:.3e}  max-rel {drel:.3e}")

with torch.no_grad():
    ce11 = m11.ce_from_lm(m11._hidden_to_lm(input_ids, 0)[0], labels)
    lm13 = m13._hidden_to_lm(input_ids, 0)
    ce13, nll = m13.ce_from_lm(lm13[0], labels, return_nll=True)
    ce13p = m13.ce_from_lm(lm13[0], labels, return_nll=False)
print(f"CE loss: v11 {ce11.item():.6f}  v13 {ce13p.item():.6f}  "
      f"diff {abs(ce11.item()-ce13p.item()):.3e}")
print(f"v13 NLL byproduct shape: {None if nll is None else tuple(nll.shape)}")

ok = maxd < 1e-4 and dlogits < 1e-4
print(f"\nVERDICT: {'EQUIVALENT (fork clean on additive path)' if ok else 'DIVERGENT — see per-block diffs above'}")
