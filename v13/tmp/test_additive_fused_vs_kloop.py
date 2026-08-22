"""Equivalence: _forward_multistate_fused (fused_e3=True, additive) vs
_forward_multistate (K-loop, additive), K=3, head decay, same inputs.

This is the ADDITIVE twin of test_fused_delta_equivalence.py (which covered
delta only). The fused_e3 path did not exist in round-1 (Jul 1) v11 code —
if the fused ADDITIVE path diverges from the K-loop, that explains the
ab1 stall (round-1 learned via K-loop).

Run: .venv/bin/python v13/tmp/test_additive_fused_vs_kloop.py
"""
import sys
import torch

sys.path.insert(0, '/home/gowrav/Development/qllm2')
from v13.model import V13LM, V13Config  # noqa: E402

torch.manual_seed(7)
dev = 'cuda'
T, B = 512, 2

def make(fused: bool):
    cfg = V13Config(
        vocab_size=50261, dim=384, n_heads=6, head_dim=64, n_layers=1,
        expand=3, dropout=0.0, max_seq_len=T, use_rope=True, use_gsp=True,
        fused_qkv=True, tie_weights=True, gradient_checkpointing=False,
        chunk_size=256, decay_mode='head', write_mode='additive', n_states=3,
        state_dt_spread=2.0, base_dt_bias=-4.0, gate_content_aware=True,
        protect_gate_bias=-3.0, fused_e3=fused, delta_erase_gate=False,
        vault_state=False, write_phase_address=False,
        gate_surprisal_lambda=0.0,
    )
    m = V13LM(cfg).to(dev).eval()
    return m

m_f = make(True)
m_k = make(False)
# identical weights
m_k.load_state_dict(m_f.state_dict())
m_f.train(False); m_k.train(False)
# dropout off already (0.0); make sure no training-mode aux
for m in (m_f, m_k):
    m.eval()

x = torch.randint(0, 50261, (B, T), device=dev)
layer = m_f.blocks[0]

# Feed the layer's PAM sublayer directly: build q/k/v via its _project path
# by calling the PAM layer with a prepared complex input.
z = m_f.embed(x)
if m_f.pos_embed is not None:
    z = m_f.pos_embed(z, step_offset=0)
z = m_f.embed_norm(z)

layer_k = m_k.blocks[0]
# Force the K-loop: temporarily disable fused on the fused model too? No —
# call each model's own PAM layer; m_f has fused_e3=True, m_k fused_e3=False.
with torch.no_grad():
    out_f, st_f = layer(z.clone(), pam_state=None, step_offset=0)
    out_k, st_k = layer_k(z.clone(), pam_state=None, step_offset=0)

d = (out_f.float() - out_k.float()).abs().max().item()
print(f"output max-abs diff (fused - kloop): {d:.6e}")
# state compare
sf, sk = st_f, st_k
if isinstance(sf, (tuple, list)) and isinstance(sk, (tuple, list)):
    for i, (a, b) in enumerate(zip(sf, sk)):
        da = (a.float() - b.float()).abs().max().item()
        print(f"  state[{i}] max-abs diff: {da:.6e}")
        d = max(d, da)
else:
    ds = (sf.float() - sk.float()).abs().max().item()
    print(f"  state max-abs diff: {ds:.6e}")
    d = max(d, ds)

ok = d < 1e-5
print(f"\nVERDICT: {'EQUIVALENT' if ok else 'DIVERGENT — fused additive path is BUGGY'}")

# If divergent: bisect by chunk — run fused with delta_chunk-like chunk_size
# variations is not applicable; instead compare first-chunk-only output by
# masking the rest of the sequence.
if not ok:
    for tlen in (1, 8, 64, 128, 256, 512):
        with torch.no_grad():
            of, _ = layer(z[:, :tlen].clone(), pam_state=None, step_offset=0)
            ok2, _ = layer_k(z[:, :tlen].clone(), pam_state=None, step_offset=0)
        print(f"  T={tlen}: diff {(of.float()-ok2.float()).abs().max().item():.4e}")
