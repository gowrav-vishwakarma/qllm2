"""M4.4: compact L0-pruned heads into a slim inference checkpoint.

When a model is trained with head gates (``head_gate=True``), the hard-concrete
L0 penalty drives unused head slots to ~0. Those slots still occupy memory and
FLOPs. This utility rebuilds a checkpoint keeping ONLY the open head slots per
layer (per-layer ``n_heads`` via ``layer_specs``), slicing every per-head
parameter of each PAM layer, and preserving inference behavior: the surviving
heads keep their learned gate value ``z`` (head_gate stays enabled with the
sliced ``log_alpha``), and dropped heads had z<threshold (≈0 contribution).

Usage:
    .venv/bin/python -m v12.compact --checkpoint checkpoints_v12_headgate/best_model.pt \
        --out checkpoints_v12_headgate/slim.pt --threshold 1e-3

Only the common geometry is supported (head decay_mode, fused_qkv); per_channel
decay or non-fused qkv raise NotImplementedError.
"""

import argparse
import copy
from dataclasses import asdict

import torch

from v12.model import V12Config, V12LM, HardConcreteGate


def _hard_concrete_z(log_alpha: torch.Tensor) -> torch.Tensor:
    """Deterministic hard-concrete gate value (matches HardConcreteGate._z)."""
    s = torch.sigmoid(log_alpha)
    s_bar = s * (HardConcreteGate.zeta - HardConcreteGate.gamma) + HardConcreteGate.gamma
    return s_bar.clamp(0.0, 1.0)


def _keep_indices(state, prefix, n_heads, threshold):
    """Head indices to keep for one PAM layer (all if it has no head gate)."""
    key = f'{prefix}head_gate.log_alpha'
    if key not in state:
        return list(range(n_heads))
    z = _hard_concrete_z(state[key])
    keep = [h for h in range(n_heads) if float(z[h]) > threshold]
    return keep or [int(z.argmax())]  # never drop every head


def _row_take(t, idx):
    return t.index_select(0, torch.tensor(idx, dtype=torch.long))


def _col_take(t, idx):
    return t.index_select(1, torch.tensor(idx, dtype=torch.long))


def _compact_pam(state, prefix, cfg_layer, keep):
    """Slice every per-head param under ``prefix`` down to ``keep`` heads."""
    if cfg_layer.decay_mode == 'per_channel':
        raise NotImplementedError("compact supports head decay_mode only")
    if not cfg_layer.fused_qkv:
        raise NotImplementedError("compact supports fused_qkv only")

    H, d, K = cfg_layer.n_heads, cfg_layer.head_dim, cfg_layer.n_states
    inner = H * d

    # qkv rows: [3*inner, dim], head h owns rows [b*inner + h*d : +d] for b in 0..2.
    qkv_rows = [b * inner + h * d + j for b in range(3) for h in keep for j in range(d)]
    for wk in ('qkv_proj.weight_real', 'qkv_proj.weight_imag'):
        state[prefix + wk] = _row_take(state[prefix + wk], qkv_rows)

    # o_proj cols: [dim, inner], head h owns cols [h*d : +d].
    o_cols = [h * d + j for h in keep for j in range(d)]
    for wk in ('o_proj.weight_real', 'o_proj.weight_imag'):
        state[prefix + wk] = _col_take(state[prefix + wk], o_cols)

    # Simple per-head row tensors ([H, *] or [H]).
    for wk in ('dt_proj.weight', 'dt_proj.bias', 'dt_bias',
               'head_gate.log_alpha', 'write_phase_w', 'write_phase_b',
               'protect_gate.weight', 'protect_gate.bias',
               'beta_proj.weight', 'beta_proj.bias'):
        full = prefix + wk
        if full in state:
            state[full] = _row_take(state[full], keep)

    # phase_proj / score_proj rows grouped by head: head h owns [h*K : +K].
    grouped = [h * K + k for h in keep for k in range(K)]
    for wk in ('phase_proj.weight', 'phase_proj.bias',
               'score_proj.weight', 'score_proj.bias'):
        full = prefix + wk
        if full in state:
            state[full] = _row_take(state[full], grouped)
    # state_dt_offset [K] is unaffected.


def compact_checkpoint(ckpt_path, out_path, threshold=1e-3, verify_seq_len=32):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = V12Config(**ckpt['config'])

    src = V12LM(cfg)
    src.load_state_dict(ckpt['model_state_dict'])
    src.eval()

    report = src.head_gate_report(threshold=threshold)
    if report is None:
        raise ValueError("checkpoint has no head gates (nothing to compact)")

    # New per-layer specs with compacted n_heads (start from the source manifest).
    old_specs = cfg.layer_specs or src._materialize_specs()
    new_specs = [copy.deepcopy(s) for s in old_specs]

    state = dict(ckpt['model_state_dict'])
    per_layer_keep = []
    for i, block in enumerate(src.blocks):
        pam = block.pam
        keep = _keep_indices(state, f'blocks.{i}.pam.', pam.num_heads, threshold)
        per_layer_keep.append(keep)
        cfg_layer = copy.deepcopy(cfg)
        cfg_layer.n_heads = pam.num_heads
        cfg_layer.head_dim = pam.head_dim
        cfg_layer.n_states = pam.n_states
        cfg_layer.decay_mode = pam.decay_mode
        cfg_layer.fused_qkv = pam.fused_qkv
        _compact_pam(state, f'blocks.{i}.pam.', cfg_layer, keep)
        new_specs[i]['n_heads'] = len(keep)

    new_cfg = copy.deepcopy(cfg)
    new_cfg.layer_specs = new_specs
    slim = V12LM(new_cfg)
    slim.load_state_dict(state, strict=True)
    slim.eval()

    # Verify inference behavior is preserved (dropped heads had z≈0).
    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (2, verify_seq_len))
    with torch.no_grad():
        a, _, _ = src(ids)
        b, _, _ = slim(ids)
    max_diff = (a - b).abs().max().item()

    src_p = src.count_parameters()['total']
    slim_p = slim.count_parameters()['total']

    out = dict(ckpt)
    out['config'] = asdict(new_cfg)
    out['model_state_dict'] = slim.state_dict()
    out['compacted_from'] = str(ckpt_path)
    out['compact_threshold'] = threshold
    out['compact_kept_heads'] = per_layer_keep
    torch.save(out, out_path)

    print(f"Compacted {ckpt_path} -> {out_path}")
    print(f"  per-layer kept heads: {[len(k) for k in per_layer_keep]} "
          f"(of {[b.pam.num_heads for b in src.blocks]})")
    print(f"  params: {src_p:,} -> {slim_p:,} ({100*(1-slim_p/src_p):.1f}% smaller)")
    print(f"  max|Δlogits| src vs slim: {max_diff:.3e}")
    return slim, max_diff


def main():
    p = argparse.ArgumentParser(description='Compact L0-pruned heads into a slim checkpoint')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--threshold', type=float, default=1e-3,
                   help='Keep a head slot when its gate z > threshold (default 1e-3)')
    p.add_argument('--verify_seq_len', type=int, default=32)
    args = p.parse_args()
    compact_checkpoint(args.checkpoint, args.out, threshold=args.threshold,
                       verify_seq_len=args.verify_seq_len)


if __name__ == '__main__':
    main()
