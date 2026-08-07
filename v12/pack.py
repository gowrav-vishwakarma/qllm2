"""Pack step: assemble ONE inference checkpoint from group checkpoints.

This is the "config decides sequential vs MoE at pack time" seam. A compose spec
(JSON) lists an ordered set of modules; pack merges their blocks into a single
grown stack, carrying each module's per-layer structure and stamping the
composition mode (``attach_mode``) per group into ``layer_specs``.

Default composition is always-on sequential depth. Groups tagged
``"attach_mode": "moe"`` are recorded as such; at run time they currently execute
sequentially (see ``V12LM._apply_moe_group``) until the router lands — so packing
a MoE group is forward-compatible and behavior is unchanged today.

Compose spec (JSON):
    {
      "base": "checkpoints_v12_grow/base/best_model.pt",
      "modules": [
        {"checkpoint": ".../facts/best_model.pt", "group_id": "facts"},
        {"checkpoint": ".../code/best_model.pt",  "group_id": "code",
         "attach_mode": "moe"}
      ]
    }

Semantics:
  - The base checkpoint supplies the shared params (embeddings, norms, LM head)
    and its foundation blocks (spec group_id == "base").
  - Each module contributes its GROWN blocks (spec group_id != "base"), appended
    in order and renumbered. ``take`` may override selection: "grown" (default),
    "all", or an explicit list of source block indices.
  - Per-module ``group_id`` / ``attach_mode`` / ``skill`` override the copied
    spec's provenance; structural fields (n_heads, ...) are preserved.

Usage:
    .venv/bin/python -m v12.pack --spec compose.json --out packed/model.pt
"""

import argparse
import copy
import json
from dataclasses import asdict

import torch

from v12.model import V12Config, V12LM

# Non-block params shared across the stack, taken from the base checkpoint.
_SHARED_PREFIXES = (
    'embed.', 'embed_norm.', 'pos_embed.', 'output_norm.',
    'lm_head_proj.', 'lm_head_norm.',
)


def _load(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    # Filter to V12Config fields; registry module checkpoints carry extra keys
    # (e.g. 'module_card') that V12Config does not accept.
    cfg = V12Config(**{k: v for k, v in ckpt['config'].items()
                       if k in V12Config.__dataclass_fields__})
    state = ckpt['model_state_dict']
    # Materialize specs (older ckpts may lack layer_specs).
    specs = cfg.layer_specs
    if not specs:
        n = cfg.n_layers
        specs = [{'group_id': 'base', 'stage': 0, 'attach_mode': 'sequential',
                  'frozen': False} for _ in range(n)]
    return cfg, state, specs


def shared_drift(base_state, other_state, tol=1e-3):
    """[(rel_frobenius, key)] for shared params ``other_state`` moved off the base.

    Composition keeps only the base's shared params, so any drift here means the
    module's blocks were tuned against a table that is about to be discarded.
    Returns [] when the other state carries no shared params at all (a published
    group module) — there is nothing left to compare by then.
    """
    out = []
    for key, base_val in base_state.items():
        if not any(key.startswith(p) for p in _SHARED_PREFIXES):
            continue
        mod_val = other_state.get(key)
        if mod_val is None or mod_val.shape != base_val.shape:
            continue
        denom = base_val.float().norm().item()
        if denom == 0:
            continue
        rel = (mod_val.float() - base_val.float()).norm().item() / denom
        if rel > tol:
            out.append((rel, key))
    out.sort(reverse=True)
    return out


def warn_shared_drift(base_state, other_state, label, *, tol=1e-3, indent='  '):
    """Print a composition-fidelity warning for drifted shared params."""
    drift = shared_drift(base_state, other_state, tol=tol)
    if not drift:
        return drift
    total = sum(1 for k in base_state if any(k.startswith(p) for p in _SHARED_PREFIXES))
    print(f"{indent}WARNING: {label} retrained {len(drift)}/{total} shared tensors; "
          f"composition keeps the BASE copy, so its blocks will read a different "
          f"table than they were trained on.")
    for rel, key in drift[:4]:
        print(f"{indent}  {key}: rel-Frobenius {rel:.4f}")
    print(f"{indent}  Retrain that stage with --freeze_shared to make packing lossless.")
    return drift


def _take_indices(specs, take):
    if take == 'all':
        return list(range(len(specs)))
    if isinstance(take, list):
        return [int(i) for i in take]
    # default: 'grown' == blocks whose group is not the foundation.
    return [i for i, s in enumerate(specs) if s.get('group_id') != 'base']


def pack(spec_path, out_path):
    with open(spec_path, 'r') as f:
        compose = json.load(f)

    base_cfg, base_state, base_specs = _load(compose['base'])
    base_keep = [i for i, s in enumerate(base_specs) if s.get('group_id', 'base') == 'base']
    if not base_keep:
        base_keep = list(range(len(base_specs)))

    combined_specs = []
    block_sources = []  # (source_state, old_idx) in final order

    # Foundation blocks from the base checkpoint.
    for old in base_keep:
        s = copy.deepcopy(base_specs[old])
        s.setdefault('group_id', 'base')
        s.setdefault('attach_mode', 'sequential')
        combined_specs.append(s)
        block_sources.append((base_state, old))

    # Grown blocks from each module.
    for module in compose.get('modules', []):
        m_cfg, m_state, m_specs = _load(module['checkpoint'])
        if m_cfg.dim != base_cfg.dim or m_cfg.head_dim != base_cfg.head_dim:
            raise ValueError(
                f"module {module['checkpoint']} geometry (dim={m_cfg.dim}, "
                f"head_dim={m_cfg.head_dim}) != base (dim={base_cfg.dim}, "
                f"head_dim={base_cfg.head_dim})"
            )
        warn_shared_drift(base_state, m_state, module['checkpoint'])
        take = module.get('take', 'grown')
        for old in _take_indices(m_specs, take):
            s = copy.deepcopy(m_specs[old])
            if 'group_id' in module:
                s['group_id'] = module['group_id']
            if 'skill' in module:
                s['skill'] = module['skill']
            s['attach_mode'] = module.get('attach_mode', s.get('attach_mode', 'sequential'))
            combined_specs.append(s)
            block_sources.append((m_state, old))

    # Build the combined config + model.
    new_cfg = copy.deepcopy(base_cfg)
    new_cfg.layer_specs = combined_specs
    new_cfg.n_layers = len(combined_specs)
    model = V12LM(new_cfg)

    # Assemble the combined state_dict: shared params from base + renumbered blocks.
    combined_state = {}
    for key, val in base_state.items():
        if any(key.startswith(p) for p in _SHARED_PREFIXES):
            combined_state[key] = val
    for new_idx, (src_state, old_idx) in enumerate(block_sources):
        old_prefix = f'blocks.{old_idx}.'
        new_prefix = f'blocks.{new_idx}.'
        for key, val in src_state.items():
            if key.startswith(old_prefix):
                combined_state[new_prefix + key[len(old_prefix):]] = val

    for module in compose.get('modules', []):
        m_cfg, m_state, _m_specs = _load(module['checkpoint'])
        gid = module.get('group_id')
        if not gid:
            continue
        aprefix = f'module_adapters.{gid}.'
        for key, val in m_state.items():
            if key.startswith(aprefix):
                combined_state[key] = val

    missing, unexpected = model.load_state_dict(combined_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"packed state mismatch: missing={missing[:5]}... "
            f"unexpected={unexpected[:5]}..."
        )

    out = {
        'config': asdict(new_cfg),
        'model_state_dict': model.state_dict(),
        'packed_from': {'base': compose['base'],
                        'modules': [m['checkpoint'] for m in compose.get('modules', [])]},
    }
    torch.save(out, out_path)

    modes = [s.get('attach_mode', 'sequential') for s in combined_specs]
    groups = [s.get('group_id') for s in combined_specs]
    print(f"Packed {len(combined_specs)} layers -> {out_path}")
    print(f"  groups:       {groups}")
    print(f"  attach_modes: {modes}")
    if 'moe' in modes:
        print("  note: moe groups run sequentially until the router lands "
              "(schema recorded for forward-compat).")
    return model


def assemble_plan(plan):
    """Assemble a resolved module stack into a live (model, new_cfg).

    The base module (role at bottom) supplies shared params + its blocks; each
    group module supplies its own blocks (renumbered). Each module's attach_mode
    is stamped per layer for the composition path. Returns (V12LM, V12Config).
    """
    base_ref = plan.stack[0]
    base_ckpt = torch.load(base_ref.ckpt_path, map_location='cpu', weights_only=False)
    base_cfg = V12Config(**{k: v for k, v in base_ckpt['config'].items()
                            if k in V12Config.__dataclass_fields__})
    base_state = base_ckpt['model_state_dict']

    combined_specs, block_sources = [], []
    for ref in plan.stack:
        state = (base_state if ref is base_ref
                 else torch.load(ref.ckpt_path, map_location='cpu',
                                 weights_only=False)['model_state_dict'])
        if ref is not base_ref:
            warn_shared_drift(base_state, state, ref.ckpt_path)
        specs = ref.card.layer_specs or [{} for _ in range(ref.n_layers)]
        for j in range(ref.n_layers):
            s = copy.deepcopy(specs[j]) if j < len(specs) else {}
            if ref.card.group_id:
                s['group_id'] = ref.card.group_id
            s['attach_mode'] = ref.card.attach_mode
            s['frozen'] = False
            combined_specs.append(s)
            block_sources.append((state, j))

    new_cfg = copy.deepcopy(base_cfg)
    new_cfg.layer_specs = combined_specs
    new_cfg.n_layers = len(combined_specs)
    model = V12LM(new_cfg)

    combined_state = {}
    for key, val in base_state.items():
        if any(key.startswith(p) for p in _SHARED_PREFIXES):
            combined_state[key] = val
    for new_idx, (src_state, old_idx) in enumerate(block_sources):
        op, np_ = f'blocks.{old_idx}.', f'blocks.{new_idx}.'
        for key, val in src_state.items():
            if key.startswith(op):
                combined_state[np_ + key[len(op):]] = val

    missing, unexpected = model.load_state_dict(combined_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"packed state mismatch: missing={missing[:5]}... unexpected={unexpected[:5]}...")
    return model, new_cfg


def pack_from_registry(target_id, constraint, registry, out_path, *, force=False):
    """Resolve a target module from the registry and assemble one inference ckpt."""
    from v12.registry import resolve

    plan = resolve(target_id, constraint, registry, force=force)
    if plan.report['warnings']:
        for w in plan.report['warnings']:
            print(f"  [warn] {w}")
    model, new_cfg = assemble_plan(plan)

    out = {
        'config': asdict(new_cfg),
        'model_state_dict': model.state_dict(),
        'packed_from': {'target': plan.report['target'], 'stack': plan.order()},
        'composition': plan.report,
    }
    torch.save(out, out_path)
    print(f"Packed {plan.report['target']} -> {out_path}")
    print(f"  stack: {plan.order()} ({plan.report['total_layers']} layers)")
    if plan.lineage:
        print(f"  lineage (finetuned, not stacked): "
              f"{[l['module_id'] for l in plan.lineage]}")
    return model


def main():
    p = argparse.ArgumentParser(description='Pack modules into one inference checkpoint')
    sub_note = ('Either --spec (compose JSON) or --target (resolve from --registry).')
    p.add_argument('--spec', default=None, help='Compose spec JSON (hand-written)')
    p.add_argument('--target', default=None, help='Registry module_id to resolve + pack')
    p.add_argument('--constraint', default='*', help='Version constraint for --target')
    p.add_argument('--registry', default='v12_registry')
    p.add_argument('--force', action='store_true', help='Downgrade resolver conflicts to warnings')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    if args.target:
        from v12.registry import Registry
        pack_from_registry(args.target, args.constraint, Registry(args.registry),
                           args.out, force=args.force)
    elif args.spec:
        pack(args.spec, args.out)
    else:
        p.error('provide --spec or --target. ' + sub_note)


if __name__ == '__main__':
    main()
