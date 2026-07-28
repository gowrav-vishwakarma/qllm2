"""Publish a trained stage checkpoint as a registry module.

A stage checkpoint contains the whole grown stack (base + previously grown groups
+ the group trained this stage). Publishing extracts ONLY the relevant blocks
into a slim, self-contained module checkpoint, stamps a ``ModuleCard`` (identity,
geometry, requires, substrate_hash from the grown specs), writes the card (sidecar
+ embedded), and registers it.

  - role=base:  extract the foundation blocks (group_id == "base") AND the shared
                params (embeddings, norms, LM head). This is the stack root.
  - role=group: extract the group's own blocks (default: the most-recently-grown
                group_id). Declares its substrate via --requires (prelayer/finetuned).

Usage:
    # publish the grammar base
    .venv/bin/python -m v12.publish --checkpoint ckpts/grammar/best_model.pt \
        --module_id grammar_by_x --version 1.0 --role base --provenance "x lab"

    # publish the fact group, requiring the grammar base beneath it
    .venv/bin/python -m v12.publish --checkpoint ckpts/fact/best_model.pt \
        --module_id fact_by_x --version 1.0 --role group --group_id fact \
        --requires "grammar_by_x@>=1.0:prelayer" --provenance "x lab"
"""

import argparse
import copy
from dataclasses import asdict

import torch

from v12.model import V12Config
from v12.pack import _SHARED_PREFIXES, warn_shared_drift
from v12.registry import (
    ModuleCard, Registry, Requirement, hash_block_states, write_card,
)


def _materialize(cfg: V12Config):
    specs = cfg.layer_specs
    if specs:
        return specs
    return [{'group_id': 'base', 'stage': 0, 'attach_mode': 'sequential', 'frozen': False}
            for _ in range(cfg.n_layers)]


def parse_requires(spec: str):
    """Parse "id@spec:mode,id2@spec2:mode2" into a list of Requirement."""
    out = []
    for item in (s for s in (spec or '').split(',') if s.strip()):
        mode = 'prelayer'
        body = item.strip()
        if ':' in body:
            body, mode = body.rsplit(':', 1)
        if '@' in body:
            mid, vspec = body.split('@', 1)
        else:
            mid, vspec = body, '*'
        out.append(Requirement(module_id=mid.strip(), version_spec=vspec.strip(), mode=mode.strip()))
    return out


def build_module(ckpt: dict, *, role='group', group_id=None, take=None):
    """Extract a slim module (state, cfg, specs, substrate_hash, self_hash)."""
    cfg = V12Config(**{k: v for k, v in ckpt['config'].items()
                       if k in V12Config.__dataclass_fields__})
    state = ckpt['model_state_dict']
    specs = _materialize(cfg)

    if take is not None:
        idxs = list(take)
    elif role == 'base':
        idxs = [i for i, s in enumerate(specs) if s.get('group_id', 'base') == 'base']
    else:
        grp = group_id or specs[-1].get('group_id')
        idxs = [i for i, s in enumerate(specs) if s.get('group_id') == grp]
    if not idxs:
        raise ValueError(f"no blocks selected (role={role}, group_id={group_id})")

    new_state = {}
    if role == 'base':
        for k, v in state.items():
            if any(k.startswith(p) for p in _SHARED_PREFIXES):
                new_state[k] = v
    for new_i, old in enumerate(idxs):
        op, np_ = f'blocks.{old}.', f'blocks.{new_i}.'
        for k, v in state.items():
            if k.startswith(op):
                new_state[np_ + k[len(op):]] = v

    new_specs = [copy.deepcopy(specs[i]) for i in idxs]
    new_cfg = copy.deepcopy(cfg)
    new_cfg.layer_specs = new_specs
    new_cfg.n_layers = len(idxs)

    self_hash = hash_block_states([(new_state, i) for i in range(len(idxs))])
    substrate_hash = None if role == 'base' else specs[idxs[0]].get('substrate_hash')
    return new_state, new_cfg, new_specs, substrate_hash, self_hash


def _check_shared_drift(state, requires, registry):
    """Warn if this stage retrained the shared params its base module owns.

    This is the last point where the drift is visible: a role=group module keeps
    only its own blocks, so by pack time the stage's shared params are gone and
    nothing downstream can tell that the blocks were trained against a different
    table. Returns the max rel-Frobenius seen (0.0 if clean/unknown).
    """
    if registry is None or not requires:
        return 0.0
    for req in requires:
        try:
            _ver, base_path, base_card = registry.find(req.module_id, req.version_spec)
        except KeyError:
            continue
        if base_card.role != 'base':
            continue
        base_state = torch.load(base_path, map_location='cpu',
                                weights_only=False)['model_state_dict']
        drift = warn_shared_drift(base_state, state,
                                  f"this stage vs base {req.module_id}@{_ver}",
                                  indent='  ')
        return max((r for r, _ in drift), default=0.0)
    return 0.0


def publish(ckpt_path, *, module_id, version, role='group', group_id=None, take=None,
            requires=None, provenance='', description='', attach_mode='sequential',
            skill=None, registry=None, out=None, overwrite=False):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    new_state, new_cfg, new_specs, substrate_hash, self_hash = build_module(
        ckpt, role=role, group_id=group_id, take=take,
    )
    skill = skill or (new_specs[-1].get('skill') if new_specs else None)
    grp = group_id or (new_specs[-1].get('group_id') if new_specs else None)
    if role != 'base':
        _check_shared_drift(ckpt['model_state_dict'], requires, registry)

    card = ModuleCard(
        module_id=module_id, version=str(version), role=role, skill=skill,
        group_id=grp, attach_mode=attach_mode, provenance=provenance,
        description=description, dim=new_cfg.dim, head_dim=new_cfg.head_dim,
        vocab_size=new_cfg.vocab_size, n_layers=new_cfg.n_layers,
        layer_specs=new_specs, self_hash=self_hash, substrate_hash=substrate_hash,
        requires=requires or [],
    )
    module_ckpt = {'config': asdict(new_cfg), 'model_state_dict': new_state}

    if registry is not None:
        path = registry.add(module_ckpt, card, overwrite=overwrite)
    else:
        path = out or (ckpt_path + f'.module_{module_id}_{version}.pt')
        torch.save(module_ckpt, path)
        write_card(path, card, embed=True)

    print(f"Published {module_id}@{version} ({role}) -> {path}")
    print(f"  layers={new_cfg.n_layers} self_hash={self_hash[:12]} "
          f"substrate_hash={(substrate_hash or '-')[:12]} "
          f"requires={[f'{r.module_id}@{r.version_spec}:{r.mode}' for r in (requires or [])]}")
    return path, card


def main():
    p = argparse.ArgumentParser(description='Publish a stage checkpoint as a registry module')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--module_id', required=True)
    p.add_argument('--version', required=True)
    p.add_argument('--role', choices=['base', 'group'], default='group')
    p.add_argument('--group_id', default=None, help='Which group to extract (default: last)')
    p.add_argument('--take', default=None, help='Explicit source block indices "lo:hi" or "0,1"')
    p.add_argument('--requires', default=None,
                   help='"id@spec:mode,..." e.g. "grammar_by_x@>=1.0:prelayer"')
    p.add_argument('--provenance', default='')
    p.add_argument('--description', default='')
    p.add_argument('--attach_mode', choices=['sequential', 'moe'], default='sequential')
    p.add_argument('--skill', default=None)
    p.add_argument('--registry', default='v12_registry')
    p.add_argument('--out', default=None, help='Write to a file instead of a registry')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    take = None
    if args.take:
        take = (list(range(*map(int, args.take.split(':')))) if ':' in args.take
                else [int(i) for i in args.take.split(',') if i != ''])
    registry = None if args.out else Registry(args.registry)
    publish(
        args.checkpoint, module_id=args.module_id, version=args.version, role=args.role,
        group_id=args.group_id, take=take, requires=parse_requires(args.requires),
        provenance=args.provenance, description=args.description,
        attach_mode=args.attach_mode, skill=args.skill, registry=registry,
        out=args.out, overwrite=args.overwrite,
    )


if __name__ == '__main__':
    main()
