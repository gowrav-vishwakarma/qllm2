"""V12 stage-loss registry: pluggable per-stage training objectives.

The depth-growth curriculum trains different layer groups on different skills
(grammar -> facts -> reasoning -> math -> ...), and the user asked for the
*objective* to be swappable per stage during ablations. This module is that
seam.

A ``StageLoss`` is described by:
  - ``config_overrides``: V12Config knobs applied before the stage runs. The V7
    trainer already consumes these (e.g. the recall program's gate-surprisal aux,
    the aux-loss weight, or head-gate L0 pressure), so these profiles change the
    effective training objective WITHOUT touching the trainer loop.
  - ``loss_fn`` (optional): a callable
        ``(model, input_ids, labels, loss_mask, **kw) -> (main_loss, aux_loss)``
    for a genuinely different objective. This is the forward-compat hook; the
    current V7 trainer computes chunked CE itself, so custom ``loss_fn``s are the
    documented extension point (wire into ``V7Trainer.train_epoch``) rather than
    something already spliced into the loop. ``ce`` leaves it None (trainer CE).

Register new objectives with ``register_stage_loss`` and select them from the
CLI with ``--stage_loss`` (default ``ce``).
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import torch


@dataclass
class StageLoss:
    name: str
    description: str
    config_overrides: Dict[str, object] = field(default_factory=dict)
    loss_fn: Optional[Callable[..., Tuple[torch.Tensor, torch.Tensor]]] = None


STAGE_LOSSES: Dict[str, StageLoss] = {}


def register_stage_loss(stage_loss: StageLoss) -> StageLoss:
    if stage_loss.name in STAGE_LOSSES:
        raise ValueError(f"stage loss '{stage_loss.name}' already registered")
    STAGE_LOSSES[stage_loss.name] = stage_loss
    return stage_loss


def get_stage_loss(name: str) -> StageLoss:
    if name not in STAGE_LOSSES:
        raise ValueError(
            f"Unknown stage loss '{name}'. Available: {sorted(STAGE_LOSSES)}"
        )
    return STAGE_LOSSES[name]


def stage_loss_names():
    return sorted(STAGE_LOSSES)


def apply_stage_loss(cfg, name: str) -> Dict[str, object]:
    """Apply a stage-loss profile's config overrides onto ``cfg`` in place.

    Returns the dict of applied (field -> value) for logging. Unknown config
    fields raise (fail fast rather than silently ignore a typo).
    """
    stage_loss = get_stage_loss(name)
    applied = {}
    for key, value in stage_loss.config_overrides.items():
        if not hasattr(cfg, key):
            raise ValueError(
                f"stage loss '{name}' override '{key}' is not a V12Config field"
            )
        setattr(cfg, key, value)
        applied[key] = value
    return applied


# ── Built-in profiles ───────────────────────────────────────────────────────

# Plain next-token cross-entropy (the trainer default). No overrides.
register_stage_loss(StageLoss(
    name='ce',
    description='Next-token cross-entropy (trainer default).',
))

# Recall-oriented: CE + gate-surprisal supervision so the GSP write gate learns
# when to write vs freeze. Good for a facts / knowledge stage.
register_stage_loss(StageLoss(
    name='ce_recall',
    description='CE + gate-surprisal recall program (facts/knowledge stages).',
    config_overrides={
        'gate_surprisal_lambda': 0.3,
        'gate_surprisal_tau': 0.5,
        'gate_surprisal_sign': 1.0,
    },
))

# Pruning: CE + hard-concrete L0 head-gate pressure so the effective head count
# is discovered for this stage's layers. Pair with a high n_heads budget.
register_stage_loss(StageLoss(
    name='ce_prune',
    description='CE + L0 head-gate sparsity pressure (discover head count).',
    config_overrides={
        'head_gate': True,
        'head_gate_l0_lambda': 0.001,
    },
))
