"""v17 external arm builders for the frozen ``v14.microbench`` harness (zero edits to v14).

Usage:
  V14_EXT_ARMS="pow=v17.ext:build_pow;pow_sync=v17.ext:build_pow_sync;pow_work=v17.ext:build_pow_work" \
    .venv/bin/python -u -m v14.microbench --arms pow,echo,tpp ...

Matches ``V14LM(cfg)``'s exact parameter count without advancing the global RNG,
just as ``v15.bench:build_grove`` and ``v14.microbench:build_model('tpp')`` do.
"""

from __future__ import annotations

import torch

from v14.units import V14LM
from v17.model import POWConfig, POWLM, _solve_mlp_hidden


def _build_matched(v14_cfg, label: str, **overrides) -> POWLM:
    with torch.random.fork_rng(devices=[]):
        target = sum(p.numel() for p in V14LM(v14_cfg).parameters() if p.requires_grad)

    base_kw = dict(
        vocab_size=v14_cfg.vocab_size,
        dim=v14_cfg.dim,
        n_layers=v14_cfg.n_layers,
        n_heads=v14_cfg.n_heads,
        head_dim=v14_cfg.head_dim,
        op_rings=max(48, (v14_cfg.dim // 4 // v14_cfg.n_heads) * v14_cfg.n_heads),
        op_heads=v14_cfg.n_heads,
        **overrides,
    )
    mlp = _solve_mlp_hidden(lambda mh: POWLM(POWConfig(mlp_hidden=mh, **base_kw)), target)
    model = POWLM(POWConfig(mlp_hidden=mlp, **base_kw))
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [{label}] dim={v14_cfg.dim} L={v14_cfg.n_layers} H={v14_cfg.n_heads}x{v14_cfg.head_dim} "
          f"mlp={mlp}: {n_p / 1e6:.2f}M params (echo {target / 1e6:.2f}M, {(n_p - target) / target:+.3%}), "
          f"O(1) state {model.state_bytes(1) / 1024:.1f} KB/seq", flush=True)
    return model


def build_pow(v14_cfg) -> POWLM:
    """Full Two-Organ POWLM (DualPhase S^1 + 3-Bounce Phase-Closure Workspace + Causal Operator)."""
    return _build_matched(v14_cfg, "pow_v17")


def build_pow_sync(v14_cfg) -> POWLM:
    """Ablation 1: Organ 1 (DualPhase S^1) only, 1-pass, no Operator Engine."""
    return _build_matched(v14_cfg, "pow_sync",
                          sync_phasor=True, n_bounces=1, operator_engine=False)


def build_pow_work(v14_cfg) -> POWLM:
    """Ablation 2: Organ 2 (2-Bounce + Causal Operator) only, Chrono without S^1 Synchrony."""
    return _build_matched(v14_cfg, "pow_work",
                          sync_phasor=False, n_bounces=2, operator_engine=True)
