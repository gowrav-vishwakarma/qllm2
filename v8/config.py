"""V8 configuration: V7 QPAM backbone + Quantum-Logic Core (QLC).

The V8 model = a V7Block stack (the "grammar/sequence" backbone reused without
modification) + a Quantum-Logic Core sitting between the backbone hidden state
and the LM head, optionally followed by joint Stage C fine-tuning.

QLC parameters live alongside the existing V7 ones. See the plan
(.cursor/plans/v8_quantum_logic_core_8df73232.plan.md) for the architectural
rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Tuple, Dict

from v7.model import V7Config


# ── QLC subconfig ────────────────────────────────────────────────────────────

@dataclass
class QLCConfig:
    """Quantum-Logic Core hyperparameters.

    Set ``enabled=False`` to make V8LM behave identically to a V7LM (the
    "passthrough" V8-A row in the ablation matrix).
    """

    enabled: bool = True

    # Sasaki Projection Memory (per-iteration facts projector)
    rank: int = 8                     # r — number of orthonormal columns in Pi_F
    n_heads: int = 1                  # multi-head reasoning (each head has its own SPM)

    # Effect-algebra bank
    bank_size: int = 2048             # M — number of rank-1 effects in the bank
    top_k: int = 4                    # how many effects feed Pi_F per iteration
    bank_temperature: float = 1.0     # softmax temperature on probe scores

    # Reasoning loop
    t_max: int = 4                    # max iterations of (Probe -> Sasaki -> Halt)
    ponder_lambda: float = 0.01       # ACT-style ponder cost coefficient
    eps_no_op: float = 1e-3           # halt-yes mass below this -> treated as no-op

    # Ablation switches (set by V8-F / V8-G rows; see plan §6)
    quantale_off: bool = False        # symmetrize Sasaki update -> 1/2 (PQP + QPQ)
    orthohalt_off: bool = False       # replace OrthoHalt with a plain MLP halt head
    use_infonce: bool = True          # Stage B InfoNCE auxiliary on selected effects

    # Numerical safety
    qr_refresh_every: int = 0         # 0 = never; >0 = re-QR Pi_F columns every K iters
    sasaki_eps: float = 1e-6


@dataclass
class V8Config:
    """V8 = backbone V7Config + QLCConfig.

    The backbone field is a *full* V7Config so existing presets and code paths
    work unchanged. ``qlc`` carries the new primitive's hyperparameters.

    Stage flag controls trainer behavior:
      * ``A``: train backbone only, QLC bypassed (passthrough).
      * ``B``: backbone frozen, QLC trainable.
      * ``C``: joint fine-tune (low LR + KL anchor to Stage A logits).
    """

    backbone: V7Config = field(default_factory=V7Config)
    qlc: QLCConfig = field(default_factory=QLCConfig)
    stage: str = "A"                  # 'A' | 'B' | 'C'
    freeze_backbone: bool = False     # Stage B sets True
    kl_anchor_weight: float = 0.0     # Stage C uses small >0 (e.g. 0.05)


# ── Presets ──────────────────────────────────────────────────────────────────

# Reuse v7 backbone presets and add QLC defaults sized to fit alongside.

def _backbone_medium_v3() -> V7Config:
    """Mirrors v6 medium-pam-v3: dim=384, 16 layers, 6 heads, head_dim=64.

    This is the apples-to-apples grammar baseline targeted by Stage A.
    """
    return V7Config(
        vocab_size=50257, dim=384, n_heads=6, head_dim=64,
        n_layers=16, expand=3, dropout=0.1, max_seq_len=2048,
        hierarchical_dt=False, cross_level=False,
        use_rope=True, use_gsp=True, fused_qkv=True, qk_norm=False,
        chunk_size=256, multi_scale_loss=False, use_reverse_assoc=False,
        gradient_checkpointing=True,
    )


def _backbone_tiny() -> V7Config:
    """~2M-param TinyStories smoke backbone."""
    return V7Config(
        vocab_size=50257, dim=64, n_heads=2, head_dim=32,
        n_layers=2, expand=2, dropout=0.0, max_seq_len=512,
        hierarchical_dt=False, cross_level=False,
        use_rope=True, use_gsp=True, fused_qkv=True, qk_norm=False,
        chunk_size=0, multi_scale_loss=False, use_reverse_assoc=False,
        gradient_checkpointing=False,
    )


PRESETS: Dict[str, V8Config] = {
    # ── Stage A target: the QPAM backbone we must reproduce (~29.95 PPL on WT103) ──
    "stageA_medium": V8Config(
        backbone=_backbone_medium_v3(),
        qlc=QLCConfig(enabled=False),  # passthrough during Stage A
        stage="A",
        freeze_backbone=False,
    ),
    # ── Stage B rows from §6 of the plan ──
    "stageB_r4": V8Config(
        backbone=_backbone_medium_v3(),
        qlc=QLCConfig(enabled=True, rank=4, bank_size=2048, top_k=2, t_max=1),
        stage="B", freeze_backbone=True,
    ),
    "stageB_r8": V8Config(
        backbone=_backbone_medium_v3(),
        qlc=QLCConfig(enabled=True, rank=8, bank_size=2048, top_k=4, t_max=1),
        stage="B", freeze_backbone=True,
    ),
    "stageB_r16": V8Config(
        backbone=_backbone_medium_v3(),
        qlc=QLCConfig(enabled=True, rank=16, bank_size=2048, top_k=4, t_max=1),
        stage="B", freeze_backbone=True,
    ),
    "stageB_M2k": V8Config(
        backbone=_backbone_medium_v3(),
        qlc=QLCConfig(enabled=True, rank=8, bank_size=2048, top_k=4, t_max=1),
        stage="B", freeze_backbone=True,
    ),
    "stageB_M8k": V8Config(
        backbone=_backbone_medium_v3(),
        qlc=QLCConfig(enabled=True, rank=8, bank_size=8192, top_k=4, t_max=1),
        stage="B", freeze_backbone=True,
    ),
    "stageB_T2": V8Config(
        backbone=_backbone_medium_v3(),
        qlc=QLCConfig(enabled=True, rank=8, bank_size=2048, top_k=4, t_max=2),
        stage="B", freeze_backbone=True,
    ),
    "stageB_T4": V8Config(
        backbone=_backbone_medium_v3(),
        qlc=QLCConfig(enabled=True, rank=8, bank_size=2048, top_k=4, t_max=4),
        stage="B", freeze_backbone=True,
    ),
    "stageB_F_quantale_off": V8Config(
        backbone=_backbone_medium_v3(),
        qlc=QLCConfig(
            enabled=True, rank=8, bank_size=2048, top_k=4, t_max=4,
            quantale_off=True,
        ),
        stage="B", freeze_backbone=True,
    ),
    "stageB_G_orthohalt_off": V8Config(
        backbone=_backbone_medium_v3(),
        qlc=QLCConfig(
            enabled=True, rank=8, bank_size=2048, top_k=4, t_max=4,
            orthohalt_off=True,
        ),
        stage="B", freeze_backbone=True,
    ),
    # ── Stage C joint fine-tune (top configs only, decided after Stage B) ──
    "stageC_T4_joint": V8Config(
        backbone=_backbone_medium_v3(),
        qlc=QLCConfig(enabled=True, rank=8, bank_size=2048, top_k=4, t_max=4),
        stage="C", freeze_backbone=False, kl_anchor_weight=0.05,
    ),
    # ── TinyStories smoke (Stage A.5 gate) ──
    "smoke_tiny_passthrough": V8Config(
        backbone=_backbone_tiny(),
        qlc=QLCConfig(enabled=False),
        stage="A", freeze_backbone=False,
    ),
    "smoke_tiny_qlc_r4_T2": V8Config(
        backbone=_backbone_tiny(),
        qlc=QLCConfig(
            enabled=True, rank=4, bank_size=128, top_k=2, t_max=2,
            ponder_lambda=0.01,
        ),
        stage="A", freeze_backbone=False,  # train end-to-end on smoke
    ),
}


def get_config(preset: str) -> V8Config:
    """Return a deep-copied V8Config preset (safe to mutate)."""
    if preset not in PRESETS:
        raise ValueError(
            f"Unknown V8 preset '{preset}'. Available: {sorted(PRESETS.keys())}"
        )
    cfg = PRESETS[preset]
    # dataclass replace gives a shallow copy of the top level; backbone/qlc are
    # immutable-ish dataclasses so we replace each.
    return V8Config(
        backbone=replace(cfg.backbone),
        qlc=replace(cfg.qlc),
        stage=cfg.stage,
        freeze_backbone=cfg.freeze_backbone,
        kl_anchor_weight=cfg.kl_anchor_weight,
    )
