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

from v8.backbone import V7Config


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

    # ── Rethink-plan additions (see v8_classical_rethink_44e4a93c.plan.md) ──
    use_complex: bool = True
    """Real-vs-complex SPM ablation. False forces imag=0 throughout the QLC."""
    out_scale_init: float = 0.1
    """Initial value of the QLC->backbone residual scale."""
    out_scale_learnable: bool = True
    """If False, ``out_scale`` is a frozen buffer (used by the out_scale sweep)."""
    renormalize_psi: bool = True
    """If False, skip the unit-sphere renorm after each Sasaki update."""
    halt_mode: str = "ortho"
    """One of: ``ortho`` (legacy), ``mlp``, ``delta``, ``entropy``."""
    unsharp_target: bool = False
    """Use unsharp E = sigma(g) u u^H in OrthoHalt so gamma carries the gate deficit."""
    quantale_order_test: bool = False
    """When True (with ``quantale_off=True``), test true PQ vs QP ordering."""
    infonce_weight: float = 0.0
    """Weight of the bank's InfoNCE auxiliary loss in Stage B."""
    infonce_every: int = 1
    """Run the InfoNCE auxiliary every N steps (1 = every step)."""
    target_alignment_weight: float = 0.0
    """v8.2: weight of the (1 - mean |u^H psi|^2) auxiliary that pulls the
    OrthoHalt target_mlp's u toward psi.

    Without this, ``|u^H psi|^2 ~= 1/d`` for the lifetime of training (random
    direction in d-space) and the alpha/gamma channels stay pinned at the noise
    floor — see AUDIT_V8.md §6 and the 4450ff0 e2e medium negative-result
    write-up in EXPERIMENTS_V8.md §10. Enabled by default in the
    ``e2e_*_reasoning`` presets at 0.05; set to 0.0 for legacy / ablation runs.
    """
    target_alignment_target: float = 1.0
    """Target for the alignment auxiliary.

    ``1.0`` preserves the v8.2 hinge loss. Lower values use an MSE target so
    alignment rises off the 1/d floor without collapsing beta to zero.
    """

    # ── V8.3 context-reasoning additions ───────────────────────────────────
    use_iteration_context: bool = False
    """Add a learned iteration embedding to the QLC probe/target input."""
    use_layer_context: bool = False
    """Add a learned backbone-layer embedding when QLC is interleaved."""
    rope_conditioned_probe: bool = False
    """Apply a RoPE-style phase rotation to the bank probe query."""
    weighted_projector: bool = False
    """Use selected bank gates to make the SPM value operator unsharp."""
    max_reason_iters: int = 16
    """Embedding table size for iteration-aware target/probe conditioning."""
    max_context_layers: int = 64
    """Embedding table size for layer-aware interleaved QLC conditioning."""

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
    unfreeze_lm_head: bool = False    # if True, lm_head_proj/norm stay trainable in Stage B
    kl_anchor_weight: float = 0.0     # Stage C uses small >0 (e.g. 0.05)

    # V8.3 interleaving controls. Empty tuple preserves legacy post-backbone
    # placement. Layer indices are zero-based V7Block indices after which QLC
    # is applied before the next block.
    qlc_insert_layers: Tuple[int, ...] = ()
    shared_interleaved_qlc: bool = True
    interleaved_out_scale: float = 0.05
    final_qlc_enabled: bool = True


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
        gradient_checkpointing=False,
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

    # ── Stage B sharpened ablations (rethink-plan §G discriminator suite) ──
    # All medium-backbone Stage B presets that toggle ONE thing at a time vs
    # the canonical "stageB_T2" baseline. Run on the 4090; each row is a
    # decisive go/no-go test.

    # G.1 Equal-FLOP baseline: passthrough but with QLC structurally absent.
    # Pair with --epochs N where N is ~T_max * baseline_epochs to spend the
    # same FLOPs the QLC rows would consume.
    "stageB_equal_flop_passthrough": V8Config(
        backbone=_backbone_medium_v3(),
        qlc=QLCConfig(enabled=False),
        stage="B", freeze_backbone=True,
    ),

    # G.2 Real-vs-complex SPM. Same QLC config as stageB_T2 but use_complex=False.
    "stageB_real_spm": V8Config(
        backbone=_backbone_medium_v3(),
        qlc=QLCConfig(
            enabled=True, rank=8, bank_size=2048, top_k=4, t_max=2,
            use_complex=False,
        ),
        stage="B", freeze_backbone=True,
    ),

    # G.3 Rank sweep (r in {1, 2, 4, 8, 16}) — see if r=1 captures most of the gain.
    "stageB_rank_r1": V8Config(
        backbone=_backbone_medium_v3(),
        qlc=QLCConfig(enabled=True, rank=1, bank_size=2048, top_k=1, t_max=2),
        stage="B", freeze_backbone=True,
    ),
    "stageB_rank_r2": V8Config(
        backbone=_backbone_medium_v3(),
        qlc=QLCConfig(enabled=True, rank=2, bank_size=2048, top_k=2, t_max=2),
        stage="B", freeze_backbone=True,
    ),
    "stageB_rank_r4": V8Config(
        backbone=_backbone_medium_v3(),
        qlc=QLCConfig(enabled=True, rank=4, bank_size=2048, top_k=4, t_max=2),
        stage="B", freeze_backbone=True,
    ),
    # r8 already covered by stageB_T2; r16 by stageB_r16 (T_max=1 there).

    # G.4 out_scale sweep — pin out_scale at fixed values; if 0.0 matches the
    # learnable case, the QLC residual is just noise (audit §4).
    "stageB_outscale0": V8Config(
        backbone=_backbone_medium_v3(),
        qlc=QLCConfig(
            enabled=True, rank=8, bank_size=2048, top_k=4, t_max=2,
            out_scale_init=0.0, out_scale_learnable=False,
        ),
        stage="B", freeze_backbone=True,
    ),
    "stageB_outscale01": V8Config(
        backbone=_backbone_medium_v3(),
        qlc=QLCConfig(
            enabled=True, rank=8, bank_size=2048, top_k=4, t_max=2,
            out_scale_init=0.1, out_scale_learnable=False,
        ),
        stage="B", freeze_backbone=True,
    ),
    "stageB_outscale1": V8Config(
        backbone=_backbone_medium_v3(),
        qlc=QLCConfig(
            enabled=True, rank=8, bank_size=2048, top_k=4, t_max=2,
            out_scale_init=1.0, out_scale_learnable=False,
        ),
        stage="B", freeze_backbone=True,
    ),

    # G.6 Empirical halt heads — DeltaHalt and EntropyHalt replace OrthoHalt.
    "stageB_halt_delta": V8Config(
        backbone=_backbone_medium_v3(),
        qlc=QLCConfig(
            enabled=True, rank=8, bank_size=2048, top_k=4, t_max=4,
            halt_mode="delta",
        ),
        stage="B", freeze_backbone=True,
    ),
    "stageB_halt_entropy": V8Config(
        backbone=_backbone_medium_v3(),
        qlc=QLCConfig(
            enabled=True, rank=8, bank_size=2048, top_k=4, t_max=4,
            halt_mode="entropy",
        ),
        stage="B", freeze_backbone=True,
    ),

    # OrthoHalt with the unsharp target gate (rethink §1) — gives gamma a
    # geometrically meaningful non-zero floor so the algebraic readout is
    # actually distinguishable from MLPHalt.
    "stageB_unsharp_ortho": V8Config(
        backbone=_backbone_medium_v3(),
        qlc=QLCConfig(
            enabled=True, rank=8, bank_size=2048, top_k=4, t_max=4,
            unsharp_target=True,
        ),
        stage="B", freeze_backbone=True,
    ),

    # True quantale-ordering test (rethink §2). Compare the LM PPL of this
    # row to the stageB_T4 baseline; if order matters, this should regress.
    "stageB_quantale_order_off": V8Config(
        backbone=_backbone_medium_v3(),
        qlc=QLCConfig(
            enabled=True, rank=8, bank_size=2048, top_k=4, t_max=4,
            quantale_off=True, quantale_order_test=True,
        ),
        stage="B", freeze_backbone=True,
    ),

    # LM-head-only-unfrozen Stage B (rethink §6). Lets the readout adapt to
    # the QLC's new state geometry; isolates "QLC weak" from "frozen readout".
    "stageB_lmhead_unfrozen": V8Config(
        backbone=_backbone_medium_v3(),
        qlc=QLCConfig(enabled=True, rank=8, bank_size=2048, top_k=4, t_max=2),
        stage="B", freeze_backbone=True, unfreeze_lm_head=True,
    ),

    # InfoNCE-on Stage B (wires the bank's auxiliary objective; rethink §5).
    "stageB_infonce_on": V8Config(
        backbone=_backbone_medium_v3(),
        qlc=QLCConfig(
            enabled=True, rank=8, bank_size=2048, top_k=4, t_max=2,
            infonce_weight=0.1, infonce_every=4,
        ),
        stage="B", freeze_backbone=True,
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

    # ── Smoke variants of the discriminator suite ──
    # These run on the tiny backbone so the rethink-plan §G suite can be
    # validated end-to-end on TinyStories before any 4090/A100 spend.
    "smoke_tiny_real_spm": V8Config(
        backbone=_backbone_tiny(),
        qlc=QLCConfig(
            enabled=True, rank=4, bank_size=128, top_k=2, t_max=2,
            use_complex=False, ponder_lambda=0.01,
        ),
        stage="A", freeze_backbone=False,
    ),
    "smoke_tiny_outscale0": V8Config(
        backbone=_backbone_tiny(),
        qlc=QLCConfig(
            enabled=True, rank=4, bank_size=128, top_k=2, t_max=2,
            out_scale_init=0.0, out_scale_learnable=False, ponder_lambda=0.01,
        ),
        stage="A", freeze_backbone=False,
    ),
    "smoke_tiny_unsharp": V8Config(
        backbone=_backbone_tiny(),
        qlc=QLCConfig(
            enabled=True, rank=4, bank_size=128, top_k=2, t_max=2,
            unsharp_target=True, ponder_lambda=0.01,
        ),
        stage="A", freeze_backbone=False,
    ),
    "smoke_tiny_halt_delta": V8Config(
        backbone=_backbone_tiny(),
        qlc=QLCConfig(
            enabled=True, rank=4, bank_size=128, top_k=2, t_max=2,
            halt_mode="delta", ponder_lambda=0.01,
        ),
        stage="A", freeze_backbone=False,
    ),
    "smoke_tiny_quantale_order_off": V8Config(
        backbone=_backbone_tiny(),
        qlc=QLCConfig(
            enabled=True, rank=4, bank_size=128, top_k=2, t_max=4,
            quantale_off=True, quantale_order_test=True, ponder_lambda=0.01,
        ),
        stage="A", freeze_backbone=False,
    ),

    # ── End-to-end single-run presets (single-run-v8-training plan v2) ──
    # One model, one continuous run from random init. No Stage A/B/C handoff.
    # Uses ``unsharp_target=True`` so ``gamma`` is non-trivial (AUDIT_V8 §1).
    # ``kl_anchor_weight`` MUST stay 0: there is no Stage A reference logits
    # in a random-init single run (`backbone_logits()` would otherwise pull
    # the model toward its own untrained logits).
    "e2e_medium_reasoning": V8Config(
        backbone=_backbone_medium_v3(),
        qlc=QLCConfig(
            enabled=True,
            rank=8, bank_size=2048, top_k=4,
            t_max=4,
            ponder_lambda=0.01,
            unsharp_target=True,
            out_scale_init=0.05,
            out_scale_learnable=True,
            renormalize_psi=True,
            halt_mode="ortho",
            # v8.2 fix bundle (see AUDIT_V8.md §6 + EXPERIMENTS_V8.md §10):
            # without these the OrthoHalt target_mlp stays at random direction
            # and alpha/gamma sit at the 1/d noise floor for the entire run.
            target_alignment_weight=0.05,
            infonce_weight=0.05,
            infonce_every=4,
        ),
        stage="A",
        freeze_backbone=False,
        unfreeze_lm_head=False,
        kl_anchor_weight=0.0,
    ),

    # TinyStories smoke mirror of e2e_medium_reasoning. Same QLC flags so the
    # recipe is sanity-gated end-to-end before any medium run is launched.
    "e2e_tiny_reasoning": V8Config(
        backbone=_backbone_tiny(),
        qlc=QLCConfig(
            enabled=True,
            rank=4, bank_size=128, top_k=2,
            t_max=4,
            ponder_lambda=0.01,
            unsharp_target=True,
            out_scale_init=0.05,
            out_scale_learnable=True,
            renormalize_psi=True,
            halt_mode="ortho",
            # v8.2 fix bundle — same rationale as e2e_medium_reasoning above.
            target_alignment_weight=0.05,
            infonce_weight=0.05,
            infonce_every=4,
        ),
        stage="A",
        freeze_backbone=False,
        unfreeze_lm_head=False,
        kl_anchor_weight=0.0,
    ),

    # ── V8.3 interleaved context-reasoning presets ──
    # QLC is injected before later PAM layers so memory-conditioned states can
    # participate in sequence mixing instead of acting only as a final adapter.
    "e2e_medium_context_reasoning": V8Config(
        backbone=_backbone_medium_v3(),
        qlc=QLCConfig(
            enabled=True,
            rank=8, bank_size=2048, top_k=4,
            t_max=4,
            ponder_lambda=0.01,
            unsharp_target=True,
            out_scale_init=0.05,
            out_scale_learnable=True,
            renormalize_psi=False,
            halt_mode="ortho",
            target_alignment_weight=0.05,
            infonce_weight=0.05,
            infonce_every=4,
            use_iteration_context=True,
            use_layer_context=True,
            rope_conditioned_probe=True,
            weighted_projector=True,
            target_alignment_target=0.75,
        ),
        stage="A",
        freeze_backbone=False,
        unfreeze_lm_head=False,
        kl_anchor_weight=0.0,
        qlc_insert_layers=(3, 7, 11),
        shared_interleaved_qlc=True,
        interleaved_out_scale=0.05,
        final_qlc_enabled=False,
    ),
    "e2e_tiny_context_reasoning": V8Config(
        backbone=_backbone_tiny(),
        qlc=QLCConfig(
            enabled=True,
            rank=4, bank_size=128, top_k=2,
            t_max=4,
            ponder_lambda=0.01,
            unsharp_target=True,
            out_scale_init=0.05,
            out_scale_learnable=True,
            renormalize_psi=False,
            halt_mode="ortho",
            target_alignment_weight=0.05,
            infonce_weight=0.05,
            infonce_every=4,
            use_iteration_context=True,
            use_layer_context=True,
            rope_conditioned_probe=True,
            weighted_projector=True,
            target_alignment_target=0.75,
        ),
        stage="A",
        freeze_backbone=False,
        unfreeze_lm_head=False,
        kl_anchor_weight=0.0,
        qlc_insert_layers=(0,),
        shared_interleaved_qlc=True,
        interleaved_out_scale=0.05,
        final_qlc_enabled=False,
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
        unfreeze_lm_head=cfg.unfreeze_lm_head,
        kl_anchor_weight=cfg.kl_anchor_weight,
        qlc_insert_layers=tuple(cfg.qlc_insert_layers),
        shared_interleaved_qlc=cfg.shared_interleaved_qlc,
        interleaved_out_scale=cfg.interleaved_out_scale,
        final_qlc_enabled=cfg.final_qlc_enabled,
    )
