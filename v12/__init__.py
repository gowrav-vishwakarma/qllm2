"""V12: Learnable phase-band heads on the Phase-Associative Memory (PAM) core.

Leaner clone of V11 that carries forward only the proven winners (complex CGU +
PAM + RoPE + GSP + ModSwish, chunked dual-form training, O(1) recurrent
inference, E3 K=3 multistate phase-interference retrieval, phase-aware GSP gate,
fused E3 + chunked CE) and adds three unified, novel mechanisms:

- M1  Learnable phase-band heads: a max head budget ``h_max`` where each head
      slot has a learned phase center + a hard-concrete on/off gate, so the
      effective number of heads (and which dims feed each head, by phase) is
      learned rather than hand-defined.
- M2  Progressive frozen-head growth: staged curriculum (grammar -> facts ->
      reasoning) that freezes trained head slots and activates new ones.
- M3  Low-interference write/addressing: fact-band head slots specialised with
      vault (no-decay) states, phase-addressed and delta-rule writes.
- M4  Depth-growth framework: spec-driven non-uniform stacks (``layer_specs``) with
      grow/freeze/manifest APIs, a small grammar base plus attachable specialist
      layer groups (facts, reasoning, math, ..., C, Java) grown in any order, a
      pluggable per-stage loss, head compaction, and pack-time composition with an
      ``attach_mode`` (sequential now; MoE router reserved).

Identity preserved: complex/phase-first, matrix-state associative memory with
complex-conjugate retrieval. NOT a transformer, NOT a vector-state SSM.
Inference stays O(1)/token with bounded state (no KV cache).
"""

from v12.model import (
    V12Config,
    V12LM,
    V12PAMLayer,
    V12Block,
    get_config,
    PRESETS,
)

__all__ = [
    "V12Config",
    "V12LM",
    "V12PAMLayer",
    "V12Block",
    "get_config",
    "PRESETS",
]
