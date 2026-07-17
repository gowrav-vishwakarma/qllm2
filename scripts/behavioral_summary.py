#!/usr/bin/env python3
"""Shared behavioral-probe aggregation for recall-program metrics.

Canonical definition of single-assoc accuracy at a context length:
  mean over all target_position cells with associations==1 at that context.

This matches scripts/recall_gate_verdict.py and must be used by every
summarizer (baselines, status scripts, pickers) so numbers cannot diverge.
"""
from __future__ import annotations


def acc_where(aggs: list, **kw) -> float | None:
    rows = [a for a in aggs if all(a.get(k) == v for k, v in kw.items())]
    if not rows:
        return None
    return sum(r['accuracy'] for r in rows) / len(rows)


def behavioral_summary(behavior: dict) -> dict:
    """Summarize a memory_probes behavioral JSON artifact."""
    aggs = behavior.get('aggregates', [])
    if not aggs:
        return {}
    contexts = sorted({a['context_tokens'] for a in aggs})
    max_ctx = contexts[-1]
    overall = sum(a['accuracy'] for a in aggs) / len(aggs)
    single_by_ctx = {c: acc_where(aggs, context_tokens=c, associations=1) for c in contexts}
    # Multi-assoc at shortest context (interference diagnostic).
    min_ctx = contexts[0]
    multi8_at_min = acc_where(aggs, context_tokens=min_ctx, associations=8)
    return {
        'contexts': contexts,
        'max_context': max_ctx,
        'overall_accuracy': overall,
        'single_assoc_by_context': single_by_ctx,
        'single_assoc_at_max_context': single_by_ctx.get(max_ctx),
        'multi8_at_min_context': multi8_at_min,
        'min_context': min_ctx,
    }
