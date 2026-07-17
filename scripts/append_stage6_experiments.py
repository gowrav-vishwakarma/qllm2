#!/usr/bin/env python3
"""Append Stage-6 results section to v11/EXPERIMENTS_V11.md (idempotent)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def fmt_arms(block) -> str:
    if not block:
        return '_pending_\n'
    rows = block.get('arms') or []
    lines = [
        '| arm | recall@2048 | multi8@min | gate |',
        '|-----|-------------|------------|------|',
    ]
    for r in rows:
        lines.append(
            f"| {r.get('arm', '?')} | {r.get('recall_at_2048', r.get('status'))} "
            f"| {r.get('multi8_at_min', '—')} | {r.get('gate', '—')} |"
        )
    return '\n'.join(lines) + '\n'


def main() -> int:
    doc = Path('v11/EXPERIMENTS_V11.md')
    text = doc.read_text()
    marker = '## Recall program — Stage-6 architecture round'
    if marker in text:
        print('[final] EXPERIMENTS_V11.md already has Stage-6 section')
        return 0

    summary = Path('logs/v11/recall_stage6_final/summary.json')
    data = json.loads(summary.read_text()) if summary.exists() else {}

    comp_lines = [
        '| model | params | @2048 | multi8@min | overall |',
        '|-------|--------|-------|------------|---------|',
    ]
    for r in data.get('comparisons') or []:
        comp_lines.append(
            f"| {r['model']} | {r.get('params')} | {r.get('single_at_max')} "
            f"| {r.get('multi8_at_min')} | {r.get('overall')} |"
        )

    combo = data.get('combo_verdict') or {}
    combo_brief = {
        'ship': combo.get('ship'),
        'recall': (combo.get('behavioral') or {}).get('single_assoc_at_max_context'),
        'multi8': (combo.get('behavioral') or {}).get('multi8_at_min_context'),
    }
    day = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    section = f"""

---

{marker} ({day}, RTX PRO 6000)

Metrics unified via `scripts/behavioral_summary.py` (mean over positions, assoc=1).
`BEHAVIOR_TRIALS=60`. Architecture levers: E2 delta compile fix (`@torch.compiler.disable`
on `_complex_triangular_solve`), vault state (`--vault_state`), phase addressing
(`--write_phase_address`).

### 6b Capacity micro (~11M, 30M tok, 100% recall curriculum)

{fmt_arms(data.get('micro'))}

### 6c Architecture arms (~100M, 300M tok, fineweb+recall_w3)

{fmt_arms(data.get('stage6_arms'))}

### 6d Combo + matched baselines

{chr(10).join(comp_lines)}

Combo verdict: `{json.dumps(combo_brief)}`

### Learnings (Stage 6)

1. **Metrics fixed:** baselines summary no longer silently reports pos=1.0 only.
2. **E2 compile hang fixed** with eager island around `linalg.solve`.
3. **Vault / phase addressing** land as flag-gated PAM-native levers (selftest parallel≡recurrent).
4. See tables above for whether interference (multi8@128) and recall@2048 moved vs Stage-4/5.

Artifacts: `checkpoints_v11_recall_micro/`, `checkpoints_v11_recall_stage6/`,
`checkpoints_v11_recall_stage6_combo/`, `checkpoints_v11_recall_matched/`,
`logs/v11/recall_stage6_final/summary.json`.
"""
    doc.write_text(text.rstrip() + '\n' + section)
    print('[final] appended Stage-6 section to EXPERIMENTS_V11.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
