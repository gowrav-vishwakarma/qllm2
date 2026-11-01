"""Render EXPERIMENT_TREE.md (repo root) from the per-version lineage files (user, 2026-11-01 19:00 IST: "a tree
that shows what extended what and the result, so we know the journey, the branch points, and never miss an
experiment, its result, commit hash, log and result file").

Each version keeps `<version>/EXPERIMENT_LINEAGE.tsv`, tab-separated, one row per experiment, header:
    id  parent  date  what  verdict  run_hash  code_at  log_or_result  section
- id       short id as in that version's EXPERIMENT_INDEX.md ("M2", "B1 run 4", "A9").
- parent   the experiment this one EXTENDED or was BRANCHED from: "<id>" in the same version, or "<version>:<id>"
           across versions ("v13_sempty:E1 echo"). Empty = a root of its version; "?" = parent not yet known.
- date     YYYY-MM-DD of the verdict (or the launch).
- what     one line: what was tried (the full write-up stays in the notebook section).
- verdict  one line: KEEP / FAIL / MET / NOT met / running + the key number(s) with params + tokens where they matter.
- run_hash the commit the run was launched from (checkpoints / logs carry it).
- code_at  the last commit that still HAS the code (for removed paths), else "current" or empty.
- log_or_result   the log / jsonl / report file (repo-relative) or "see section".
- section  the notebook section (EXPERIMENTS_*.md) or board entry that holds the write-up.
Rows may be added by any session; the tree is re-rendered with `python3 scripts/exp_tree.py` and committed.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ['v13_sempty', 'v14', 'v15', 'v16']
COLS = ['id', 'parent', 'date', 'what', 'verdict', 'run_hash', 'code_at', 'log_or_result', 'section']


def load():
    nodes = {}
    for v in VERSIONS:
        p = ROOT / v / 'EXPERIMENT_LINEAGE.tsv'
        if not p.exists():
            continue
        with open(p, newline='') as f:
            rd = csv.DictReader(f, delimiter='\t')
            missing = [c for c in COLS if c not in rd.fieldnames]
            if missing:
                sys.exit(f'{p}: missing columns {missing}')
            for r in rd:
                if not r['id'].strip():
                    continue
                key = f"{v}:{r['id'].strip()}"
                if key in nodes:
                    sys.exit(f'duplicate id {key}')
                r = {k: (r[k] or '').strip() for k in COLS}
                r['version'] = v
                nodes[key] = r
    return nodes


def resolve_parent(nodes, key):
    r = nodes[key]
    p = r['parent']
    if p in ('', '?'):
        return None
    if ':' not in p:
        p = f"{r['version']}:{p}"
    if p not in nodes:
        print(f'WARN {key}: parent {p!r} not found; shown as a root', file=sys.stderr)
        return None
    return p


def render(nodes):
    children = {k: [] for k in nodes}
    roots = {v: [] for v in VERSIONS}
    unknown = []
    for k in nodes:
        p = resolve_parent(nodes, k)
        if p is None:
            roots[nodes[k]['version']].append(k)
            if nodes[k]['parent'] == '?':
                unknown.append(k)
        else:
            children[p].append(k)
    for lst in list(children.values()) + list(roots.values()):
        lst.sort(key=lambda k: (nodes[k]['date'], nodes[k]['id']))

    out = ['# Experiment tree: what extended what, with the result, hashes and files',
           '',
           'Rendered by `python3 scripts/exp_tree.py` from `<version>/EXPERIMENT_LINEAGE.tsv` (edit the TSV, not this file).',
           'Each node: **id** (date) what -> **verdict**; run `hash` / code at `hash`; log / result; notebook section.',
           'A child is indented under the experiment it extended or branched from; `<- vX:id` marks a branch from another version.',
           'Params and tokens are in the verdict where they matter; claims only within a tokens tier (user rule 2026-11-01).',
           '']
    n_nodes = 0

    def line(k, depth):
        nonlocal n_nodes
        n_nodes += 1
        r = nodes[k]
        hs = []
        if r['run_hash']:
            hs.append(f"run `{r['run_hash']}`")
        if r['code_at'] and r['code_at'] != r['run_hash']:
            hs.append(f"code at `{r['code_at']}`")
        cross = ''
        if ':' in r['parent'] and not r['parent'].startswith(r['version'] + ':'):
            cross = f" <- {r['parent']}"
        parts = [f"**{r['id']}**{cross}" + (f" ({r['date']})" if r['date'] else ''), r['what'], f"-> **{r['verdict']}**"]
        tail = '; '.join(x for x in [', '.join(hs), r['log_or_result'], r['section']] if x)
        s = '  ' * depth + '- ' + ' '.join(p for p in parts if p) + (f"; {tail}" if tail else '')
        out.append(s)
        for c in children[k]:
            line(c, depth + 1)

    for v in VERSIONS:
        if not roots[v] and not any(nodes[k]['version'] == v for k in nodes):
            continue
        out.append(f'## {v}')
        out.append('')
        if not roots[v]:
            out.append(f'(every {v} experiment branches from another version; find them by `<- {v}:` is not used, look for ids under their parents above)')
        for k in roots[v]:
            line(k, 0)
        out.append('')
    if unknown:
        out.append('## Parent not yet known (`parent` = ?)')
        out.append('')
        out.extend(f"- {k}" for k in sorted(unknown))
        out.append('')
    counts = {v: sum(1 for k in nodes if nodes[k]['version'] == v) for v in VERSIONS}
    out.append(f'{n_nodes} experiments: ' + ', '.join(f'{v} {c}' for v, c in counts.items() if c) + '.')
    return '\n'.join(out) + '\n'


if __name__ == '__main__':
    nodes = load()
    text = render(nodes)
    (ROOT / 'EXPERIMENT_TREE.md').write_text(text)
    print(f'EXPERIMENT_TREE.md: {len(nodes)} nodes')
