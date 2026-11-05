"""Library's eval items for the SCI-1 13-gram decontamination filter (Mind's ask, 2026-11-02): one JSON list of strings.

Contents: A26 templates + every rendered A26 item of its registered seed (protocol sentence, question, the 8 rows; the
filler is the 5 % content-hash holdout, excluded from training by exclude_holdout, so it is NOT dumped); v16/bench.py
cell templates; v14/styles.py natural wordings (A21 / A25); v16/prompts story + long-prompt texts (the full prompts).
NOT included on purpose: v14/protocol_docs.py's TRAINING wordings (A26's G1 arm uses them), or the filter would delete
the protocol training source. Most templates are shorter than 13 words, so a 13-gram filter cannot catch them; their
protection is that no training generator uses them (protocol_docs.check(): 0 shared 5-grams with A26).

    python -m v14.decontam_items --out v14/decontam_items.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'v16')):
    if p not in sys.path:
        sys.path.insert(0, p)


def items() -> list:
    from transformers import AutoTokenizer
    from memory_probes.spread import filler_tokens
    from v14 import a26_protocol as A
    from v14 import styles as S
    tok = AutoTokenizer.from_pretrained('gpt2')
    out = []
    out += [t for fam in A.PROTO.values() for t in fam] + A.QUEST + [r.strip() for r in A.ROWS]
    fill = filler_tokens(str(A.FILL))
    for d in (2048, 8192, 32768):                                   # A26's registered docs (seed 26000)
        for ci, cell in enumerate(('unit', 'round', 'convert')):
            rng = random.Random(26000 * 1000 + d + ci)
            for _ in range(40):
                doc = A.make_doc(rng, tok, fill, cell, d)
                out += [doc['proto'], doc['question'], doc['pack']]
                out += doc['rows_text']                                     # the 8 result rows as inserted
    out += ['Definition: {k} refers to {v}.', 'Query: {k} refers to', 'Result: job: {k}; status: {v}; owner: {v}; region: {v}; label: {v}.',
            'Query: job: {k}; {f}:']
    for name in ('STATEMENTS', 'QUESTIONS', 'FIELD_QUESTIONS'):
        for x in getattr(S, name):
            out += [x] if isinstance(x, str) else [y for y in x if isinstance(y, str)]
    out += A.PROTO_FRESH + A.QUEST_FRESH + [r.strip() for r in A.ROWS_FRESH]          # A29b fresh wordings (test only)
    from v14 import simple_chat as SC                                                    # the SIMPLE-CHAT items (test only)
    out += [it[0] if isinstance(it, tuple) else ' '.join(it) for _, it, _ in SC.items()]
    for f in sorted((ROOT / 'v16' / 'prompts').glob('*.txt')):
        out.append(f.read_text())
    seen, uniq = set(), []
    for x in out:
        x = x.strip()
        if x and x not in seen:
            seen.add(x); uniq.append(x)
    return uniq


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=str(ROOT / 'v14' / 'decontam_items.json'))
    a = ap.parse_args()
    it = items()
    json.dump(it, open(a.out, 'w'), indent=0, ensure_ascii=False)
    n13 = sum(1 for x in it if len(x.split()) >= 13)
    print(f'{len(it)} strings, {n13} with >= 13 words, {sum(len(x) for x in it):,} chars -> {a.out}')
