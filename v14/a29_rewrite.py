"""A29: the package COMPILES a prose protocol into compact demos at query time (no training).

A26 UNIT docs (seed 26000). The planner finds the protocol in the session TEXT with a parser (a sentence naming a quantity
and a unit from a fixed unit list; it never sees the test's ground truth), writes "Report rule." + 3 worked lines
"x -> x u" (random x) right before the question, takes the value out of the fetched row (A23's substitution) and the mind
answers "\\n{v} ->" with the skill contrast (twin = the same session without the inserted pack; A23 / apply_value).
Arms per doc: rewrite (that path) and a9 (A26 'held': the plain package answer), the same fetched row.
Reported: the parser's hit rate (right unit found from prose alone) beside the answers. Labelled TEXT + MIND: the
library compiles, the mind copies the value and writes the unit.
    python v14/a29_rewrite.py --ckpt <weights> --wording train|a26 --lengths 2048,8192 --docs 100 --out logs/...jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'v16')):
    if p not in sys.path:
        sys.path.insert(0, p)
from memory_probes.spread import filler_tokens  # noqa: E402
from v14.a26_protocol import EXAM, FILL, MAX_NEW, make_doc, planner_value, score  # noqa: E402
from v16.package import Package  # noqa: E402

UNITS = ['mg', 'g', 'kg', 'ng', 'μg', 'mL', 'L', 'μL', 'mM', 'μM', 'nM', 'nm', 'mm', 'cm']
QWORDS = {'mass': ['mass', 'masses', 'weight'], 'volume': ['volume', 'volumes'], 'concentration': ['concentration', 'concentrations'],
          'length': ['length', 'lengths']}
UNIT_RE = re.compile(r'\b(?:in|is|as|use|means)\s+(?:\w+\s+)?\(?(' + '|'.join(sorted(map(re.escape, UNITS), key=len, reverse=True))
                     + r')(?![A-Za-z])')   # "measured in kg", "the unit is mL", "use L as the unit", "always means g", "in milligrams (mg)"


def parse_protocol(text: str, quantity_word: str):
    """the unit a prose protocol sets for a quantity: the LAST sentence that names the quantity (or its plural) and a unit
    from UNITS right after "in" / "is" / "as" (one optional word between, e.g. "in milligrams (mg)"), else None. Text only."""
    best = None
    for sent in re.split(r'(?<=[.!?])\s+|\n+', text):
        low = sent.lower()
        if any(re.search(r'\b' + w + r'\b', low) for w in QWORDS.get(quantity_word, [quantity_word])):
            m = UNIT_RE.findall(sent)
            if m:
                best = m[-1]
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--wording', default='a26', choices=['a26', 'train', 'fresh'])
    ap.add_argument('--lengths', default='2048,8192'); ap.add_argument('--docs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=26000); ap.add_argument('--device', default='cuda'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import subprocess
    print('HEADER ' + json.dumps(dict(unit='A29', hash=subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                                                                      cwd=ROOT).stdout.strip(), args=vars(a))), flush=True)
    pkg = Package(a.ckpt, device=a.device); tok = pkg.tok
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    fill = filler_tokens(str(FILL)); outf = open(a.out, 'a')
    for d in [int(x) for x in a.lengths.split(',')]:
        rng = random.Random(a.seed * 1000 + d + 0)
        for di in range(a.docs):
            t0 = time.time()
            doc = make_doc(rng, tok, fill, 'unit', d, a.wording)
            q = EXAM.format(q=doc['question'])
            ans9, info = pkg.answer(doc['held'], q)
            row_text = info.get('row_text') or ''
            base = dict(d=d, doc=di, wording=a.wording, quantity=doc['q'], unit=doc['unit'], target=doc['target'],
                        proto=doc['proto'], question=doc['question'], path=info['path'], row=row_text[-90:])
            outf.write(json.dumps(dict(base, arm='a9', answer=ans9, **score('unit', ans9, doc))) + '\n')
            unit = parse_protocol(tok.decode(doc['held']), doc['q'])         # the library's parser, from the prose only
            v = planner_value(row_text, doc['q'])
            if unit is None or v is None:
                ans, ok = '', dict(value_ok=0, conv_ok=0, right=0)
            else:
                drng = random.Random(di * 7919 + d)
                pack = 'Report rule.' + ''.join(f'\n{x / 10:.1f} -> {x / 10:.1f} {unit}' for x in drng.sample(range(11, 999), 3))
                sess = doc['held'] + enc('\n\n') + enc(pack)
                ans, _ = pkg.apply_value(sess, {'pack': pack, 'query': '\n{v} ->'}, v, max_new=MAX_NEW)
                ok = score('unit', ans, doc)
            outf.write(json.dumps(dict(base, arm='rewrite', answer=ans, parsed_unit=unit, parser_ok=int(unit == doc['unit']),
                                       planner_value=v, **ok)) + '\n')
            outf.flush()
            print(d, di, f'{time.time() - t0:.1f} s', flush=True)
    print('A29 DONE', flush=True)


if __name__ == '__main__':
    main()
