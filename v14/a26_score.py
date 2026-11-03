"""Score A26 logs: right (value ok / convention ok) per body x arm x cell x length; UNIT split by trained / held-out quantity.
    python v14/a26_score.py logs/v14_a26_*.jsonl [more.jsonl]"""
import collections
import json
import sys

rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
G = collections.defaultdict(list)
for r in rows:
    G[(r['body'], r['arm'], r['cell'], r['d'])].append(r)
ds = sorted({r['d'] for r in rows})
f = lambda R, k: sum(r[k] for r in R) / len(R) if R else float('nan')  # noqa: E731
order = sorted({(r['body'], r['arm']) for r in rows})
QOF = {**{u: 'T' for u in ['mg', 'g', 'kg', 'ng', 'μg', 'mL', 'L', 'μL']}, **{u: 'H' for u in ['mM', 'μM', 'nM', 'nm', 'mm', 'cm']}}
for cell in ('unit', 'round', 'convert'):
    if not any(r['cell'] == cell for r in rows):
        continue
    print(f"\n== {cell}: right (value ok / convention ok)")
    print('body arm'.ljust(24) + ''.join(f'{d // 1024}k'.ljust(20) for d in ds))
    for b, a in order:
        line = f'{b} {a}'.ljust(24)
        for d in ds:
            R = G.get((b, a, cell, d), [])
            line += (f"{f(R, 'right'):.2f} ({f(R, 'value_ok'):.2f}/{f(R, 'conv_ok'):.2f}) n{len(R)}" if R else '-').ljust(20)
        print(line)
print('\n== UNIT split: trained quantities (mass / volume) | held-out (concentration / length): right')
for b, a in order:
    line = f'{b} {a}'.ljust(24)
    for d in ds:
        R = G.get((b, a, 'unit', d), [])
        tr = [r for r in R if QOF.get(r['unit']) == 'T']; ho = [r for r in R if QOF.get(r['unit']) == 'H']
        line += (f"{f(tr, 'right'):.2f} n{len(tr)} | {f(ho, 'right'):.2f} n{len(ho)}" if R else '-').ljust(24)
    print(line)
