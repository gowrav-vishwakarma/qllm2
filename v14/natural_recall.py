"""NATURAL RECALL probe (Lead 2026-11-19 07:1x IST; PRE-REGISTERED in EXPERIMENTS_V14 "NATURAL RECALL"): a fact stated once in
ordinary prose, asked 2k / 8k / 16k tokens later with a natural question. The natural variant of t40 (a 20-way "Key: value"
template), per the user's realistic-tests rule.

Doc = [filler 256] [3 facts in one short prose paragraph, shuffled: TARGET, a SAME-TYPE LURE about another person, an OTHER-TYPE
fact] [held-out filler of D tokens] [question]. Five fact types (city, car colour, lesson weekday, job, pet), 3 statement wordings
each; four question styles: direct, cloze, indirect, synonym (the cue word replaced by a synonym, e.g. moved -> relocated).
Scores per doc (one teacher-forced pass for the candidate read, plus up to 4 greedy steps for the free answer):
  acc   = the gold answer's first token beats every other candidate of its type (8 per type; chance 0.125);
  lure  = the lure's answer wins instead (binding failure);
  free  = greedy decode (4 tokens) starts with the gold answer word (case-insensitive).
Same docs for every body: rng = Random(19000 * 1000 + D * 10 + style_index). D = 64 is the readability control.
    python v14/natural_recall.py --ckpt <weights> --dists 64,2048,8192,16384 --docs 200 --out logs/v14_natrec_<body>_<hash>_<stamp>.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)
from memory_probes.spread import filler_tokens  # noqa: E402
from v14.a26_protocol import FILL  # noqa: E402

NAMES = ['Marta', 'Daniel', 'Priya', 'Tomas', 'Helen', 'Victor', 'Ingrid', 'Samuel', 'Leila', 'Oscar', 'Nadia', 'Gregory',
         'Elena', 'Patrick', 'Yusuf', 'Clara', 'Arthur', 'Rosa', 'Felix', 'Irene']
# type -> (answers, statement wordings, questions by style).  {n} = name, {a} = answer.
TYPES = {
    'city': (['Lisbon', 'Vienna', 'Chicago', 'Madrid', 'Toronto', 'Berlin', 'Sydney', 'Dublin'],
             ['{n} packed up the old flat last spring and moved to {a}, where the rent was lower.',
              'After the wedding, {n} settled in {a} for good.',
              'It was {a} that {n} finally chose as a new home, after years of travelling.'],
             {'direct': 'Which city did {n} move to?\nAnswer:',
              'cloze': 'These days {n} lives in',
              'indirect': 'I keep forgetting where {n} ended up living. It was',
              'synonym': 'Which town did {n} relocate to?\nAnswer:'}),
    'colour': (['red', 'blue', 'green', 'black', 'white', 'silver', 'yellow', 'orange'],
               ['{n} bought a second-hand car last month, a {a} one with a dent in the door.',
                'The car {n} drives to work is painted {a}, which the neighbours keep teasing about.',
                'Everyone recognises {n} in the car park: the only {a} car on the street.'],
               {'direct': 'What colour is the car {n} drives?\nAnswer:',
                'cloze': 'The car that {n} drives is painted',
                'indirect': 'I could not remember the colour of {n}\'s car, but I think it was',
                'synonym': 'What shade is {n}\'s vehicle?\nAnswer:'}),
    'day': (['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'weekend'],
            ['{n} takes a piano lesson every {a}, straight after lunch.',
             'Every {a} afternoon {n} walks across town for a piano lesson.',
             'The piano teacher sees {n} once a week, on {a}.'],
            {'direct': 'On which day does {n} have a piano lesson?\nAnswer:',
             'cloze': '{n} goes to the piano lesson every',
             'indirect': 'I never remember when {n} has piano. I believe it is on',
             'synonym': 'Which day is {n}\'s keyboard class?\nAnswer:'}),
    'job': (['nurse', 'pilot', 'teacher', 'lawyer', 'chef', 'farmer', 'dentist', 'plumber'],
            ['{n} has worked as a {a} for almost ten years now.',
             'By day {n} is a {a}, and by night a keen amateur painter.',
             'Ask anyone in the village and they will tell you {n} is the local {a}.'],
            {'direct': 'What does {n} do for a living?\nAnswer: a',
             'cloze': 'By profession, {n} is a',
             'indirect': 'I forgot what {n}\'s job was. {n} works as a',
             'synonym': 'What is {n}\'s occupation?\nAnswer: a'}),
    'pet': (['cat', 'dog', 'parrot', 'rabbit', 'tortoise', 'hamster', 'goat', 'snake'],
            ['{n} keeps a {a} at home and talks about it constantly.',
             'The only company {n} has in the evenings is a pet {a}.',
             'Visitors to {n}\'s house are always greeted by the family {a}.'],
            {'direct': 'What kind of pet does {n} have?\nAnswer: a',
             'cloze': 'At home, {n} keeps a pet',
             'indirect': 'I can never recall what animal {n} owns. It is a',
             'synonym': 'Which animal companion lives with {n}?\nAnswer: a'}),
}
STYLES = ['direct', 'cloze', 'indirect', 'synonym']


def first_tokens(enc):
    """type -> {answer: first token of ' answer'}; asserts the 8 first tokens of a type are distinct."""
    out = {}
    for t, (ans, _, _) in TYPES.items():
        m = {a: enc(' ' + a)[0] for a in ans}
        assert len(set(m.values())) == len(ans), f'first-token collision in {t}: {m}'
        out[t] = m
    return out


def make_doc(rng, enc, fill, D, style):
    t = rng.choice(list(TYPES)); ans, wording, qs = TYPES[t]
    n_tgt, n_lure, n_oth = rng.sample(NAMES, 3)
    a_tgt, a_lure = rng.sample(ans, 2)
    t2 = rng.choice([x for x in TYPES if x != t]); a2 = rng.choice(TYPES[t2][0])
    facts = [rng.choice(wording).format(n=n_tgt, a=a_tgt), rng.choice(wording).format(n=n_lure, a=a_lure),
             rng.choice(TYPES[t2][1]).format(n=n_oth, a=a2)]
    rng.shuffle(facts)
    para = ' ' + ' '.join(facts)
    s0 = rng.randrange(0, len(fill) - 300 - D - 10)
    pre = [int(x) for x in fill[s0:s0 + 256]]
    gap = [int(x) for x in fill[s0 + 300:s0 + 300 + D]]
    q = qs[style].format(n=n_tgt)
    ids = pre + enc(para) + enc('\n\n') + gap + enc('\n\n' + q)
    return dict(type=t, style=style, name=n_tgt, gold=a_tgt, lure=a_lure, other_type=t2, question=q, ids=ids, D=D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--dists', default='64,2048,8192,16384')
    ap.add_argument('--docs', type=int, default=200); ap.add_argument('--device', default='cuda'); ap.add_argument('--out', required=True)
    ap.add_argument('--free_steps', type=int, default=4)
    a = ap.parse_args()
    h = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    print('HEADER ' + json.dumps(dict(unit='NATURAL RECALL', hash=h, args=vars(a))), flush=True)
    from run_memory_behavioral import _load_v13_sempty
    from run_skillpack_probe import Body
    dev = torch.device(a.device)
    m, tok, _ = _load_v13_sempty(Path(a.ckpt), 'baseline_real_pm', dev); m.eval(); body = Body(m, dev)
    print('arch', body.arch, flush=True)
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    ft = first_tokens(enc); fill = filler_tokens(str(FILL)); out = open(a.out, 'w'); summ = {}
    for D in [int(x) for x in a.dists.split(',')]:
        recs = []
        for si, style in enumerate(STYLES):
            rng = random.Random(19000 * 1000 + D * 10 + si)
            for di in range(a.docs // len(STYLES)):
                d = make_doc(rng, enc, fill, D, style); cands = ft[d['type']]
                lg = body.last_logits(d['ids'])
                cl = {w: float(lg[t]) for w, t in cands.items()}
                top = max(cl, key=cl.get)
                gen, ids = [], list(d['ids'])
                nxt = int(lg.argmax())
                for _ in range(a.free_steps):
                    gen.append(nxt); ids.append(nxt)
                    if len(gen) == a.free_steps:
                        break
                    nxt = int(body.last_logits(ids).argmax())
                text = tok.decode(gen).strip().lower()
                rec = dict(D=D, doc=di, style=style, type=d['type'], gold=d['gold'], lure=d['lure'], name=d['name'],
                           n_tokens=len(d['ids']), acc=int(top == d['gold']), lure_win=int(top == d['lure']),
                           free=int(text.startswith(d['gold'].lower())), free_text=text,
                           margin=cl[d['gold']] - max(v for w, v in cl.items() if w != d['gold']), hash=h)
                recs.append(rec); out.write(json.dumps(rec) + '\n'); out.flush()
        mean = lambda k, rs: sum(r[k] for r in rs) / max(1, len(rs))  # noqa: E731
        s = dict(acc=mean('acc', recs), free=mean('free', recs), lure=mean('lure_win', recs), n=len(recs),
                 by_style={st: dict(acc=mean('acc', [r for r in recs if r['style'] == st]),
                                    free=mean('free', [r for r in recs if r['style'] == st])) for st in STYLES})
        summ[D] = s
        print(f"D {D}: acc {s['acc']:.3f} free {s['free']:.3f} lure {s['lure']:.3f} (n {s['n']}, chance 0.125) | "
              + ' '.join(f"{st} {v['acc']:.2f}/{v['free']:.2f}" for st, v in s['by_style'].items()), flush=True)
    print('SUMMARY ' + json.dumps(summ), flush=True)
    print('NATREC DONE', flush=True)


if __name__ == '__main__':
    main()
