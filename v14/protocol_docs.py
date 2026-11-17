"""Protocol docs: a synthetic TRAINING source for the SCI-1 mind (Lead's YES 2026-11-02 ~11:00 IST; A26's trained skill).

A research-notebook document states ONE reporting convention (a protocol), then prose and 3-6 result rows, then 1-3
questions whose answers APPLY the convention to a row's value (answer spans marked with v7.data's ANSWER_MARK, so they
get the answer loss weight like the recall / b2 docs). 20 % of docs have NO protocol, and their answers copy the row
as is: the convention must be read, not assumed.

TRAINING families (the only ones generated):
    UNIT     mass (mg, g, kg, ng, μg), volume (mL, L, μL): "values are written without the unit, which is u"
    ROUND    to the nearest whole number; to two decimal places (any quantity below)
    CONVERT  volume L -> mL; distance km -> m (x 1000)
HELD OUT for A26 (never generated, checked by check()): UNIT concentration / length, ROUND to one decimal,
CONVERT g -> mg (and every mass conversion), plus A26's protocol wordings, question styles and row templates.

Two styles, so the skill is not one template:
    natural (70 %): ~10 protocol wordings per family, 8 question styles, 5 row templates, 2 QA forms, prose gaps
        drawn log-uniform from 2 to 150 filler sentences (lab-notebook bank + v7's recall filler bank);
    compact (30 %): "Report rule." + 2-8 demos "x -> y", an optional prose gap, then 1-3 queries "x ->" (the format
        A26's TRAINED arm uses on the held-out families).

Interface (like v7.data's synthetic iterators; Mind wires it as a source):
    protocol_text_iter(seed=0, skip_docs=0, doc_counters=None, counter_key='proto', bump=None) -> Iterator[str]
Check:  python -m v14.protocol_docs   (leak check on 20,000 docs + answer-token share + 3 sample docs)
"""
from __future__ import annotations

import random
import re
from typing import Dict, Iterator, Optional

ANSWER_MARK = '\x00'                                   # the answer marker of v7's data module (asserted equal in check())
SEED_OFFSET = 10 ** 8                                  # generator streams never touch A26's test seeds (26000-series)


def _ans(text: str) -> str:
    return f'{ANSWER_MARK}{text}{ANSWER_MARK}'


# ---- training families -------------------------------------------------------------------------------------------
UNIT_Q = {'mass': ('masses', 'Masses', ['mg', 'g', 'kg', 'ng', 'μg']),
          'volume': ('volumes', 'Volumes', ['mL', 'L', 'μL'])}
ROUND_Q = {'mass': ['mg', 'g', 'kg'], 'volume': ['mL', 'L'], 'time': ['s', 'min', 'h'], 'temperature': ['°C']}
ROUND_RULES = {'int': 'the nearest whole number', 'dec2': 'two decimal places'}
CONVERT = {'volume': ('volumes', 'Volumes', 'L', 'mL'), 'distance': ('distances', 'Distances', 'km', 'm')}
PLURAL = {'mass': 'masses', 'volume': 'volumes', 'time': 'times', 'temperature': 'temperatures', 'distance': 'distances'}

UNIT_W = ["Every {q} listed in this log was measured in {u}.",
          "The {qp} in the tables below use {u} as the unit.",
          "Throughout this report, {q} is expressed in {u}.",
          "We record each {q} in {u}; the unit is not repeated.",
          "{Qp} are written without units here; the unit is {u}.",
          "Measurement convention: {q} in {u}.",
          "In what follows, a {q} reading always means {u}.",
          "Lab standard for this project: {qp} are logged in {u}.",
          "Read every {q} number in this document as {u}.",
          "For brevity, {q} values omit the unit, which is {u}."]
ROUND_W = ["When reporting, round each value to {r}.",
           "Values in the summary are rounded to {r}.",
           "Our reporting standard is {r}.",
           "Give every reported number to {r}.",
           "Final values must be stated to {r}.",
           "All reported figures are rounded to {r}.",
           "Before reporting a value, round it to {r}.",
           "Report results to {r}, not with the raw precision.",
           "The lab reports each number rounded to {r}.",
           "Precision rule for answers: {r}."]
CONVERT_W = ["Report every {q} in {u2}, even when it was measured in {u1}.",
             "In the summary, {qp} are converted to {u2}.",
             "Our convention is to state {qp} in {u2}.",
             "Convert each {q} to {u2} before reporting it.",
             "All {qp} in the answers are given in {u2}.",
             "Express every reported {q} in {u2}.",
             "{Qp} are reported in {u2} (the instrument reads {u1}).",
             "When asked for a {q}, answer in {u2}.",
             "Answers quote {qp} in {u2}, whatever unit the row shows.",
             "The standard unit for reported {qp} is {u2}."]
QUESTIONS = ["Which {q} did sample {s} have?", "State the {q} for sample {s}.", "How large was the {q} of {s}?",
             "Sample {s}: {q}?", "Tell me the {q} measured for {s}.", "What is the reported {q} of {s}?",
             "{s} had which {q}?", "Look up sample {s} and give its {q}."]
QA_FORMS = ["\n\nQuestion: {q}\nAnswer:", "\nQ: {q}\nA:"]
ROW_T = [" Run {s} on {m} gave a {q} of {v}.", " {m}, sample {s}: {q} reading {v}.",
         " The {q} of {s} ({m}) came out at {v}.", " Entry {s} | {m} | {q} {v}.",
         " We measured {s} ({m}) and logged a {q} of {v}."]
MATERIALS = ['cell lysate', 'wash buffer', 'sample pool 2', 'crude extract', 'stock A', 'aliquot 6', 'reference batch',
             'eluate 2', 'filtered broth', 'tissue digest', 'resin slurry', 'cold fraction', 'mother liquor',
             'standard mix', 'test batch 4', 'pilot culture']
LAB_FILLER = ["The bench was cleaned before the next series.", "Gloves were changed between samples.",
              "The instrument was calibrated in the morning.", "Two tubes were labelled again because the ink ran.",
              "The incubator door stayed closed during the run.", "A note was added to the shared log.",
              "The centrifuge was balanced with a water tube.", "Reagents were taken from the second shelf.",
              "The fume hood alarm was checked and silent.", "Readings were copied into the spreadsheet later.",
              "The control looked as expected.", "Nothing unusual happened during transport.",
              "The timer was reset after each step.", "Waste went into the yellow container.",
              "The protocol sheet was kept next to the scale.", "A colleague double-checked the labels.",
              "The room temperature was stable all day.", "Pipette tips were replaced from a fresh box.",
              "One vial was set aside for a repeat.", "The lid of the cooler was taped shut."]

# ---- what A26 holds out (generation must never produce these) ----------------------------------------------------
HELD_OUT_UNITS = ['mM', 'μM', 'nM', 'nm', 'mm', 'cm']
HELD_OUT_QUANTITIES = ['concentration', 'length']


def _tenths(rng, lo=5, hi=999):
    while True:
        x = rng.randint(lo, hi)
        if x % 10:
            return x


def _sample_ids(rng, n):
    out = set()
    while len(out) < n:
        out.add(rng.choice('ABCDEFGHJKLMNPRSTUVWXYZ') + rng.choice(['', rng.choice('ABCDEFGHJKLMNPRSTUVWXYZ')])
                + str(rng.randint(1, 99)))
    out = sorted(out)                                  # a set's order depends on PYTHONHASHSEED: sort, then shuffle by rng
    rng.shuffle(out)
    return out


def _round_half_up(x_milli: int, rule: str) -> str:
    """x_milli = value * 1000 (an int); returns the rounded value text."""
    if rule == 'int':
        return str((x_milli + 500) // 1000)
    v = (x_milli + 5) // 10                            # hundredths
    return f'{v // 100}.{v % 100:02d}'


def item(rng, fam: str, sub: str):
    """(row value text, answer text with the convention applied, answer text without a protocol)."""
    if fam == 'unit':
        v = f'{_tenths(rng) / 10:.1f}'
        return v, f'{v} {sub}', v
    if fam == 'round':
        rule, unit = sub
        while True:                                    # 3 decimals, last digit not 0 or 5, no tie for either rule
            x = rng.randint(1000, 99999)
            if x % 10 not in (0, 5) and x % 1000 != 500:
                break
        raw = f'{x // 1000}.{x % 1000:03d}'
        return f'{raw} {unit}', f'{_round_half_up(x, rule)} {unit}', f'{raw} {unit}'
    q, (u1, u2) = fam, sub                             # convert: x 1000
    t = _tenths(rng, 11, 9999)                         # u1 with up to 4 decimals -> u2 with one decimal
    big = f'{t / 10:.1f}'
    small = f'{t / 10000:.4f}'.rstrip('0').rstrip('.')
    return f'{small} {u1}', f'{big} {u2}', f'{small} {u1}'


def _filler(rng, n):
    from v7.data import _recall_filler
    out = []
    for _ in range(n):
        out.append(rng.choice(LAB_FILLER) if rng.random() < 0.5 else _recall_filler(rng, 1))
    return ' '.join(out)


def _gap(rng, lo=2, hi=150):
    return int(round(lo * ((hi / lo) ** rng.random())))


def _family(rng):
    fam = rng.choice(['unit', 'round', 'convert'])
    if fam == 'unit':
        q = rng.choice(list(UNIT_Q)); qp, Qp, units = UNIT_Q[q]
        u = rng.choice(units)
        return fam, q, u, dict(q=q, qp=qp, Qp=Qp, u=u), rng.choice(UNIT_W)
    if fam == 'round':
        rule = rng.choice(list(ROUND_RULES)); q = rng.choice(list(ROUND_Q)); u = rng.choice(ROUND_Q[q])
        return fam, q, (rule, u), dict(r=ROUND_RULES[rule]), rng.choice(ROUND_W)
    q = rng.choice(list(CONVERT)); qp, Qp, u1, u2 = CONVERT[q]
    return 'convert', q, (u1, u2), dict(q=q, qp=qp, Qp=Qp, u1=u1, u2=u2), rng.choice(CONVERT_W)


def build_doc(rng, with_meta: bool = False, max_gap: int = 150, compact_only: bool = False):
    fam, q, sub, slots, wording = _family(rng)
    meta = dict(family=fam, quantity=q, sub=sub, parts=[])
    fam_key = q if fam == 'convert' else fam
    has_proto = rng.random() >= 0.2
    if compact_only or rng.random() < 0.3:             # compact style (compact_only short-circuits: the default stream's rng use is unchanged) ("Report rule." demos + queries)
        k = rng.randint(2, 8)
        lines = []
        for _ in range(k):
            rv, ap, _ = item(rng, fam_key, sub)
            lines.append(f'\n{rv.split()[0]} -> {ap}')
        text = 'Report rule.' + ''.join(lines)
        meta['parts'] += lines
        if rng.random() < 0.5:
            text += '\n\n' + _filler(rng, _gap(rng, hi=max_gap)) + '\n'
        for _ in range(rng.randint(1, 3)):
            rv, ap, _ = item(rng, fam_key, sub)
            text += f'\n{rv.split()[0]} ->' + _ans(f' {ap}')
            meta['parts'].append(f'{rv} -> {ap}')
        return (text + '\n', dict(meta, style='compact')) if with_meta else text + '\n'
    return _natural_body(rng, q, sub, fam_key, wording.format(**slots) if has_proto else '', has_proto, max_gap, meta, with_meta)


SCOPE_SHARE = 0.20                  # proto4k2 (PRE-REGISTERED 11-17): share of OTHER-quantity docs (scope contrast) at short gaps


SCOPE_HELD_ASKED = 'volume'          # Argon / Lead 11-17: the unit direction "asked volume, mass protocol" is HELD OUT of all scope
                                     # training (the scope probe scores it); training sees only "asked mass, volume protocol"


def _scoped_family(rng):
    """a family whose protocol names a quantity (unit / convert), and the protocol of its OTHER quantity. Unit family: the asked
    quantity is never SCOPE_HELD_ASKED (held-out direction). Convert (not probed): both directions."""
    while True:
        fam, q, sub, slots, wording = _family(rng)
        if fam == 'convert' or (fam == 'unit' and q != SCOPE_HELD_ASKED):
            break
    if fam == 'unit':
        oq = next(k for k in UNIT_Q if k != q); qp, Qp, units = UNIT_Q[oq]
        return fam, q, sub, oq, rng.choice(UNIT_W).format(q=oq, qp=qp, Qp=Qp, u=rng.choice(units))
    oq = next(k for k in CONVERT if k != q); qp, Qp, u1, u2 = CONVERT[oq]
    return fam, q, sub, oq, rng.choice(CONVERT_W).format(q=oq, qp=qp, Qp=Qp, u1=u1, u2=u2)


def build_doc_v2(rng, with_meta: bool = False, max_gap: int = 300):
    """proto4k2 (2026-11-17, SCOPE): SCOPE_SHARE of docs are natural-style docs whose ONE protocol is about the OTHER quantity of
    the family (unit / convert); the asked quantity has no convention, so the answers are the raw values (unit: bare number, then
    '.'). The rest are build_doc's docs unchanged (its own 80 / 20 protocol / no-protocol and compact / natural mix)."""
    if rng.random() >= SCOPE_SHARE:
        out = build_doc(rng, with_meta=with_meta, max_gap=max_gap)
        if with_meta:
            out[1].setdefault('mode', 'own' if out[1].get('protocol') else 'none')
        return out
    fam, q, sub, oq, proto = _scoped_family(rng)
    meta = dict(family=fam, quantity=q, sub=sub, parts=[], mode='other', other_q=oq)
    return _natural_body(rng, q, sub, q if fam == 'convert' else fam, proto, False, max_gap, meta, with_meta)


def _natural_body(rng, q, sub, fam_key, proto, conv, max_gap, meta, with_meta):
    """build_doc's natural-style body (rng use unchanged). proto '' = no protocol sentence; conv False = raw answers."""
    n_rows = rng.randint(3, 6)
    ids = _sample_ids(rng, n_rows); mats = rng.sample(MATERIALS, n_rows)
    rows, answers = [], []
    for i in range(n_rows):
        rv, ap, raw = item(rng, fam_key, sub)
        rows.append(rng.choice(ROW_T).format(s=ids[i], m=mats[i], q=q, v=rv))
        answers.append(ap if conv else raw)
    parts = []
    if rng.random() < 0.5:
        parts.append(_filler(rng, rng.randint(1, 3)))
    if proto:
        parts.append(proto)
    parts.append(_filler(rng, _gap(rng, hi=max_gap)))
    for r in rows:
        parts.append(r.strip())
        if rng.random() < 0.6:
            parts.append(_filler(rng, rng.randint(1, 20)))
    parts.append(_filler(rng, rng.randint(0, 40)) if rng.random() < 0.7 else '')
    text = ' '.join(p for p in parts if p)
    qa = rng.choice(QA_FORMS)
    meta['parts'] += ([proto] if proto else []) + rows
    for i in rng.sample(range(n_rows), rng.randint(1, min(3, n_rows))):
        question = rng.choice(QUESTIONS).format(q=q, s=ids[i])
        text += qa.format(q=question) + _ans(f' {answers[i]}') + '.'
        meta['parts'] += [question, answers[i]]
    meta.update(style='natural', protocol=bool(proto) and conv)
    return (text + '\n', meta) if with_meta else text + '\n'


FAR_Q = (2048, 3000)        # proto_far: protocol end -> first marked answer, in tokens (Mind: the doc must fit one 4096 window)
FAR_MAX = 3200              # proto_far: whole doc <= 3,200 tokens (ANSWER_IN_WINDOW: the answer must sit in the window)
_TOK = {}


def _ntok(text):
    if 'tok' not in _TOK:
        from v7.data import get_chat_tokenizer
        _TOK['tok'] = get_chat_tokenizer()
    return len(_TOK['tok'].encode(text.replace(ANSWER_MARK, ''), add_special_tokens=False))


def build_far_doc(rng, with_meta: bool = False):
    """proto_far (2026-11-07, Mind's far block): ONE protocol (natural style, trained families / wordings only, always stated),
    then a filler gap sized so the FIRST marked answer sits FAR_Q tokens after the protocol, then 3-6 rows (short filler
    between) and 1-3 questions; the whole doc <= FAR_MAX tokens. Lengths are measured with the chat tokenizer while building
    (exact, deterministic given rng). The stream is bit-identical to 2026-11-07's (v14/test_proto_far2.py checks a hash)."""
    fam, q, sub, slots, wording = _family(rng)
    return _far_body(rng, fam, q, sub, wording.format(**slots), True, dict(mode='own'), with_meta)


FAR2_SHARES = (0.65, 0.25, 0.10)    # proto_far2 modes: own protocol / no protocol / OTHER-quantity protocol (PRE-REGISTERED 11-17)


FAR3_SHARES = (0.55, 0.25, 0.20)    # proto_far3 (SCOPE, PRE-REGISTERED 11-17): own / none / other; 'other' redraws round -> ~20 % realized


def build_far_doc_v3(rng, with_meta: bool = False):
    """proto_far3 (2026-11-17, SCOPE): proto_far2's design with FAR3_SHARES; an 'other' draw takes a family whose protocol names a
    quantity (unit / convert, never round), so the other-quantity share is the full 20 %. Own rng stream (far_version 3)."""
    r = rng.random()
    mode = 'own' if r < FAR3_SHARES[0] else 'none' if r < FAR3_SHARES[0] + FAR3_SHARES[1] else 'other'
    if mode == 'other':
        fam, q, sub, oq, proto = _scoped_family(rng)
        return _far_body(rng, fam, q, sub, proto, False, dict(mode='other', other_q=oq), with_meta)
    fam, q, sub, slots, wording = _family(rng)
    return _far_body(rng, fam, q, sub, wording.format(**slots) if mode == 'own' else '', mode == 'own',
                     dict(mode=mode, other_q=None), with_meta)


def build_far_doc_v2(rng, with_meta: bool = False):
    """proto_far2 (2026-11-17, Lead after the 42.75B no-protocol false units): proto_far's far docs, but only 65 % carry the
    protocol for the asked quantity. 25 % have NO protocol, and 10 % a protocol for the OTHER quantity of the same family. Those 35 %
    answer with the raw value (unit family: the bare number, then '.'), the form proto4k's no-protocol docs use. Round wordings
    name no quantity, so a round doc that draws 'other' becomes 'none'. Same gap / length rules as proto_far; own rng stream."""
    r = rng.random()
    mode = 'own' if r < FAR2_SHARES[0] else 'none' if r < FAR2_SHARES[0] + FAR2_SHARES[1] else 'other'
    fam, q, sub, slots, wording = _family(rng)
    if mode == 'other' and (fam == 'round' or (fam == 'unit' and q == SCOPE_HELD_ASKED)):
        mode = 'none'                                                # round names no quantity; asked-volume is the held-out direction
    proto, other_q = '', None
    if mode == 'own':
        proto = wording.format(**slots)
    elif mode == 'other' and fam == 'unit':
        other_q = next(k for k in UNIT_Q if k != q); qp, Qp, units = UNIT_Q[other_q]
        proto = rng.choice(UNIT_W).format(q=other_q, qp=qp, Qp=Qp, u=rng.choice(units))
    elif mode == 'other':                                            # convert: the other CONVERT quantity
        other_q = next(k for k in CONVERT if k != q); qp, Qp, u1, u2 = CONVERT[other_q]
        proto = rng.choice(CONVERT_W).format(q=other_q, qp=qp, Qp=Qp, u1=u1, u2=u2)
    return _far_body(rng, fam, q, sub, proto, mode == 'own', dict(mode=mode, other_q=other_q), with_meta)


def _far_body(rng, fam, q, sub, proto, conv, extra, with_meta):
    """the far doc after the family draw (proto_far's original body; rng use unchanged). proto '' = no protocol sentence;
    conv False = answers are the raw values (no convention applied)."""
    fam_key = q if fam == 'convert' else fam
    n_rows = rng.randint(3, 6)
    ids = _sample_ids(rng, n_rows); mats = rng.sample(MATERIALS, n_rows)
    rows, answers = [], []
    for i in range(n_rows):
        rv, ap, raw = item(rng, fam_key, sub)
        rows.append(rng.choice(ROW_T).format(s=ids[i], m=mats[i], q=q, v=rv).strip())
        answers.append(ap if conv else raw)
    lead = _filler(rng, rng.randint(1, 3)) if rng.random() < 0.5 else ''
    tail_rows = ' '.join(r + (' ' + _filler(rng, rng.randint(1, 4)) if rng.random() < 0.6 else '') for r in rows)
    qa = rng.choice(QA_FORMS)
    asked = rng.sample(range(n_rows), rng.randint(1, min(3, n_rows)))
    qtext = ''.join(qa.format(q=rng.choice(QUESTIONS).format(q=q, s=ids[i])) + _ans(f' {answers[i]}') + '.' for i in asked)
    target = rng.randint(*FAR_Q)
    head = (lead + ' ' + proto).strip()
    fixed_after = _ntok(' ' + tail_rows + qa.format(q='x'))          # rows + the first question's lead-in
    gap, used = [], 0
    while used + fixed_after < target and len(gap) < 600:            # grow the gap a sentence at a time (running count)
        g = _filler(rng, 1); gap.append(g); used += _ntok(' ' + g)
    text = ' '.join(p for p in (head, ' '.join(gap), tail_rows) if p) + qtext
    while _ntok(text) > FAR_MAX and gap:                             # cap the whole doc (drop gap sentences from the end)
        gap.pop()
        text = ' '.join(p for p in (head, ' '.join(gap), tail_rows) if p) + qtext
    meta = dict(family=fam, quantity=q, sub=sub, style='far', protocol=bool(proto) and conv,
                parts=([proto] if proto else []) + rows, **extra)
    return (text + '\n', meta) if with_meta else text + '\n'


def protocol_text_iter(*, seed: int = 0, skip_docs: int = 0, doc_counters: Optional[Dict[str, int]] = None,
                       counter_key: str = 'proto', bump=None, max_gap: Optional[int] = None,
                       compact_only: bool = False, far: bool = False, far_version: int = 1,
                       scope_version: int = 1) -> Iterator[str]:
    """Infinite, deterministic stream of protocol docs (marked answer spans). max_gap = the largest prose gap in filler
    sentences (log-uniform from 2; ~12 tokens each): 150 = the T 2048 recipe, 300 = 'proto4k' (protocol -> rows up to
    ~3.5-4k tokens, for SEQ 4096). compact_only = 'protoc': ONLY the compact "Report rule." style (demos + 1-3 queries, the converted
    value as the marked answer), prose gap <= 20 sentences unless max_gap is given; trained families only. far = 'proto_far':
    build_far_doc only (protocol -> first answer 2,048-3,000 tokens, doc <= 3,200), its own rng stream. far_version 2 =
    'proto_far2' (build_far_doc_v2: 25 % no protocol, 10 % other-quantity protocol), its own rng stream (offset 8888).
    far_version 3 = 'proto_far3' (build_far_doc_v3, SCOPE: 55 / 25 / 20), offset 9999. scope_version 2 (not far) = 'proto4k2'
    (build_doc_v2: 20 % other-quantity docs at short gaps, the rest build_doc's), offset 5555."""
    if max_gap is None:
        max_gap = 20 if compact_only else 150
    far2, far3, sc2 = far and far_version == 2, far and far_version == 3, not far and scope_version == 2
    off = (9999 if far3 else 8888 if far2 else 7777) if far else (5555 if sc2 else 0)
    rng = random.Random((seed * 2654435761 + SEED_OFFSET + 31337 + off) & 0xFFFFFFFF)
    n = 0
    while True:
        if far:
            doc = build_far_doc_v3(rng) if far3 else build_far_doc_v2(rng) if far2 else build_far_doc(rng)
        elif sc2:
            doc = build_doc_v2(rng, max_gap=max_gap)
        else:
            doc = build_doc(rng, max_gap=max_gap, compact_only=compact_only)
        n += 1
        if n <= skip_docs:
            continue
        if bump is not None:
            bump(doc_counters, counter_key)
        elif doc_counters is not None:
            doc_counters[counter_key] = doc_counters.get(counter_key, 0) + 1
        yield doc


def _skeleton(t: str) -> str:
    return re.sub(r'\{[^}]*\}', '#', t).strip().lower()


def _ngrams(t: str, n=5):
    w = re.findall(r"[a-z#]+", _skeleton(t))
    return {' '.join(w[i:i + n]) for i in range(len(w) - n + 1)}


def check(n_docs: int = 20000, far: bool = False, far_version: int = 1, scope_version: int = 1) -> dict:
    """leak check against A26's held-out families and wordings (on the generated protocol / rows / questions / answers,
    not on filler prose) + the share of answer characters. far = the proto_far docs (build_far_doc)."""
    from v7.data import ANSWER_MARK as AM
    from v14 import a26_protocol as A
    assert AM == ANSWER_MARK
    a26_t = [t for fam in A.PROTO.values() for t in fam] + A.QUEST + A.ROWS
    ours = UNIT_W + ROUND_W + CONVERT_W + QUESTIONS + ROW_T
    shared_skel = {_skeleton(t) for t in ours} & {_skeleton(t) for t in a26_t}
    shared_5g = set().union(*[_ngrams(t) for t in ours]) & set().union(*[_ngrams(t) for t in a26_t])
    held_unit = re.compile(r'(?<![A-Za-zμ])(' + '|'.join(map(re.escape, HELD_OUT_UNITS)) + r')(?![A-Za-z])')
    rng = random.Random(1)
    bad, fams, styles, modes, chars_ans, chars = [], {}, {}, {}, 0, 0
    for _ in range(n_docs):
        FB = {1: build_far_doc, 2: build_far_doc_v2, 3: build_far_doc_v3}[far_version]
        d, m = (FB(rng, with_meta=True) if far else build_doc_v2(rng, with_meta=True) if scope_version == 2
                else build_doc(rng, with_meta=True))
        modes[m.get('mode', '-')] = modes.get(m.get('mode', '-'), 0) + 1
        k = f"{m['family']}:{m['quantity']}:{m['sub'][0] if m['family'] == 'round' else m['sub']}"
        fams[k] = fams.get(k, 0) + 1
        styles[m['style']] = styles.get(m['style'], 0) + 1
        part = ' '.join(m['parts'])
        if (m['quantity'] in HELD_OUT_QUANTITIES or held_unit.search(part)
                or (m['family'] == 'convert' and m['quantity'] == 'mass')
                or 'one decimal' in part.lower() or 'nearest 0.1' in part.lower()):
            bad.append(part[:200])
        spans = d.split(ANSWER_MARK)[1::2]
        chars_ans += sum(len(x) for x in spans); chars += len(d.replace(ANSWER_MARK, ''))
    res = dict(docs=n_docs, families=dict(sorted(fams.items())), styles=styles, modes=modes, held_out_hits=len(bad),
               shared_template_skeletons=sorted(shared_skel), shared_5gram_with_a26=sorted(shared_5g),
               answer_char_share=round(chars_ans / chars, 4), examples_bad=bad[:3])
    res['leak_free'] = int(not bad and not shared_skel and not shared_5g)
    return res


if __name__ == '__main__':
    import json
    rng = random.Random(7)
    for _ in range(3):
        print(build_doc(rng).replace(ANSWER_MARK, '⟦').strip()[:1200], '\n' + '-' * 80)
    print(json.dumps(check(), indent=1, ensure_ascii=False))
