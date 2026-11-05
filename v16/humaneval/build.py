"""python build.py <out_dir> <step>   (out_dir = where run.sh put the dumps)
Build the mobile-friendly human-eval site: site/index.html + site/data/<section>.json (loaded on demand).
Every item a section has, with filters (all / right / wrong), search and paging. Run with the worktree venv (datasets)."""
import json, re, glob, sys
from pathlib import Path
S = Path(sys.argv[1]); STEP = sys.argv[2]; HERE = Path(__file__).parent; R = Path('/home/gowrav/Development/qllm2'); D = S / 'site' / 'data'; D.mkdir(parents=True, exist_ok=True)
full = json.load(open(S / 'm4_full.json')) if (S / 'm4_full.json').exists() else json.load(open(S / 'm4.json'))
meta = full['meta']; secs = []

def jl(p): return [json.loads(l) for l in open(p)] if Path(p).exists() else []
def put(sid, group, title, intro, kind, items):
    if not items: return
    json.dump(items, open(D / f'{sid}.json', 'w'))
    n_ok = sum(1 for x in items if x.get('ok') is True); has = any('ok' in x for x in items)
    secs.append(dict(id=sid, group=group, title=title, intro=intro, kind=kind, n=len(items), ok=n_ok if has else None))

# Generations (open prompts)
gg = {}
for g in full['gens']: gg.setdefault(g['group'], []).append(g)
for k, v in gg.items():
    put('gen_' + re.sub(r'\W+', '', k).lower(), 'Open prompts (read and judge)', k, f'{len(v)} prompts. Greedy and sampled answers.',
        'gen', [dict(name=g['name'], prompt=g['prompt'], greedy=g['greedy'], sampled=g['sampled']) for g in v])
# Benchmarks
names = dict(hellaswag='HellaSwag: pick the sensible ending', arc_easy='ARC-Easy: school science', piqa='PIQA: physical common sense')
for t, rows in full['bench'].items():
    if t == 'arc_easy' and rows and 'll' in rows[0]:   # the CHECK reports ARC-e as acc (raw log-likelihood), HS / PIQA as acc_norm
        for r in rows:
            r['pick'] = max(range(len(r['ll'])), key=lambda c: r['ll'][c]); r['ok'] = r['pick'] == r['gold']; r['score'] = r['ll']
    put(t, 'Standard benchmarks (all questions)', names[t],
        'Every question of the test split. Green = correct answer; blue outline = model\'s pick (score per character, higher = preferred).',
        'mc', [dict(prompt=r['q'], choices=r['choices'], gold=r['gold'], pick=r['pick'], ok=r['ok'], score=r['score']) for r in rows])
# Math (GSM8K, think_eval rows; questions re-attached from the same test split, same order)
kf = sorted(glob.glob(str(R / f'logs/sci1/v13_sempty_think_sci1_step_{STEP}_*.json')))
k = json.load(open(kf[-1])) if kf else {'rows': []}
try:
    from datasets import load_dataset
    qs = list(load_dataset('openai/gsm8k', 'main', split='test'))
except Exception:
    qs = []
items = []
for i, r in enumerate(k['rows']):
    q = qs[i // 2]['question'] if qs and i // 2 < len(qs) else ''
    items.append(dict(prompt=q, mode='thinking ON' if r['mode'] == 'on' else 'thinking OFF', answer=r['text'],
                      gold=r['gold'], pred=r['pred'], ok=r['ok']))
put('math', 'Standard benchmarks (all questions)', 'GSM8K math word problems', f'{len(items)//2} problems, each asked twice: '
    'with thinking ON and OFF (thinking is being removed). Gold = correct number.', 'math', items)
# Rule-learning (s8), skill pack, needles: the exact scored items, recorded from the real scripts
put('s8', 'Our intelligence tests', 'Learning a rule from 8 examples (s8)', 'The model sees 8 examples of an unseen rule '
    '(letter shift / affine number map), sometimes with 2k–4k tokens of unrelated text in between, then must answer a new case. '
    'Picks among 8 candidates (chance 12.5 %).', 'cand', jl(S / 'rec_s8.jsonl'))
put('skill', 'Our intelligence tests', 'Skill pack: remember a skill over long text', 'A skill (rule + examples) is shown once at '
    'the start, then 0–8k tokens of unrelated text, then a question. Two arms: pack held in memory vs pack placed right before the question.',
    'cand', jl(S / 'rec_skill.jsonl'))
put('needle', 'Our intelligence tests', 'Needles: recall a fact from far back', 'Facts (key → value) are hidden at the start / '
    'middle / end of 128–8192 tokens of text; the model must recall the value. Picks among 8 candidates.', 'cand', jl(S / 'rec_needle.jsonl'))
# Protocol (Library A26, milestone 3)
a26 = sorted(glob.glob(str(R / 'logs/v14_a26_sci1m*_*.jsonl')), key=lambda f: re.search(r'sci1m(\d+)', f).group(1))
last = re.search(r'sci1m\d+', a26[-1]).group(0) if a26 else ''; a26_name = last
rows = [json.loads(l) for f in a26 if last in f for l in open(f)]
put('protocol', 'Our intelligence tests', f'Following a stated rule (latest A26 run: {a26_name})', 'Library\'s test: a lab rule is stated, a record '
    'follows (sometimes far later), then a question. Right = value AND unit / rounding as the rule says.', 'proto',
    [dict(prompt=r.get('question', ''), rule=r.get('proto', ''), row=r.get('row', ''), answer=r.get('answer', ''),
          expected=f"{r.get('target', '')} {r.get('unit', '')}", ok=bool(r.get('right')),
          tag=f"distance {r.get('d')} tokens · {r.get('arm')} · {r.get('cell')}") for r in rows])
# Model alone vs model + library (Library's per-item package rows, if present for this step)
pf = S / 'pairs.jsonl'
if not pf.exists():
    cand = sorted(glob.glob(str(R / f'logs/v14_pkg_suite_*_pairs.jsonl')))
    cand = [c for c in cand if any(json.loads(l).get('step') == str(int(STEP)) for l in open(c).readlines()[:1])]
    pf = Path(cand[-1]) if cand else None
if pf and pf.exists():
    rows = [json.loads(l) for l in open(pf)]
    names = {'rxk': ('Rule held in memory x fact fetched by the library', 'The rule (8 examples) is in memory; the record with the value is fetched at question time.'),
             'rewrite': ('Told rule rewritten next to the question', 'The library turns the prose rule into worked lines placed next to the question.'),
             'told': ('Told rule held in memory (twin / fork read)', 'Model alone = plain read; with library = read against the forked no-rule twin.'),
             'recall': ('Recall 32k tokens back', 'Model alone = memory only; with library = the exact-text organ and planner.')}
    for cell, (title, intro) in names.items():
        it = [dict(prompt=r.get('question', ''), context=r.get('context', ''), gold=r.get('gold', ''),
                   bare=r.get('bare_answer', ''), bare_ok=r.get('bare_ok') in (1, '1', True),
                   pkg=r.get('package_answer', ''), pkg_ok=r.get('package_ok') in (1, '1', True),
                   ok=r.get('package_ok') in (1, '1', True),
                   tag=f"{r.get('task', '')} · {int(r.get('L', 0)):,} tokens") for r in rows if r.get('cell') == cell]
        if it:
            nb = sum(x['bare_ok'] for x in it); npk = sum(x['pkg_ok'] for x in it)
            put('pair_' + cell, 'Model alone vs model + library (same items)', title,
                intro + f' Model alone right {nb}/{len(it)}, with library {npk}/{len(it)}. Green border = right with the library.', 'pair', it)

# Progress
items = []
for fn in sorted(glob.glob(str(R / 'logs/sci1/v13_sempty_gens_sci1_step_*.txt'))):
    st = int(re.search(r'step_(\d+)', fn).group(1)); items.append(dict(name=f'Checkpoint {st // 15259} (step {st:,})', text=open(fn).read()))
put('progress', 'Progress', 'Same 5 prompts at checkpoints 1–4', 'How the answers changed over training.', 'text', items)

page = (HERE / 'page.html').read_text().replace('__SECS__', json.dumps(secs)).replace('__META__', json.dumps(meta))
(S / 'site/index.html').write_text(page)
print('built', [(s['id'], s['n'], s['ok']) for s in secs])
