"""SCI-1 real-data sources for a scientist that calls tools, thinks briefly and writes analysis code (user 2026-11-02 11:10 IST).

All three render into the chat text the mind already reads (ChatML with plain-text tags, no new tokens), with the tool format
shared with the package (Library's v0: a tool result is read like a fetched row):

    <|im_start|>assistant\\n<tool_call>\\n{"name": ..., "arguments": {...}}\\n</tool_call><|im_end|>\\n
    <|im_start|>tool\\n<tool_response>\\n{"name": ..., "content": ...}\\n</tool_response><|im_end|>\\n

- ``tools``  : glaive-function-calling-v2 (Apache-2.0) + ToolACE (Apache-2.0) + our own library-fetch docs (call -> one result
               row -> an answer that uses it). Normal LM weight (no answer marks).
- ``think``  : MetaMathQA (MIT) + GSM8K train (MIT), each item ONCE in THINK ON ('Reasoning: on' system line + <think> short
               rationale </think> + answer) or OFF (answer only), ~50 / 50; responses > ``max_chars`` skipped.
- ``pysci``  : Common Pile Stack-Edu (filtered, per-file permissive licences): Python files that import a scientific / analysis
               library (numpy, scipy, pandas, matplotlib, sklearn, ...); code only (no executed output in this source).
Train / holdout split = v7.data.pretrain_holdout_bucket on the rendered text. In-memory sources are shuffled once with a fixed
seed, so skip_docs resumes exactly.
"""
from __future__ import annotations

import ast, json, random, re
from typing import Dict, Iterator, List, Optional

IM_S, IM_E = '<|im_start|>', '<|im_end|>'
SCI_IMPORT = re.compile(r'^\s*(?:import|from)\s+(numpy|scipy|pandas|matplotlib|sklearn|seaborn|sympy|statsmodels|astropy|Bio|'
                        r'xarray|networkx|skimage|torch|h5py|netCDF4|pymc|emcee|lmfit|uncertainties|pint)\b', re.M)


def _turn(role: str, text: str) -> str:
    return f'{IM_S}{role}\n{text.strip()}{IM_E}\n'


def _call_block(name: str, arguments) -> str:
    return '<tool_call>\n' + json.dumps({'name': name, 'arguments': arguments}, ensure_ascii=False) + '\n</tool_call>'


def _resp_block(name: str, content) -> str:
    return '<tool_response>\n' + json.dumps({'name': name, 'content': content}, ensure_ascii=False) + '\n</tool_response>'


def _glaive_docs() -> List[str]:
    from huggingface_hub import hf_hub_download
    rows = json.load(open(hf_hub_download('glaiveai/glaive-function-calling-v2', 'glaive-function-calling-v2.json', repo_type='dataset')))
    out = []
    for r in rows:
        sys_txt = (r.get('system') or '').replace('SYSTEM:', '', 1).strip()
        parts = re.split(r'\n*(USER|ASSISTANT|FUNCTION RESPONSE): ', '\n' + (r.get('chat') or ''))
        doc, last_call = _turn('system', sys_txt) if sys_txt else '', 'tool'
        for role, txt in zip(parts[1::2], parts[2::2]):
            txt = txt.replace('<|endoftext|>', '').strip()
            if role == 'USER':
                doc += _turn('user', txt)
            elif role == 'ASSISTANT' and txt.startswith('<functioncall>'):
                body = txt[len('<functioncall>'):].strip()
                try:
                    j = json.loads(body.replace("'{", '{').replace("}'", '}'))
                    last_call = j.get('name', 'tool'); doc += _turn('assistant', _call_block(last_call, j.get('arguments', {})))
                except ValueError:
                    doc += _turn('assistant', '<tool_call>\n' + body + '\n</tool_call>')
            elif role == 'ASSISTANT':
                doc += _turn('assistant', txt)
            else:
                try:
                    content = json.loads(txt)
                except ValueError:
                    content = txt
                doc += _turn('tool', _resp_block(last_call, content))
        if '<tool_call>' in doc or len(doc) > 200:
            out.append(doc)
    return out


def _toolace_docs() -> List[str]:
    from huggingface_hub import hf_hub_download
    rows = json.load(open(hf_hub_download('Team-ACE/ToolACE', 'data.json', repo_type='dataset')))
    out = []
    for r in rows:
        doc, names = _turn('system', r.get('system', '')), ['tool']
        for c in r.get('conversations', []):
            v = c.get('value', '')
            if c['from'] == 'assistant' and v.strip().startswith('[') and v.strip().endswith(')]'):
                try:
                    calls = ast.parse(v.strip()).body[0].value.elts
                    blocks = []
                    for call in calls:
                        nm = ast.unparse(call.func); names = [nm]
                        blocks.append(_call_block(nm, {k.arg: ast.literal_eval(k.value) for k in call.keywords}))
                    doc += _turn('assistant', '\n'.join(blocks))
                    continue
                except (SyntaxError, ValueError, AttributeError, IndexError):
                    pass
            if c['from'] == 'tool':
                try:
                    res = json.loads(v)
                    res = res if isinstance(res, list) else [res]
                    doc += _turn('tool', '\n'.join(_resp_block(x.get('name', names[0]), x.get('results', x)) for x in res))
                except (ValueError, AttributeError):
                    doc += _turn('tool', _resp_block(names[0], v))
            else:
                doc += _turn(c['from'], v)
        out.append(doc)
    return out


FETCH_FIELDS = ('mass', 'temperature', 'pH', 'date', 'author', 'location', 'voltage', 'yield', 'sample count', 'instrument',
                'wavelength', 'pressure', 'duration', 'species', 'batch')
FETCH_ASK = ('What was the {f} recorded for {k}?', 'Look up the {f} of {k}.', 'Find {k} in my notes and tell me its {f}.',
             'Which {f} did we log for {k}?', 'I need the {f} for {k}, please check the session notes.')
FETCH_PROSE = ('{k} showed a {f} reading of {v}.', 'We noted {v} as the {f} of {k} before lunch.', 'Entry for {k}: {f} came out at {v}, '
               'checked twice.', 'The {f} for {k} ({b}) was {v}.', 'After calibration, {k} gave a {f} of {v}.')
FETCH_SAY = ('The {f} recorded for {k} is {v}.', 'According to the notes, {k} has a {f} of {v}.', '{k}: {f} = {v}.',
             'The session record shows {v} as the {f} for {k}.')


def _fetch_docs(n: int, seed: int) -> List[str]:
    """Our own format: user asks -> library.fetch call -> ONE result row -> an answer that uses the row (no answer marks)."""
    from v7.data import _recall_key
    rng, out = random.Random(seed), []
    vals = {'mass': lambda: f'{rng.uniform(0.1, 900):.2f} g', 'temperature': lambda: f'{rng.uniform(-80, 400):.1f} C',
            'pH': lambda: f'{rng.uniform(1, 13):.2f}', 'date': lambda: f'2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}',
            'voltage': lambda: f'{rng.uniform(0.1, 48):.2f} V', 'yield': lambda: f'{rng.uniform(5, 99):.1f} %',
            'sample count': lambda: str(rng.randint(3, 400)), 'wavelength': lambda: f'{rng.randint(200, 1100)} nm',
            'pressure': lambda: f'{rng.uniform(0.2, 9):.2f} bar', 'duration': lambda: f'{rng.randint(2, 180)} min'}
    words = ('north lab', 'cold room', 'bench 4', 'Dr. Okafor', 'Dr. Lindqvist', 'spectrometer B', 'E. coli K-12', 'zebrafish',
             'batch 17', 'site C', 'the field station', 'HPLC-2')
    for _ in range(n):
        k, f = _recall_key(rng), rng.choice(FETCH_FIELDS)
        v = vals[f]() if f in vals else rng.choice(words)
        # Half the results are rendered records, half are raw session sentences (the package returns the session's raw text).
        row = f'Record: {k} | {f}: {v}.' if rng.random() < 0.5 else rng.choice(FETCH_PROSE).format(k=k, f=f, v=v, b=rng.choice(words))
        out.append(_turn('user', rng.choice(FETCH_ASK).format(f=f, k=k))
                   + _turn('assistant', _call_block('library.fetch', {'query': f'{k} {f}'}))
                   + _turn('tool', _resp_block('library.fetch', row))
                   + _turn('assistant', rng.choice(FETCH_SAY).format(f=f, k=k, v=v)))
    return out


THINK_ON = 'Reasoning: on'   # the user's switch (user 11:30 IST via Lead): present = think, absent = answer only; default OFF


def _think_render(q: str, why: str, ans: str, on: bool) -> str:
    if on:
        return _turn('system', THINK_ON) + _turn('user', q) + _turn('assistant', f'<think>\n{why.strip()}\n</think>\nThe answer is {ans}.')
    return _turn('user', q) + _turn('assistant', f'The answer is {ans}.')


def _think_docs(max_chars: int = 1200, seed: int = 43) -> List[str]:
    """Each item is rendered ONCE, in THINK ON (system line + <think> trace + answer) or OFF (same prompt, answer only), ~50 / 50
    by a seeded draw, so the marker is the only difference. Nothing else in the mix carries <think> (smoltalk2's is stripped)."""
    from huggingface_hub import hf_hub_download
    from datasets import load_dataset
    out, rng = [], random.Random(seed + 7)
    for r in json.load(open(hf_hub_download('meta-math/MetaMathQA', 'MetaMathQA-395K.json', repo_type='dataset'))):
        resp = r.get('response', '')
        if len(resp) > max_chars or 'The answer is:' not in resp:
            continue
        why, ans = resp.rsplit('The answer is:', 1)
        out.append(_think_render(r['query'], why, ans.strip(), rng.random() < 0.5))
    for r in load_dataset('openai/gsm8k', 'main', split='train'):
        a = r['answer']
        if len(a) > max_chars or '####' not in a:
            continue
        why, ans = a.rsplit('####', 1)
        why = re.sub(r'<<[^>]*>>', '', why)              # GSM8K's calculator annotations
        out.append(_think_render(r['question'], why, ans.strip(), rng.random() < 0.5))
    return out


def _memory_iter(name: str, build, *, seed: int, exclude_holdout: bool, only_holdout: bool, holdout_pct: int, skip_docs: int,
                 doc_counters: Optional[Dict[str, int]], counter_key: str, bump) -> Iterator[str]:
    from v7.data import pretrain_holdout_bucket
    docs = build(); random.Random(seed).shuffle(docs)
    print(f'  [stream] {name}: {len(docs):,} docs in memory; skip {skip_docs:,}', flush=True)
    left, epoch = int(skip_docs), 0
    while True:                                          # small sources repeat (each pass reshuffled), never run dry
        for d in docs:
            hold = pretrain_holdout_bucket(d, holdout_pct)
            if (exclude_holdout and hold) or (only_holdout and not hold):
                continue
            if left > 0:
                left -= 1
                continue
            if bump is not None:
                bump(doc_counters, counter_key)
            yield d
        epoch += 1; random.Random(seed + epoch).shuffle(docs)
        if only_holdout:
            return


def tools_text_iter(*, seed: int, skip_docs: int = 0, doc_counters=None, counter_key: str = 'tools', bump=None,
                    exclude_holdout: bool = True, only_holdout: bool = False, holdout_pct: int = 5) -> Iterator[str]:
    def build():
        g, t = _glaive_docs(), _toolace_docs()
        return g + t + _fetch_docs((len(g) + len(t)) // 8, seed)   # library-fetch docs ~11 % of the tool docs
    return _memory_iter('tools', build, seed=seed, exclude_holdout=exclude_holdout, only_holdout=only_holdout,
                        holdout_pct=holdout_pct, skip_docs=skip_docs, doc_counters=doc_counters, counter_key=counter_key, bump=bump)


def think_text_iter(*, seed: int, skip_docs: int = 0, doc_counters=None, counter_key: str = 'think', bump=None,
                    exclude_holdout: bool = True, only_holdout: bool = False, holdout_pct: int = 5) -> Iterator[str]:
    return _memory_iter('think', _think_docs, seed=seed, exclude_holdout=exclude_holdout, only_holdout=only_holdout,
                        holdout_pct=holdout_pct, skip_docs=skip_docs, doc_counters=doc_counters, counter_key=counter_key, bump=bump)


def pysci_text_iter(*, skip_docs: int = 0, doc_counters=None, counter_key: str = 'pysci', exclude_holdout: bool = True,
                    only_holdout: bool = False, holdout_pct: int = 5, max_chars: int = 60_000) -> Iterator[str]:
    """Stack-Edu Python files that import a scientific / analysis library (code only)."""
    import gzip
    from huggingface_hub import HfFileSystem
    from v7.data import _hf_glob, _jsonl_stream, _stream_retry, pretrain_holdout_bucket
    files = _hf_glob('common-pile/stackv2_edu_filtered', 'stack-edu-*.json.gz')
    fs = HfFileSystem()

    def lang(f):
        def first():
            with fs.open(f, 'rb', block_size=1 << 20) as fh:
                return json.loads(gzip.GzipFile(fileobj=fh).readline()).get('metadata', {}).get('language')
        return _stream_retry(first, f'peek {f}')
    py = [f for f in files if lang(f) == 'Python']

    def extract(row):
        t = row.get('text') or ''
        if (row.get('metadata') or {}).get('language') != 'Python' or len(t) > max_chars or not SCI_IMPORT.search(t):
            return ()
        hold = pretrain_holdout_bucket(t, holdout_pct)
        if (exclude_holdout and hold) or (only_holdout and not hold):
            return ()
        return (t,)
    yield from _jsonl_stream('pysci', py, extract, skip_docs=skip_docs, doc_counters=doc_counters, counter_key=counter_key)
