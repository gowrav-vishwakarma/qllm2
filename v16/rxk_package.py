"""A23 (registered 2026-11-01 14:19 IST): pkg.apply on Mind's R x K docs (reproduction) + the story rule check.

Docs as scripts/run_rxk_probe.py builds them (seeds 8800-8849, affine_map_new + shift_wide_new): session = skill pack +
filler + " Record: K | x: v."; the key question is the implicit line "K ->"; the package fetches K's row, takes v out,
writes the trained line "<v> ->" and answers with the skill contrast (L >= 4096: echo dilation s = 8). Free = the first
answer token is the target. Compared doc by doc with Mind's exp_c4 / exp_dil8_c4 (logs/diag/v13_sempty_rxk_*_99d2951).
Story check: Old Pell's rule as the story's WORDS vs as a register skill in the trained affine format, asked "4 ->".

    python v16/rxk_package.py --out logs/v16_rxk_package_<hash>_<stamp>
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'v16')):
    sys.path.insert(0, p)
from memory_probes.spread import _SEP, filler_tokens, split_units  # noqa: E402
from run_skillpack_probe import filler_block  # noqa: E402
from v14.rules_ood import build_example as b2_example  # noqa: E402
from v7.data import _recall_key  # noqa: E402
from v16.package import Package  # noqa: E402

CK = {'b2p2': '/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/b2p2_main_38e3c29/best_model.pt',
      'cu2a': '/home/gowrav/Development/qllm2/checkpoints_v13_sempty/cu2a_main_24b2969/best_model.pt'}
MIND_LOG = '/home/gowrav/Development/qllm2/logs/diag/v13_sempty_rxk_{}_99d2951.jsonl'
RULE_WORDS = ("Old Pell, whose rule for prices was famous in Harrow: every loaf cost twice its weight in coins plus one "
              "coin for the oven.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--minds', default='b2p2,cu2a'); ap.add_argument('--lengths', default='2048,8192')
    ap.add_argument('--seed', type=int, default=8800); ap.add_argument('--docs', type=int, default=50)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    filler = filler_tokens(str(ROOT / '.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt'))
    rows, summary = [], {}
    for name in a.minds.split(','):
        pkg = Package(CK[name], device=a.device); tok = pkg.tok
        enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
        sep, nl = enc(_SEP), enc('\n')
        mind = {}
        for l in open(MIND_LOG.format(name)):
            r = json.loads(l); mind[(r['sub'], r['doc'], r['L'])] = r
        for task in ('affine_map_new', 'shift_wide_new'):
            for seed in range(a.seed, a.seed + a.docs):
                ex = b2_example(tok, task, 8, seed); header, units = split_units(ex)
                pack = enc(header) + sum((enc(u) for u in units[:-1]), [])
                lead, v, tail = re.match(r'^(\s*)(\S+)(\s*->.*)$', units[-1], re.S).groups()
                K = _recall_key(random.Random(seed * 31 + 7))
                field = 'x' if task.startswith('affine') else 'letter'
                row = enc(f' Record: {K} | {field}: {v}.') + nl
                for L in [int(x) for x in a.lengths.split(',')]:
                    fill = sep + filler_block(filler, L, seed, L) + sep
                    ans, info = pkg.apply(pack + fill + row, f'{lead}{K}{tail}', {'pack': pack, 'query': lead + '{v}' + tail},
                                          dilate=8.0 if L >= 4096 else None)
                    ok = int(info.get('first_token') == ex.target_token_id)
                    mr = mind.get((task, seed, L), {})
                    ref = (mr.get('exp_dil8_c4') if L >= 4096 else mr.get('exp_c4')) or [None]
                    rec = dict(mind=name, task=task, doc=seed, L=L, key=K, value_true=v, value_fetched=info.get('value'),
                               path=info.get('path'), answer=ans, free=ok, mind_harness_free=ref[0])
                    rows.append(rec)
                print(name, task, 'done', flush=True)
        for L in [int(x) for x in a.lengths.split(',')]:
            R = [r for r in rows if r['mind'] == name and r['L'] == L]
            got = sum(r['free'] for r in R) / len(R)
            ref = [r['mind_harness_free'] for r in R if r['mind_harness_free'] is not None]
            summary[f'{name}/{L}'] = dict(n=len(R), package_free=round(got, 3),
                                          harness_free=round(sum(ref) / len(ref), 3) if ref else None,
                                          value_right=round(sum(r['value_fetched'] == r['value_true'] for r in R) / len(R), 3))
            print(name, L, summary[f'{name}/{L}'], flush=True)
        # the story check: the rule as WORDS vs as a REGISTER skill (trained affine format), asked for weight 4
        story = (ROOT / 'v16' / 'prompts' / 'story_L2048_s1.txt').read_text()
        demos3 = "Map rule.\n2 -> 5\n3 -> 7\n5 -> 11"                              # as registered
        demos8 = "Map rule.\n0 -> 1\n1 -> 3\n2 -> 5\n3 -> 7\n5 -> 11\n6 -> 13\n7 -> 15\n8 -> 17"   # the trained shot count
        for label, sess, skill in (('rule as the story\'s words', story, {'pack': RULE_WORDS, 'query': '\n{v} ->'}),
                                   ('rule as 3 register demonstrations', demos3 + '\n\n' + story, {'pack': demos3, 'query': '\n{v} ->'}),
                                   ('rule as 8 register demonstrations', demos8 + '\n\n' + story, {'pack': demos8, 'query': '\n{v} ->'})):
            ans, info = pkg.apply_value(sess, skill, '4')
            summary[f'{name}/story: {label}'] = dict(answer=ans, skill_found=info.get('skill_found', 1), says_9=int('9' in ans))
            print(name, 'story', label, repr(ans), flush=True)
    json.dump(dict(summary=summary, rows=rows), open(a.out + '.json', 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
