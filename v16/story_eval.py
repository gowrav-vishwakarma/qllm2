"""Story comprehension stress test (user, Nov 1): a complete story in chapters, other real text between the chapters
to reach 2k / 4k / 8k tokens, then exam-style questions asked in different manners (who, where, how many and why,
order of events, explain, true or false, a rule from the story applied to a new case, an inference, a summary), each in
chat form (ChatML, as in training) and in plain "Question: ... Answer:" form. Free generation, outputs shown. Runs any
body (PAM / T++ / Mamba-2) ALONE, i.e. the mind without the library; the package pass is Library's runner on the same files.
Usage: python v16/story_eval.py --ckpt <path> --arch pam|transformer|mamba2 --lengths 2048,4096,8192 --device cuda
"""
from __future__ import annotations
import argparse, random, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'v16'))
from run_memory_behavioral import _load_v13_sempty  # noqa: E402
from memory_probes.spread import filler_tokens  # noqa: E402
from bench import Reader  # noqa: E402

NL = 198
IM_START, IM_END = '<|im_start|>', '<|im_end|>'

CHAPTERS = [
"""The Lantern of Harrow Mill

Chapter One. The village of Harrow sat where the river Dunn bent twice before reaching the sea. Mira Vell, who was fourteen that spring, lived with her mother and her brother Tomas in the stone house nearest the old mill. The mill had not turned in nine years. Everyone said it belonged to the miller's widow, Agnes Thorne, though she had not set foot in it since her husband drowned. On the first warm evening of April, Mira climbed through the broken shutter of the mill to look for swallows' nests and found, under a heap of sacks, a brass lantern with a green glass door. Inside the lantern, folded small, was a letter addressed to nobody at all. It said only: "The bridge is not safe. Do not let them cross on market day." It was signed with a single letter, H.""",
"""Chapter Two. Mira told no one but Tomas. Tomas was seventeen and worked for the baker, Old Pell, whose rule for prices was famous in Harrow: every loaf cost twice its weight in coins plus one coin for the oven. A loaf of weight two cost five coins, a loaf of weight three cost seven, and a loaf of weight five cost eleven. Tomas laughed at the letter and said the bridge had stood for two hundred years. But three nights later the storm came. The river rose over the lower fields, and the Vell family lost eleven sheep, not to the water but to the wolves that came down from Cold Ridge while the fences lay flat. After the storm Mira hid the brass key she had found hanging inside the lantern beneath the third step of the church stair, because the church was the only building the flood had not touched.""",
"""Chapter Three. The stranger arrived on the Tuesday after the flood, the same day the tax collector always came, which is how people remembered it. He gave his name as Halden and said he was an engineer sent by the county to look at the bridge. He walked its length twice, knelt at the eastern pier, and then went to the inn without a word. That night he found Tomas at the baker's and promised him that if Tomas kept the villagers off the bridge until the county men came, he would pay for Tomas's passage to the city and a year of schooling there. Tomas, who wanted nothing more, agreed. The fair was set for the Saturday. The bridge was closed on the Friday, because Halden had found that the eastern pier had been hollowed out by the flood and would not bear a loaded cart.""",
"""Chapter Four. The fair went ahead on the village green instead of across the river, and it was the smallest fair anyone could remember: four stalls, a fiddler, and Old Pell selling loaves at his usual rule. Agnes Thorne came down from her house for the first time in years and stood a long while looking at the mill. When Mira showed her the lantern, the widow wept, and said the letter was in her husband's hand, written the week before he drowned, and that the H stood for Harrow, not for a man. The county men came on the Monday and began to rebuild the pier. Tomas left for the city on the following Tuesday, carrying one loaf of weight four, for which Old Pell charged him nothing. Mira kept the lantern. She never did find the swallows' nests.""",
]

QUESTIONS = [
    ('who_found', "Who found the brass lantern in the old mill?", 'Mira'),
    ('where_key', "After the storm, where did Mira hide the brass key?", 'third step'),
    ('howmany_why', "How many sheep did the Vell family lose that spring, and what actually killed them?", 'eleven'),
    ('promise', "What did the stranger Halden promise Tomas in return for keeping people off the bridge?", 'city'),
    ('order', "Put these three events in the order they happened: the fair, the storm, Halden's arrival.", 'storm'),
    ('explain_why', "Explain in one sentence why the bridge was closed on the Friday.", 'pier'),
    ('rule_apply', "Old Pell's price rule is given in the story. By that rule, how many coins does a loaf of weight four cost?", '9'),
    ('inference_day', "The tax collector always came on the same weekday as Halden arrived. On which weekday did Tomas leave for the city?", 'Tuesday'),
    ('true_false', "True or false: the mill belonged to Agnes Thorne's daughter.", 'False'),
    ('letter_H', "What did the letter H at the bottom of the letter stand for, according to Agnes Thorne?", 'Harrow'),
    ('summary', "Summarise what happened at the fair in two sentences.", 'green'),
]


def build_prompt(tok, filler, L, seed):
    rng = random.Random(seed)
    story_ids = [len(tok.encode(c, add_special_tokens=False)) for c in CHAPTERS]
    pad_total = max(0, L - sum(story_ids) - 60)
    gaps = 3                                      # other text between the four chapters
    pieces = []
    for i, ch in enumerate(CHAPTERS):
        pieces.append(ch)
        if i < gaps:
            n = pad_total // gaps
            s0 = rng.randrange(0, len(filler) - n - 1)
            other = tok.decode(filler[s0:s0 + n].tolist()).strip()
            pieces.append("\n\n[Other papers from the same drawer]\n\n" + other + "\n\n[The story continues]\n\n")
    return ''.join(pieces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--arch', default='pam', choices=['pam', 'transformer', 'mamba2'])
    ap.add_argument('--lengths', default='2048,4096,8192'); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--max_new', type=int, default=40)
    ap.add_argument('--forms', default='chat,plain')
    a = ap.parse_args()
    model, tok, cfg = _load_v13_sempty(Path(a.ckpt), 'baseline_real_pm', torch.device(a.device)); model.eval()
    R = Reader(model, a.arch)
    fill = filler_tokens(str(ROOT / '.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt'))
    im_end = tok.convert_tokens_to_ids(IM_END) if IM_END in tok.get_vocab() else None
    name = Path(a.ckpt).parent.name; ts = time.strftime('%Y%m%d_%H%M')
    rep = ROOT / 'logs' / f'v16_story_{name}_{ts}.md'
    out = [f"# Story comprehension, {name} ({a.arch}) ALONE (no library), seed {a.seed}, greedy, max_new {a.max_new}\n"]
    for L in [int(x) for x in a.lengths.split(',')]:
        text = build_prompt(tok, fill, L, a.seed)
        pf = ROOT / 'v16' / 'prompts' / f'story_L{L}_s{a.seed}.txt'; pf.parent.mkdir(exist_ok=True); pf.write_text(text)
        base = tok.encode(text, add_special_tokens=False)
        out.append(f"\n## L = {L} (prompt {len(base)} tokens, file {pf.relative_to(ROOT)})\n\n| question | form | expected | generated |\n|---|---|---|---|")
        hits = {f: 0 for f in a.forms.split(',')}
        for qname, q, exp in QUESTIONS:
            for form in a.forms.split(','):
                if form == 'chat':
                    qt = f"\n\n{IM_START}user\n{q}{IM_END}\n{IM_START}assistant\n"; stops = tuple(t for t in (NL, im_end) if t is not None)
                else:
                    qt = f"\n\nQuestion: {q}\nAnswer:"; stops = (NL,)
                ids = base + tok.encode(qt, add_special_tokens=False)
                gen = R.greedy(ids, max_new=a.max_new, stops=stops)
                txt = tok.decode(gen).replace('\n', ' / ').replace('|', '\\|').strip()
                hit = exp.lower() in txt.lower(); hits[form] += int(hit)
                out.append(f"| {qname} | {form} | {exp} | {'OK' if hit else '--'} `{txt}` |")
                print(f"L={L} {qname:14s} {form:5s} exp={exp:10s} {'OK' if hit else '--'} {txt[:100]!r}", flush=True)
        out.append(f"\nkeyword hits: " + ', '.join(f"{f} {hits[f]}/{len(QUESTIONS)}" for f in hits))
        print(f"L={L} keyword hits: " + ', '.join(f"{f} {hits[f]}/{len(QUESTIONS)}" for f in hits), flush=True)
        rep.write_text('\n'.join(out) + '\n')
    print('report', rep)


if __name__ == '__main__':
    main()
