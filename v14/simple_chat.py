"""SIMPLE-CHAT probe (PRE-REGISTERED 2026-11-05, EXPERIMENTS_V14 "SIMPLE-CHAT"). Items written by Library, held out from every chat
source (they go into decontam_items.json). ChatML user / assistant, greedy, <= 48 new tokens, stop at <|im_end|> (or a new
<|im_start|>). Strict scoring per type:
  yesno     the whole answer is YES or NO (case-insensitive, optional final period) AND correct
  short     the first line, normalised, equals a gold form, with nothing else beyond 5 words
  instr     a verifiable format check passes
  smalltalk the reply is non-empty, <= 40 words, has no role-token leak, no trigram repeated 3 times, and ends at a sentence end
    python v14/simple_chat.py --ckpt <weights> --out logs/v14_simple_chat_<tag>_<hash>_<stamp>.jsonl [--device cuda]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'v16')):
    if p not in sys.path:
        sys.path.insert(0, p)

YN = " Answer only YES or NO."
YESNO = [("Is the sun a star?", "YES"), ("Is ice colder than boiling water?", "YES"), ("Do fish breathe with lungs?", "NO"),
         ("Is a triangle a shape with four sides?", "NO"), ("Is Paris in France?", "YES"), ("Is the moon bigger than the earth?", "NO"),
         ("Can a dog fly without help?", "NO"), ("Is seven an odd number?", "YES"), ("Is water made of hydrogen and oxygen?", "YES"),
         ("Is a whale a fish?", "NO"), ("Does a week have seven days?", "YES"), ("Is snow usually white?", "YES"),
         ("Is the Pacific an ocean?", "YES"), ("Is ten smaller than three?", "NO"), ("Do plants need light to grow?", "YES"),
         ("Is a spider an insect?", "NO"), ("Is gold a metal?", "YES"), ("Does the earth go around the sun?", "YES"),
         ("Is a square round?", "NO"), ("Is English a language?", "YES"), ("Is fire cold?", "NO"), ("Is a year twelve months long?", "YES"),
         ("Can people breathe under water without equipment?", "NO"), ("Is Mount Everest a mountain?", "YES"),
         ("Is a kilometre shorter than a metre?", "NO"), ("Is milk usually white?", "YES"), ("Do cats bark?", "NO"),
         ("Is the number zero positive?", "NO"), ("Is a carrot a vegetable?", "YES"), ("Is Tokyo in Japan?", "YES"),
         ("Do birds have feathers?", "YES"), ("Is salt sweet?", "NO"), ("Is iron magnetic?", "YES"), ("Is a minute sixty seconds?", "YES"),
         ("Is the Sahara covered in ice?", "NO"), ("Can a car run without any fuel or battery?", "NO"),
         ("Is two plus two equal to four?", "YES"), ("Is a bicycle an animal?", "NO"), ("Is Earth a planet?", "YES"),
         ("Do humans have three hearts?", "NO"), ("Is rain made of water?", "YES"), ("Is a lemon sour?", "YES"),
         ("Is the Atlantic a desert?", "NO"), ("Is a hundred larger than ninety?", "YES"), ("Do trees grow underground only?", "NO"),
         ("Is copper a good conductor of electricity?", "YES"), ("Is Saturn the closest planet to the sun?", "NO"),
         ("Is glass usually transparent?", "YES"), ("Is a penguin able to swim?", "YES"), ("Is Monday a month?", "NO"),
         ("Does sound travel through a vacuum?", "NO"), ("Is DNA found in living cells?", "YES"), ("Is a circle a polygon?", "NO"),
         ("Is the heart a muscle?", "YES"), ("Is oxygen a gas at room temperature?", "YES"), ("Is a bat a bird?", "NO"),
         ("Is Rome the capital of Italy?", "YES"), ("Is eight divided by two equal to five?", "NO"),
         ("Do bees make honey?", "YES"), ("Is the sky green on a clear day?", "NO")]
SHORT = [("What is the capital of France?", ["paris"]), ("How many legs does a spider have?", ["8", "eight"]),
         ("What is the largest planet in our solar system?", ["jupiter"]), ("What colour is grass?", ["green"]),
         ("How many days are in a week?", ["7", "seven"]), ("What is the freezing point of water in Celsius?", ["0", "zero", "0 c", "0 degrees"]),
         ("What is the chemical symbol for gold?", ["au"]), ("What is three times four?", ["12", "twelve"]),
         ("What is the capital of Japan?", ["tokyo"]), ("Which planet is known as the red planet?", ["mars"]),
         ("How many hours are in a day?", ["24", "twenty-four", "twenty four"]), ("What gas do plants take in from the air?", ["carbon dioxide", "co2"]),
         ("What is the opposite of hot?", ["cold"]), ("What is the boiling point of water in Celsius?", ["100", "one hundred"]),
         ("How many continents are there?", ["7", "seven"]), ("What is the capital of Germany?", ["berlin"]),
         ("What is ten minus three?", ["7", "seven"]), ("What do bees make?", ["honey"]), ("What is the chemical symbol for water?", ["h2o"]),
         ("How many sides does a triangle have?", ["3", "three"]), ("What is the capital of Spain?", ["madrid"]),
         ("Which organ pumps blood around the body?", ["heart", "the heart"]), ("What is half of twenty?", ["10", "ten"]),
         ("What colour is the sky on a clear day?", ["blue"]), ("What is the capital of Italy?", ["rome"]),
         ("How many minutes are in an hour?", ["60", "sixty"]), ("What is the closest star to Earth?", ["the sun", "sun"]),
         ("What is five plus six?", ["11", "eleven"]), ("What is the largest ocean on Earth?", ["pacific", "the pacific", "pacific ocean", "the pacific ocean"]),
         ("What is the chemical symbol for iron?", ["fe"]), ("How many months are in a year?", ["12", "twelve"]),
         ("What is the capital of the United Kingdom?", ["london"]), ("What animal says moo?", ["cow", "a cow"]),
         ("What is nine divided by three?", ["3", "three"]), ("What shape has four equal sides and four right angles?", ["square", "a square"]),
         ("What is the capital of Canada?", ["ottawa"]), ("How many wheels does a bicycle have?", ["2", "two"]),
         ("What is frozen water called?", ["ice"]), ("What is the first letter of the English alphabet?", ["a"]),
         ("What is two squared?", ["4", "four"])]
INSTR = [  # (prompt, check name, arg)
    ("Name a fruit. Reply with exactly one word.", 'words', 1), ("Describe the sea in exactly three words.", 'words', 3),
    ("Say hello in exactly two words.", 'words', 2), ("Give me exactly four words about winter.", 'words', 4),
    ("Write exactly five words about a cat.", 'words', 5), ("Answer in exactly one word: what do you drink when thirsty?", 'words', 1),
    ("In exactly three words, describe a sunny day.", 'words', 3), ("Reply with exactly two words about rain.", 'words', 2),
    ("Write one short sentence about trees, all in lowercase letters.", 'lower', None),
    ("Tell me what you had for breakfast, using only lowercase letters.", 'lower', None),
    ("Describe a dog in lowercase letters only.", 'lower', None), ("Write a greeting in all lowercase.", 'lower', None),
    ("Write one sentence about the moon using no capital letters.", 'lower', None),
    ("Say something about music in lowercase only.", 'lower', None),
    ("List three colours, separated by commas, and nothing else.", 'list', 3),
    ("List two animals separated by a comma, nothing else.", 'list', 2),
    ("Give four fruits as a comma-separated list, nothing else.", 'list', 4),
    ("Name three countries, separated by commas, and nothing else.", 'list', 3),
    ("List two metals separated by a comma and nothing else.", 'list', 2),
    ("Write five numbers separated by commas, nothing else.", 'list', 5),
    ("Start your answer with the word Sure. What is a cloud?", 'start', 'sure'),
    ("Begin your reply with the word Yes. Is a tomato red?", 'start', 'yes'),
    ("Start with the word Okay and then say what a planet is.", 'start', 'okay'),
    ("Begin your answer with the word Hello. Who are you?", 'start', 'hello'),
    ("Start your reply with the word Well. Why is the sky blue?", 'start', 'well'),
    ("Begin with the word Today and describe the weather.", 'start', 'today'),
    ("Reply with a single number only: how many legs does a cat have?", 'number', 4),
    ("Answer with a number only: what is six plus seven?", 'number', 13),
    ("Give only a number: how many days are in a week?", 'number', 7),
    ("Reply with just a number: what is ten times ten?", 'number', 100),
    ("Answer with a single number: how many sides does a square have?", 'number', 4),
    ("Only a number please: what is fifteen minus five?", 'number', 10),
    ("Reply with a number and nothing else: how many hours are in a day?", 'number', 24),
    ("Just the number: what is two plus two?", 'number', 4),
    ("Reply with exactly one word: the colour of snow.", 'words', 1),
    ("List three vegetables separated by commas, nothing else.", 'list', 3),
    ("Write a short sentence about books in lowercase only.", 'lower', None),
    ("Start your answer with the word Indeed. Is water wet?", 'start', 'indeed'),
    ("Answer with only a number: what is nine minus four?", 'number', 5),
    ("In exactly two words, name a big animal.", 'words', 2)]
SMALLTALK = [[t] for t in ["Hi! How are you today?", "Good morning! Did you sleep well?", "What's your favourite season and why?",
                           "I just finished a long run. Any tips for recovery?", "Hey, what are you up to?",
                           "I'm feeling a bit tired today.", "Thanks for your help earlier!", "Do you like rainy days?",
                           "What would you recommend for a quiet evening?", "I'm learning to cook. Any easy dish to start with?"]] + [
    ["Hello!", "Can you suggest a good book to read?"], ["Hi there.", "What's a nice thing to do on a weekend?"],
    ["Good evening.", "I had a busy day at work."], ["Hey!", "Do you think it will rain tomorrow?"],
    ["Hello, friend.", "What's your favourite colour?"], ["Hi.", "I'm planning a small trip. Any advice?"],
    ["Good afternoon!", "I'm bored. Any ideas?"], ["Hi!", "What makes a good morning routine?"],
    ["Hello.", "Tell me something cheerful."], ["Hey there.", "How do you stay focused?"],
    ["Hi!", "I got a new plant.", "How often should I water it?"], ["Hello.", "I'm making tea.", "Do you prefer tea or coffee?"],
    ["Hey.", "I started learning the guitar.", "Any tips for a beginner?"], ["Hi there!", "I just moved to a new city.", "How can I meet people?"],
    ["Good morning.", "It's my friend's birthday.", "What's a simple gift idea?"], ["Hello!", "I want to exercise more.", "Where should I start?"],
    ["Hi.", "I'm reading about space.", "What's one fun fact?"], ["Hey!", "I baked bread today.", "Any tips to make it softer?"],
    ["Hello.", "I can't decide what to watch.", "Comedy or drama?"], ["Hi!", "I'm nervous about an exam.", "Any advice?"]]
IM_S, IM_E = '<|im_start|>', '<|im_end|>'


def chat(turns):
    """user turns, with a short neutral assistant reply between them (the probe scores only the last reply)."""
    out = ''
    for i, t in enumerate(turns):
        out += f'{IM_S}user\n{t}{IM_E}\n{IM_S}assistant\n'
        if i < len(turns) - 1:
            out += f'Sure.{IM_E}\n'
    return out


def norm(s):
    return re.sub(r'[^a-z0-9 \-]', '', s.lower()).strip()


def check(kind, item, ans):
    a = ans.strip()
    if kind == 'yesno':
        m = re.fullmatch(r'(?i)\s*(yes|no)\.?\s*', a)
        return int(bool(m) and m.group(1).upper() == item[1])
    if kind == 'short':
        first = norm(a.split('\n')[0]) if a else ''
        return int(any(first in (g, 'the ' + g, 'it is ' + g) for g in item[1]))
    if kind == 'instr':
        _, name, arg = item; w = re.findall(r"[A-Za-z0-9'\-]+", a)
        if name == 'words':
            return int(len(w) == arg)
        if name == 'lower':
            return int(bool(a) and a == a.lower() and any(c.isalpha() for c in a))
        if name == 'list':
            parts = [p.strip(' .') for p in a.split('\n')[0].split(',')]
            return int(len(parts) == arg and all(p and len(p.split()) <= 3 for p in parts))
        if name == 'start':
            return int(bool(w) and w[0].lower() == arg)
        if name == 'number':
            return int(bool(re.fullmatch(r'\s*' + str(arg) + r'\.?\s*', a)))
    if kind == 'smalltalk':
        words = a.split(); tri = collections.Counter(tuple(words[i:i + 3]) for i in range(len(words) - 2))
        return int(bool(words) and len(words) <= 40 and IM_S not in a and 'assistant\n' not in a and 'user\n' not in a
                   and (max(tri.values()) < 3 if tri else True) and a.rstrip()[-1:] in '.!?')
    return 0


def items():
    return ([('yesno', (q + YN, a), chat([q + YN])) for q, a in YESNO] + [('short', (q, g), chat([q])) for q, g in SHORT]
            + [('instr', it, chat([it[0]])) for it in INSTR] + [('smalltalk', t, chat(t)) for t in SMALLTALK])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--out', required=True); ap.add_argument('--device', default='cuda')
    ap.add_argument('--max_new', type=int, default=48)
    a = ap.parse_args()
    from v16.package import Package
    pkg = Package(a.ckpt, device=a.device); tok = pkg.tok; mind = pkg.mind
    stop = set(pkg.stops[2:]) | {tok.convert_tokens_to_ids(IM_S)} if IM_S in tok.get_vocab() else set(pkg.stops[2:])
    out = open(a.out, 'w'); res = collections.defaultdict(list)
    for kind, item, prompt in items():
        ids = tok.encode(prompt, add_special_tokens=False)
        lg, st, off = mind.feed(ids, None, 0); gen = []
        for _ in range(a.max_new):
            t = int(lg.argmax())
            if t in stop:
                break
            gen.append(t); lg, st, off = mind.feed([t], st, off)
        ans = tok.decode(gen)
        ok = check(kind, item, ans); res[kind].append(ok)
        out.write(json.dumps(dict(kind=kind, prompt=prompt, answer=ans, right=ok)) + '\n')
    summ = {k: round(sum(v) / len(v), 3) for k, v in res.items()}
    summ['mean'] = round(sum(summ.values()) / len(summ), 3); summ['n'] = {k: len(v) for k, v in res.items()}
    print('SIMPLE-CHAT ' + json.dumps(summ), flush=True)


if __name__ == '__main__':
    main()
