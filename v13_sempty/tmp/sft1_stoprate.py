"""sft1 chat stop-rate (EXPERIMENTS_SEMPY "sft1": stop-on-<|im_end|> rate on the chat probes >= 0.95; base measured too).
The human-eval dump caps answers at 60-120 tokens, so it cannot tell "did not stop" from "still answering"; this measures it:
  stop  = <|im_end|> emitted within MAX_NEW new tokens, with no <|endoftext|> / <|im_start|> before it
  leak  = <think> / </think> / <tool_call> / tool-schema text in the answer of a prompt that offers no tools
  nonen = answer has < 90 % ASCII letters among its letters
Prompts: the 20 chat-format prompts of v16/humaneval/dump.py + 40 natural chat questions (varied styles). Greedy + 2 sampled
seeds (T 0.8, top-k 50, top-p 0.9, rep 1.2, as the dump).
  python v13_sempty/tmp/sft1_stoprate.py <ckpt.pt> <out.jsonl>"""
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import torch  # noqa: E402
from run_memory_behavioral import _load_v13_sempty  # noqa: E402

MAX_NEW = 768
src = (ROOT / 'v16/humaneval/dump.py').read_text().split('\ndef main():')[0]
g = {'__name__': 'dump_prompts', 'sys': sys}
exec(src.replace('W, CK, OUT = Path(sys.argv[1]), sys.argv[2], sys.argv[3]', 'W = Path(".")')
        .replace("N = int(sys.argv[4]) if len(sys.argv) > 4 else 40", 'N = 0')
        .replace('import torch\nfrom mc_eval import Scorer, load_task', ''), g)
U = g['U']
P = [dict(name=p['name'], group=p['group'], prompt=p['prompt'], tools='You can call tools' in p['prompt'] or '<tool_response>' in p['prompt'])
     for p in g['P'] if p['prompt'].rstrip().endswith('<|im_start|>assistant')]
NEW = [
    'hi! how are you today?', 'Can you summarize what mitochondria do in two sentences?', 'whats the boiling point of water at sea level',
    'I have a job interview tomorrow. Any quick advice?', 'Explain recursion to a ten-year-old.', 'Is a tomato a fruit or a vegetable?',
    'Write a haiku about the ocean.', 'What causes the seasons on Earth?', 'Translate "good morning" into French.',
    'How many legs does a spider have?', 'Give me a short definition of entropy.', 'What is the capital of Australia?',
    'Why do we need sleep?', 'Suggest a name for a black cat.', 'What is the difference between weather and climate?',
    'List three renewable energy sources.', 'How do vaccines work?', 'Can you explain what an API is?',
    'My plant leaves are turning yellow. What might be wrong?', 'What is 12 times 12?', 'Who wrote Romeo and Juliet?',
    'Describe the water cycle briefly.', 'What does a CPU do?', 'Tell me a fun fact about octopuses.',
    'How should I store fresh basil?', 'In one sentence: what is machine learning?', 'What is the speed of light?',
    'Help me write a polite email declining a meeting.', 'Why is the ocean salty?', 'What is an atom made of?',
    'Recommend a good habit for studying.', 'thanks for your help!', 'How does a rainbow form?',
    'Explain the difference between a virus and a bacterium.', 'What is photosynthesis? Keep it short.', 'Can dogs see colors?',
    'What is the Pythagorean theorem?', 'Name two noble gases.', 'How do I calculate the area of a circle?',
    'What year did humans first land on the Moon?']
P += [dict(name=f'nat{i + 1}', group='Natural chat', prompt=U(q), tools=False) for i, q in enumerate(NEW)]
LEAK = re.compile(r'<think>|</think>|<tool_call>|<tool_response>|"arguments"\s*:|"name"\s*:\s*"')


def main():
    ck, out = sys.argv[1], sys.argv[2]
    dev = torch.device('cuda')
    m, tok, _ = _load_v13_sempty(Path(ck), 'baseline_real_pm', dev); m.eval()
    im_end = tok.convert_tokens_to_ids('<|im_end|>')
    t0 = time.time(); rows = []
    for p in P:
        ids = torch.tensor([tok.encode(p['prompt'])], device=dev)
        for mode, seed in (('greedy', None), ('s1', 1234), ('s2', 4321)):
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if seed is None:
                    o = m.generate(ids, max_new_tokens=MAX_NEW, temperature=0.0, eos_token_id=im_end)
                else:
                    torch.manual_seed(seed)
                    o = m.generate(ids, max_new_tokens=MAX_NEW, temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.2,
                                   eos_token_id=im_end)
            new = o[0, ids.shape[1]:].tolist()
            txt = tok.decode(new)
            ans = txt.split('<|im_end|>')[0]
            stop = im_end in new and '<|endoftext|>' not in ans and '<|im_start|>' not in ans
            letters = [c for c in ans if c.isalpha()]
            nonen = bool(letters) and sum(c.isascii() for c in letters) / len(letters) < 0.9
            leak = (not p['tools']) and bool(LEAK.search(ans))
            rows.append(dict(name=p['name'], group=p['group'], tools=p['tools'], mode=mode, stop=stop, n_new=len(new), leak=leak,
                             nonen=nonen, answer=ans[:1500]))
        print(p['name'], [r['stop'] for r in rows[-3:]], [r['n_new'] for r in rows[-3:]], flush=True)
    with open(out, 'w') as f:
        for r in rows:
            f.write(json.dumps(dict(ckpt=ck, **r)) + '\n')
    for sel, lab in ((lambda r: True, 'all'), (lambda r: r['mode'] == 'greedy', 'greedy'), (lambda r: r['mode'] != 'greedy', 'sampled')):
        x = [r for r in rows if sel(r)]
        print(f'{lab:8s} n {len(x)} stop {sum(r["stop"] for r in x) / len(x):.3f} leak {sum(r["leak"] for r in x)} '
              f'nonen {sum(r["nonen"] for r in x)} mean_len {sum(r["n_new"] for r in x) / len(x):.0f}', flush=True)
    print(f'done {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
