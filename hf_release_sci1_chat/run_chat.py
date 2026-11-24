#!/usr/bin/env python3
"""Multi-turn ChatML chat with SCI-1 (self-contained: torch + transformers only).

    python run_chat.py                                   # interactive
    python run_chat.py --prompt "What is the capital of France?"     # one turn, then exit
    python run_chat.py --prompt "Hi" --prompt "Tell me more"         # several turns, then exit

The model sees the conversation in ChatML, as in training:

    [<|im_start|>system\\n{system}<|im_end|>\\n]
    <|im_start|>user\\n{message}<|im_end|>\\n<|im_start|>assistant\\n{reply}<|im_end|>\\n ...

and a reply stops at <|im_end|>. The recurrent state is carried from turn to
turn (O(1) memory: the conversation is never re-read), so each turn only
feeds its own new tokens.

Interactive commands:
  /new            new session (forget the conversation)
  /system TEXT    set the system message and start a new session (empty = none)
  /greedy         greedy decoding (default)     /sample   sampled (t 0.8, top-k 50, top-p 0.9, rep 1.2)
  /max N          max new tokens per reply (default 200)
  /history        show the raw ChatML the model has seen
  /quit           exit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from modeling_pam2 import IM_END, IM_START, TokenStreamer, load_model, load_tokenizer  # noqa: E402

SAMPLED = dict(temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.2)
GREEDY = dict(temperature=0.0, top_k=0, top_p=0.0, repetition_penalty=1.0)


class ChatSession:
    """One conversation. Keeps the model state between turns plus the token history."""

    def __init__(self, model, tok, system: str = '', device='cpu'):
        self.model, self.tok, self.system, self.device = model, tok, system, device
        self.end_id = tok.convert_tokens_to_ids(IM_END)
        self.state = None          # model state after every token in `history` except `pending`
        self.pending = []          # tokens already in the transcript, not yet fed to the model
        self.history = []          # every token of the conversation (for /history and the penalty)

    def reply(self, message: str, *, max_new_tokens=200, seed=None, stream=False, **sampling):
        text = ''
        if not self.history and self.system:
            text += f'{IM_START}system\n{self.system}{IM_END}\n'
        text += f'{IM_START}user\n{message.strip()}{IM_END}\n{IM_START}assistant\n'
        fresh = self.tok.encode(text)
        feed = self.pending + fresh          # `pending` is already in `history`
        x = torch.tensor([feed], device=self.device)
        ctx = torch.tensor([self.history + fresh], device=self.device)
        gen = None
        if seed is not None and sampling.get('temperature', 0) > 0:
            gen = torch.Generator(device=self.device).manual_seed(seed)
        streamer = TokenStreamer(self.tok.decode, stop_ids=(self.end_id,)) if stream else None
        t0 = time.time()
        out, self.state = self.model.generate(
            x, max_new_tokens=max_new_tokens, eos_token_id=self.end_id, state=self.state,
            return_state=True, penalty_context=ctx, generator=gen, streamer=streamer, **sampling)
        dt = time.time() - t0
        new = out[0, len(feed):].tolist()
        stopped = bool(new) and new[-1] == self.end_id
        body = new[:-1] if stopped else new
        # The state has consumed everything except the last returned token; the closing
        # <|im_end|>\n of this turn goes in front of the next turn's tokens.
        self.history += fresh + body + [self.end_id] + self.tok.encode('\n')
        self.pending = ([] if stopped else new[-1:]) + [self.end_id] + self.tok.encode('\n')
        reply = self.tok.decode(body).strip()
        return reply, {'new_tokens': len(new), 'stopped_on_im_end': stopped, 'seconds': dt,
                       'context_tokens': len(self.history)}

    def transcript(self) -> str:
        return self.tok.decode(self.history)


def main() -> None:
    ap = argparse.ArgumentParser(description='SCI-1 multi-turn chat (ChatML)')
    ap.add_argument('--checkpoint', default=str(HERE / 'sci1_chat_386m.pt'))
    ap.add_argument('--system', default='', help='system message (default: none, as in the repo chat loop)')
    ap.add_argument('--prompt', action='append', default=None,
                    help='non-interactive: one user turn (repeat for several turns), print, exit')
    ap.add_argument('--sample', action='store_true', help='sampled decoding (default greedy)')
    ap.add_argument('--temperature', type=float, default=None)
    ap.add_argument('--top_k', type=int, default=None)
    ap.add_argument('--top_p', type=float, default=None)
    ap.add_argument('--repetition_penalty', type=float, default=None)
    ap.add_argument('--max_new_tokens', type=int, default=200)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--device', default='auto', help='auto | cpu | cuda | cuda:N')
    ap.add_argument('--dtype', default='float32', choices=['float32', 'bfloat16'])
    args = ap.parse_args()

    sampling = dict(SAMPLED if args.sample else GREEDY)
    for k in ('temperature', 'top_k', 'top_p', 'repetition_penalty'):
        if getattr(args, k) is not None:
            sampling[k] = getattr(args, k)
    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available())
                          else ('cpu' if args.device == 'auto' else args.device))
    t0 = time.time()
    model = load_model(args.checkpoint, device=device, dtype=getattr(torch, args.dtype))
    tok = load_tokenizer()
    print(f'[SCI-1 {model.count_parameters() / 1e6:.0f}M params, {device}, {args.dtype}, '
          f'loaded in {time.time() - t0:.1f}s]', file=sys.stderr)
    session = ChatSession(model, tok, args.system, device)

    def turn(msg):
        print('model> ', end='', flush=True)
        reply, info = session.reply(msg, max_new_tokens=args.max_new_tokens, seed=args.seed,
                                    stream=True, **sampling)
        print(f"\n   [{info['new_tokens']} tok, {info['seconds']:.1f}s, context {info['context_tokens']} tok, "
              f"stopped_on_im_end={info['stopped_on_im_end']}]\n", flush=True)

    if args.prompt:
        for msg in args.prompt:
            print(f'you> {msg}')
            turn(msg)
        return

    print('SCI-1 chat. /new = new session, /quit = exit, /help = commands\n')
    while True:
        try:
            u = input('you> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not u:
            continue
        if u.startswith('/'):
            cmd, _, arg = u.partition(' ')
            if cmd == '/quit':
                break
            elif cmd == '/new':
                session = ChatSession(model, tok, session.system, device)
                print('-- new session --\n')
            elif cmd == '/system':
                session = ChatSession(model, tok, arg.strip(), device)
                print(f'-- system set, new session: {arg.strip()!r} --\n')
            elif cmd == '/greedy':
                sampling = dict(GREEDY)
                print('-- greedy --')
            elif cmd == '/sample':
                sampling = dict(SAMPLED)
                print('-- sampled (t 0.8, top-k 50, top-p 0.9, rep 1.2) --')
            elif cmd == '/max':
                args.max_new_tokens = int(arg or 200)
                print(f'-- max {args.max_new_tokens} tokens --')
            elif cmd == '/history':
                print(session.transcript() or '(empty)')
            else:
                print(__doc__)
            continue
        turn(u)


if __name__ == '__main__':
    main()
