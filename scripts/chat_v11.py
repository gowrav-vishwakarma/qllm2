#!/usr/bin/env python3
"""Interactive chat with a V11 checkpoint (ChatML, post-SFT or base)."""

import argparse
import select
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v7.data import DEFAULT_SYSTEM, IM_END, format_chat_prompt_from_messages, get_chat_tokenizer
from v11.model import V11LM, get_config

BANNER_WIDTH = 60
PASTE_IDLE_SEC = 0.08
PASTE_MAX_SEC = 1.0


def print_welcome(*, system_prompt: str) -> None:
    print('V11 multi-turn chat')
    print('  Paste multi-line prompts as one message')
    print('  Empty line → new chat    exit → quit')
    if system_prompt != DEFAULT_SYSTEM:
        preview = system_prompt.replace('\n', ' ')
        if len(preview) > 72:
            preview = preview[:69] + '...'
        print(f'  System: {preview}')


def print_new_chat(session_id: int, *, first: bool = False) -> None:
    line = '=' * BANNER_WIDTH
    print(line)
    if first:
        print(f'  NEW CHAT (session {session_id})')
    else:
        print(f'  NEW CHAT (session {session_id}) — history cleared')
    print(line)


def _stdin_has_buffered_input() -> bool:
    """True when more input is already waiting (e.g. rest of a paste)."""
    try:
        return bool(select.select([sys.stdin], [], [], 0)[0])
    except (ValueError, OSError):
        return False


def _readline_from_stdin() -> str:
    line = sys.stdin.readline()
    if line == '':
        raise EOFError
    return line.rstrip('\n\r')


def _discard_pending_stdin() -> None:
    """Drop lines buffered while the model was generating (interactive paste spill)."""
    if not sys.stdin.isatty():
        return
    while _stdin_has_buffered_input():
        if sys.stdin.readline() == '':
            break


def _absorb_paste_lines(first_line: str) -> str:
    """Collect the rest of a multi-line terminal paste into one message."""
    lines = [first_line]
    if not sys.stdin.isatty():
        while _stdin_has_buffered_input():
            extra = sys.stdin.readline()
            if extra == '':
                break
            lines.append(extra.rstrip('\n\r'))
        return '\n'.join(lines)

    # Interactive paste often arrives in one burst; wait briefly so we do not
    # start generating while trailing lines are still landing in stdin.
    last_data = time.monotonic()
    deadline = time.monotonic() + PASTE_MAX_SEC
    while time.monotonic() < deadline:
        if _stdin_has_buffered_input():
            extra = sys.stdin.readline()
            if extra == '':
                break
            lines.append(extra.rstrip('\n\r'))
            last_data = time.monotonic()
        elif time.monotonic() - last_data >= PASTE_IDLE_SEC:
            break
        else:
            time.sleep(0.01)

    return '\n'.join(lines)


def read_user_input() -> Tuple[str, Optional[str]]:
    """Read one user turn. Returns (kind, text) where kind is exit|new_chat|message."""
    _discard_pending_stdin()

    sys.stdout.write('\nUser> ')
    sys.stdout.flush()
    first = _readline_from_stdin()

    if first.strip().lower() == 'exit':
        return 'exit', None
    if not first.strip():
        return 'new_chat', None

    return 'message', _absorb_paste_lines(first)


def generate_reply(
    model,
    tokenizer,
    messages: list[dict],
    *,
    device: torch.device,
    im_end_id: int,
    max_new_tokens: int,
    temperature: float,
    max_prompt_tokens: Optional[int] = None,
    default_system: str = DEFAULT_SYSTEM,
) -> str:
    prompt = format_chat_prompt_from_messages(messages, default_system=default_system)
    ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    if max_prompt_tokens is not None and ids.shape[1] > max_prompt_tokens:
        print(
            f'  (warning: prompt is {ids.shape[1]} tokens; '
            f'limit is {max_prompt_tokens}. Start a new chat to avoid truncation.)'
        )
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.15,
            eos_token_id=im_end_id,
        )
    gen_ids = out[0, ids.shape[1]:].tolist()
    reply = tokenizer.decode(gen_ids, skip_special_tokens=False)
    return reply.split(IM_END, 1)[0].strip()


def _load_system_prompt(args) -> str:
    if args.system_file:
        text = Path(args.system_file).read_text(encoding='utf-8').strip()
        if not text:
            raise SystemExit(f'--system-file is empty: {args.system_file}')
        return text
    if args.system:
        return args.system.strip()
    return DEFAULT_SYSTEM


def main():
    p = argparse.ArgumentParser(description='Chat with a V11 checkpoint')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--preset', default='v11_e3_k3')
    p.add_argument('--max_new_tokens', type=int, default=256)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument(
        '--system',
        default=None,
        help='Custom system prompt for all chats in this session',
    )
    p.add_argument(
        '--system-file',
        default=None,
        help='Read custom system prompt from a file (overrides --system)',
    )
    args = p.parse_args()

    system_prompt = _load_system_prompt(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = get_chat_tokenizer()
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)

    cfg = get_config(args.preset)
    cfg.vocab_size = len(tokenizer)
    model = V11LM(cfg)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    m = model._orig_mod if hasattr(model, '_orig_mod') else model

    max_prompt_tokens = cfg.max_seq_len - args.max_new_tokens
    session_id = 1
    messages: list[dict] = []

    print_welcome(system_prompt=system_prompt)
    print_new_chat(session_id, first=True)

    while True:
        try:
            kind, user = read_user_input()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if kind == 'exit':
            break

        if kind == 'new_chat':
            session_id += 1
            messages = []
            print_new_chat(session_id)
            continue

        messages.append({'role': 'user', 'content': user.strip()})
        reply = generate_reply(
            m,
            tokenizer,
            messages,
            device=device,
            im_end_id=im_end_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_prompt_tokens=max_prompt_tokens,
            default_system=system_prompt,
        )
        messages.append({'role': 'assistant', 'content': reply})
        print(f'Assistant> {reply}')
        _discard_pending_stdin()


if __name__ == '__main__':
    main()
