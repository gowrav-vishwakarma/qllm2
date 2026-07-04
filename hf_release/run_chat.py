#!/usr/bin/env python3
"""Chat with QLLM V11 PAM checkpoint (self-contained HF release)."""

from __future__ import annotations

import argparse
import select
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

from modeling_qllm import V11LM, load_model

IM_START = '<|im_start|>'
IM_END = '<|im_end|>'
THINK_START = '<think>'
THINK_END = '</think>'
DEFAULT_SYSTEM = 'You are a helpful assistant.'
NO_THINK_HINT = (
    'Answer directly in plain text. Do not use '
    f'{THINK_START} or {THINK_END} reasoning blocks.'
)

BANNER_WIDTH = 60
PASTE_IDLE_SEC = 0.08
PASTE_MAX_SEC = 1.0


def get_chat_tokenizer(vocab_size: int = 50261):
    """Build ChatML tokenizer matching checkpoint vocab (50259 legacy or 50261 v2)."""
    tok = AutoTokenizer.from_pretrained('gpt2')
    extras = [IM_START, IM_END]
    if vocab_size >= 50261:
        extras.extend([THINK_START, THINK_END])
    tok.add_special_tokens({'additional_special_tokens': extras})
    tok.pad_token = tok.eos_token
    if len(tok) != vocab_size:
        raise ValueError(
            f'Tokenizer vocab {len(tok)} does not match checkpoint vocab_size {vocab_size}'
        )
    return tok


def format_chat_prompt(user_text: str, system: str = DEFAULT_SYSTEM) -> str:
    return (
        f'{IM_START}system\n{system}{IM_END}\n'
        f'{IM_START}user\n{user_text.strip()}{IM_END}\n'
        f'{IM_START}assistant\n'
    )


def format_chat_messages(
    messages: list[dict],
    *,
    default_system: str = DEFAULT_SYSTEM,
) -> str:
    sys_msg = default_system
    for msg in messages:
        if (msg.get('role') or '').strip().lower() == 'system' and (msg.get('content') or '').strip():
            sys_msg = (msg.get('content') or '').strip()
            break
    parts = [f'{IM_START}system\n{sys_msg}{IM_END}\n']
    for msg in messages:
        role = (msg.get('role') or '').strip().lower()
        content = (msg.get('content') or '').strip()
        if role in ('system', '') or not content:
            continue
        parts.append(f'{IM_START}{role}\n{content}{IM_END}\n')
    return ''.join(parts)


def format_chat_prompt_from_messages(
    messages: list[dict],
    *,
    default_system: str = DEFAULT_SYSTEM,
) -> str:
    return format_chat_messages(messages, default_system=default_system) + f'{IM_START}assistant\n'


def strip_thinking(text: str) -> str:
    """Drop ``<think>…</think>`` blocks (and unclosed tails)."""
    out = text
    while THINK_START in out:
        start = out.find(THINK_START)
        end = out.find(THINK_END, start + len(THINK_START))
        if end == -1:
            out = out[:start]
            break
        out = out[:start] + out[end + len(THINK_END):]
    return out.strip()


def print_welcome(*, system_prompt: str, no_think: bool = False) -> None:
    print('QLLM V11 PAM multi-turn chat')
    print('  Paste multi-line prompts as one message')
    print('  Empty line → new chat    exit → quit')
    if no_think:
        print('  Thinking blocks: disabled (--no-think)')
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
    if not sys.stdin.isatty():
        return
    while _stdin_has_buffered_input():
        if sys.stdin.readline() == '':
            break


def _absorb_paste_lines(first_line: str) -> str:
    lines = [first_line]
    if not sys.stdin.isatty():
        while _stdin_has_buffered_input():
            extra = sys.stdin.readline()
            if extra == '':
                break
            lines.append(extra.rstrip('\n\r'))
        return '\n'.join(lines)

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


def read_user_input() -> tuple[str, Optional[str]]:
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
    model: V11LM,
    tokenizer,
    messages: list[dict],
    *,
    device: torch.device,
    im_end_id: int,
    max_new_tokens: int,
    temperature: float,
    max_prompt_tokens: Optional[int] = None,
    default_system: str = DEFAULT_SYSTEM,
    strip_thinking_blocks: bool = False,
) -> tuple[str, bool, int, str]:
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
    n_new = out.shape[1] - ids.shape[1]
    gen_ids = out[0, ids.shape[1]:].tolist()
    stopped = im_end_id in gen_ids
    reply = tokenizer.decode(gen_ids, skip_special_tokens=False)
    raw = reply.split(IM_END, 1)[0].strip()
    if strip_thinking_blocks:
        reply = strip_thinking(raw)
    else:
        reply = raw
    return reply, stopped, n_new, raw


def _load_system_prompt(args) -> str:
    if args.system_file:
        text = Path(args.system_file).read_text(encoding='utf-8').strip()
        if not text:
            raise SystemExit(f'--system-file is empty: {args.system_file}')
        return text
    if args.system:
        return args.system.strip()
    return DEFAULT_SYSTEM


def main() -> None:
    p = argparse.ArgumentParser(description='Chat with QLLM V11 PAM (HF release)')
    p.add_argument('--checkpoint', default='qllm_v11_e3k3_chat.pt')
    p.add_argument('--config', default='config.json')
    p.add_argument('--prompt', default='')
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
    p.add_argument(
        '--no-think',
        action='store_true',
        help='Discourage and strip <think> blocks from replies',
    )
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, device=device)
    tokenizer = get_chat_tokenizer(model.config.vocab_size)
    supports_think = model.config.vocab_size >= 50261

    system_prompt = _load_system_prompt(args)
    no_think = args.no_think
    if no_think and not supports_think:
        print('Note: --no-think ignored (legacy vocab 50259 has no thinking tokens)')
        no_think = False
    if no_think:
        system_prompt = f'{system_prompt}\n\n{NO_THINK_HINT}'.strip()

    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    max_prompt_tokens = model.config.max_seq_len - args.max_new_tokens

    if args.prompt:
        messages = [{'role': 'user', 'content': args.prompt.strip()}]
        reply, stopped, n_new, _ = generate_reply(
            model,
            tokenizer,
            messages,
            device=device,
            im_end_id=im_end_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_prompt_tokens=max_prompt_tokens,
            default_system=system_prompt,
            strip_thinking_blocks=no_think,
        )
        print(f'User> {args.prompt}')
        print(f'Assistant> {reply}')
        print(f'(new_tokens={n_new}, stopped_on_im_end={stopped})')
        return

    session_id = 1
    messages: list[dict] = []

    print_welcome(system_prompt=system_prompt, no_think=no_think)
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
        reply, stopped, n_new, _ = generate_reply(
            model,
            tokenizer,
            messages,
            device=device,
            im_end_id=im_end_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_prompt_tokens=max_prompt_tokens,
            default_system=system_prompt,
            strip_thinking_blocks=no_think,
        )
        messages.append({'role': 'assistant', 'content': reply})
        print(f'Assistant> {reply}')
        if not stopped:
            print(f'  (warning: did not stop on {IM_END}, new_tokens={n_new})')
        _discard_pending_stdin()


if __name__ == '__main__':
    main()
