"""External LLM "brain" client + offline conversation dataset builder.

The duplex model is a *voice interface*, not the reasoner: semantics are
outsourced to any OpenAI-compatible local endpoint (ollama, llama.cpp, vLLM,
LM Studio). This module is a thin, dependency-light client (`requests`; uses the
`openai` SDK if installed) plus an offline builder that turns ASR-corpus
transcripts into (user, assistant-reply) conversation data for Stage C.

Two-layer conversation memory (see plan Stage D): verbatim history lives here in
the LLM's text chat context; the PAM state carries only acoustic/conversational
gist. So long-conversation recall is guaranteed by the brain, not the 100M state.

Endpoints (examples):
    ollama    : OLLAMA -> http://localhost:11434/v1 , model e.g. qwen2.5:3b-instruct
    llama.cpp : http://localhost:8080/v1
    vLLM      : http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional

DEFAULT_SYSTEM = (
    'You are a concise, friendly voice assistant. Reply in the same language as '
    'the user (Hindi, Gujarati, or English). Keep replies short and speakable — '
    'one or two sentences, no markdown, no lists, no emojis.'
)


@dataclass
class BrainConfig:
    base_url: str = field(default_factory=lambda: os.environ.get(
        'BRAIN_BASE_URL', 'http://localhost:11434/v1'))
    api_key: str = field(default_factory=lambda: os.environ.get('BRAIN_API_KEY', 'ollama'))
    model: str = field(default_factory=lambda: os.environ.get(
        'BRAIN_MODEL', 'qwen2.5:3b-instruct'))
    temperature: float = 0.7
    max_tokens: int = 256
    timeout: float = 120.0
    system_prompt: str = DEFAULT_SYSTEM


class BrainClient:
    """OpenAI-compatible chat client (streaming + non-streaming)."""

    def __init__(self, config: Optional[BrainConfig] = None):
        self.config = config or BrainConfig()

    # ── low-level chat ------------------------------------------------------
    def chat(self, messages: List[Dict], stream: bool = False, **overrides):
        payload = {
            'model': overrides.get('model', self.config.model),
            'messages': messages,
            'temperature': overrides.get('temperature', self.config.temperature),
            'max_tokens': overrides.get('max_tokens', self.config.max_tokens),
            'stream': stream,
        }
        url = self.config.base_url.rstrip('/') + '/chat/completions'
        headers = {'Authorization': f'Bearer {self.config.api_key}',
                   'Content-Type': 'application/json'}
        if stream:
            return self._chat_stream(url, headers, payload)
        return self._chat_once(url, headers, payload)

    def _chat_once(self, url, headers, payload) -> str:
        import requests
        r = requests.post(url, headers=headers, json=payload, timeout=self.config.timeout)
        r.raise_for_status()
        data = r.json()
        return data['choices'][0]['message']['content'].strip()

    def _chat_stream(self, url, headers, payload) -> Iterator[str]:
        import requests
        with requests.post(url, headers=headers, json=payload, stream=True,
                           timeout=self.config.timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith('data:'):
                    continue
                chunk = line[len('data:'):].strip()
                if chunk == '[DONE]':
                    break
                try:
                    obj = json.loads(chunk)
                    delta = obj['choices'][0].get('delta', {}).get('content')
                    if delta:
                        yield delta
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    # ── convenience --------------------------------------------------------
    def reply(self, user_text: str, history: Optional[List[Dict]] = None,
              stream: bool = False, system: Optional[str] = None):
        messages = [{'role': 'system', 'content': system or self.config.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({'role': 'user', 'content': user_text})
        return self.chat(messages, stream=stream)

    def probe(self) -> bool:
        try:
            out = self.reply('Say OK.')
            print(f'brain OK ({self.config.model} @ {self.config.base_url}): {out[:60]!r}')
            return True
        except Exception as e:  # noqa: BLE001
            print(f'brain probe FAILED: {type(e).__name__}: {e}')
            return False


class Conversation:
    """Rolling verbatim history for the brain (the long-term memory layer)."""

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM, max_turns: int = 24):
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.turns: List[Dict] = []

    def add_user(self, text: str) -> None:
        self.turns.append({'role': 'user', 'content': text})
        self._trim()

    def add_assistant(self, text: str) -> None:
        self.turns.append({'role': 'assistant', 'content': text})
        self._trim()

    def _trim(self) -> None:
        if len(self.turns) > 2 * self.max_turns:
            self.turns = self.turns[-2 * self.max_turns:]

    def messages(self) -> List[Dict]:
        return [{'role': 'system', 'content': self.system_prompt}] + self.turns


# ── Offline conversation dataset builder (Stage C teacher data) ──────────────

def build_conversation_dataset(
    user_rows: Iterable[Dict],
    out_path: str,
    client: Optional[BrainClient] = None,
    system_prompt: str = DEFAULT_SYSTEM,
    reply_fn: Optional[Callable[[str, str], str]] = None,
    max_examples: int = 0,
) -> int:
    """Generate assistant replies for user utterances -> jsonl.

    Each line: {"user": <transcript>, "reply": <llm reply>, "lang": <lang>}.
    `user_rows` items must have 'text' (and optionally 'lang'). Pass either a
    `client` (real endpoint) or a `reply_fn(user_text, lang)->str` (tests/offline).
    Rows that error are skipped so a flaky endpoint never aborts a long build.
    """
    if client is None and reply_fn is None:
        client = BrainClient()

    def _reply(text: str, lang: str) -> str:
        if reply_fn is not None:
            return reply_fn(text, lang)
        return client.reply(text, system=system_prompt)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, 'w', encoding='utf-8') as f:
        for row in user_rows:
            text = str(row.get('text', '')).strip()
            if not text:
                continue
            lang = str(row.get('lang', 'en'))
            try:
                reply = _reply(text, lang).strip()
            except Exception as e:  # noqa: BLE001
                print(f'  skip (brain error): {type(e).__name__}: {e}')
                continue
            if not reply:
                continue
            f.write(json.dumps({'user': text, 'reply': reply, 'lang': lang},
                               ensure_ascii=False) + '\n')
            n += 1
            if n % 50 == 0:
                print(f'  built {n} conversations...')
            if max_examples and n >= max_examples:
                break
    print(f'Wrote {n} conversations -> {out}')
    return n


def load_conversation_dataset(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _cli():
    p = argparse.ArgumentParser(description='Brain client + conversation builder')
    p.add_argument('--probe', action='store_true', help='ping the endpoint')
    p.add_argument('--build_dataset', action='store_true')
    p.add_argument('--out', default='data/duplex_conversations.jsonl')
    p.add_argument('--languages', default='hindi,gujarati')
    p.add_argument('--n_per_lang', type=int, default=500)
    p.add_argument('--n_english', type=int, default=500)
    p.add_argument('--base_url', default='')
    p.add_argument('--model', default='')
    args = p.parse_args()

    cfg = BrainConfig()
    if args.base_url:
        cfg.base_url = args.base_url
    if args.model:
        cfg.model = args.model
    client = BrainClient(cfg)

    if args.probe:
        client.probe()
        return

    if args.build_dataset:
        from v11.duplex.audio_data import load_asr_rows
        langs = [s.strip() for s in args.languages.split(',') if s.strip()]
        rows = load_asr_rows(languages=langs, n_per_lang=args.n_per_lang,
                             include_english=args.n_english > 0, n_english=args.n_english)
        user_rows = [{'text': r['text'], 'lang': r.get('lang', 'en')} for r in rows]
        build_conversation_dataset(user_rows, args.out, client=client)
        return

    p.print_help()


if __name__ == '__main__':
    _cli()
