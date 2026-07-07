"""Gradio UI: talk to V11 duplex checkpoints (audio -> listen/speak/backchannel)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import numpy as np

from v11.duplex.infer import (
    checkpoint_label,
    discover_checkpoints,
    load_checkpoint_meta,
    load_duplex_from_checkpoint,
    predict_thinking,
)

# Cached loaded session (one checkpoint at a time).
_SESSION = {
    'path': None,
    'model': None,
    'encoder': None,
    'has_audio': False,
    'preset': 'duplex_5m',
    'device': None,
}

# Full voice-interface session (Stage C/D unified model).
_VOICE = {'path': None, 'session': None}


def _refresh_checkpoints(root: str):
    paths = discover_checkpoints(root)
    labels = [checkpoint_label(p) for p in paths] if paths else ['(none found)']
    return gr.update(choices=labels, value=labels[0])


def _resolve_path(label: str, root: str) -> str:
    if not label or label == '(none found)':
        return ''
    for p in discover_checkpoints(root):
        if checkpoint_label(p) == label:
            return p
    return ''


def _load_session(checkpoint_label_str: str, preset: str, whisper: str, root: str):
    path = _resolve_path(checkpoint_label_str, root)
    if not path:
        raise gr.Error('No checkpoint selected.')

    if _SESSION['path'] == path and _SESSION['model'] is not None:
        return _checkpoint_info(path)

    model, encoder, has_audio, preset_name = load_duplex_from_checkpoint(
        path, preset=preset if preset != 'auto' else None, whisper=whisper,
    )
    _SESSION.update({
        'path': path,
        'model': model,
        'encoder': encoder,
        'has_audio': has_audio,
        'preset': preset_name,
        'device': next(model.parameters()).device,
    })
    return _checkpoint_info(path)


def _checkpoint_info(path: str) -> str:
    meta = load_checkpoint_meta(path)
    lines = [f'Loaded `{checkpoint_label(path)}`']
    if meta:
        lines.append(f"- preset: `{meta.get('preset', '?')}`")
        if 'dataset' in meta:
            lines.append(f"- dataset: `{meta.get('dataset')}`")
        if 'languages' in meta:
            lines.append(f"- languages: `{meta.get('languages')}`")
        if 'best_val_think_acc' in meta:
            lines.append(f"- best val think acc: **{meta['best_val_think_acc']:.1%}**")
        elif 'final_val_think_acc' in meta:
            lines.append(f"- val think acc: **{meta['final_val_think_acc']:.1%}**")
    mode = 'Stage 1 audio (Whisper)' if _SESSION['has_audio'] else 'Stage 0 text proxy'
    lines.append(f'- mode: **{mode}**')
    return '\n'.join(lines)


def analyze(
    checkpoint_label_str: str,
    preset: str,
    whisper: str,
    root: str,
    audio: Optional[Tuple[int, np.ndarray]],
    text_context: str,
) -> Tuple[str, str]:
    info = _load_session(checkpoint_label_str, preset, whisper, root)
    if _SESSION['model'] is None:
        return info, 'Model not loaded.'

    if _SESSION['has_audio']:
        if audio is None:
            return info, 'Record or upload audio — this checkpoint expects Stage 1 audio input.'
    elif not text_context.strip():
        text_context = 'hello there'

    pred = predict_thinking(
        _SESSION['model'],
        _SESSION['encoder'],
        audio=audio,
        text_context=text_context,
        device=_SESSION['device'],
        checkpoint=_SESSION['path'] or '',
        preset=_SESSION['preset'],
    )
    reaction = {
        'listen': 'User has the floor — assistant should **listen** (wait).',
        'speak': 'Assistant should **speak** (take a turn).',
        'backchannel': 'Assistant should **backchannel** (short ack while user continues).',
    }[pred.thinking]
    return info, pred.to_markdown() + f'\n\n> {reaction}'


def _load_voice_session(label: str, tokenizer_dir: str, whisper: str, mimi: str,
                        use_brain: bool, base_url: str, model_name: str, root: str) -> str:
    from v11.duplex.infer_stream import DuplexSession, load_session
    path = _resolve_path(label, root)
    if not path:
        raise gr.Error('No checkpoint selected.')
    key = (path, use_brain, base_url, model_name, tokenizer_dir)
    if _VOICE['path'] == key and _VOICE['session'] is not None:
        return f'Voice session ready: `{checkpoint_label(path)}`'
    loaded = load_session(path, tokenizer_dir=tokenizer_dir or '', whisper=whisper, mimi=mimi)
    brain = None
    if use_brain:
        from v11.duplex.brain import BrainClient, BrainConfig
        cfg = BrainConfig()
        if base_url:
            cfg.base_url = base_url
        if model_name:
            cfg.model = model_name
        brain = BrainClient(cfg)
    _VOICE['session'] = DuplexSession(loaded, brain=brain)
    _VOICE['path'] = key
    brain_note = f'brain=`{brain.config.model}`' if brain else 'brain=off (echo)'
    return f'Voice session ready: `{checkpoint_label(path)}` ({brain_note})'


def voice_chat(audio, lang: str):
    import numpy as np
    if _VOICE['session'] is None:
        raise gr.Error('Load a voice checkpoint first.')
    if audio is None:
        raise gr.Error('Record or upload speech.')
    sr, wav = audio
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    peak = float(np.abs(wav).max()) or 1.0
    wav = wav / peak
    result = _VOICE['session'].converse(wav, int(sr), lang=lang)
    transcript = (f"**You:** {result['transcript'] or '(empty)'}\n\n"
                  f"**Assistant:** {result['reply'] or '(no reply)'}")
    reply_audio = None
    if result['audio'].size:
        reply_audio = (result['sample_rate'], result['audio'])
    return transcript, reply_audio


def _reset_voice():
    if _VOICE['session'] is not None:
        _VOICE['session'].reset()
    return '*Conversation reset.*', None


def build_ui(default_root: str = '.') -> gr.Blocks:
    ckpts = discover_checkpoints(default_root)
    labels = [checkpoint_label(p) for p in ckpts] or ['(none found)']
    default_ckpt = labels[0]

    with gr.Blocks(title='V11 Duplex Voice Demo') as demo:
        gr.Markdown('# V11 Duplex — Voice Demo')
        with gr.Tabs():
            with gr.Tab('Voice interface (Stage C/D)'):
                gr.Markdown(
                    'Talk to the unified duplex model: **speech in → transcript → '
                    'LLM brain → spoken reply**. One PAM state spans the session; '
                    'verbatim history lives in the brain. Point the brain at any '
                    'OpenAI-compatible endpoint (ollama, llama.cpp, vLLM).'
                )
                with gr.Row():
                    v_ckpt = gr.Dropdown(choices=labels, value=default_ckpt, label='Checkpoint (duplex_100m)')
                    v_refresh = gr.Button('Refresh list')
                    v_lang = gr.Dropdown(choices=['hindi', 'gujarati', 'english'],
                                         value='hindi', label='Language')
                with gr.Row():
                    v_tok = gr.Textbox(value='', label='Tokenizer dir (blank = from checkpoint)')
                    v_whisper = gr.Textbox(value='openai/whisper-small', label='Whisper model')
                    v_mimi = gr.Textbox(value='kyutai/mimi', label='Mimi codec')
                with gr.Row():
                    v_use_brain = gr.Checkbox(value=False, label='Use external LLM brain')
                    v_base = gr.Textbox(value='http://localhost:11434/v1', label='Brain base_url')
                    v_model = gr.Textbox(value='qwen2.5:3b-instruct', label='Brain model')
                v_load = gr.Button('Load voice checkpoint', variant='secondary')
                v_status = gr.Markdown('*Load a duplex_100m checkpoint, then record and speak.*')
                v_audio = gr.Audio(sources=['microphone', 'upload'], type='numpy', label='Your speech')
                with gr.Row():
                    v_go = gr.Button('Talk', variant='primary')
                    v_reset = gr.Button('Reset conversation')
                v_transcript = gr.Markdown()
                v_reply_audio = gr.Audio(label='Assistant reply', autoplay=True)

                v_refresh.click(fn=lambda: _refresh_checkpoints(default_root), outputs=v_ckpt)
                v_load.click(
                    fn=lambda c, tk, w, m, ub, bu, bm: _load_voice_session(
                        c, tk, w, m, ub, bu, bm, default_root),
                    inputs=[v_ckpt, v_tok, v_whisper, v_mimi, v_use_brain, v_base, v_model],
                    outputs=v_status,
                )
                v_go.click(fn=voice_chat, inputs=[v_audio, v_lang],
                           outputs=[v_transcript, v_reply_audio])
                v_reset.click(fn=_reset_voice, outputs=[v_transcript, v_reply_audio])

            with gr.Tab('Turn detection (legacy POC)'):
                gr.Markdown(
                    'Legacy 5M/10M/25M turn-taking checkpoints: predicts '
                    '**listen**, **speak**, or **backchannel**.'
                )
                with gr.Row():
                    checkpoint = gr.Dropdown(choices=labels, value=default_ckpt, label='Checkpoint')
                    refresh = gr.Button('Refresh list')
                    preset = gr.Dropdown(
                        choices=['auto', 'duplex_5m', 'duplex_10m', 'duplex_25m'],
                        value='auto', label='Preset override',
                    )
                with gr.Row():
                    whisper = gr.Textbox(value='openai/whisper-small', label='Whisper model (Stage 1)')
                    root = gr.Textbox(value=default_root, label='Search root', visible=False)
                load_btn = gr.Button('Load checkpoint', variant='secondary')
                status = gr.Markdown('*Select a checkpoint and load, then record audio.*')
                audio = gr.Audio(sources=['microphone', 'upload'], type='numpy', label='Your speech')
                text_context = gr.Textbox(
                    label='Text context (Stage 0 / fallback)',
                    placeholder='Used when checkpoint has no audio encoder weights',
                )
                go = gr.Button('Analyze', variant='primary')
                output = gr.Markdown()

                refresh.click(fn=lambda: _refresh_checkpoints(default_root), outputs=checkpoint)
                load_btn.click(
                    fn=lambda c, p, w: _load_session(c, p, w, default_root),
                    inputs=[checkpoint, preset, whisper],
                    outputs=status,
                )
                go.click(
                    fn=lambda c, p, w, a, t: analyze(c, p, w, default_root, a, t),
                    inputs=[checkpoint, preset, whisper, audio, text_context],
                    outputs=[status, output],
                )

    return demo


def main():
    p = argparse.ArgumentParser(description='V11 duplex Gradio talk demo')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=7860)
    p.add_argument('--share', action='store_true')
    p.add_argument('--root', default='.', help='Repo root to search for checkpoints')
    args = p.parse_args()

    root = str(Path(args.root).resolve())
    demo = build_ui(default_root=root)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()
