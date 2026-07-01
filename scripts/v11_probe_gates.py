#!/usr/bin/env python3
"""Diagnose PAM memory utilization on a trained V11 checkpoint.

Logs, per PAM layer, the learned GSP write-protection probability `p`
(protect_gate) and the decay `gamma` distribution (from dt_proj + dt_bias),
split between high-surprisal "content" tokens and low-surprisal "filler"
tokens. Complements the state-rank probe: rank says how much capacity is used,
the gates say whether the model learned *when* to write vs protect.

Usage:
    uv run python scripts/v11_probe_gates.py \
        --checkpoint checkpoints_v11_e3_k3_chat_pretrain/best_model.pt \
        --preset v11_e3_k3_chat --tokens 4096
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v7.model import cabs, to_real_concat
from v11.model import V11LM, get_config


def _load_eval_tokens(tokenizer, n_tokens: int) -> torch.Tensor:
    """Load a slice of WikiText-103 validation tokens (cached by data loader)."""
    from v7.data import load_wikitext103_val

    val_ds, _ = load_wikitext103_val(seq_len=min(n_tokens, 2048))
    ids = val_ds[0]['input_ids']
    flat = ids.view(-1)[:n_tokens]
    return flat


def _cfg_from_checkpoint(preset: str, ckpt: dict):
    """Build a config from the preset, then overlay the checkpoint's saved config.

    Ensures vocab_size and gate flags (e.g. gate_content_aware) match the trained
    weights exactly, so load_state_dict succeeds regardless of preset drift.
    """
    cfg = get_config(preset)
    saved = ckpt.get('config') or {}
    for k, v in saved.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.dropout = 0.0
    cfg.gradient_checkpointing = False
    return cfg


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description='Probe PAM gates on a trained checkpoint')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--preset', default='v11_e3_k3_chat')
    p.add_argument('--tokens', type=int, default=4096)
    p.add_argument('--surprisal_pct', type=float, default=0.25,
                   help='Top/bottom fraction by next-token loss = content/filler')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from v7.data import get_chat_tokenizer

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = _cfg_from_checkpoint(args.preset, ckpt)

    tokenizer = get_chat_tokenizer()
    if len(tokenizer) != cfg.vocab_size:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  cfg from ckpt: vocab={cfg.vocab_size} gate_content_aware="
          f"{getattr(cfg, 'gate_content_aware', None)}")

    model = V11LM(cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    ids = _load_eval_tokens(tokenizer, args.tokens).to(device).unsqueeze(0)  # [1,T]
    T = ids.shape[1]

    # Per-token next-token loss to classify content (high loss) vs filler (low loss).
    logits, _, _ = model(ids)
    shift_logits = logits[:, :-1].reshape(-1, logits.size(-1)).float()
    shift_labels = ids[:, 1:].reshape(-1)
    per_tok_loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')  # [T-1]
    k = max(1, int((T - 1) * args.surprisal_pct))
    content_idx = torch.topk(per_tok_loss, k).indices
    filler_idx = torch.topk(-per_tok_loss, k).indices

    # Capture each PAM layer's input via forward pre-hooks.
    captured: dict = {}

    def make_hook(i):
        def hook(_module, inputs):
            captured[i] = inputs[0].detach()  # x_in to pam: [B,T,dim,2]
        return hook

    handles = []
    for i, block in enumerate(model.blocks):
        handles.append(block.pam.register_forward_pre_hook(make_hook(i)))
    model(ids)
    for h in handles:
        h.remove()

    print('=' * 88)
    print(f'PAM gate diagnosis  ckpt={args.checkpoint}  tokens={T}  preset={args.preset}')
    print('  p   = mean GSP write-protect prob (sigmoid(protect_gate)); high p => "do not write"')
    print('  gam = mean decay gamma (->1 keeps state, ->0 forgets fast)')
    print('  content = top {:.0%} surprisal tokens, filler = bottom {:.0%}'.format(
        args.surprisal_pct, args.surprisal_pct))
    print('=' * 88)
    header = f"{'layer':>5} | {'p_all':>7} {'p_content':>9} {'p_filler':>9} | {'gam_all':>8} {'dt_bias':>8}"
    print(header)
    print('-' * len(header))

    def _gate_input(pam, x):
        # Match the model: content-aware gate reads real+imag (2*dim); else magnitude.
        if getattr(pam, 'gate_content_aware', getattr(cfg, 'gate_content_aware', False)):
            return to_real_concat(x)
        return cabs(x)

    rows = []
    for i, block in enumerate(model.blocks):
        pam = block.pam
        x = captured[i]  # [1,T,dim,2]
        if pam.use_gsp:
            p_all = torch.sigmoid(pam.protect_gate(_gate_input(pam, x)))  # [1,T,H]
            p_all_t = p_all.mean(dim=-1).squeeze(0)  # [T]
            pa = p_all_t.mean().item()
            pc = p_all_t[content_idx].mean().item()
            pf = p_all_t[filler_idx].mean().item()
        else:
            pa = pc = pf = float('nan')

        # gamma (head-decay path): softplus(dt_proj + dt_bias) -> exp(-dt); GSP blends to p.
        x_flat = to_real_concat(x)
        dt = pam.dt_proj(x_flat)  # [1,T,H] (head mode)
        dt = F.softplus(dt + pam.dt_bias)
        base_gamma = torch.exp(-dt)  # without GSP blend
        if pam.use_gsp:
            p_blend = torch.sigmoid(pam.protect_gate(_gate_input(pam, x))).transpose(1, 2)  # [1,H,T]
            gam = (base_gamma.transpose(1, 2) * (1 - p_blend) + p_blend).mean().item()
        else:
            gam = base_gamma.mean().item()
        dtb = pam.dt_bias.float().mean().item()

        print(f"{i:>5} | {pa:>7.3f} {pc:>9.3f} {pf:>9.3f} | {gam:>8.4f} {dtb:>8.3f}")
        rows.append((i, pa, pc, pf, gam, dtb))

    # Summary
    valid_p = [r[1] for r in rows if r[1] == r[1]]
    if valid_p:
        print('-' * len(header))
        print(f"  mean p across layers: {sum(valid_p)/len(valid_p):.3f}  "
              f"(init bias -3.0 => p~0.047; higher = model learned to protect)")
        # content should be protected MORE than filler if gate is working
        deltas = [r[2] - r[3] for r in rows if r[2] == r[2]]
        if deltas:
            md = sum(deltas) / len(deltas)
            print(f"  mean (p_content - p_filler): {md:+.3f}  "
                  f"(positive = protects content more than filler = healthy)")


if __name__ == '__main__':
    main()
