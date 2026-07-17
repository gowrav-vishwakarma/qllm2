"""
V12 training entrypoint.

Reuses the V7 trainer (`v7.train.V7Trainer`) with extended data pipeline for
Phase C (DCLM-Edu pretrain + SmolTalk2 SFT).

Usage:
    .venv/bin/python -m v12.train --preset tiny --epochs 1 --max_samples 200 \
        --dataset tinystories --num_workers 2 --gen_every 0           # smoke
    .venv/bin/python -m v12.train --preset v12_e3_k3 --stage pretrain \
        --dataset dclm_edu --token_budget 2000000000 --resume_from ckpt.pt
    .venv/bin/python -m v12.train --preset v12_e3_k3 --stage sft \
        --dataset smoltalk2 --resume_from ckpt.pt --lr 5e-5 --epochs 1

Note: default batch sizes here target a 24GB RTX-4090 (local dev), not the
96GB server. Scale --batch_size / grad-accum up on the server.
"""

import argparse
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from v12.model import V12LM, get_config, PRESETS
from v12.losses import apply_stage_loss, get_stage_loss, stage_loss_names
from v7.train import (
    V7Trainer, seed_everything, _notify_training_failure, _is_oom_error,
    _notify_discord, _notify_discord_long,
)
from v7.data import (
    load_wikitext103,
    load_wikitext103_val,
    load_tinystories,
    load_dclm_edu,
    load_fineweb_edu,
    load_pretrain_mix,
    load_smoltalk2,
    load_tulu3_sft,
    TeeLogger,
)


def build_argparser():
    p = argparse.ArgumentParser(description='V12 PAM (new memory dynamics) training')
    p.add_argument('--preset', type=str, default='v12_baseline', choices=list(PRESETS.keys()))
    p.add_argument('--dataset', type=str, default='wikitext103',
                   choices=['wikitext103', 'tinystories', 'dclm_edu', 'fineweb_edu',
                            'pretrain_mix', 'smoltalk2', 'tulu3', 'fact'])
    p.add_argument('--stage', type=str, default='lm',
                   choices=['lm', 'pretrain', 'sft'],
                   help='lm=legacy WikiText path; pretrain=web stream; sft=chat masked CE')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=4,
                   help='Default 4 targets a 24GB RTX-4090 for the ~100M preset at '
                        'seq_len 2048. Raise to 18-32 on the 96GB server.')
    p.add_argument('--lr', type=float, default=None,
                   help='Default: 1e-4 pretrain/lm, 5e-5 sft')
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_steps', type=int, default=1000)
    p.add_argument('--seq_len', type=int, default=None)
    p.add_argument('--dropout', type=float, default=None)
    p.add_argument('--gradient_clip', type=float, default=1.0)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--compile_mode', type=str, default='default',
                   choices=['default', 'reduce-overhead', 'max-autotune'])
    p.add_argument('--amp_dtype', type=str, default='auto', choices=['auto', 'bf16', 'fp16'])
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--max_samples', type=int, default=9999999)
    p.add_argument('--token_budget', type=int, default=0,
                   help='Stop after this many training tokens (pretrain pilot)')
    p.add_argument('--edu_score_min', type=int, default=3,
                   help='Minimum edu score for DCLM-Edu / FineWeb-Edu rows')
    p.add_argument('--pretrain_sources', type=str, default='dclm,fineweb',
                   help='Comma list for --dataset pretrain_mix '
                        '(subset of: dclm,fineweb,smoltalk2_mid)')
    p.add_argument('--pretrain_weights', type=str, default=None,
                   help='Comma list of mix weights matching --pretrain_sources (e.g. 85,10,5)')
    p.add_argument('--blend_warmup_tokens', type=int, default=0,
                   help='Draw web-only until this many tokens, then apply full blend '
                        '(grammar/knowledge warmup for --dataset pretrain_mix)')
    p.add_argument('--fineweb_name', type=str, default='sample-10BT',
                   help='FineWeb-Edu config (e.g. sample-10BT, sample-100BT)')
    p.add_argument('--holdout_pct', type=int, default=5,
                   help='Hash-bucket holdout %% for in-distribution pretrain val')
    p.add_argument('--dclm_skip_docs', type=int, default=0,
                   help='Skip first N DCLM docs after filters (resume fresh shard)')
    p.add_argument('--fineweb_skip_docs', type=int, default=0,
                   help='Skip first N FineWeb docs after filters (resume fresh shard)')
    p.add_argument('--smoltalk2_mid_skip_rows', type=int, default=0,
                   help='Skip first N smoltalk2 Mid rows (pretrain reasoning blend cursor)')
    p.add_argument('--sft_filter', type=str, default='hard', choices=['none', 'hard'])
    p.add_argument('--think_fraction', type=float, default=0.15,
                   help='Fraction of reasoning (think) conversations to keep in smoltalk2 SFT')
    p.add_argument('--smoltalk2_skip_rows', type=int, default=0,
                   help='Skip first N smoltalk2 SFT rows (SFT stage cursor)')
    p.add_argument('--gen_every', type=int, default=5000)
    p.add_argument('--save_every_steps', type=int, default=0,
                   help='Save latest.pt every N steps (0=off; 5000 for streaming pretrain)')
    p.add_argument('--gen_prompt', type=str, default='In 1923 , the University of')
    p.add_argument('--log_interval', type=int, default=50)
    p.add_argument('--log_dir', type=str, default='logs')
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints_v12')
    p.add_argument('--resume', type=str, default=None,
                   help='Full resume (model + optimizer + scheduler)')
    p.add_argument('--resume_from', type=str, default=None,
                   help='Load model weights only (fresh optimizer; for SFT stage)')
    p.add_argument('--warmstart_chatml', action='store_true',
                   help='After loading weights, seed ChatML token rows (50257/50258) '
                        'from the mean of trained GPT-2 rows (50257 base vocab)')
    p.add_argument('--no_resume_cursor', action='store_true',
                   help='Load resume weights but do NOT auto-seed data skip cursors '
                        '(fast warm-start A/B; accepts data overlap with prior training)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no_grad_ckpt', action='store_true')
    p.add_argument('--fused_ce', action='store_true',
                   help='Memory-lean chunked linear+CE loss (no [B*T,vocab] logits). '
                        'Exact-equivalent; frees ~30GB at B=18/T=2048 for bigger batch.')
    p.add_argument('--fused_ce_chunk', type=int, default=4096,
                   help='Row-chunk size for fused CE (tokens per chunk).')
    p.add_argument('--chunk_size', type=int, default=None)
    p.add_argument('--activation', type=str, default=None,
                   choices=['modrelu', 'swish', 'phase_mod'])
    p.add_argument('--decay_mode', type=str, default=None, choices=['head', 'per_channel'])
    p.add_argument('--write_mode', type=str, default=None, choices=['additive', 'delta'])
    p.add_argument('--n_states', type=int, default=None)
    p.add_argument('--delta_chunk', type=int, default=None)
    p.add_argument('--gate_content_aware', action='store_true',
                   help='GSP write gate reads real+imag (2*dim) vs magnitude-only')
    p.add_argument('--no_gate_content_aware', action='store_true',
                   help='Force magnitude-only GSP gate (ablation baseline)')
    p.add_argument('--base_dt_bias', type=float, default=None,
                   help='Override uniform decay bias (default -4.0)')
    p.add_argument('--protect_gate_bias', type=float, default=None,
                   help='Override GSP protect-gate init bias (default -3.0)')
    p.add_argument('--routing_content_aware', action='store_true',
                   help='E3 phase/score router reads real+imag (2*dim) vs magnitude-only')
    p.add_argument('--no_routing_content_aware', action='store_true',
                   help='Force magnitude-only E3 routing (ablation baseline)')
    p.add_argument('--state_compete', action='store_true',
                   help='E3 magnitude competition via score_proj softmax over K states')
    p.add_argument('--no_state_compete', action='store_true',
                   help='Disable state_compete even if preset enables it')
    p.add_argument('--phase_init', type=str, default=None, choices=['zero', 'spread', 'ortho'],
                   help='E3 phase_proj init: zero (default), spread (0,±2π/3), ortho')
    p.add_argument('--route_balance_lambda', type=float, default=None,
                   help='MoE-style load-balance on batch-mean routing (needs state_compete)')
    p.add_argument('--aux_loss_weight', type=float, default=None,
                   help='Trainer weight for route_balance aux loss (default 1.0)')
    # ── Recall program (V12): memory horizon + gate supervision ──────────────
    p.add_argument('--gamma_floor', type=float, default=None,
                   help='Min base per-step decay (recall program). 0=off, ~0.98 lengthens memory')
    p.add_argument('--gate_surprisal_lambda', type=float, default=None,
                   help='Weight of gate-surprisal aux loss (0=off). Ties protect gate to surprisal')
    p.add_argument('--gate_surprisal_tau', type=float, default=None,
                   help='Temperature (nats) for surprisal->protect target (default 1.0)')
    p.add_argument('--gate_surprisal_sign', type=float, default=None,
                   help='+1: low-surprisal->protect (recall-oriented, default); -1: content->protect')
    # Stage-6 architecture levers
    p.add_argument('--vault_state', action='store_true',
                   help='Pin one K-state to γ=1 (no decay); writes still GSP-gated')
    p.add_argument('--no_vault_state', action='store_true',
                   help='Disable vault_state even if preset enables it')
    p.add_argument('--vault_state_idx', type=int, default=None,
                   help='Which of the K states is the vault (default 0)')
    p.add_argument('--write_phase_address', action='store_true',
                   help='Key-conditioned write phase + matching query phase on read')
    p.add_argument('--no_write_phase_address', action='store_true',
                   help='Disable write_phase_address even if preset enables it')
    # M1: learnable phase-band heads (treat n_heads as a max budget H_max).
    p.add_argument('--head_gate', action='store_true',
                   help='Enable hard-concrete L0 head gates so effective head count is learned')
    p.add_argument('--head_gate_l0_lambda', type=float, default=None,
                   help='Weight of the expected-L0 sparsity penalty on head gates')
    # M2: progressive frozen-head growth curriculum.
    p.add_argument('--active_heads', type=str, default=None,
                   help='Head slots to train this stage: "lo:hi", "0,1,2", or "all". '
                        'Others are frozen (zero grad). Use with weight_decay=0.')
    p.add_argument('--open_heads', type=str, default=None,
                   help='Head slots that CONTRIBUTE this stage ("lo:hi"/"0,1"). '
                        'Reserved (closed) slots output 0. Default: 0..max(active_heads).')
    p.add_argument('--freeze_embeddings', action='store_true',
                   help='M2: freeze token embeddings (and tied LM head) this stage')
    p.add_argument('--freeze_lm_head', action='store_true',
                   help='M2: freeze lm_head_proj / lm_head_norm this stage')
    p.add_argument('--freeze_cgu', action='store_true',
                   help='M2: freeze CGU channel mixers this stage')
    # M4.3: pluggable per-stage training objective (see v12/losses.py).
    p.add_argument('--stage_loss', type=str, default='ce', choices=stage_loss_names(),
                   help='Stage objective profile. ce=plain CE (default); other '
                        'profiles apply config overrides the trainer honors.')
    # M4.2 / M4.5: progressive depth growth. Grow specialist layer group(s) on top
    # of a loaded (frozen) base, in a swappable skill order.
    p.add_argument('--grow_layers', type=str, default=None,
                   help='Grow layers on top of the resumed stack. Shorthand '
                        '"skill:count[:head_budget],skill2:count2" (e.g. '
                        '"facts:4,reasoning:4"), or "@path.json" for full specs.')
    p.add_argument('--layer_head_budget', type=int, default=None,
                   help='Default n_heads for grown layers when a group omits it '
                        '(defaults to the base preset n_heads).')
    p.add_argument('--freeze_layers', type=str, default=None,
                   help='Freeze whole blocks this stage: "lo:hi", "0,1,2", or '
                        '"base" (freeze everything below the grown layers).')
    p.add_argument('--stage_skill', type=str, default=None,
                   help='Skill/group label for --grow_layers entries lacking one.')
    p.add_argument('--attach_mode', type=str, default='sequential',
                   choices=['sequential', 'moe'],
                   help='Composition mode recorded for grown layers. sequential='
                        'always-on depth (default); moe=reserved pack-time router '
                        '(runs sequentially until the router lands).')
    # Playable module system: build the frozen base from the registry instead of
    # a single --resume_from checkpoint.
    p.add_argument('--substrate', type=str, default=None,
                   help='Resolve + assemble a frozen substrate stack from the '
                        'registry before growing. Comma list "id@spec[,id2@spec2]" '
                        '(e.g. "grammar_by_x@1,fact_by_y@>=1"). New layers grow on top.')
    p.add_argument('--registry', type=str, default='v12_registry',
                   help='Registry root dir for --substrate resolution.')
    p.add_argument('--substrate_force', action='store_true',
                   help='Downgrade substrate resolver conflicts (hash/geometry) to warnings.')
    return p


def _resolve_stage_dataset(stage: str, dataset: str) -> str:
    if stage == 'pretrain' and dataset == 'wikitext103':
        return 'dclm_edu'
    if stage == 'sft' and dataset == 'wikitext103':
        return 'smoltalk2'
    return dataset


def _resize_embeddings_for_vocab(model, state):
    """Grow loaded token embeddings to the model's vocab (e.g. +ChatML tokens).

    Weights are tied (lm_head reuses the embedding), so only the two complex
    embedding matrices need resizing. New rows are initialized to match
    ``ComplexEmbed`` (normal, std=0.02); existing rows are copied verbatim.
    """
    target_state = model.state_dict()
    for key in ('embed.embed_real.weight', 'embed.embed_imag.weight'):
        if key not in state or key not in target_state:
            continue
        loaded = state[key]
        target = target_state[key]
        if tuple(loaded.shape) == tuple(target.shape):
            continue
        if loaded.shape[0] < target.shape[0] and loaded.shape[1] == target.shape[1]:
            grown = target.clone()
            torch.nn.init.normal_(grown, std=0.02)
            grown[: loaded.shape[0]] = loaded
            state[key] = grown
            print(
                f"  resized {key}: {tuple(loaded.shape)} -> {tuple(grown.shape)} "
                f"(+{target.shape[0] - loaded.shape[0]} new rows)"
            )
        else:
            raise ValueError(
                f"Cannot resize {key}: loaded {tuple(loaded.shape)} vs "
                f"target {tuple(target.shape)}"
            )


# GPT-2 base vocab before special tokens; rows at this index and above are
# special (im_start=50257, im_end=50258, <think>=50259, </think>=50260).
_CHATML_BASE_VOCAB = 50257
_CHATML_TOKEN_IDS = (50257, 50258, 50259, 50260)


def _warmstart_chatml_embeddings(state, base_vocab: int = _CHATML_BASE_VOCAB):
    """Overwrite untrained ChatML embedding rows with the mean of base-vocab rows.

    Used when a from-scratch ``v12_e3_k3_chat`` pretrain never saw ``<|im_start|>`` /
    ``<|im_end|>`` in raw text, so those tied rows stayed at random init.
    """
    for key in ('embed.embed_real.weight', 'embed.embed_imag.weight'):
        if key not in state:
            continue
        w = state[key]
        if w.shape[0] <= base_vocab:
            continue
        mean_row = w[:base_vocab].mean(dim=0, keepdim=True)
        for idx in _CHATML_TOKEN_IDS:
            if idx < w.shape[0]:
                w[idx] = mean_row.squeeze(0)
        print(
            f"  warmstart ChatML {key}: rows {_CHATML_TOKEN_IDS} "
            f"<- mean of [: {base_vocab}]"
        )


def _drop_shape_mismatches(model, state):
    """Drop checkpoint tensors whose shape differs from the model (e.g. a
    content-aware protect_gate grown from dim -> 2*dim). Those params keep their
    fresh init and re-train; everything else loads normally."""
    target = model.state_dict()
    dropped = []
    for key in list(state.keys()):
        if key in target and tuple(state[key].shape) != tuple(target[key].shape):
            dropped.append((key, tuple(state[key].shape), tuple(target[key].shape)))
            del state[key]
    for key, src, dst in dropped:
        print(f"  reinit {key}: ckpt {src} != model {dst} (kept fresh init)")
    return dropped


def _parse_grow_layers(spec: str, *, default_skill=None, default_budget=None,
                       attach_mode='sequential', head_gate=False,
                       write_phase_address=False):
    """Parse --grow_layers into a list of layer_spec dicts.

    Shorthand: "skill:count[:head_budget]" comma-separated, e.g.
    "facts:4,reasoning:4:8". Or "@path.json" for a JSON list of full specs.
    When ``head_gate`` is set, each grown layer is stamped dynamic-head
    (learnable/prunable head count, per-head phase bands) so ``head_budget`` acts
    as H_max and the group is compacted after training.
    """
    def _stamp(entry):
        entry.setdefault('attach_mode', attach_mode)
        if head_gate:
            entry.setdefault('head_gate', True)
        if write_phase_address:
            entry.setdefault('write_phase_address', True)
        return entry

    spec = spec.strip()
    if spec.startswith('@'):
        import json
        with open(spec[1:], 'r') as f:
            specs = json.load(f)
        if not isinstance(specs, list):
            raise ValueError(f"--grow_layers JSON must be a list, got {type(specs)}")
        return [_stamp(s) for s in specs]
    specs = []
    for group in (g for g in spec.split(',') if g.strip()):
        parts = group.split(':')
        skill = parts[0].strip() or default_skill
        count = int(parts[1]) if len(parts) > 1 and parts[1] != '' else 1
        budget = int(parts[2]) if len(parts) > 2 and parts[2] != '' else default_budget
        for _ in range(count):
            entry = {'skill': skill, 'group_id': skill}
            if budget is not None:
                entry['n_heads'] = budget
            specs.append(_stamp(entry))
    return specs


def _parse_substrate(spec: str):
    """Parse --substrate "id@spec[,id2@spec2]" into [(module_id, version_spec)]."""
    out = []
    for item in (s.strip() for s in spec.split(',') if s.strip()):
        mid, vspec = item.split('@', 1) if '@' in item else (item, '*')
        out.append((mid.strip(), vspec.strip()))
    return out


def _parse_layer_range(spec: str, *, base_len: int, total: int):
    """Parse --freeze_layers into a list of block indices."""
    spec = spec.strip()
    if spec == 'base':
        return list(range(base_len))
    if ':' in spec:
        lo, hi = spec.split(':')
        return list(range(int(lo), int(hi)))
    return [int(i) for i in spec.split(',') if i != '']


def _load_checkpoint_weights(model, path: str, *, warmstart_chatml: bool = False):
    print(f"\nLoading weights from {path}...")
    checkpoint = torch.load(path, weights_only=False)
    state = checkpoint['model_state_dict']
    _resize_embeddings_for_vocab(model, state)
    if warmstart_chatml:
        _warmstart_chatml_embeddings(state)
    dropped = _drop_shape_mismatches(model, state)
    model.load_state_dict(state, strict=not dropped)
    return checkpoint


def main():
    args = build_argparser().parse_args()

    if args.stage == 'sft' and args.lr is None:
        args.lr = 5e-5
    elif args.lr is None:
        args.lr = 1e-4

    args.dataset = _resolve_stage_dataset(args.stage, args.dataset)
    token_budget = args.token_budget if args.token_budget > 0 else None

    env_path = Path(__file__).resolve().parent.parent / '.env'
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and line.startswith('DISCORD_HOOK='):
                os.environ['DISCORD_HOOK'] = line.split('=', 1)[1].strip().strip('\'"')
    if os.environ.get('DISCORD_HOOK'):
        print('[Discord] Webhook configured -- notifications enabled', file=sys.stderr)
    else:
        print('[Discord] No webhook (set DISCORD_HOOK in .env to enable)', file=sys.stderr)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"{args.preset}_{args.stage}_{args.dataset}"
    log_path = log_dir / f'v12_{run_tag}.log'
    tee = TeeLogger(log_path, mode='a' if args.resume else 'w')
    sys.stdout = tee
    print(f"Wall clock start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('=' * 60)
    print('  V12: PAM with new memory dynamics')
    print(f"  Preset: {args.preset} | Stage: {args.stage} | Dataset: {args.dataset}")
    print('=' * 60)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    seed_everything(args.seed)

    cfg = get_config(args.preset)
    if args.seq_len is not None:
        cfg.max_seq_len = args.seq_len
    if args.dropout is not None:
        cfg.dropout = args.dropout
    if args.no_grad_ckpt:
        cfg.gradient_checkpointing = False
    if args.chunk_size is not None:
        cfg.chunk_size = args.chunk_size
    if args.activation is not None:
        cfg.activation = args.activation
    if args.decay_mode is not None:
        cfg.decay_mode = args.decay_mode
    if args.write_mode is not None:
        cfg.write_mode = args.write_mode
    if args.n_states is not None:
        cfg.n_states = args.n_states
    if args.delta_chunk is not None:
        cfg.delta_chunk = args.delta_chunk
    if args.no_gate_content_aware:
        cfg.gate_content_aware = False
    elif args.gate_content_aware:
        cfg.gate_content_aware = True
    if args.base_dt_bias is not None:
        cfg.base_dt_bias = args.base_dt_bias
    if args.protect_gate_bias is not None:
        cfg.protect_gate_bias = args.protect_gate_bias
    if args.no_routing_content_aware:
        cfg.routing_content_aware = False
    elif args.routing_content_aware:
        cfg.routing_content_aware = True
    if args.no_state_compete:
        cfg.state_compete = False
    elif args.state_compete:
        cfg.state_compete = True
    if args.phase_init is not None:
        cfg.phase_init = args.phase_init
    if args.route_balance_lambda is not None:
        cfg.route_balance_lambda = args.route_balance_lambda
    if args.aux_loss_weight is not None:
        cfg.aux_loss_weight = args.aux_loss_weight
    if args.gamma_floor is not None:
        cfg.gamma_floor = args.gamma_floor
    if args.gate_surprisal_lambda is not None:
        cfg.gate_surprisal_lambda = args.gate_surprisal_lambda
    if args.gate_surprisal_tau is not None:
        cfg.gate_surprisal_tau = args.gate_surprisal_tau
    if args.gate_surprisal_sign is not None:
        cfg.gate_surprisal_sign = args.gate_surprisal_sign
    if args.no_vault_state:
        cfg.vault_state = False
    elif args.vault_state:
        cfg.vault_state = True
    if args.vault_state_idx is not None:
        cfg.vault_state_idx = args.vault_state_idx
    if args.no_write_phase_address:
        cfg.write_phase_address = False
    elif args.write_phase_address:
        cfg.write_phase_address = True
    # M1: learnable head count (treat n_heads as a max budget; L0-prune slots).
    if args.head_gate:
        cfg.head_gate = True
    if args.head_gate_l0_lambda is not None:
        cfg.head_gate_l0_lambda = args.head_gate_l0_lambda

    # M4.3: apply the selected stage-loss profile (config overrides the trainer
    # honors). 'ce' is a no-op default.
    _stage_loss = get_stage_loss(args.stage_loss)
    _applied = apply_stage_loss(cfg, args.stage_loss)
    print(f"\nStage loss: {args.stage_loss} — {_stage_loss.description}")
    if _applied:
        print(f"  overrides applied: {_applied}")

    # A recall objective (gate-surprisal / contrastive) only lives on the fused-CE
    # path; without --fused_ce it silently no-ops. Auto-enable it and warn loudly.
    _recall_on = (getattr(cfg, 'gate_surprisal_lambda', 0.0) > 0
                  or getattr(cfg, 'fact_contrastive_lambda', 0.0) > 0)
    if _recall_on and not args.fused_ce:
        print("  [warn] stage loss needs the fused-CE path for its recall aux; "
              "auto-enabling --fused_ce.")
        args.fused_ce = True

    print(f"\nConfig: {asdict(cfg)}")
    print(f"Memory dynamics: decay_mode={cfg.decay_mode}, write_mode={cfg.write_mode}, "
          f"n_states={cfg.n_states}, chunk_size={cfg.chunk_size}")
    print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}, LR: {args.lr}")
    if token_budget:
        print(f"Token budget: {token_budget:,}")

    seq_len = cfg.max_seq_len
    max_samples = args.max_samples if args.max_samples < 9999999 else None
    print(f"\nLoading {args.dataset} (seq_len={seq_len})...")

    per_source_tokens: dict = {}
    per_source_docs: dict = {}
    secondary_val_loader = None
    wiki_val_ds = None

    # Auto-seed skip cursors from a resumed checkpoint so continued pretrain does
    # not re-read consumed docs/rows (per-source freshness). Explicit CLI skips win.
    resume_ckpt_path = args.resume or args.resume_from
    resumed_docs: dict = {}
    if getattr(args, 'no_resume_cursor', False):
        resume_ckpt_path = None  # keep weights, skip cursor auto-seed
    if resume_ckpt_path and Path(resume_ckpt_path).exists():
        try:
            _rc = torch.load(resume_ckpt_path, map_location='cpu', weights_only=False)
            resumed_docs = dict(_rc.get('per_source_docs') or {})
            del _rc
        except Exception as _e:  # noqa: BLE001
            print(f"  [cursor] could not read per_source_docs from resume ckpt: {_e}")
    if resumed_docs:
        if args.dclm_skip_docs == 0 and 'dclm' in resumed_docs:
            args.dclm_skip_docs = int(resumed_docs['dclm'])
        if args.fineweb_skip_docs == 0 and 'fineweb' in resumed_docs:
            args.fineweb_skip_docs = int(resumed_docs['fineweb'])
        if args.smoltalk2_mid_skip_rows == 0 and 'smoltalk2_mid' in resumed_docs:
            args.smoltalk2_mid_skip_rows = int(resumed_docs['smoltalk2_mid'])
        print(f"  [cursor] seeded skips from resume ckpt: {resumed_docs}")

    if args.dataset == 'wikitext103':
        train_ds, val_ds, tokenizer = load_wikitext103(max_samples=max_samples, seq_len=seq_len)
    elif args.dataset == 'tinystories':
        train_ds, val_ds, tokenizer = load_tinystories(
            max_samples=max_samples or 20000, seq_len=seq_len)
    elif args.dataset == 'dclm_edu':
        train_ds, val_ds, tokenizer = load_dclm_edu(
            seq_len=seq_len,
            edu_score_min=args.edu_score_min,
            token_budget=token_budget,
            holdout_pct=args.holdout_pct,
        )
        wiki_val_ds, _ = load_wikitext103_val(seq_len=seq_len)
    elif args.dataset == 'fineweb_edu':
        train_ds, val_ds, tokenizer = load_fineweb_edu(
            seq_len=seq_len,
            edu_score_min=args.edu_score_min,
            token_budget=token_budget,
            chat_vocab=(args.preset.endswith('_chat')),
            fineweb_name=args.fineweb_name,
            holdout_pct=args.holdout_pct,
            dclm_skip_docs=0,
            fineweb_skip_docs=args.fineweb_skip_docs,
            mix_seed=args.seed,
            token_counters=per_source_tokens,
        )
        wiki_val_ds, _ = load_wikitext103_val(seq_len=seq_len)
    elif args.dataset == 'pretrain_mix':
        sources = tuple(s.strip() for s in args.pretrain_sources.split(',') if s.strip())
        weights = (
            tuple(float(w) for w in args.pretrain_weights.split(','))
            if args.pretrain_weights else None
        )
        skip_docs_map = {
            'dclm': args.dclm_skip_docs,
            'fineweb': args.fineweb_skip_docs,
            'smoltalk2_mid': args.smoltalk2_mid_skip_rows,
        }
        train_ds, val_ds, tokenizer = load_pretrain_mix(
            seq_len=seq_len,
            edu_score_min=args.edu_score_min,
            token_budget=token_budget,
            sources=sources,
            weights=weights,
            chat_vocab=(args.preset.endswith('_chat')),
            fineweb_name=args.fineweb_name,
            holdout_pct=args.holdout_pct,
            mix_seed=args.seed,
            skip_docs=skip_docs_map,
            blend_warmup_tokens=args.blend_warmup_tokens,
            token_counters=per_source_tokens,
            doc_counters=per_source_docs,
        )
        wiki_val_ds, _ = load_wikitext103_val(seq_len=seq_len)
    elif args.dataset == 'smoltalk2':
        train_ds, val_ds, tokenizer = load_smoltalk2(
            seq_len=seq_len,
            max_samples=max_samples,
            sft_filter=args.sft_filter,
            think_fraction=args.think_fraction,
            smoltalk2_skip_rows=args.smoltalk2_skip_rows,
        )
    elif args.dataset == 'tulu3':
        train_ds, val_ds, tokenizer = load_tulu3_sft(
            seq_len=seq_len,
            max_samples=max_samples,
            sft_filter=args.sft_filter,
        )
    elif args.dataset == 'fact':
        from v12.fact_data import load_fact_recall
        train_ds, val_ds, tokenizer = load_fact_recall(
            seq_len=seq_len,
            token_budget=token_budget,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Match model vocab to the data tokenizer (ChatML + reasoning specials -> 50261).
    tok_vocab = len(tokenizer)
    if tok_vocab != cfg.vocab_size:
        print(f"Adjusting vocab_size: {cfg.vocab_size} -> {tok_vocab} (tokenizer)")
        cfg.vocab_size = tok_vocab
    if args.preset.endswith('_chat') and tok_vocab != 50261:
        raise ValueError(
            f"Chat preset expects the ChatML+reasoning tokenizer (vocab 50261), "
            f"got {tok_vocab}. Check get_chat_tokenizer()."
        )

    from torch.utils.data import DataLoader
    use_cuda = torch.cuda.is_available()
    is_streaming = (
        args.dataset in ('dclm_edu', 'fineweb_edu', 'pretrain_mix')
        and not getattr(train_ds, 'pretrain_cached', False)
    )
    nw = 0 if is_streaming else (args.num_workers if use_cuda else 0)
    dl_kwargs = {}
    if nw > 0:
        dl_kwargs['persistent_workers'] = True
        dl_kwargs['prefetch_factor'] = 4
    gen = torch.Generator()
    gen.manual_seed(args.seed)

    def _wi(wid):
        np.random.seed(args.seed + wid)
        random.seed(args.seed + wid)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=not is_streaming,
        num_workers=nw,
        pin_memory=use_cuda,
        generator=gen,
        worker_init_fn=_wi if nw > 0 else None,
        **dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(nw, 2) if nw > 0 else 0,
        pin_memory=use_cuda,
        worker_init_fn=_wi if nw > 0 else None,
    )
    if wiki_val_ds is not None:
        secondary_val_loader = DataLoader(
            wiki_val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=use_cuda,
        )
        print(
            f"Val: in-distro holdout ({len(val_loader)} batches), "
            f"WikiText secondary ({len(secondary_val_loader)} batches)"
        )
    try:
        n_train_batches = len(train_loader)
    except TypeError:
        n_train_batches = None
    print(f"Train batches: {n_train_batches or 'streaming'}, Val batches: {len(val_loader)}")

    if args.substrate:
        from v12.registry import Registry, resolve_stack
        from v12.pack import assemble_plan
        reg = Registry(args.registry)
        plan = resolve_stack(_parse_substrate(args.substrate), reg,
                             force=args.substrate_force)
        for w in plan.report['warnings']:
            print(f"[substrate warn] {w}")
        model, sub_cfg = assemble_plan(plan)
        if sub_cfg.vocab_size != cfg.vocab_size:
            print(f"[substrate] WARNING: substrate vocab_size={sub_cfg.vocab_size} != "
                  f"tokenizer/preset vocab_size={cfg.vocab_size}")
        cfg = model.config  # adopt substrate architecture (dim/vocab/specs)
        print(f"[substrate] {plan.report['target']} -> {plan.order()} "
              f"({plan.report['total_layers']} frozen base layers)")
    else:
        model = V12LM(cfg)
    params = model.count_parameters()
    print(f"\nModel parameters: {params}")
    print(f"Total: {params['total']:,} ({params['total']/1e6:.1f}M)")

    start_epoch = 0
    checkpoint = None
    if args.substrate and (args.resume or args.resume_from):
        print("[substrate] note: --substrate provides the base; ignoring "
              "--resume/--resume_from weight load for the base stack.")
    elif args.resume:
        checkpoint = _load_checkpoint_weights(
            model, args.resume, warmstart_chatml=args.warmstart_chatml,
        )
        if token_budget:
            start_epoch = checkpoint.get('epoch', 0)
        else:
            start_epoch = checkpoint.get('epoch', 0) + 1
    elif args.resume_from:
        checkpoint = _load_checkpoint_weights(
            model, args.resume_from, warmstart_chatml=args.warmstart_chatml,
        )

    # M4.2/4.5: progressive DEPTH growth. Grow specialist layer group(s) on top of
    # the loaded (frozen) base stack, then freeze the base so only new layers learn.
    if args.grow_layers:
        base_len = len(model.blocks)
        grow_specs = _parse_grow_layers(
            args.grow_layers,
            default_skill=args.stage_skill,
            default_budget=(args.layer_head_budget if args.layer_head_budget is not None
                            else cfg.n_heads),
            attach_mode=args.attach_mode,
            head_gate=args.head_gate,
            write_phase_address=args.write_phase_address,
        )
        new_idx = model.grow_layers(grow_specs)
        print(f"[M4 grow] +{len(new_idx)} layers {new_idx} on base of {base_len} "
              f"(skills={[s.get('skill') for s in grow_specs]}, "
              f"attach_mode={args.attach_mode}); total layers={len(model.blocks)}")
        if args.freeze_layers is None:
            # Default: freeze the whole base so only the grown group trains.
            args.freeze_layers = 'base'
        # Keep the trainer's config in sync for checkpoint provenance.
        cfg.layer_specs = model.config.layer_specs
        cfg.n_layers = model.config.n_layers

    if args.freeze_layers:
        # For "base", freeze everything below the grown group (new_idx[0]); with no
        # growth this stage, "base" would freeze the whole stack (rarely useful).
        base_len = new_idx[0] if args.grow_layers else len(model.blocks)
        frozen = _parse_layer_range(
            args.freeze_layers, base_len=base_len, total=len(model.blocks),
        )
        model.freeze_layers(frozen)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[M4 freeze] froze blocks {frozen}; trainable params={n_trainable:,}. "
              f"(requires_grad=False => excluded from optimizer, no wd drift)")

    # M2: progressive frozen-head growth. --active_heads picks which head slots
    # train this stage; the rest are frozen (zero grad on their fused slices).
    if args.active_heads is not None and args.active_heads != 'all':
        def _parse_heads(spec):
            if ':' in spec:
                lo, hi = spec.split(':')
                return list(range(int(lo), int(hi)))
            return [int(i) for i in spec.split(',') if i != '']
        active = _parse_heads(args.active_heads)
        open_idx = _parse_heads(args.open_heads) if args.open_heads else None
        model.set_stage_active_heads(
            active,
            open_indices=open_idx,
            freeze_embeddings=args.freeze_embeddings,
            freeze_lm_head=args.freeze_lm_head,
            freeze_cgu=args.freeze_cgu,
        )
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[M2 stage] active head slots={active} "
              f"(of {cfg.n_heads}); trainable params={n_trainable:,}. "
              f"weight_decay={args.weight_decay} (use 0 to protect frozen slices)")

    if token_budget:
        est_steps = token_budget // max(args.batch_size * seq_len, 1) + args.warmup_steps
        max_epochs = max(args.epochs, 9999)
    else:
        est_steps = None
        max_epochs = args.epochs

    trainer = V7Trainer(
        model, train_loader, val_loader, tokenizer,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_clip=args.gradient_clip,
        max_epochs=max_epochs,
        checkpoint_dir=args.checkpoint_dir,
        amp_dtype_str=args.amp_dtype,
        compile_model=args.compile,
        compile_mode=args.compile_mode,
        gen_every=args.gen_every,
        gen_prompt=args.gen_prompt,
        log_interval=args.log_interval,
        save_every_steps=args.save_every_steps,
        start_epoch=start_epoch,
        run_label=f'V12/{args.preset}/{args.stage}',
        log_path=str(log_path),
        token_budget=token_budget,
        total_steps_override=est_steps,
        secondary_val_loader=secondary_val_loader,
        per_source_tokens=per_source_tokens,
        per_source_docs=per_source_docs,
        fused_ce=args.fused_ce,
        fused_ce_chunk=args.fused_ce_chunk,
    )
    if checkpoint and args.resume and 'optimizer_state_dict' in checkpoint:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.global_step = checkpoint.get('global_step', 0)
        trainer.global_tokens = checkpoint.get('global_tokens', 0)
        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        trainer.best_val_ppl = checkpoint.get('best_val_ppl', float('inf'))
        trainer.best_wiki_val_loss = checkpoint.get('best_wiki_val_loss', float('inf'))
        trainer.best_wiki_val_ppl = checkpoint.get('best_wiki_val_ppl', float('inf'))
        saved_src = checkpoint.get('per_source_tokens') or {}
        trainer.per_source_tokens.update(saved_src)
        per_source_tokens.update(saved_src)
        saved_docs = checkpoint.get('per_source_docs') or {}
        trainer.per_source_docs.update(saved_docs)

    _summary = [
        f"Host: {os.uname().nodename}",
        f"Preset: {args.preset} | Stage: {args.stage} | Dataset: {args.dataset}",
        f"Memory dynamics: decay_mode={cfg.decay_mode} | write_mode={cfg.write_mode} "
        f"| n_states={cfg.n_states} | chunk_size={cfg.chunk_size}",
        f"Complex dim: {cfg.dim} | Layers: {cfg.n_layers} | "
        f"PAM heads={cfg.n_heads} d={cfg.head_dim} | CGU expand={cfg.expand}",
        f"Activation: {cfg.activation} | RoPE: {cfg.use_rope} | GSP: {cfg.use_gsp} | "
        f"grad_ckpt: {cfg.gradient_checkpointing}",
        f"Params: {params['total']:,} ({params['total']/1e6:.1f}M)",
        f"Epochs: {args.epochs} | Batch: {args.batch_size} | "
        f"Batches/epoch: {n_train_batches or 'stream'} | seq_len={cfg.max_seq_len}",
        f"LR: {args.lr} | warmup={args.warmup_steps} | wd={args.weight_decay} | "
        f"grad_clip={args.gradient_clip} | dropout={cfg.dropout}",
        f"Token budget: {token_budget or 'none'} | edu_score_min: {args.edu_score_min} | "
        f"sft_filter: {args.sft_filter}",
        f"Save every: {args.save_every_steps} steps | AMP: {args.amp_dtype} | Compile: {args.compile}",
        f"Resume: {args.resume or 'none'} | Weights from: {args.resume_from or 'scratch'} | "
        f"warmstart_chatml: {args.warmstart_chatml}",
        f"Log: {log_path.resolve()}",
        f"Checkpoint dir: {Path(args.checkpoint_dir).resolve()}",
    ]
    _hdr = '**V12 Training started**' if not args.resume else '**V12 Training resumed**'
    _notify_discord_long(_hdr + '\n```\n' + '\n'.join(_summary) + '\n```')

    trainer.train()
    print(f"\nWall clock end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout = tee._stdout
    tee.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        _notify_training_failure('stopped by user')
        raise
    except Exception as e:
        _notify_training_failure('OOM' if _is_oom_error(e) else 'failed', e)
        raise
