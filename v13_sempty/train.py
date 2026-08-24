"""Self-contained trainer for v13_sempty (no V7Trainer).

AdamW (betas 0.9/0.95), 2-D-only weight decay, warmup-cosine, grad-clip,
optional AMP, fused CE + gate-surprisal BCE. Dataset/tokenizer loading is
reused from ``v7.data``; the step loop lives here.

Default device is CPU so a live GPU training run is not disturbed.
Use ``--device cuda`` explicitly if a free GPU is available.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from v13_sempty.config import PRESETS, get_config
from v13_sempty.model import V13LM

_NO_DECAY_SUFFIXES = {'dt_bias'}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_param_groups(model: nn.Module, weight_decay: float):
    """Split into decay/no-decay groups (same rules as v7.data.build_param_groups)."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        suffix = name.split('.')[-1]
        if suffix in _NO_DECAY_SUFFIXES:
            no_decay.append(param)
        elif suffix in ('bias', 'bias_real', 'bias_imag'):
            no_decay.append(param)
        elif param.dim() >= 2 and suffix in ('weight', 'weight_real', 'weight_imag'):
            decay.append(param)
        else:
            no_decay.append(param)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


def build_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    total_steps = max(total_steps, 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def resolve_amp_dtype(amp_dtype_str: str, device: torch.device):
    if device.type != 'cuda':
        return None
    if amp_dtype_str == 'off':
        return None
    if amp_dtype_str == 'bf16':
        return torch.bfloat16
    if amp_dtype_str == 'fp16':
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def gate_surprisal_loss(gate_probs, nll, labels, loss_mask, m_cfg):
    """BCE between per-layer protect prob and a surprisal-derived target."""
    valid = labels != -100
    if loss_mask is not None:
        valid = valid & (loss_mask > 0)
    finite = torch.isfinite(nll)
    valid = valid & finite
    if valid.any():
        median_ce = nll[valid].median()
    else:
        median_ce = torch.zeros((), device=nll.device, dtype=nll.dtype)
    tau = max(getattr(m_cfg, 'gate_surprisal_tau', 1.0), 1e-3)
    sign = getattr(m_cfg, 'gate_surprisal_sign', 1.0)
    target_p = torch.sigmoid(sign * (median_ce - nll) / tau).detach()
    target_p = torch.nan_to_num(target_p, nan=0.5, posinf=1.0, neginf=0.0)
    gp = gate_probs.float().clamp(1e-4, 1 - 1e-4)
    target = target_p.float().unsqueeze(0).expand_as(gp)
    vmask = valid.unsqueeze(0).expand_as(gp).to(gp.dtype)
    with torch.amp.autocast(device_type=gp.device.type, enabled=False):
        bce = F.binary_cross_entropy(gp, target, reduction='none')
    return (bce * vmask).sum() / vmask.sum().clamp_min(1.0)


class Trainer:
    def __init__(
        self,
        model: V13LM,
        train_loader,
        *,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 10,
        total_steps: int = 100,
        gradient_clip: float = 1.0,
        amp_dtype_str: str = 'off',
        fused_ce: bool = True,
        fused_ce_chunk: int = 4096,
        device: Optional[torch.device] = None,
        log_interval: int = 1,
    ):
        self.model = model
        self.train_loader = train_loader
        self.gradient_clip = gradient_clip
        self.fused_ce = fused_ce
        self.fused_ce_chunk = fused_ce_chunk
        self.log_interval = log_interval
        self.device = device or torch.device('cpu')
        self.model.to(self.device)

        groups = build_param_groups(model, weight_decay)
        self.optimizer = torch.optim.AdamW(
            groups, lr=learning_rate, betas=(0.9, 0.95),
            fused=(self.device.type == 'cuda'),
        )
        self.scheduler = build_lr_scheduler(self.optimizer, warmup_steps, total_steps)
        self.amp_dtype = resolve_amp_dtype(amp_dtype_str, self.device)
        self.use_amp = self.amp_dtype is not None
        self.scaler = (
            torch.amp.GradScaler('cuda')
            if self.use_amp and self.amp_dtype == torch.float16
            else None
        )
        self.global_step = 0
        self._last_gate_loss = 0.0

    def _step_loss(self, input_ids, labels, loss_mask=None):
        cfg = self.model.config
        if self.fused_ce:
            lm, aux_loss, gate_probs = self.model._hidden_to_lm(input_ids)
            main = self.model.ce_from_lm(
                lm, labels, loss_mask=loss_mask, chunk=self.fused_ce_chunk,
                return_nll=True,
            )
            main_loss, nll = main
        else:
            logits, _, aux_loss = self.model(input_ids, labels=labels)
            main_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1),
            )
            gate_probs, nll = None, None
        loss = main_loss
        if aux_loss.detach().abs().item() > 0:
            loss = loss + getattr(cfg, 'aux_loss_weight', 1.0) * aux_loss
        gsl = getattr(cfg, 'gate_surprisal_lambda', 0.0)
        if self.fused_ce and gate_probs is not None and gsl > 0 and nll is not None:
            gate_loss = gate_surprisal_loss(gate_probs, nll, labels, loss_mask, cfg)
            loss = loss + gsl * gate_loss
            self._last_gate_loss = float(gate_loss.detach())
        return loss, main_loss

    def step(self, batch) -> float:
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        loss_mask = batch.get('loss_mask')
        if loss_mask is not None:
            loss_mask = loss_mask.to(self.device)

        with torch.amp.autocast(
            self.device.type,
            enabled=self.use_amp,
            dtype=self.amp_dtype or torch.float16,
        ):
            loss, main_loss = self._step_loss(input_ids, labels, loss_mask)

        if not torch.isfinite(loss).all():
            raise RuntimeError(f"non-finite loss at step {self.global_step}: {loss}")

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        return float(main_loss.detach())

    def train(self, max_steps: Optional[int] = None) -> list:
        self.model.train()
        losses = []
        for batch in self.train_loader:
            loss = self.step(batch)
            losses.append(loss)
            if self.log_interval and self.global_step % self.log_interval == 0:
                print(
                    f"step {self.global_step}  loss={loss:.4f}  "
                    f"lr={self.optimizer.param_groups[0]['lr']:.2e}  "
                    f"gate={self._last_gate_loss:.4f}",
                    flush=True,
                )
            if max_steps is not None and self.global_step >= max_steps:
                break
        return losses


def synthetic_loader(vocab_size: int, batch_size: int, seq_len: int, n_batches: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab_size, (n_batches * batch_size, seq_len), generator=g)
    labels = torch.randint(0, vocab_size, (n_batches * batch_size, seq_len), generator=g)
    ds = TensorDataset(ids, labels)

    def _collate(rows):
        x = torch.stack([r[0] for r in rows])
        y = torch.stack([r[1] for r in rows])
        return {'input_ids': x, 'labels': y}

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)


def load_real_dataset(name: str, seq_len: int, max_samples: int):
    from v7.data import load_tinystories, load_wikitext103
    if name == 'tinystories':
        train_ds, val_ds, tok = load_tinystories(max_samples=max_samples, seq_len=seq_len)
    elif name == 'wikitext103':
        train_ds, val_ds, tok = load_wikitext103(max_samples=max_samples, seq_len=seq_len)
    else:
        raise ValueError(f"unknown dataset {name}")
    return train_ds, val_ds, tok


def build_argparser():
    p = argparse.ArgumentParser(description='v13_sempty self-contained trainer')
    p.add_argument('--preset', type=str, default='tiny', choices=list(PRESETS.keys()))
    p.add_argument('--dataset', type=str, default='synthetic',
                   choices=['synthetic', 'tinystories', 'wikitext103'])
    p.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--seq_len', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_steps', type=int, default=2)
    p.add_argument('--gradient_clip', type=float, default=1.0)
    p.add_argument('--amp_dtype', type=str, default='off',
                   choices=['off', 'auto', 'bf16', 'fp16'])
    p.add_argument('--fused_ce', action='store_true', default=True)
    p.add_argument('--no_fused_ce', action='store_true')
    p.add_argument('--fused_ce_chunk', type=int, default=4096)
    p.add_argument('--max_samples', type=int, default=64)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints_v13_sempty')
    return p


def main():
    args = build_argparser().parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA requested but not available; falling back to CPU')
        args.device = 'cpu'
    seed_everything(args.seed)
    cfg = get_config(args.preset)
    cfg.gradient_checkpointing = False
    if args.seq_len:
        cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)

    device = torch.device(args.device)
    print(f"device={device} preset={args.preset} dataset={args.dataset}")

    if args.dataset == 'synthetic':
        vocab = min(cfg.vocab_size, 256)
        cfg.vocab_size = vocab
        loader = synthetic_loader(vocab, args.batch_size, args.seq_len, args.steps, args.seed)
        tokenizer = None
    else:
        train_ds, _, tokenizer = load_real_dataset(
            args.dataset, args.seq_len, args.max_samples,
        )
        tok_vocab = len(tokenizer)
        if tok_vocab != cfg.vocab_size:
            print(f"Adjusting vocab_size: {cfg.vocab_size} -> {tok_vocab}")
            cfg.vocab_size = tok_vocab
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = V13LM(cfg)
    params = model.count_parameters()
    print(f"params: {params['total']:,} ({params['total']/1e6:.2f}M)")

    fused = args.fused_ce and not args.no_fused_ce
    trainer = Trainer(
        model, loader,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        total_steps=args.steps,
        gradient_clip=args.gradient_clip,
        amp_dtype_str=args.amp_dtype,
        fused_ce=fused,
        fused_ce_chunk=args.fused_ce_chunk,
        device=device,
    )
    losses = trainer.train(max_steps=args.steps)
    print(f"losses: {[round(x, 4) for x in losses]}")
    if len(losses) >= 2:
        print(f"delta loss (last-first) = {losses[-1] - losses[0]:+.4f}")
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'config': asdict(cfg),
            'losses': losses,
        },
        ckpt_dir / 'latest.pt',
    )
    print(f"saved {ckpt_dir / 'latest.pt'}")


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
