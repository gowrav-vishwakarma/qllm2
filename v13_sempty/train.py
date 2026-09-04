"""Self-contained trainer for v13_sempty (no V7Trainer).

AdamW (betas 0.9/0.95), 2-D-only weight decay, warmup-cosine, grad-clip,
optional AMP, fused CE. Dataset/tokenizer loading is
reused from ``v7.data``; the step loop lives here.

Default device is CPU so a live GPU training run is not disturbed.
Use ``--device cuda`` explicitly if a free GPU is available.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from v13_sempty.model import LM, _retention_capture
from v13_sempty.config import PRESETS, get_config
from v13_sempty.triton_kernels import kernel_enabled, set_kernel_enabled

_NO_DECAY_SUFFIXES = {'dt_bias'}


def _git_hash() -> str:
    import subprocess
    try:
        root = Path(__file__).resolve().parent.parent
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=root,
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'],
                               cwd=root, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() + ('-dirty' if dirty.stdout.strip() else '')
    except Exception:  # noqa: BLE001
        return 'unknown'


def _print_run_header(args, cfg, model, params, device, loader, val_loader,
                      tokenizer):
    """V13-style rich header at the top of every log.

    Records everything needed to reproduce/interpret the run later: wall clock,
    commit, full CLI args, full model config, geometry (batch/seq/tokens/steps/
    implied-epochs), dataset sizes, parameter breakdown, and the environment
    (device, GPU, AMP, kernel flags). Kept as plain prints so it lands in the
    tee'd .log verbatim.
    """
    bs, sl = args.batch_size, args.seq_len
    tok_per_step = bs * sl
    try:
        steps_per_epoch = len(loader)
    except TypeError:
        steps_per_epoch = None
    epochs_implied = (args.steps / steps_per_epoch) if steps_per_epoch else None
    total_tokens = args.steps * tok_per_step

    bar = '=' * 72
    print(bar)
    print('  v13_sempty trainer  (real-arm PAM, O(1) memory)')
    print(f"  wall clock start : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  commit           : {_git_hash()}")
    print(f"  preset / dataset : {args.preset} / {args.dataset}")
    print(bar)

    # --- environment --------------------------------------------------------
    gpu = 'cpu'
    if device.type == 'cuda' and torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(device)
    print('[env]')
    print(f"  device={device} gpu={gpu} torch={torch.__version__}")
    print(f"  fused_pam={kernel_enabled()} fused_ce={args.fused_ce and not args.no_fused_ce} "
          f"ce_gemm_dtype={args.ce_gemm_dtype} amp={args.amp_dtype} "
          f"grad_ckpt={cfg.gradient_checkpointing}")

    # --- geometry -----------------------------------------------------------
    print('[geometry]')
    print(f"  batch_size={bs} seq_len={sl} tokens/step={tok_per_step:,}")
    _epstr = f"{epochs_implied:.2f}" if epochs_implied is not None else "n/a"
    print(f"  steps={args.steps:,} steps/epoch={steps_per_epoch} "
          f"epochs_implied={_epstr}")
    print(f"  planned_tokens={total_tokens:,}  "
          f"lr={args.lr} warmup={args.warmup_steps} wd={args.weight_decay} "
          f"grad_clip={args.gradient_clip}")

    # --- dataset ------------------------------------------------------------
    print('[data]')
    _tr = None
    try:
        _tr = len(loader.dataset)
    except (AttributeError, TypeError):
        pass
    _va = None
    if val_loader is not None:
        try:
            _va = len(val_loader.dataset)
        except (AttributeError, TypeError):
            pass
    print(f"  train_chunks={_tr} val_chunks={_va} "
          f"recall_frac={args.recall_frac} "
          f"vocab={cfg.vocab_size} tokenizer={'gpt2' if tokenizer else 'synthetic'}")

    # --- params -------------------------------------------------------------
    print('[params]')
    _pm = f"  total={params['total']:,} ({params['total']/1e6:.2f}M dense)"
    if 'cond_mem_table' in params:
        _pm += (f" + {params['cond_mem_table']:,} table "
                f"(total_with_table {params['total_with_table']/1e6:.2f}M)")
    print(_pm)

    # --- ladder (only non-default arch flags) -------------------------------
    _ladder = [k for k in ('short_conv', 'n_states', 'vault', 'delta', 'cond_mem')
               if getattr(cfg, k) not in (False, 1)]
    if _ladder:
        print('[ladder] ' + " ".join(f"{k}={getattr(cfg, k)}" for k in _ladder))

    # --- full config + args (verbatim, for exact repro) ---------------------
    print('[config] ' + str(asdict(cfg)))
    print('[args] ' + str(vars(args)))
    print(bar, flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_param_groups(model: nn.Module, weight_decay: float, learning_rate: float = 1e-4,
                       table_lr_mult: float = 5.0):
    """Split into decay/no-decay groups (same rules as v7.data.build_param_groups).

    A4 cond_mem lookup tables get their own group: no weight decay and lr x5
    (Engram convention). LambdaLR preserves the per-group lr ratio.
    """
    table_ids = model.cond_mem_table_param_ids() if hasattr(model, 'cond_mem_table_param_ids') else set()
    decay, no_decay, table = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in table_ids:
            table.append(param)
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
    groups = [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]
    if table:
        groups.append({'params': table, 'weight_decay': 0.0,
                       'lr': learning_rate * table_lr_mult})
    return groups


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


class Trainer:
    def __init__(
        self,
        model: LM,
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
        ce_gemm_dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        log_interval: int = 1,
        val_loader=None,
        tokenizer=None,
        gen_every: int = 0,
        gen_prompt: str = 'The',
        gen_max_tokens: int = 80,
        save_every_steps: int = 0,
        diag_every: int = 500,
        val_every: int = 0,
        max_val_batches: Optional[int] = None,
        checkpoint_dir: Optional[Path] = None,
        run_label: str = 'v13_sempty',
    ):
        self.model = model
        self.train_loader = train_loader
        self.gradient_clip = gradient_clip
        self.fused_ce = fused_ce
        self.fused_ce_chunk = fused_ce_chunk
        # Training head GEMMs: None = autocast dtype (bf16). Validation always
        # runs the fp32 head so the reported NLL/PPL is exact.
        self.ce_gemm_dtype = ce_gemm_dtype
        self.log_interval = log_interval
        self.device = device or torch.device('cpu')
        self.model.to(self.device)

        groups = build_param_groups(model, weight_decay, learning_rate=learning_rate)
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
        self.global_tokens = 0
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.gen_every = gen_every
        self.gen_prompt = gen_prompt
        self.gen_max_tokens = gen_max_tokens
        self.save_every_steps = save_every_steps
        self.diag_every = diag_every
        self.max_val_batches = max_val_batches
        self.checkpoint_dir = checkpoint_dir
        self.run_label = run_label
        self.best_val_loss = float('inf')
        self.best_val_ppl = float('inf')
        self.val_every = val_every
        self._blocks = self.model.blocks
        self._set_capture(False)
        self._last_grad_norms = None

    def _step_loss(self, input_ids, labels, loss_mask=None):
        """One training step's loss: fused chunked CE, or plain CE fallback."""
        if self.fused_ce:
            lm, _aux_loss = self.model._hidden_to_lm(input_ids)
            return self.model.ce_from_lm(
                lm, labels, loss_mask=loss_mask, chunk=self.fused_ce_chunk,
                gemm_dtype=self.ce_gemm_dtype,
            )
        logits, _, _aux = self.model(input_ids, labels=labels)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

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
            loss = self._step_loss(input_ids, labels, loss_mask)

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
        if self.diag_every > 0 and (self.global_step + 1) % self.diag_every == 0:
            self._last_grad_norms = self._block_grad_norms()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        return float(loss.detach())

    # ── logging / diagnostics (mirrors v7.V7Trainer cadence) ────────────────

    def _set_capture(self, on: bool):
        """Toggle the realized-retention hook on every PAM layer."""
        for b in self._blocks:
            pam = getattr(b, 'pam', None)
            if pam is not None:
                pam.capture_decay = on

    @torch.no_grad()
    def _block_grad_norms(self) -> list:
        """Per-block grad L2, captured while grads are still live."""
        norms = []
        for b in self._blocks:
            g = 0.0
            for p in b.parameters():
                if p.grad is not None:
                    g += float(p.grad.detach().float().pow(2).sum())
            norms.append(g ** 0.5)
        return norms

    @torch.no_grad()
    def _val_loss(self) -> Optional[float]:
        """Token-weighted val NLL over up to max_val_batches batches."""
        if self.val_loader is None or len(self.val_loader) == 0:
            return None
        self.model.eval()
        total, ntok = 0.0, 0
        for i, batch in enumerate(self.val_loader):
            if self.max_val_batches is not None and i >= self.max_val_batches:
                break
            x = batch['input_ids'].to(self.device)
            y = batch['labels'].to(self.device)
            lm, _ = self.model._hidden_to_lm(x)          # fp32, no autocast: exact val
            loss = self.model.ce_from_lm(
                lm, y, loss_mask=batch.get('loss_mask'),
                chunk=self.fused_ce_chunk, gemm_dtype=torch.float32,
            )
            total += float(loss.detach()) * x.numel()
            ntok += x.numel()
        self.model.train()
        return total / max(ntok, 1)

    @torch.no_grad()
    def _generate_sample(self, prompt: str, max_tokens: int) -> str:
        self.model.eval()
        ids = self.tokenizer.encode(prompt)
        x = torch.tensor([ids], device=self.device)
        out = self.model.generate(
            x, max_new_tokens=max_tokens, temperature=0.8, top_k=50,
            top_p=0.9, repetition_penalty=1.2,
        )
        self.model.train()
        return self.tokenizer.decode(out[0].tolist())

    def _save_ckpt(self, name: str):
        if self.checkpoint_dir is None:
            return
        d = Path(self.checkpoint_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / name
        tmp = d / (name + '.tmp')
        ck = self.model.config
        ckpt = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'global_tokens': self.global_tokens,
            'best_val_loss': self.best_val_loss,
            'best_val_ppl': self.best_val_ppl,
            'config': asdict(ck),
        }
        torch.save(ckpt, tmp)
        os.replace(tmp, path)
        print(f"  [checkpoint] step {self.global_step} -> {path}", flush=True)

    def _diagnostics(self) -> dict:
        """Per-layer learnable-scale + retention + grad/weight-norm snapshot.

        This is the 'which variable is helping' panel: cgu_scale (transform
        path), pam_scale (memory path), realized retention (how long the
        notebook holds), and per-block grad/weight L2 (where it's learning).
        """
        # realized retention: one no-grad forward with the capture hook on
        self._set_capture(True)
        _retention_capture.clear()
        try:
            it = iter(self.train_loader)
            try:
                b0 = next(it)
            except StopIteration:
                return {}
            x = b0['input_ids'].to(self.device)
            with torch.no_grad():
                _ = self.model(x)
        finally:
            self._set_capture(False)
        layer_ret = [float(r.mean()) for r in _retention_capture]

        g_norms = self._last_grad_norms
        if g_norms is None or len(g_norms) != len(self._blocks):
            g_norms = [0.0] * len(self._blocks)

        out = {
            'cgu_scale': [],
            'pam_scale': [],
            'dt_bias': [],
            'g_norm': [],
            'w_norm': [],
            'ret': [],
        }
        for i, b in enumerate(self._blocks):
            out['cgu_scale'].append(float(b.cgu_scale.detach()))
            out['pam_scale'].append(float(b.pam_scale.detach()))
            out['dt_bias'].append(float(b.pam.dt_bias.detach().mean()))
            out['g_norm'].append(g_norms[i])
            w = 0.0
            for p in b.parameters():
                w += float(p.detach().float().pow(2).sum())
            out['w_norm'].append(w ** 0.5)
            out['ret'].append(layer_ret[i] if i < len(layer_ret) else float('nan'))
        return out

    def _log_diag(self, d: dict):
        if not d:
            return
        def _row(key, fmt):
            return ' '.join(fmt.format(v) for v in d[key])
        print(
            f"  [diag] step {self.global_step} "
            f"cgu={_row('cgu_scale', '{:+.2f}')}  "
            f"pam={_row('pam_scale', '{:+.2f}')}",
            flush=True,
        )
        print(
            f"  [diag] dtbias={_row('dt_bias', '{:.2f}')}  "
            f"ret={_row('ret', '{:.2f}')}",
            flush=True,
        )
        print(
            f"  [diag] gnorm={_row('g_norm', '{:.1e}')}  "
            f"wnorm={_row('w_norm', '{:.1f}')}",
            flush=True,
        )

    def train(self, max_steps: Optional[int] = None) -> list:
        self.model.train()
        losses = []
        train_start = time.time()
        log_start = time.time()
        log_tokens = 0
        try:
            steps_per_epoch = len(self.train_loader)
        except TypeError:
            steps_per_epoch = None
        # ETA/progress track the whole run when max_steps spans many epochs.
        n_batches = max_steps if max_steps is not None else steps_per_epoch
        # Epoch bookkeeping so the log reads like the V11/V13 per-epoch logs
        # even though we drive the loop off a single global step budget.
        total_epochs = (math.ceil(max_steps / steps_per_epoch)
                        if (max_steps and steps_per_epoch) else None)
        self._steps_per_epoch = steps_per_epoch
        self._total_epochs = total_epochs
        epoch_loss_sum, epoch_loss_n = 0.0, 0
        epoch_start = time.time()

        def _epoch_stream():
            """Re-iterate the loader across epochs (re-shuffles each pass) until
            max_steps; a single pass when max_steps is None."""
            while True:
                for b in self.train_loader:
                    yield b
                if max_steps is None:
                    return

        for batch_idx, batch in enumerate(_epoch_stream()):
            loss = self.step(batch)
            losses.append(loss)
            batch_tokens = batch['input_ids'].numel()
            self.global_tokens += batch_tokens
            log_tokens += batch_tokens
            epoch_loss_sum += loss
            epoch_loss_n += 1

            if self.log_interval and self.global_step % self.log_interval == 0:
                self._log_line(loss, batch_idx, n_batches, train_start,
                               log_start, log_tokens)
                log_start = time.time()
                log_tokens = 0

            if (
                self.gen_every > 0 and self.global_step > 0
                and self.global_step % self.gen_every == 0
                and self.tokenizer is not None
            ):
                try:
                    text = self._generate_sample(self.gen_prompt, self.gen_max_tokens)
                    print(f"  [gen @ step {self.global_step}, "
                          f"{self.global_tokens:,} tok] prompt: {self.gen_prompt}")
                    print(f"    {text[:600]}", flush=True)
                except Exception as e:
                    print(f"  [gen @ step {self.global_step}] failed: {e}", flush=True)

            if (
                self.val_every > 0 and self.global_step > 0
                and self.global_step % self.val_every == 0
                and self.val_loader is not None
            ):
                try:
                    vl = self._val_loss()
                except Exception as e:
                    vl = None
                    print(f"  [val @ step {self.global_step}] failed: {e}", flush=True)
                if vl is not None:
                    ppl = math.exp(min(vl, 20))
                    tag = ''
                    if vl < self.best_val_loss:
                        self.best_val_loss = vl
                        self.best_val_ppl = ppl
                        tag = ' *best*'
                        self._save_ckpt('best_model.pt')
                    print(
                        f"  [val @ step {self.global_step}] "
                        f"val_loss={vl:.4f} val_ppl={ppl:.2f}{tag} "
                        f"(best {self.best_val_ppl:.2f})",
                        flush=True,
                    )

            if (
                self.save_every_steps > 0 and self.global_step > 0
                and self.global_step % self.save_every_steps == 0
            ):
                self._save_ckpt('latest.pt')

            if self.diag_every > 0 and self.global_step > 0 \
                    and self.global_step % self.diag_every == 0:
                try:
                    self._log_diag(self._diagnostics())
                except Exception as e:
                    print(f"  [diag @ step {self.global_step}] failed: {e}", flush=True)

            # Epoch-boundary banner (mirrors the V11/V13 "Epoch N/M" summary so
            # the logs align epoch-for-epoch with the reference tables).
            if steps_per_epoch and self.global_step % steps_per_epoch == 0:
                ep_done = self.global_step // steps_per_epoch
                ep_avg = epoch_loss_sum / max(epoch_loss_n, 1)
                ep_time = time.time() - epoch_start
                print(
                    f"=== Epoch {ep_done}/{total_epochs} done | step "
                    f"{self.global_step} | train_loss={ep_avg:.4f} "
                    f"ppl={math.exp(min(ep_avg, 20)):.2f} | "
                    f"gtok={self.global_tokens:,} | {ep_time/60:.1f} min | "
                    f"best_val_ppl={self.best_val_ppl:.2f} ===",
                    flush=True,
                )
                epoch_loss_sum, epoch_loss_n = 0.0, 0
                epoch_start = time.time()

            if max_steps is not None and self.global_step >= max_steps:
                break
        return losses

    def _log_line(self, loss, batch_idx, n_batches, train_start,
                  log_start, log_tokens):
        lr = self.optimizer.param_groups[0]['lr']
        ppl = math.exp(min(loss, 20))
        elapsed = time.time() - train_start
        avg_tok_s = self.global_tokens / elapsed if elapsed > 0 else 0
        inst_tok_s = log_tokens / max(time.time() - log_start, 1e-9)
        if n_batches:
            pct = 100.0 * (batch_idx + 1) / n_batches
            remaining = elapsed / (batch_idx + 1) * (n_batches - batch_idx - 1)
            eta_m, eta_s = divmod(int(remaining), 60)
            eta_str = f"ETA {eta_m}m{eta_s:02d}s"
            prog = f"[{batch_idx + 1}/{n_batches} {pct:3.0f}%]"
        else:
            eta_str, prog = "ETA n/a", f"[{self.global_step}]"
        # Which epoch this step falls in (1-indexed), matching the V11 logs.
        spe = getattr(self, '_steps_per_epoch', None)
        if spe:
            cur_ep = (self.global_step - 1) // spe + 1
            tot_ep = getattr(self, '_total_epochs', None)
            ep_str = f"ep{cur_ep}/{tot_ep} " if tot_ep else f"ep{cur_ep} "
        else:
            ep_str = ""
        line = (
            f"step {self.global_step} {ep_str}{prog}  loss={loss:.4f} ppl={ppl:.1f} "
            f"lr={lr:.2e} | {inst_tok_s:.0f} tok/s (avg {avg_tok_s:.0f}) "
            f"{eta_str} | gtok={self.global_tokens:,}"
        )
        if self.device.type == 'cuda':
            mem = torch.cuda.memory_allocated() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            line += f" | GPU {mem:.1f}/{peak:.1f}GB"
        print(line, flush=True)


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
    p.add_argument('--ce_gemm_dtype', type=str, default='auto', choices=['auto', 'fp32'],
                   help='head GEMM precision in training: auto = the autocast dtype '
                        '(bf16), fp32 = exact. Validation is always fp32.')
    p.add_argument('--fused_pam', dest='fused_pam', action='store_true', default=True,
                   help='Triton fused PAM scan for the real arm (default on)')
    p.add_argument('--no_fused_pam', dest='fused_pam', action='store_false',
                   help='use the plain-torch PAM scan instead of the Triton kernel')
    p.add_argument('--max_samples', type=int, default=64)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints_v13_sempty')
    p.add_argument('--gradient_checkpointing', action='store_true', default=False,
                   help='recompute blocks in backward; the memory lever for 16-layer runs')
    p.add_argument('--log_interval', type=int, default=50,
                   help='steps between training log lines (v7 default 50)')
    p.add_argument('--val_every', type=int, default=500,
                   help='steps between validation PPL evals (0=off)')
    p.add_argument('--gen_every', type=int, default=2000,
                   help='steps between in-loop generation samples (0=off)')
    p.add_argument('--gen_prompt', type=str, default='In 1923, the University of')
    p.add_argument('--gen_max_tokens', type=int, default=80)
    p.add_argument('--save_every_steps', type=int, default=2000,
                   help='steps between latest.pt checkpoints (0=only final)')
    p.add_argument('--diag_every', type=int, default=2000,
                   help='steps between per-layer diagnostic panels (0=off)')
    p.add_argument('--max_val_batches', type=int, default=128,
                   help='cap on val batches per eval (0=all)')
    p.add_argument('--recall_frac', type=float, default=0.0,
                   help='fraction of TRAIN samples replaced by synthetic recall '
                        'docs (v7._build_recall_doc). 0=off. Val is never mixed.')
    # Architecture-ladder overrides (real arm; see EXPERIMENTS_SEMPY). Each
    # toggles a cfg field on top of the chosen preset.
    p.add_argument('--short_conv', action='store_true', help='A1: depthwise conv on qkv')
    p.add_argument('--n_states', type=int, default=None, help='A2: PAM states per head')
    p.add_argument('--vault', action='store_true', help='A2b: pinned state + protect gate')
    p.add_argument('--delta', action='store_true', help='A3: delta erase/write')
    p.add_argument('--cond_mem', action='store_true', help='A4: conditional n-gram memory')
    return p


def main():
    args = build_argparser().parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA requested but not available; falling back to CPU')
        args.device = 'cpu'
    seed_everything(args.seed)
    cfg = get_config(args.preset)
    cfg.gradient_checkpointing = args.gradient_checkpointing
    if args.seq_len:
        cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)
    # Architecture-ladder overrides.
    if args.short_conv:
        cfg.short_conv = True
    if args.n_states is not None:
        cfg.n_states = args.n_states
    if args.vault:
        cfg.vault = True
    if args.delta:
        cfg.delta = True
    if args.cond_mem:
        cfg.cond_mem = True
    device = torch.device(args.device)
    set_kernel_enabled(args.fused_pam)

    val_loader = None
    if args.dataset == 'synthetic':
        vocab = min(cfg.vocab_size, 256)
        cfg.vocab_size = vocab
        loader = synthetic_loader(vocab, args.batch_size, args.seq_len, args.steps, args.seed)
        tokenizer = None
    else:
        train_ds, val_ds, tokenizer = load_real_dataset(
            args.dataset, args.seq_len, args.max_samples,
        )
        tok_vocab = len(tokenizer)
        if tok_vocab != cfg.vocab_size:
            print(f"Adjusting vocab_size: {cfg.vocab_size} -> {tok_vocab}")
            cfg.vocab_size = tok_vocab
        if args.recall_frac > 0.0:
            from v13_sempty.data_mix import RecallMixDataset
            train_ds = RecallMixDataset(
                train_ds, frac=args.recall_frac, seq_len=args.seq_len,
                tokenizer=tokenizer, seed=args.seed)
            print(f"recall mix: {args.recall_frac:.1%} of train samples "
                  f"are synthetic recall docs (val unmixed)")
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = LM(cfg)
    params = model.count_parameters()
    _print_run_header(args, cfg, model, params, device, loader, val_loader,
                      tokenizer)

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
        ce_gemm_dtype=torch.float32 if args.ce_gemm_dtype == 'fp32' else None,
        device=device,
        log_interval=args.log_interval,
        val_loader=val_loader,
        tokenizer=tokenizer,
        gen_every=args.gen_every,
        gen_prompt=args.gen_prompt,
        gen_max_tokens=args.gen_max_tokens,
        save_every_steps=args.save_every_steps,
        val_every=args.val_every,
        diag_every=args.diag_every,
        max_val_batches=args.max_val_batches if args.max_val_batches > 0 else None,
        checkpoint_dir=Path(args.checkpoint_dir),
        run_label=args.preset,
    )
    losses = trainer.train(max_steps=args.steps)
    print(f"\nTraining complete. steps={trainer.global_step} "
          f"tokens={trainer.global_tokens:,} "
          f"best_val_ppl={trainer.best_val_ppl:.2f}")
    if len(losses) >= 2:
        print(f"train loss: {losses[0]:.4f} -> {losses[-1]:.4f} "
              f"(delta {losses[-1] - losses[0]:+.4f})")
    trainer._save_ckpt('latest.pt')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
